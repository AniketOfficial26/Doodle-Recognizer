"""Image preprocessing utilities for doodle recognition."""

import numpy as np
import cv2
from typing import Tuple


class ImagePreprocessor:
    """Handles image preprocessing for doodle recognition."""
    
    def __init__(self, target_size: Tuple[int, int] = (28, 28)):
        self.target_size = target_size
    
    def preprocess_from_flat(self, image_flat: np.ndarray, width: int, height: int) -> np.ndarray:
        """Main preprocessing method that tries multiple approaches."""
        # Primary: operate directly on normalized float input
        processed = self._preprocess_from_normalized_float(image_flat, width, height)
        if float(processed.max()) > 0.0:
            return processed

        # Fallback 1: OpenCV adaptive (uint8) path
        img_2d = np.array(image_flat, dtype='float32').reshape((height, width))
        img_uint8 = np.clip(img_2d * 255.0, 0, 255).astype(np.uint8)
        processed_cv = self._preprocess_canvas_cv(img_uint8)
        if float(processed_cv.max()) > 0.0:
            return processed_cv

        # Fallback 2: simple resize
        try:
            print("preprocess fallback: both normalized and cv pipelines were empty; using simple resize")
        except Exception:
            pass
        return self._preprocess_simple_resize(img_uint8)
    
    def _preprocess_canvas_cv(self, img: np.ndarray) -> np.ndarray:
        """Convert an arbitrary canvas image to a centered 28x28 grayscale tensor."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = cv2.medianBlur(gray, 3)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If background is white (most pixels white), invert so strokes are white on black
        white_ratio = float(np.mean(thresh == 255))
        if white_ratio > 0.5:
            thresh = cv2.bitwise_not(thresh)
        # Slightly dilate to keep thin strokes visible
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
        # Ensure strokes are white on black background
        if float(np.mean(thresh == 255)) > 0.5:
            thresh = cv2.bitwise_not(thresh)

        cnt_res = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnt_res) == 2:
            contours, _ = cnt_res
        else:
            _, contours, _ = cnt_res
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10:
                x, y, w, h = cv2.boundingRect(c)
                digit = thresh[y:y + h, x:x + w]
            else:
                digit = thresh
        else:
            digit = thresh

        h, w = digit.shape
        scale = max(h, w) / float(max(self.target_size)) if max(self.target_size) > 0 else 1.0
        if scale > 0:
            new_w = max(1, int(round(w / scale)))
            new_h = max(1, int(round(h / scale)))
            digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            digit = cv2.resize(digit, self.target_size, interpolation=cv2.INTER_AREA)

        canvas = np.zeros(self.target_size, dtype=np.uint8)
        x_off = (self.target_size[1] - digit.shape[1]) // 2
        y_off = (self.target_size[0] - digit.shape[0]) // 2
        canvas[y_off:y_off + digit.shape[0], x_off:x_off + digit.shape[1]] = digit

        canvas = canvas.astype('float32') / 255.0
        return canvas.reshape(1, self.target_size[0], self.target_size[1], 1)
    
    def _preprocess_from_normalized_float(self, image_flat: np.ndarray, width: int, height: int) -> np.ndarray:
        """Robust preprocessing using the frontend's normalized float [0,1] data."""
        img = np.array(image_flat, dtype='float32').reshape((height, width))
        img = np.clip(img, 0.0, 1.0)

        max_val = float(img.max())
        if max_val <= 0.01:
            # No signal
            return np.zeros((1, self.target_size[0], self.target_size[1], 1), dtype='float32')

        # Adaptive foreground mask
        thresh = max(0.1, 0.2 * max_val)
        mask = img >= thresh

        if not np.any(mask):
            return np.zeros((1, self.target_size[0], self.target_size[1], 1), dtype='float32')

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y0, y1 = int(y_indices[0]), int(y_indices[-1]) + 1
        x0, x1 = int(x_indices[0]), int(x_indices[-1]) + 1
        cropped = img[y0:y1, x0:x1]

        h, w = cropped.shape
        target_h, target_w = self.target_size
        if h == 0 or w == 0:
            return np.zeros((1, target_h, target_w, 1), dtype='float32')

        # Preserve aspect ratio to fit inside target
        scale = min(target_h / h, target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w), dtype='float32')
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        return canvas.reshape(1, target_h, target_w, 1)
    
    def _preprocess_simple_resize(self, img: np.ndarray) -> np.ndarray:
        """Simple, robust fallback: resize to target size and normalize."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        # If background is white, invert so strokes are bright
        if float(np.mean(small)) > 127:
            small = cv2.bitwise_not(small)
        small = small.astype('float32') / 255.0
        return small.reshape(1, self.target_size[0], self.target_size[1], 1)

# doodle_app.py
"""
Doodle Classifier Application Class
Implements OOP principles with encapsulation and abstraction
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
import requests
from io import BytesIO
from PIL import Image, ImageOps
import json
from typing import Dict, List, Tuple, Any, Optional

from config import (
    MODEL_PATH, CLASS_NAMES, TARGET_IMAGE_SIZE, 
    HUGGINGFACE_API_URL, DEFAULT_API_KEY, ENABLE_DEBUG_LOGGING
)


class DoodleRecognizer:
    """
    Main class for doodle recognition with encapsulation and abstraction
    """
    
    def __init__(self, model_path: str = MODEL_PATH, class_names: List[str] = CLASS_NAMES):
        """Initialize the DoodleRecognizer"""
        self._model_path = model_path
        self._class_names = class_names
        self._target_size = TARGET_IMAGE_SIZE
        self._model = None
        self._api_key = os.environ.get('HUGGINGFACE_API_KEY', DEFAULT_API_KEY)
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Private method to load the trained TensorFlow model
        """
        try:
            self._model = tf.keras.models.load_model(self._model_path)
            print(f"Model loaded successfully from {self._model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self._model = None
    
    def load_model(self) -> bool:
        """
        Public method to reload the model
        Returns True if successful, False otherwise
        
        Returns:
            bool: Success status of model loading
        """
        self._load_model()
        return self._model is not None
    
    def _log_debug_info(self, message: str, data: Any = None) -> None:
        """
        Private method for debug logging
        
        Args:
            message (str): Debug message
            data (Any): Optional data to log
        """
        if ENABLE_DEBUG_LOGGING:
            print(f"DEBUG: {message}")
            if data is not None:
                print(f"DEBUG DATA: {data}")
    
    def _resize_image_array(self, image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Private method to resize image array using PIL
        
        Args:
            image_array (np.ndarray): Input image array
            target_size (Tuple[int, int]): Target size (height, width)
            
        Returns:
            np.ndarray: Resized image array
        """
        try:
            # Convert numpy array to PIL Image
            # Normalize to 0-255 range for PIL
            image_normalized = ((image_array - image_array.min()) * 255 / 
                              (image_array.max() - image_array.min())).astype(np.uint8)
            
            # Create PIL Image
            pil_image = Image.fromarray(image_normalized, mode='L')  # 'L' mode for grayscale
            
            # Resize using PIL
            resized_pil = pil_image.resize(target_size, Image.LANCZOS)
            
            # Convert back to numpy array and normalize to 0-1 range
            resized_array = np.array(resized_pil, dtype=np.float32) / 255.0
            
            return resized_array
            
        except Exception as e:
            self._log_debug_info(f"Error in image resizing: {e}")
            # Fallback: use simple numpy interpolation
            return self._simple_resize(image_array, target_size)
    
    def _simple_resize(self, image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Private method for simple image resizing fallback
        
        Args:
            image_array (np.ndarray): Input image array
            target_size (Tuple[int, int]): Target size (height, width)
            
        Returns:
            np.ndarray: Resized image array
        """
        from scipy.ndimage import zoom
        
        current_shape = image_array.shape
        zoom_factors = (target_size[0] / current_shape[0], target_size[1] / current_shape[1])
        
        try:
            return zoom(image_array, zoom_factors, order=1)  # Linear interpolation
        except ImportError:
            # If scipy is not available, use basic numpy slicing
            self._log_debug_info("Using basic numpy resizing")
            h_indices = np.linspace(0, current_shape[0] - 1, target_size[0]).astype(int)
            w_indices = np.linspace(0, current_shape[1] - 1, target_size[1]).astype(int)
            return image_array[np.ix_(h_indices, w_indices)]

    def _normalize_image_data(self, image_array: np.ndarray) -> np.ndarray:
        """
        Private method to normalize image data
        
        Args:
            image_array (np.ndarray): Input image array
            
        Returns:
            np.ndarray: Normalized image array
        """
        if image_array.mean() < 50:
            self._log_debug_info("WARNING: Image appears to be mostly black/empty")
        
        # Check if data is already normalized
        if image_array.max() <= 1:
            self._log_debug_info("Data already normalized (0-1 range)")
            return image_array.astype('float32')
        else:
            self._log_debug_info("Data needs normalization (0-255 range)")
            return image_array.astype('float32') / 255.0
    
    def _process_raw_pixel_data(self, image_data: Dict[str, Any]) -> np.ndarray:
        """
        Private method to process raw pixel data from canvas
        
        Args:
            image_data (Dict): Dictionary containing image, width, and height
            
        Returns:
            np.ndarray: Processed image array
        """
        try:
            width = int(image_data['width'])
            height = int(image_data['height'])
            pixel_count = width * height
            
            self._log_debug_info(f"Processing raw pixel data: {width}x{height} ({pixel_count} pixels)")
            
            # Convert to numpy array
            pixel_array = np.array(image_data['image'], dtype='float32')
            
            # Validate pixel count
            if len(pixel_array) != pixel_count:
                raise ValueError(f'Pixel count mismatch. Expected {pixel_count}, got {len(pixel_array)}')
            
            self._log_debug_info(f"Input pixel range: {pixel_array.min()}-{pixel_array.max()}")
            
            # Normalize if needed
            if pixel_array.max() > 1.0:
                pixel_array = pixel_array / 255.0
            
            # Reshape to 2D and add padding if needed to make it square
            processed_image = pixel_array.reshape((height, width))
            
            # Calculate padding to make it square
            max_dim = max(width, height)
            pad_width = ((0, max_dim - height), (0, max_dim - width))
            processed_image = np.pad(processed_image, pad_width, mode='constant', constant_values=0)
            
            # Resize to target size using PIL
            processed_image = self._resize_image_array(processed_image, self._target_size)
            
            # Add batch and channel dimensions
            processed_image = processed_image.reshape(1, self._target_size[0], self._target_size[1], 1)
            
            self._log_debug_info(f"Processed image shape: {processed_image.shape}")
            self._log_debug_info(f"Processed image range: {processed_image.min()}-{processed_image.max()}")
            
            return processed_image
            
        except Exception as e:
            self._log_debug_info(f"Error in raw pixel processing: {e}")
            raise
    
    def preprocess_image(self, image_data: Any) -> np.ndarray:
        """
        Public method to preprocess image data for prediction
        Abstracts the complex preprocessing logic
        
        Args:
            image_data: Image data in various formats
            
        Returns:
            np.ndarray: Preprocessed image array ready for prediction
        """
        try:
            # Check if we received raw pixel data with dimensions
            if isinstance(image_data, dict) and 'width' in image_data and 'height' in image_data:
                self._log_debug_info(f"Processing canvas data with dimensions: {image_data['width']}x{image_data['height']}")
                return self._process_raw_pixel_data(image_data)
            
            # Handle standard image data
            self._log_debug_info("Processing standard image data")
            
            # Convert to numpy array and check dimensions
            raw_array = np.array(image_data)
            self._log_debug_info(f"Raw input shape: {raw_array.shape}")
            self._log_debug_info(f"Raw input min/max: {raw_array.min()}/{raw_array.max()}")
            self._log_debug_info(f"Target size: {self._target_size}")
            
            # Check if the input data matches expected dimensions
            expected_pixels = self._target_size[0] * self._target_size[1]
            if raw_array.size == expected_pixels:
                # Data is already the right size, just reshape
                image_array = raw_array.reshape(self._target_size[0], self._target_size[1])
                self._log_debug_info(f"Data matches target size, reshaping to {self._target_size}")
            else:
                # Data needs to be resized
                self._log_debug_info(f"Data size mismatch. Got {raw_array.size} pixels, expected {expected_pixels}")
                
                # Try to infer original dimensions
                if raw_array.size == 28 * 28:
                    # Original 28x28 data, need to resize to 96x96
                    self._log_debug_info("Detected 28x28 input, resizing to 96x96")
                    image_array = raw_array.reshape(28, 28)
                    image_array = self._resize_image_array(image_array, self._target_size)
                elif int(np.sqrt(raw_array.size))**2 == raw_array.size:
                    # Square image, try to reshape and resize
                    original_size = int(np.sqrt(raw_array.size))
                    self._log_debug_info(f"Detected {original_size}x{original_size} input, resizing to {self._target_size}")
                    image_array = raw_array.reshape(original_size, original_size)
                    image_array = self._resize_image_array(image_array, self._target_size)
                else:
                    # Unknown format, try to reshape to target size directly
                    self._log_debug_info("Unknown format, attempting direct reshape")
                    image_array = raw_array.reshape(self._target_size[0], self._target_size[1])
            
            # Normalize the data
            image_array = self._normalize_image_data(image_array)
            
            # Add batch dimension and channel dimension
            image_array = image_array.reshape(1, self._target_size[0], self._target_size[1], 1)
            
            # Final debug logging
            self._log_debug_info(f"Final image shape: {image_array.shape}")
            self._log_debug_info(f"Final image min/max: {image_array.min():.3f}/{image_array.max():.3f}")
            
            return image_array
            
        except Exception as e:
            self._log_debug_info(f"Error in image preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_predictions(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Private method to process raw model predictions
        
        Args:
            predictions (np.ndarray): Raw prediction array from model
            
        Returns:
            Dict[str, Any]: Processed prediction results
        """
        # Get top 3 predictions for better insight
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            top_3_predictions.append({
                'class': self._class_names[idx],
                'confidence': float(predictions[idx])
            })
        
        # Get the best prediction
        predicted_class_index = np.argmax(predictions)
        predicted_class = self._class_names[predicted_class_index]
        confidence = float(predictions[predicted_class_index])
        
        # Debug logging
        self._log_debug_info(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
        top_3_str = [f"{p['class']}({p['confidence']:.3f})" for p in top_3_predictions]
        self._log_debug_info(f"Top 3: {top_3_str}")
        
        return {
            'label': predicted_class,
            'confidence': confidence,
            'top_predictions': top_3_predictions,
            'all_predictions': {
                self._class_names[i]: float(predictions[i]) 
                for i in range(len(self._class_names))
            }
        }
    
    def predict(self, image_data: Any) -> Dict[str, Any]:
        """Predict the class of a doodle image"""
        if self._model is None:
            raise RuntimeError('Model not loaded')
        
        processed_image = self.preprocess_image(image_data)
        predictions = self._model.predict(processed_image, verbose=0)
        return self._process_predictions(predictions[0])
    
    def get_labels(self) -> List[str]:
        """
        Public method to get the list of class labels
        
        Returns:
            List[str]: Copy of class names to prevent external modification
        """
        return self._class_names.copy()
    
    def is_model_loaded(self) -> bool:
        """
        Public method to check if model is successfully loaded
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self._model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Public method to get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information including status and class count
        """
        return {
            'model_loaded': self.is_model_loaded(),
            'model_path': self._model_path,
            'classes': self.get_labels(),
            'classes_count': len(self._class_names),
            'target_size': self._target_size
        }
    
    def _prepare_image_for_api(self, image_data: str) -> bytes:
        """
        Private method to prepare image data for external API
        
        Args:
            image_data (str): Base64 encoded image data
            
        Returns:
            bytes: Processed image bytes for API
        """
        try:
            # Handle base64 data URL format
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode and process image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Resize image to a reasonable size for the model
            image = image.resize((224, 224))
            
            # Save to bytes as JPEG
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return buffered.getvalue()
            
        except Exception as e:
            self._log_debug_info(f"Error preparing image for API: {e}")
            raise
    
    def _generate_confidence_prompt(self, predicted_class: str, confidence: float) -> str:
        """
        Private method to generate prompt based on confidence level
        
        Args:
            predicted_class (str): Predicted class name
            confidence (float): Prediction confidence
            
        Returns:
            str: Generated prompt
        """
        if confidence < 0.3:
            return "I'm not sure what this is. It might be a"
        elif confidence < 0.7:
            return f"This looks like a {predicted_class} but I'm not completely sure. It could be a"
        else:
            return f"This is a {predicted_class}. It looks like a"
    
    def _call_interpretation_api(self, img_bytes: bytes) -> requests.Response:
        """
        Private method to call the Hugging Face API
        
        Args:
            img_bytes (bytes): Image data as bytes
            
        Returns:
            requests.Response: API response
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/octet-stream"
        }
        
        return requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            data=img_bytes,
            params={"wait_for_model": True, "max_new_tokens": 100}
        )
    
    def get_ai_interpretation(self, image_data: str, predicted_class: str, confidence: float) -> str:
        """
        Public method to get AI interpretation of the drawing using external API
        
        Args:
            image_data (str): Base64 encoded image data
            predicted_class (str): Predicted class from the model
            confidence (float): Prediction confidence
            
        Returns:
            str: AI interpretation of the drawing
        """
        try:
            # Check if API is properly configured
            if self._api_key == DEFAULT_API_KEY:
                self._log_debug_info("Warning: Using default API key")
                return "AI analysis is not properly configured. Please set up your Hugging Face API key."
            
            # Prepare image and make API call
            processed_image = self._prepare_image_for_api(image_data)
            response = self._call_interpretation_api(processed_image)
            
            # Generate prompt based on confidence
            prompt = self._generate_confidence_prompt(predicted_class, confidence)
            
            # Process API response
            if response.status_code == 200:
                result = response.json()
                generated_text = None
                
                if isinstance(result, dict) and 'generated_text' in result:
                    generated_text = result['generated_text']
                elif isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                    generated_text = result[0]['generated_text']
                
                if generated_text:
                    return f"{prompt} {generated_text}"
                else:
                    self._log_debug_info(f"Unexpected API response format: {result}")
                    return f"{prompt} but I'm having trouble describing it further."
            else:
                self._log_debug_info(f"BLIP API error ({response.status_code}): {response.text}")
                return f"I'm having trouble analyzing this drawing right now. (API Error: {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            self._log_debug_info(f"Request error in AI interpretation: {str(e)}")
            return "I'm having trouble connecting to the AI service right now. Please try again later."
        except Exception as e:
            self._log_debug_info(f"Error in AI interpretation: {str(e)}")
            return "I encountered an error while analyzing this drawing."
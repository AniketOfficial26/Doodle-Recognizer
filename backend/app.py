from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # Allow server to start without TF for GenAI-only usage
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import uvicorn
import os
import base64
import difflib
from typing import Any

# Load .env if present for easier local configuration
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional: Google Gemini (Generative AI)
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # Library not installed; endpoint will report helpful error


# Initialize FastAPI app
app = FastAPI(
    title="Doodle Recognition API",
    description="FastAPI backend for 28x28 doodle prediction",
    version="1.0.0",
)

# CORS (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once at startup (if TensorFlow is available)
_here = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_here, 'doodle_recognizer_simple.keras')
model = None
if tf is not None:
    try:
        print("Loading Keras model ...")
        model = tf.keras.models.load_model(_model_path)
        print("Model loaded.")
    except Exception as _e:
        print(f"Warning: Failed to load Keras model: {_e}")
else:
    print("TensorFlow not installed; prediction endpoints will be unavailable.")

# Class labels must match training
class_names = ['apple', 'airplane', 'cat', 'car', 'dog', 'flower', 'star', 'tree', 'umbrella', 'fish']


# Schemas
class PredictionRequest(BaseModel):
    image: List[float]
    width: int = 28
    height: int = 28


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    all_predictions: Dict[str, float]


class TestResponse(BaseModel):
    message: str
    model_loaded: bool
    classes: List[str]


class InterpretRequest(BaseModel):
    image: Any
    prediction: str
    confidence: float


class InterpretResponse(BaseModel):
    interpretation: str


class GenAIGuessRequest(BaseModel):
    image: str  # data URL (e.g., "data:image/png;base64,...")
    prompt: str | None = None


class GenAIGuessResponse(BaseModel):
    guess: str


def preprocess_canvas_cv(img: np.ndarray, target_size=(28, 28)) -> np.ndarray:
    """Convert an arbitrary canvas image to a centered 28x28 grayscale tensor.

    Returns array shaped (1, 28, 28, 1) with values in [0,1].
    """
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
    scale = max(h, w) / float(max(target_size)) if max(target_size) > 0 else 1.0
    if scale > 0:
        new_w = max(1, int(round(w / scale)))
        new_h = max(1, int(round(h / scale)))
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        digit = cv2.resize(digit, target_size, interpolation=cv2.INTER_AREA)

    canvas = np.zeros(target_size, dtype=np.uint8)
    x_off = (target_size[1] - digit.shape[1]) // 2
    y_off = (target_size[0] - digit.shape[0]) // 2
    canvas[y_off:y_off + digit.shape[0], x_off:x_off + digit.shape[1]] = digit

    canvas = canvas.astype('float32') / 255.0
    return canvas.reshape(1, target_size[0], target_size[1], 1)


def preprocess_from_flat_with_cv(image_flat: np.ndarray, width: int, height: int) -> np.ndarray:
    # Primary: operate directly on normalized float input to avoid bad thresholds
    processed = preprocess_from_normalized_float(image_flat, width, height, target_size=(28, 28))
    if float(processed.max()) > 0.0:
        return processed

    # Fallback 1: OpenCV adaptive (uint8) path
    img_2d = np.array(image_flat, dtype='float32').reshape((height, width))
    img_uint8 = np.clip(img_2d * 255.0, 0, 255).astype(np.uint8)
    processed_cv = preprocess_canvas_cv(img_uint8)
    if float(processed_cv.max()) > 0.0:
        return processed_cv

    # Fallback 2: simple resize
    try:
        print("preprocess fallback: both normalized and cv pipelines were empty; using simple resize")
    except Exception:
        pass
    return preprocess_simple_resize(img_uint8, target_size=(28, 28))


def preprocess_from_normalized_float(image_flat: np.ndarray, width: int, height: int, target_size=(28, 28)) -> np.ndarray:
    """Robust preprocessing using the frontend's normalized float [0,1] data.

    - Expects background near 0 and strokes near 1
    - Crops to the tight bounding box of foreground and centers on 28x28
    - Preserves aspect ratio
    """
    img = np.array(image_flat, dtype='float32').reshape((height, width))
    img = np.clip(img, 0.0, 1.0)

    max_val = float(img.max())
    if max_val <= 0.01:
        # No signal
        return np.zeros((1, target_size[0], target_size[1], 1), dtype='float32')

    # Adaptive foreground mask
    thresh = max(0.1, 0.2 * max_val)
    mask = img >= thresh

    if not np.any(mask):
        return np.zeros((1, target_size[0], target_size[1], 1), dtype='float32')

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y0, y1 = int(y_indices[0]), int(y_indices[-1]) + 1
    x0, x1 = int(x_indices[0]), int(x_indices[-1]) + 1
    cropped = img[y0:y1, x0:x1]

    h, w = cropped.shape
    target_h, target_w = target_size
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


def preprocess_simple_resize(img: np.ndarray, target_size=(28, 28)) -> np.ndarray:
    """Simple, robust fallback: resize to 28x28 and normalize; ensure strokes are bright.

    Accepts uint8 image (H,W) or (H,W,3/4). Returns (1,28,28,1) float32 in [0,1].
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    # If background is white, invert so strokes are bright
    if float(np.mean(small)) > 127:
        small = cv2.bitwise_not(small)
    small = small.astype('float32') / 255.0
    return small.reshape(1, target_size[0], target_size[1], 1)


# ---------------------- Gemini (Generative AI) Setup ----------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Fast and sufficient for captions/recognition tasks
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        print("Gemini model initialized.")
    except Exception as _e:
        print(f"Warning: Failed to initialize Gemini model: {_e}")
        gemini_model = None
else:
    if genai is None:
        print("google-generativeai not installed; /genai_guess endpoint will be unavailable.")
    else:
        print("GEMINI_API_KEY not set; /genai_guess endpoint will be unavailable.")


def _parse_data_url(data_url: str) -> tuple[str, bytes]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("Expected a data URL string starting with 'data:'")
    try:
        header, b64data = data_url.split(",", 1)
        # header example: data:image/png;base64
        mime = header.split(";")[0][5:] if ";" in header else header[5:]
        return mime, base64.b64decode(b64data)
    except Exception as exc:
        raise ValueError(f"Invalid data URL: {exc}")


def _extract_text_from_genai_result(result: Any) -> str:
    # Try the high-level convenience property first
    try:
        text = getattr(result, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    # Try candidates structure
    try:
        candidates = getattr(result, "candidates", []) or []
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    maybe_text = getattr(part, "text", None)
                    if isinstance(maybe_text, str) and maybe_text.strip():
                        return maybe_text.strip()
    except Exception:
        pass
    return ""


@app.get("/test", response_model=TestResponse)
async def test():
    return TestResponse(message="Backend is working!", model_loaded=model is not None, classes=class_names)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "num_classes": len(class_names)}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available on server (TensorFlow not installed or model failed to load)")
        expected = req.width * req.height
        if len(req.image) != expected:
            raise HTTPException(status_code=400, detail=f"Invalid data length. Expected {expected}, got {len(req.image)}")

        pixel_array = np.array(req.image, dtype='float32')
        # Debug basic stats about incoming data
        try:
            print(f"/predict received {len(req.image)} px, range {pixel_array.min():.3f}-{pixel_array.max():.3f}, mean {pixel_array.mean():.3f}")
        except Exception:
            pass
        processed = preprocess_from_flat_with_cv(pixel_array, req.width, req.height)

        # Log processed stats (avoid rejecting to reduce false negatives)
        try:
            print(f"processed shape {processed.shape}, range {processed.min():.3f}-{processed.max():.3f}, mean {processed.mean():.3f}")
        except Exception:
            pass

        preds = model.predict(processed, verbose=0)
        best_idx = int(np.argmax(preds[0]))
        label = class_names[best_idx]
        confidence = float(preds[0][best_idx])

        top_indices = np.argsort(preds[0])[-3:][::-1]
        top_predictions = [{"class": class_names[i], "confidence": float(preds[0][i])} for i in top_indices]

        return PredictionResponse(
            label=label,
            confidence=confidence,
            top_predictions=top_predictions,
            all_predictions={class_names[i]: float(preds[0][i]) for i in range(len(class_names))},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download_processed")
async def download_processed(req: PredictionRequest):
    try:
        expected = req.width * req.height
        if len(req.image) != expected:
            raise HTTPException(status_code=400, detail=f"Invalid data length. Expected {expected}, got {len(req.image)}")

        processed = preprocess_from_flat_with_cv(np.array(req.image, dtype='float32'), req.width, req.height)
        image_2d = processed[0, :, :, 0]

        img_uint8 = (image_2d * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        return StreamingResponse(BytesIO(buf.getvalue()), media_type="image/png", headers={"Content-Disposition": "attachment; filename=processed.png"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interpret", response_model=InterpretResponse)
async def interpret(req: InterpretRequest):
    try:
        conf = float(req.confidence)
        pred = req.prediction
        if conf < 0.3:
            msg = f"I'm not sure what this is. It might be a {pred}."
        elif conf < 0.7:
            msg = f"This looks like a {pred}, but I'm not completely sure."
        else:
            msg = f"This is a {pred}. Nice drawing!"
        return InterpretResponse(interpretation=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/genai_guess", response_model=GenAIGuessResponse)
async def genai_guess(req: GenAIGuessRequest):
    if gemini_model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini not configured. Please install 'google-generativeai' and set GEMINI_API_KEY environment variable."
            ),
        )
    try:
        mime, image_bytes = _parse_data_url(req.image)
        # Convert to PIL Image for Gemini
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")

        allowed = ", ".join(class_names)
        default_prompt = (
            "You are a doodle recognizer. The image is a simple sketch on a white background. "
            "Choose the best matching label ONLY from this list: "
            f"{allowed}. "
            "Respond with exactly one label from the list (lowercase, no punctuation, no extra words)."
        )
        prompt = (req.prompt or default_prompt).strip()

        result = gemini_model.generate_content([prompt, pil_img])
        text = _extract_text_from_genai_result(result)
        if not text:
            text = "unknown"

        # Normalize and enforce to allowed classes
        normalized = text.lower().strip()
        if normalized not in class_names:
            # Try closest match among allowed classes
            match = difflib.get_close_matches(normalized, class_names, n=1, cutoff=0.5)
            normalized = match[0] if match else "unknown"

        return GenAIGuessResponse(guess=normalized)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")


@app.get("/genai_status")
async def genai_status():
    return {
        "google_generativeai_installed": genai is not None,
        "gemini_api_key_present": bool(GEMINI_API_KEY),
        "gemini_model_ready": gemini_model is not None,
        "model_name": getattr(getattr(gemini_model, "model_name", None), "__str__", lambda: None)() if gemini_model else None,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
from PIL import Image
import cv2
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Doodle Recognition API",
    description="A FastAPI service for recognizing hand-drawn doodles using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
print("Loading model...")
model = tf.keras.models.load_model('doodle_recognizer_simple.keras')
print("Model loaded successfully!")

# Class names
class_names = ['apple', 'airplane', 'cat', 'car', 'dog', 'flower', 'star', 'tree', 'umbrella', 'fish']
print(f"Available classes: {class_names}")

# Pydantic models for request/response
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

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    classes: List[str]
    model_summary: List[str]
    input_shape: List[Any]
    output_shape: List[Any]

class TestModelResponse(BaseModel):
    message: str
    results: Dict[str, Any]

class InterpretRequest(BaseModel):
    image: Any
    prediction: str
    confidence: float

class InterpretResponse(BaseModel):
    interpretation: str

def preprocess_canvas_cv(img: np.ndarray, target_size=(28, 28)):
    """Preprocess a binary/grayscale drawing into 28x28, centered and normalized.

    - Accepts a NumPy image (H,W) or (H,W,3/4)
    - Returns (1,28,28,1) float32 in [0,1]
    """
    # 1) Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2) Noise reduction
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Otsu threshold with robust polarity handling
    #    We prefer sparse white strokes on a black background.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.mean(thresh == 255))
    if white_ratio > 0.5:
        # If the canvas is mostly white, invert to make strokes white and background black
        thresh = cv2.bitwise_not(thresh)

    # 4) Crop to largest contour (handle OpenCV 3 vs 4 return signatures)
    cnt_res = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt_res) == 2:
        contours, _hier = cnt_res
    else:
        _img, contours, _hier = cnt_res
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            x, y, w, h = cv2.boundingRect(c)
            digit = thresh[y:y + h, x:x + w]
        else:
            digit = thresh
    else:
        digit = thresh

    # 5) Resize while preserving aspect ratio
    h, w = digit.shape
    scale = max(h, w) / float(max(target_size)) if max(target_size) > 0 else 1.0
    if scale > 0:
        new_w = max(1, int(round(w / scale)))
        new_h = max(1, int(round(h / scale)))
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        digit = cv2.resize(digit, target_size, interpolation=cv2.INTER_AREA)

    # 6) Center on 28x28 canvas (black background)
    canvas = np.zeros(target_size, dtype=np.uint8)
    x_off = (target_size[1] - digit.shape[1]) // 2
    y_off = (target_size[0] - digit.shape[0]) // 2
    canvas[y_off:y_off + digit.shape[0], x_off:x_off + digit.shape[1]] = digit

    # 7) Normalize and reshape
    canvas = canvas.astype('float32') / 255.0
    return canvas.reshape(1, target_size[0], target_size[1], 1)

def preprocess_from_flat_with_cv(image_flat: np.ndarray, width: int, height: int):
    """Helper to go from flat normalized [0,1] array to 28x28 using OpenCV pipeline."""
    img_2d = np.array(image_flat, dtype='float32').reshape((height, width))
    img_uint8 = np.clip(img_2d * 255.0, 0, 255).astype(np.uint8)
    return preprocess_canvas_cv(img_uint8)

@app.get("/test", response_model=TestResponse)
async def test():
    """Simple test endpoint"""
    return TestResponse(
        message="Backend is working!",
        model_loaded=model is not None,
        classes=class_names
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Simplified prediction endpoint"""
    try:
        print(f"Received prediction request with {len(request.image)} pixels ({request.width}x{request.height})")
        print(f"Data range: {min(request.image):.3f} to {max(request.image):.3f}")
        print(f"Data mean: {np.mean(request.image):.3f}")
        
        # Validate data length
        expected_length = request.width * request.height
        if len(request.image) != expected_length:
            raise HTTPException(
                status_code=400, 
                detail=f'Invalid data length. Expected {expected_length}, got {len(request.image)}'
            )
        
        # Use OpenCV-based preprocessing (crop, aspect preserve, center)
        pixel_array = np.array(request.image, dtype='float32')
        processed_image = preprocess_from_flat_with_cv(pixel_array, request.width, request.height)
        image_2d = processed_image[0, :, :, 0]

        # Debug: Print a sample of the image actually fed to the model
        print("Sample of image (5x5 center):")
        center_y, center_x = request.height // 2, request.width // 2
        sample = image_2d[center_y-2:center_y+3, center_x-2:center_x+3]
        for row in sample:
            print(f"  {[f'{val:.3f}' for val in row]}")
        
        print(f"Final image shape: {processed_image.shape}")
        print(f"Final image range: {processed_image.min():.3f} to {processed_image.max():.3f}")
        print(f"Final image mean: {processed_image.mean():.3f}")
        
        # Check if image is mostly empty (all zeros or very low values)
        if processed_image.mean() < 0.01:
            print("Warning: Image appears to be mostly empty")
            raise HTTPException(
                status_code=400,
                detail="No drawing detected. Please draw something on the canvas."
            )
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        # Debug: Print raw predictions
        print(f"Raw predictions: {predictions[0]}")
        print(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
        
        return PredictionResponse(
            label=predicted_class,
            confidence=confidence,
            top_predictions=top_3_predictions,
            all_predictions={
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_processed")
async def download_processed(request: PredictionRequest):
    """Return the processed model input as a downloadable PNG (grayscale 28x28)."""
    try:
        expected_length = request.width * request.height
        if len(request.image) != expected_length:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid data length. Expected {expected_length}, got {len(request.image)}'
            )

        # Use the same OpenCV preprocessing and take the 2D view
        processed_image = preprocess_from_flat_with_cv(
            np.array(request.image, dtype='float32'), 
            request.width, 
            request.height
        )
        image_2d = processed_image[0, :, :, 0]

        # Convert to PNG bytes (0=black background, 255=white strokes)
        img_uint8 = (image_2d * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)

        return StreamingResponse(
            BytesIO(buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=processed.png"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interpret", response_model=InterpretResponse)
async def interpret(request: InterpretRequest):
    """Return a simple AI interpretation string for the drawing.

    This is a minimal local implementation to keep the UI happy without external APIs.
    """
    try:
        pred = request.prediction
        conf = float(request.confidence)

        if conf < 0.3:
            msg = f"I'm not sure what this is. It might be a {pred}."
        elif conf < 0.7:
            msg = f"This looks like a {pred}, but I'm not completely sure."
        else:
            msg = f"This is a {pred}. Nice drawing!"

        return InterpretResponse(interpretation=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_model", response_model=TestModelResponse)
async def test_model():
    """Test the model with different inputs"""
    try:
        results = {}
        
        # Test 1: All zeros (blank image)
        blank_image = np.zeros((1, 28, 28, 1), dtype='float32')
        blank_predictions = model.predict(blank_image, verbose=0)
        blank_class = class_names[np.argmax(blank_predictions[0])]
        blank_confidence = float(blank_predictions[0].max())
        results['blank'] = {
            'prediction': blank_class,
            'confidence': blank_confidence,
            'all_predictions': {class_names[i]: float(blank_predictions[0][i]) for i in range(len(class_names))}
        }
        
        # Test 2: All ones (full image)
        full_image = np.ones((1, 28, 28, 1), dtype='float32')
        full_predictions = model.predict(full_image, verbose=0)
        full_class = class_names[np.argmax(full_predictions[0])]
        full_confidence = float(full_predictions[0].max())
        results['full'] = {
            'prediction': full_class,
            'confidence': full_confidence,
            'all_predictions': {class_names[i]: float(full_predictions[0][i]) for i in range(len(class_names))}
        }
        
        # Test 3: Center dot
        dot_image = np.zeros((1, 28, 28, 1), dtype='float32')
        dot_image[0, 14, 14, 0] = 1.0
        dot_predictions = model.predict(dot_image, verbose=0)
        dot_class = class_names[np.argmax(dot_predictions[0])]
        dot_confidence = float(dot_predictions[0].max())
        results['dot'] = {
            'prediction': dot_class,
            'confidence': dot_confidence,
            'all_predictions': {class_names[i]: float(dot_predictions[0][i]) for i in range(len(class_names))}
        }
        
        # Test 4: Simple star pattern
        star_image = np.zeros((1, 28, 28, 1), dtype='float32')
        # Create a simple star pattern
        for i in range(28):
            for j in range(28):
                if (i == 14) or (j == 14) or (abs(i - 14) == abs(j - 14)):
                    star_image[0, i, j, 0] = 1.0
        star_predictions = model.predict(star_image, verbose=0)
        star_class = class_names[np.argmax(star_predictions[0])]
        star_confidence = float(star_predictions[0].max())
        results['star_pattern'] = {
            'prediction': star_class,
            'confidence': star_confidence,
            'all_predictions': {class_names[i]: float(star_predictions[0][i]) for i in range(len(class_names))}
        }
        
        # Test 5: Simple circle
        circle_image = np.zeros((1, 28, 28, 1), dtype='float32')
        center_x, center_y = 14, 14
        radius = 8
        for i in range(28):
            for j in range(28):
                distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if radius - 1 <= distance <= radius + 1:
                    circle_image[0, i, j, 0] = 1.0
        circle_predictions = model.predict(circle_image, verbose=0)
        circle_class = class_names[np.argmax(circle_predictions[0])]
        circle_confidence = float(circle_predictions[0].max())
        results['circle'] = {
            'prediction': circle_class,
            'confidence': circle_confidence,
            'all_predictions': {class_names[i]: float(circle_predictions[0][i]) for i in range(len(class_names))}
        }
        
        return TestModelResponse(
            message="Model test results",
            results=results
        )
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model"""
    try:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        return ModelInfoResponse(
            model_loaded=model is not None,
            classes=class_names,
            model_summary=model_summary,
            input_shape=list(model.input_shape),
            output_shape=list(model.output_shape)
        )
        
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    print("Starting FastAPI Doodle Recognition API...")
    print(f"Model loaded: {model is not None}")
    print(f"Available classes: {class_names}")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5001,
        reload=True,
        log_level="info"
    )
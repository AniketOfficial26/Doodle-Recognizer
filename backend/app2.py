from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)

# Load the model
print("Loading model...")
model = tf.keras.models.load_model('doodle_recognizer_simple.keras')
print("Model loaded successfully!")

# Class names
class_names = ['apple', 'airplane', 'cat', 'car', 'dog', 'flower', 'star', 'tree', 'umbrella', 'fish']
print(f"Available classes: {class_names}")
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

    # 3) Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

    # 6) Center on 28x28 canvas
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

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'Backend is working!',
        'model_loaded': model is not None,
        'classes': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Simplified prediction endpoint"""
    try:
        # Get data from request
        data = request.get_json()
        print(f"Received data keys: {list(data.keys())}")
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get image data
        image_data = data['image']
        width = data.get('width', 28)
        height = data.get('height', 28)
        
        print(f"Processing {len(image_data)} pixels ({width}x{height})")
        print(f"Data range: {min(image_data):.3f} to {max(image_data):.3f}")
        print(f"Data mean: {np.mean(image_data):.3f}")
        
        # Validate data length
        expected_length = width * height
        if len(image_data) != expected_length:
            return jsonify({'error': f'Invalid data length. Expected {expected_length}, got {len(image_data)}'}), 400
        
        # Use OpenCV-based preprocessing (crop, aspect preserve, center)
        pixel_array = np.array(image_data, dtype='float32')
        processed_image = preprocess_from_flat_with_cv(pixel_array, width, height)
        image_2d = processed_image[0, :, :, 0]

        # Debug: Print a sample of the image actually fed to the model
        print("Sample of image (5x5 center):")
        center_y, center_x = height // 2, width // 2
        sample = image_2d[center_y-2:center_y+3, center_x-2:center_x+3]
        for row in sample:
            print(f"  {[f'{val:.3f}' for val in row]}")
        
        print(f"Final image shape: {processed_image.shape}")
        print(f"Final image range: {processed_image.min():.3f} to {processed_image.max():.3f}")
        print(f"Final image mean: {processed_image.mean():.3f}")
        
        # Check if image is mostly empty (all zeros or very low values)
        if processed_image.mean() < 0.01:
            print("Warning: Image appears to be mostly empty")
            return jsonify({
                'error': 'No drawing detected. Please draw something on the canvas.',
                'label': 'no_drawing',
                'confidence': 0.0
            }), 400
        
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
        
        return jsonify({
            'label': predicted_class,
            'confidence': confidence,
            'top_predictions': top_3_predictions,
            'all_predictions': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_processed', methods=['POST'])
def download_processed():
    """Return the processed model input as a downloadable PNG (grayscale 28x28)."""
    try:
        data = request.get_json()
        if not isinstance(data, dict) or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        width = int(data.get('width', 28))
        height = int(data.get('height', 28))

        expected_length = width * height
        if not isinstance(image_data, list) or len(image_data) != expected_length:
            return jsonify({'error': f'Invalid data length. Expected {expected_length}, got {len(image_data) if isinstance(image_data, list) else "unknown"}'}), 400

        # Use the same OpenCV preprocessing and take the 2D view
        processed_image = preprocess_from_flat_with_cv(np.array(image_data, dtype='float32'), width, height)
        image_2d = processed_image[0, :, :, 0]

        # Convert to PNG bytes (0=black background, 255=white strokes)
        img_uint8 = (image_2d * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='processed.png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_model', methods=['GET'])
def test_model():
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
        
        return jsonify({
            'message': 'Model test results',
            'results': results
        })
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        return jsonify({
            'model_loaded': model is not None,
            'classes': class_names,
            'model_summary': model_summary,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        })
        
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting simplified Flask app...")
    print(f"Model loaded: {model is not None}")
    print(f"Available classes: {class_names}")
    app.run(debug=True, host='127.0.0.1', port=5001)
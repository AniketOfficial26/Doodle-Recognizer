from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('doodle_classifier_improved.h5')
# Class names - MAKE SURE these match your training labels exactly
class_names = ['apple', 'airplane', 'cat', 'car', 'dog', 'flower', 'star', 'tree', 'umbrella', 'fish']

def preprocess_image(image_data):
    """
    Preprocess the image data to match training format
    """
    try:
        # DEBUG: Check raw input data
        raw_array = np.array(image_data)
        print(f"Raw input shape: {raw_array.shape}")
        print(f"Raw input min/max: {raw_array.min()}/{raw_array.max()}")
        print(f"Raw input mean: {raw_array.mean():.3f}")
        print(f"Raw input sample values: {raw_array.flatten()[:10]}")
        
        # Convert to numpy array and reshape
        image_array = np.array(image_data).reshape(28, 28)
        
        # Check if image needs inversion (white background with black drawing)
        if image_array.mean() < 50:  # Very dark image
            print("WARNING: Image appears to be mostly black/empty")
            print("Sample of reshaped data:", image_array[10:15, 10:15])
        
        # CRITICAL FIX: Your data is already 0-1, don't divide by 255!
        # Since raw min/max is 0/1, data is already normalized
        if raw_array.max() <= 1:
            print("Data already normalized (0-1 range)")
            image_array = image_array.astype('float32')  # Don't divide by 255
        else:
            print("Data needs normalization (0-255 range)")
            image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension and channel dimension
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Debug: Print some stats
        print(f"Final image shape: {image_array.shape}")
        print(f"Final image min/max: {image_array.min():.3f}/{image_array.max():.3f}")
        print(f"Final image mean: {image_array.mean():.3f}")
        
        return image_array
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def preprocess_canvas_drawing(canvas_data):
    """
    Alternative preprocessing for canvas drawings (if data comes differently)
    """
    try:
        # Handle different input formats
        if isinstance(canvas_data, str):
            # If it's a base64 string, decode it
            if canvas_data.startswith('data:image'):
                canvas_data = canvas_data.split(',')[1]
            image_bytes = base64.b64decode(canvas_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(canvas_data, list):
            # If it's already pixel data
            return preprocess_image(canvas_data)
        else:
            # Assume it's PIL Image or similar
            image = canvas_data
        
        # Convert to grayscale and resize to 28x28
        if image.mode != 'L':
            image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image).astype('float32')
        
        # Invert if background is white (drawing is black on white)
        if image_array.mean() > 127:
            image_array = 255 - image_array
        
        # Normalize to 0-1 range
        image_array = image_array / 255.0
        
        # Reshape for model
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Debug info
        print(f"Canvas image shape: {image_array.shape}")
        print(f"Canvas image min/max: {image_array.min():.3f}/{image_array.max():.3f}")
        print(f"Canvas image mean: {image_array.mean():.3f}")
        
        return image_array
        
    except Exception as e:
        print(f"Error in canvas preprocessing: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get JSON data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Try different preprocessing methods based on data format
        try:
            # First try the standard preprocessing
            processed_image = preprocess_image(data['image'])
        except:
            try:
                # If that fails, try canvas preprocessing
                processed_image = preprocess_canvas_drawing(data['image'])
            except Exception as e:
                return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions for better insight
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            top_3_predictions.append({
                'class': class_names[idx],
                'confidence': float(predictions[0][idx])
            })
        
        # Get the best prediction
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Debug logging
        print(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
        top_3_str = [f"{p['class']}({p['confidence']:.3f})" for p in top_3_predictions]
        print(f"Top 3: {top_3_str}")

        # Return response with additional debug info
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
        print("Error during prediction:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """
    Special endpoint for canvas drawings (base64 images)
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process canvas drawing
        processed_image = preprocess_canvas_drawing(data['image'])
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            'label': predicted_class,
            'confidence': confidence,
            'top_predictions': top_3_predictions
        })
        
    except Exception as e:
        print("Error in canvas prediction:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'message': 'Backend is working!', 
        'classes': class_names,
        'model_status': model_status
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_count': len(class_names)
    })

if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"Model loaded: {model is not None}")
    print(f"Classes: {class_names}")
    app.run(debug=True, host='0.0.0.0', port=5000)
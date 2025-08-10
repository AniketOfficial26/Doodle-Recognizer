from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Class names (without model loading)
class_names = ['apple', 'airplane', 'cat', 'car', 'dog', 'flower', 'star', 'tree', 'umbrella', 'fish']
print(f"Available classes: {class_names}")

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'Backend is working!',
        'model_loaded': False,
        'classes': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Simplified prediction endpoint without model"""
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
        
        # Mock prediction (for testing)
        import random
        predicted_class = random.choice(class_names)
        confidence = random.uniform(0.5, 0.9)
        
        return jsonify({
            'label': predicted_class,
            'confidence': confidence,
            'top_predictions': [
                {'class': predicted_class, 'confidence': confidence}
            ],
            'all_predictions': {class_name: random.uniform(0, 1) for class_name in class_names}
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting simplified Flask app (no model)...")
    print(f"Available classes: {class_names}")
    app.run(debug=True, host='127.0.0.1', port=8080)

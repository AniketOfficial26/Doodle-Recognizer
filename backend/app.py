from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model (ensure path is correct)
model = tf.keras.models.load_model('doodle_model.h5')

# Class names should match your training labels
class_names = ['apple', 'airplane', 'cat', 'car', 'dog', 'flower', 'star', 'tree', 'umbrella', 'fish']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Convert the pixel array to numpy array
        image_data = np.array(data['image'])
        
        # Reshape to 28x28 and add batch dimension and channel dimension
        # Frontend sends flattened array, so reshape it
        image_data = image_data.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = model.predict(image_data)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Return response with 'label' key to match frontend expectation
        return jsonify({
            'label': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend is working!'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
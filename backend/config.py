# config.py
"""
Configuration file for the Doodle Classifier application
"""

# Model configuration
MODEL_PATH = 'doodle_recognizer_mobilenetv2_finetuned.h5'

# Class names - MAKE SURE these match your training labels exactly
CLASS_NAMES = [
    'apple', 'airplane', 'cat', 'car', 'dog', 
    'flower', 'star', 'tree', 'umbrella', 'fish'
]

# Image processing configuration
TARGET_IMAGE_SIZE = (96, 96)  # Updated for retrained model
BATCH_SIZE = 1
NUM_CHANNELS = 1

# API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
DEFAULT_API_KEY = "YOUR_HUGGINGFACE_API_KEY"

# Flask configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Logging configuration
ENABLE_DEBUG_LOGGING = True
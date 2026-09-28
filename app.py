import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64
import h5py
import json

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model settings
IMG_SIZE = (176, 208)
MODEL_PATH = r"Models\efficientnetb5_weighted_final.h5"

# Class names and descriptions
CLASS_INFO = {
    'No Impairment': {
        'level': 0,
        'color': '#28a745',
        'description': 'No signs of cognitive impairment detected. Brain structure appears normal.'
    },
    'Very Mild Impairment': {
        'level': 1,
        'color': '#ffc107',
        'description': 'Very mild cognitive decline. May include slight memory lapses.'
    },
    'Mild Impairment': {
        'level': 2,
        'color': '#fd7e14',
        'description': 'Mild cognitive impairment. Noticeable memory and cognitive difficulties.'
    },
    'Moderate Impairment': {
        'level': 3,
        'color': '#dc3545',
        'description': 'Moderate cognitive decline. Significant impact on daily activities.'
    }
}

class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Load model globally
print("Loading model...")
model = None

def strip_quantization_config(config):
    """Recursively remove quantization_config from layer configs"""
    if isinstance(config, dict):
        # Remove quantization_config if present
        config.pop('quantization_config', None)
        # Recursively process nested dicts and lists
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                strip_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            if isinstance(item, (dict, list)):
                strip_quantization_config(item)
    return config

def load_model():
    global model
    if model is None:
        try:
            # Try loading directly first
            model = keras.models.load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully!")
        except (TypeError, ValueError) as e:
            if 'quantization_config' in str(e):
                print("Detected quantization_config issue, loading with fix...")
                # Load the h5 file and fix the config
                with h5py.File(MODEL_PATH, 'r') as f:
                    model_config = f.attrs.get('model_config')
                    if model_config is None:
                        raise ValueError("No model config found in file")
                    
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode('utf-8')
                    
                    config = json.loads(model_config)
                    # Strip quantization_config from all layers
                    config = strip_quantization_config(config)
                    
                    # Rebuild the model from the cleaned config
                    model = keras.models.model_from_json(json.dumps(config))
                    
                    # Load the weights
                    model.load_weights(MODEL_PATH)
                    print("Model loaded with config fix!")
            else:
                raise e
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))  # width, height
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def get_prediction(img_array):
    """Get model prediction"""
    model = load_model()
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx]) * 100
    
    # Get all class probabilities
    all_probs = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}
    
    return predicted_class, confidence, all_probs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP)'}), 400
    
    try:
        # Read and preprocess image
        image_data = file.read()
        img_array, img = preprocess_image(image_data)
        
        # Get prediction
        predicted_class, confidence, all_probs = get_prediction(img_array)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Get class info
        class_info = CLASS_INFO[predicted_class]
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'level': class_info['level'],
            'color': class_info['color'],
            'description': class_info['description'],
            'all_probabilities': {k: round(v, 2) for k, v in all_probs.items()},
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Create upload folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Preload model
    load_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)

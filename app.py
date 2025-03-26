from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load model - CHANGE THIS TO YOUR MODEL PATH
model = load_model('your_model.keras')  # or 'model.keras'
print("✅ Model loaded successfully!")
print(f"Model input shape: {model.input_shape}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=model.input_shape[1:3])
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        pred = model.predict(img_array)[0][0]
        
        return {
            'success': True,
            'real': float(pred) * 100,
            'fake': (1 - float(pred)) * 100,
            'image_path': img_path.replace('\\', '/')  # Fix path for web
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if file exists
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file = request.files['file']
        
        # Check filename
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Only JPG/PNG allowed'}), 400
        
        # Create upload folder if not exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Make prediction
        result = predict_image(save_path)
        
        if not result['success']:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

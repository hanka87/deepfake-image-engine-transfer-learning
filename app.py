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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model
model = load_model('deepfake_3_model.keras')
print("✅ Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    return {
        'real': float(pred) * 100,
        'fake': (1 - float(pred)) * 100,
        'image_path': img_path
    }

# [Keep all your existing imports and configuration]

# [Keep your model loading and helper functions]

# Add this new endpoint (keep your existing / route)
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    if file and allowed_file(file.filename):
        # Save with unique filename
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Process image
        results = predict_image(save_path)
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'})

# [Keep your existing / route and main block]

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
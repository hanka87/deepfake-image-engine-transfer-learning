# deepfake-image-engine-transfer-learning

Deepfake Image Detection Engine - Real vs Fake Faces (140K Dataset)

Welcome to the Deepfake Image Detection project!
This repository contains an end-to-end pipeline designed to detect real and fake human face images using deep learning and machine learning models. The project leverages the 140K Real and Fake Faces Dataset from Kaggle.

I created 2 notebooks with one of them using "Transfer learnign" to achieve SOTA(State of the art ) accuracy. 
Both the notebooks are there , and also I have uploaded also a .keras file for direct use to make projects for anyone 

Dataset Overview:
- Dataset Name: 140K Real and Fake Faces Dataset
- Source: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- Contents: 70,000 real images and 70,000 fake (AI-generated) images
- Format: JPEG images organized into train/test/validation folders

Project Highlights:
- ✅ Built an end-to-end deep learning pipeline
- ✅ Implemented multiple ML/DL models: CNN, InceptionResNetV2, Xception, EfficientNet, Gradient Boosting, XGBoost, Random Forest, and LSTM
- ✅ Integrated Explainable AI (Grad-CAM) for visual interpretability
- ✅ Applied Transfer Learning and Hyperparameter Optimization
- ✅ Achieved accuracy up to 95% on validation data and optimized performance
- ✅ Flask+react web app built for real-time user uploads and predictions
- ✅ Supported with adversarial training to handle advanced deepfakes
- ✅ Performance-tuned for faster inference speed (30% improvement)

Steps Involved:
1. Data Loading and Preprocessing
2. Model Building (Multiple ML/DL architectures)
3. Transfer Learning on pre-trained models
4. Training, Evaluation, and Hyperparameter Tuning
5. Explainability using Grad-CAM
6. Model Optimization (Pruning, Quantization)
7. Flask-based Web App Deployment
8. Real-time Predictions with user-uploaded images
9. Performance Testing and Validation on unseen datasets

Technologies Used:
- Python, TensorFlow, Keras, OpenCV
- Scikit-learn, XGBoost, LightGBM, Numpy, Pandas, Matplotlib
- Flask for Web Deployment
- Explainable AI with Grad-CAM

This project serves as a robust framework for detecting deepfake images and can be extended for further research and deployment in real-world applications.


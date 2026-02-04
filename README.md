This project is an AI-powered medical imaging system that detects bone fractures from X-ray images using Deep Learning. The model is built with PyTorch and deployed through an interactive Streamlit web application that allows users to upload X-ray images and receive instant predictions.

The goal of this project is to demonstrate how AI can assist in early fracture screening by providing fast, automated image analysis to support medical professionals.

Features
CNN-based fracture classification

Upload X-ray images for analysis

⚡ Real-time prediction

Confidence score for each prediction

Simple Streamlit web interface

Works on both CPU and GPU

Model Overview

The system uses a Convolutional Neural Network (CNN) for binary image classification.

Prediction Classes:

Fractured

Not Fractured

Architecture Includes:

Convolutional layers for feature extraction

ReLU activation for non-linearity

MaxPooling layers for downsampling

Fully connected layers for final classification

All input images are resized to 224 × 224 pixels before being passed to the model.

🖥️ Web Application (Streamlit)

The Streamlit app provides a user-friendly interface where users can:

Upload an X-ray image

Let the AI model analyze it

Instantly see the prediction result with confidence score

This makes the project easy to test without technical knowledge.

📂 Project Structure
├── model.py                     # CNN model architecture  
├── test.py                      # Streamlit app for inference  
├── main.ipynb                   # Training and experimentation notebook  
├── final_fracture_model.pth     # Trained model weights  

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/xray-fracture-detection.git
cd xray-fracture-detection


Install required libraries:

pip install torch torchvision streamlit pillow

▶️ Run the Application
streamlit run test.py


After running, open the local URL shown in the terminal and upload an X-ray image to get predictions.

🎯 Future Improvements

Multi-class fracture detection (different bone types)

Explainable AI using Grad-CAM heatmaps

Model performance optimization

Cloud deployment for remote access

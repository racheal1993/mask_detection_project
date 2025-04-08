# 😷 Mask Detection Project

A deep learning-based computer vision project to detect whether a person is wearing a face mask or not in real-time, using OpenCV, TensorFlow/Keras, and a trained CNN model.

## 🚀 Features

- Real-time face mask detection using webcam
- Trained Convolutional Neural Network (CNN) model
- OpenCV for face detection
- Sound alert system for people without masks (optional)
- Easy to deploy on local machine

## 🧠 Tech Stack

- Python 3.12
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Pyttsx3 / Winsound (optional for alerts)

## 📁 Project Structure

## 🛠️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/racheal1993/mask_detection_project.git
   cd mask_detection_project
## create a virtual environment
 python -m venv venv
venv\Scripts\activate      # On Windows
## install dependencies
  pip install -r requirements.txt

## train the model
  python train_mask_detector.py
## to run the app 
  python run app.py

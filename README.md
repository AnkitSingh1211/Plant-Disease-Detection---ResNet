# 🌿 Plant Disease Detection using ResNet-14

## 📌 Overview
This project implements a **Plant Disease Detection System** using a **ResNet-14** deep learning model. The goal is to classify plant diseases from leaf images with high accuracy. The system helps farmers and researchers detect diseases early, improving crop health and yield.

## 🖼️ Model Architecture: ResNet-14
ResNet-14 is a lightweight variant of the **ResNet (Residual Network)** architecture, optimized for efficient and accurate plant disease classification. It employs residual connections to mitigate vanishing gradient issues.

## 🚀 Features
- **Deep Learning-Based Detection**: Uses **ResNet-14** for high-accuracy classification.
- **Multi-Class Classification**: Detects various plant diseases from leaf images.
- **Transfer Learning Support**: Fine-tuned ResNet for improved performance.
- **Real-Time Prediction**: Predict diseases from new images using a trained model.
- **Visualization & Explainability**: Heatmaps (Grad-CAM) for model interpretability.

## 📊 Dataset
- The model is trained on a **PlantVillage dataset**, which contains images of healthy and diseased plant leaves.
- **Preprocessing Steps:**
  - Image resizing
  - Data augmentation (rotation, flipping, brightness adjustment)
  - Normalization for efficient model training

## 🏗️ Model Training
1. Load the dataset and preprocess images.
2. Implement **ResNet-14** with PyTorch/TensorFlow.
3. Train the model using **categorical cross-entropy loss** and **Adam optimizer**.
4. Evaluate performance with accuracy, precision, recall, and F1-score.
5. Deploy the trained model for real-time predictions.

## 🛠️ Technologies Used
- **Deep Learning Frameworks**: TensorFlow / PyTorch
- **Image Processing**: OpenCV, PIL
- **Visualization**: Matplotlib, Grad-CAM for explainability

## 🖥️ Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/AnkitSingh1211/Plant-Disease-Detection---ResNet.git
cd Plant-Disease-Detection---ResNet

![Plant-Disease-Detection](Screenshot%202025-02-12%20220744.png)

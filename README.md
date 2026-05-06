# 💵 Fake Currency Detection System (AI-Powered)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)

An advanced, AI-driven system designed to detect counterfeit currency with high precision using Deep Learning and Computer Vision. 

## 🚀 Project Overview
Traditional counterfeit detection methods often rely on manual inspection or simple UV lights. This project leverages **Convolutional Neural Networks (CNNs)** and **Transfer Learning** (specifically the `MobileNetV2` architecture) to automatically distinguish between authentic and counterfeit banknotes with a high degree of accuracy.

The system is highly optimized, making it computationally lightweight and suitable for deployment on low-resource environments (like mobile devices or simple web servers) without compromising precision.

## ✨ Key Features
* **Realistic Synthetic Data Generation:** Since datasets of professionally forged banknotes are rare, this project features a custom data engineering pipeline using `OpenCV`. It generates synthetic counterfeits by applying physical defect simulations to real banknote images:
  * Color shifting (cheap ink simulation)
  * Blur and Gaussian noise (poor micro-printing)
  * Feature blockouts (missing watermarks/security threads)
  * Paper texture overlays
* **Strict Data Splitting:** Data is rigorously split into `80% Train`, `10% Validation`, and `10% Test` (unseen data) to prevent data leakage and ensure unbiased evaluation.
* **Transfer Learning:** Utilizes Google's `MobileNetV2` as a base feature extractor, combined with a custom Dense head and Dropout regularization to prevent overfitting.
* **Exceptional Accuracy:** Achieved **95.21% overall accuracy** on the unseen test set, with **100% precision in detecting fake currency** (Zero False Positives for authentic notes).

## 🛠️ Installation & Usage
1. Clone the repository:

git clone [https://github.com/DT7A1/Fake-Currency-Detection.git](https://github.com/YourUsername/Fake-Currency-Detection.git)

cd Fake-Currency-Detection

2. Install dependencies:

pip install -r requirements.txt

3. Download the Pre-trained Model:

Navigate to the Releases section of this repository.

Download currency_model.h5 and place it inside the models/ directory.

4. Run the System:
To test a single image, edit the image path in test.py and run:

cd src
python test.py

To retrain the model or view the full evaluation summary:

cd src
python project.py

## 📊 Results
Training Accuracy: ~94.6%

Validation Accuracy: ~94.9%

Testing Accuracy: 95.21%

The model demonstrates extremely high reliability, making it robust enough for real-world application prototyping.

## Dataset

Original dataset sourced from Kaggle:

USD Bill Classification Dataset by Aishwarya Hiremath

Licensed under MIT License

Link: https://www.kaggle.com/datasets/aishwaryatechie/usd-bill-classification-dataset

## 📂 Project Structure
```text
Fake-Currency-Detection/
│
├── models/                # Contains the pre-trained .h5 model (Download from Releases)
├── outputs/               # Evaluation metrics, confusion matrix, and learning curves
├── samples/               # Sample images for quick testing
├── src/                   
│   ├── project.py         # Main training and evaluation pipeline
│   ├── test.py            # Script for testing single images
│   └── app.py             # (Optional) Flask web interface
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

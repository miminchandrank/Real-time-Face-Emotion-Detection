# 😊 Real-Time Face Emotion Detection Using Deep Learning

A real-time facial emotion detection system that uses **Convolutional Neural Networks (CNN)** and **OpenCV** to recognize human emotions from live webcam input. This project showcases how deep learning and computer vision can be integrated to interpret emotional states in real time — enabling AI to read human expressions.

🎥 **Live Demo Video**: [Watch on LinkedIn](https://www.linkedin.com/posts/miminchandrank_deeplearning-computervision-ai-activity-7319195495975829504-Ut_o?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFD4aN8BBSizqogKnOr2eBg_WSmXdqUej4w)

---

## 🔍 Overview

This system detects human faces using Haar Cascade and classifies facial expressions into **seven key emotions** using a trained CNN model:

- 😄 Happy  
- 😢 Sad  
- 😠 Angry  
- 😲 Surprised  
- 😐 Neutral  
- 😨 Fear  
- 🤢 Disgust

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – For real-time video capture and face detection
- **TensorFlow / Keras** – For building and training the deep learning model
- **NumPy / Pandas** – Data handling and preprocessing
- **Haar Cascade Classifier** – For detecting face boundaries

---

## 🚀 Features

- 🎯 Real-time emotion classification via webcam  
- 🧠 Lightweight and fast CNN model  
- 🎛️ Dynamic confidence scoring and emotion labeling on live video frames  
- 💡 Easily extendable for emotion-driven applications (e.g., chatbots, smart interfaces, sentiment tracking)

---

## 🧠 Model Details

The CNN model is trained on the **FER-2013** dataset (or a similar custom dataset) and includes:

- 3 Convolutional + MaxPooling layers  
- Dropout layers for regularization  
- Fully connected Dense layers with **softmax** activation for emotion classification  

> ✅ **Accuracy achieved**: ~65–70% on baseline FER dataset (with scope for tuning and improvement)


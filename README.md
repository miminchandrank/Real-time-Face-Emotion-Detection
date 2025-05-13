# 😊 Real-Time Face Emotion Detection Using Deep Learning

A real-time facial emotion detection system that uses **Convolutional Neural Networks (CNN)** and **OpenCV** to recognize human emotions from live webcam input. This project showcases how deep learning and computer vision can be integrated to interpret emotional states in real time — enabling AI to read human expressions.

🎥 **Demo Video**: [Insert link to demo video or GIF here]

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
- 🎛️ Dynamic confidence scoring and labeling on frames
- 💡 Easily extendable for emotion-driven applications like chatbots, sentiment monitoring, etc.

---

## 🧠 Model Details

The CNN model is trained on the **FER-2013** dataset (or custom dataset) and consists of:

- 3 Convolutional + MaxPooling layers
- Dropout layers for regularization
- Dense layers with softmax activation for multi-class classification

> ✅ Accuracy achieved: ~65–70% (FER dataset baseline, with scope for fine-tuning)

---

## 🖼️ Sample Output

![sample_output](sample_frame.png)

---

## 📁 Project Structure


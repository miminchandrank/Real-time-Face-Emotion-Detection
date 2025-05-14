import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained emotion detection model
model = load_model("emotiondetector.h5")
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)  # Try 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Webcam successfully opened")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi)
        emotion = labels[np.argmax(prediction)]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting app")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

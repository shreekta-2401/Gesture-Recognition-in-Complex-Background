import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import brightness

# Load trained model
model = joblib.load('gesture_model.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# def adjust_brightness(direction):
#     if direction == 'up':
#         os.system("brightness -i 0.1")  # Increase brightness by 0.1 (50%)
#     elif direction == 'down':
#         os.system("brightness -d 0.1")  # Decrease brightness by 0.1 (50%)


def adjust_brightness(direction):
    if direction == 'up':
        os.system("osascript -e 'tell application \"System Events\" to key code 120'")
    elif direction == 'down':
        os.system("osascript -e 'tell application \"System Events\" to key code 122'")



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Predict gesture
            if landmarks:
                landmarks = np.array(landmarks).reshape(1, -1)
                gesture = model.predict(landmarks)[0]
                adjust_brightness(gesture)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

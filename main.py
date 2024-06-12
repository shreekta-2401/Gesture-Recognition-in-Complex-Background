import cv2
import mediapipe as mp
import numpy as np
from background_model import create_background_model, update_background_model, update_threshold
from gesture_recognition import recognize_gesture

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load or create the background model
background_model = create_background_model()
if background_model is None:
    print("Failed to create background model.")
    exit(1)

background_model = background_model.astype(np.float32)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters for background subtraction
alpha = 0.7
T_initial = 30
T_threshold = 30
Tbs = np.full((background_model.shape[0], background_model.shape[1]), T_initial, dtype=np.float32)

# Variables for gesture recognition
prev_y = None
swipe_up_counter = 0
swipe_down_counter = 0
gesture_threshold = 5

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
            prev_y, swipe_up_counter, swipe_down_counter = recognize_gesture(
                hand_landmarks, prev_y, swipe_up_counter, swipe_down_counter, gesture_threshold
            )

    diff_frame = cv2.absdiff(background_model.astype(np.uint8), frame)
    diff_frame_gray = np.linalg.norm(diff_frame, axis=2)
    moving_mask = diff_frame_gray > Tbs

    background_model = update_background_model(background_model, frame, moving_mask, alpha)
    Tbs = update_threshold(Tbs, diff_frame_gray, moving_mask, alpha)

    _, thresh_diff_frame = cv2.threshold(diff_frame_gray.astype(np.uint8), T_threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Hand Detection', frame)
    cv2.imshow('Image Differencing', thresh_diff_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

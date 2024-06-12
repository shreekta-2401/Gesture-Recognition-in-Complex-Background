import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Previous frame
previous_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # Draw hand annotations on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize previous_frame if it is None
    if previous_frame is None:
        previous_frame = gray_frame
        continue

    # Compute the absolute difference between the current frame and the previous frame
    diff_frame = cv2.absdiff(previous_frame, gray_frame)

    # Apply a binary threshold to the difference image
    _, thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

    # Display the original frame
    cv2.imshow('Hand Detection', frame)

    # Display the image differencing result
    cv2.imshow('Image Differencing', thresh_frame)

    # Update previous_frame
    previous_frame = gray_frame

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

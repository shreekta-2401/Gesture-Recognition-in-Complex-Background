import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Load the saved background model
background_model = cv2.imread('background_model.png')

if background_model is None:
    print("Failed to load background model.")
    exit(1)

# Convert background model to float for updating
background_model = background_model.astype(np.float32)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters for background subtraction
alpha = 0.7
T_initial = 30
T_threshold = 30
Tbs = np.full((background_model.shape[0], background_model.shape[1]), T_initial, dtype=np.float32)

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

    # Perform image differencing with the background model
    diff_frame = cv2.absdiff(background_model.astype(np.uint8), frame)
    
    # Calculate Euclidean distance for each pixel and convert to grayscale
    diff_frame_gray = np.linalg.norm(diff_frame, axis=2)

    # Apply the background subtraction rule
    moving_mask = diff_frame_gray > Tbs

    # Update the background model
    background_model[~moving_mask] = (alpha * background_model[~moving_mask] +
                                      (1 - alpha) * frame[~moving_mask])
    
    # Update the threshold
    Tbs[~moving_mask] = alpha * Tbs[~moving_mask] + (1 - alpha) * diff_frame_gray[~moving_mask]

    # Apply a binary threshold to the difference image
    _, thresh_diff_frame = cv2.threshold(diff_frame_gray.astype(np.uint8), T_threshold, 255, cv2.THRESH_BINARY)

    # Display the original frame with hand annotations
    cv2.imshow('Hand Detection', frame)

    # Display the image differencing result
    cv2.imshow('Image Differencing', thresh_diff_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

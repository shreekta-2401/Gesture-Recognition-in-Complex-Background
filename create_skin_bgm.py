import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for background model
background_model = None
frame_count = 0
max_frames = 30
collected_frames = 0

while cap.isOpened() and collected_frames < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # If no hands are detected, update the background model
    if not results.multi_hand_landmarks:
        if background_model is None:
            background_model = np.zeros_like(frame, dtype=np.float32)
        
        background_model += frame.astype(np.float32)
        collected_frames += 1

    # Display the current frame
    cv2.imshow('Background Model Creation', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Average the accumulated frames to create the background model
if collected_frames > 0:
    background_model /= collected_frames
    background_model = background_model.astype(np.uint8)

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the background model to a file
cv2.imwrite('background_model.png', background_model)
print("Background model saved as 'background_model.png'")

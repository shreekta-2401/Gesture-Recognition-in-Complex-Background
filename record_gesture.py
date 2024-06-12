import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters
gesture_type = input("Enter gesture type (up/down): ")
data_file = f"{gesture_type}_gestures.csv"

# Open the CSV file to save landmarks
with open(data_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gesture'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)])

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

                # Save landmarks to CSV
                writer.writerow([gesture_type] + landmarks)

        cv2.imshow('Record Gestures', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

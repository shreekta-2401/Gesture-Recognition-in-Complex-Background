import cv2
import numpy as np

def create_background_model(max_frames=30):
    cap = cv2.VideoCapture(0)
    background_model = None
    collected_frames = 0

    while cap.isOpened() and collected_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if background_model is None:
            background_model = np.zeros_like(frame, dtype=np.float32)

        background_model += frame.astype(np.float32)
        collected_frames += 1

        cv2.imshow('Background Model Creation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if collected_frames > 0:
        background_model /= collected_frames
        background_model = background_model.astype(np.uint8)

    cap.release()
    cv2.destroyAllWindows()
    return background_model

def update_background_model(background_model, frame, moving_mask, alpha=0.7):
    background_model[~moving_mask] = (alpha * background_model[~moving_mask] +
                                      (1 - alpha) * frame[~moving_mask])
    return background_model

def update_threshold(Tbs, diff_frame_gray, moving_mask, alpha=0.7):
    Tbs[~moving_mask] = alpha * Tbs[~moving_mask] + (1 - alpha) * diff_frame_gray[~moving_mask]
    return Tbs

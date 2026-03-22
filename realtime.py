"""
Real-time Sign Language Detection: MediaPipe Hands + YOLO classifier.
Uses temporal smoothing for accurate, stable output. Press Q to quit.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Suppress TF Lite info/warnings
os.environ["GLOG_minloglevel"] = "3"           # Suppress MediaPipe C++ warnings

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from ultralytics import YOLO

# --- Accuracy settings ---
SMOOTH_WINDOW = 10          # Frames to average probabilities over (smooths noise)
CONSECUTIVE_FRAMES = 3       # Same gesture must win this many frames in a row before updating
DISPLAY_CONFIDENCE = 0.6     # Only show when smoothed confidence above this
HAND_PADDING = 30            # Padding around hand crop (more context often helps)
NO_HAND_RESET_FRAMES = 15    # Clear displayed label after no hand for this many frames

# Load trained model
model = YOLO(r"C:\Users\DELL\runs\classify\train4\weights\best.pt")
class_names = list(model.names.values())
num_classes = len(class_names)

# MediaPipe Hands (Solutions API - use mediapipe==0.10.9)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Running average of class probabilities (for temporal smoothing)
prob_history = deque(maxlen=SMOOTH_WINDOW)
no_hand_count = 0
display_label = None
display_confidence = 0.0
consecutive_same = 0
last_winner_id = -1

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        no_hand_count = 0
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            # Use larger padding for better classifier input
            pad = HAND_PADDING
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(w, xmax + pad)
            ymax = min(h, ymax + pad)

            hand_crop = frame[ymin:ymax, xmin:xmax]

            if hand_crop.size != 0:
                # Ensure minimum size to avoid tiny crops
                if hand_crop.shape[0] < 32 or hand_crop.shape[1] < 32:
                    continue
                results_cls = model(hand_crop, verbose=False)

                probs = results_cls[0].probs.data.cpu().numpy()
                prob_history.append(probs.copy())

                if len(prob_history) >= 2:
                    # Smoothed probabilities = average over recent frames
                    smoothed = np.mean(prob_history, axis=0)
                    winner_id = int(np.argmax(smoothed))
                    winner_conf = float(smoothed[winner_id])

                    if winner_id == last_winner_id:
                        consecutive_same += 1
                    else:
                        consecutive_same = 1
                        last_winner_id = winner_id

                    # Update displayed label only when stable and confident
                    if consecutive_same >= CONSECUTIVE_FRAMES and winner_conf >= DISPLAY_CONFIDENCE:
                        display_label = class_names[winner_id]
                        display_confidence = winner_conf
                else:
                    # Not enough history yet: still require consecutive + confidence
                    winner_id = int(np.argmax(probs))
                    winner_conf = float(probs[winner_id])
                    if winner_id == last_winner_id:
                        consecutive_same += 1
                    else:
                        consecutive_same = 1
                        last_winner_id = winner_id
                    if consecutive_same >= CONSECUTIVE_FRAMES and winner_conf >= DISPLAY_CONFIDENCE:
                        display_label = class_names[winner_id]
                        display_confidence = winner_conf
    else:
        no_hand_count += 1
        if no_hand_count >= NO_HAND_RESET_FRAMES:
            prob_history.clear()
            display_label = None
            display_confidence = 0.0
            consecutive_same = 0
            last_winner_id = -1

    # Draw the stable, smoothed label
    if display_label is not None:
        text = f"{display_label} ({display_confidence:.2f})"
        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        # Small hint
        cv2.putText(
            frame,
            "Hold gesture steady for best result",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

    cv2.imshow("Real-Time Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

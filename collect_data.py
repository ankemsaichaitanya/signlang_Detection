"""
Collect hand gesture images for training.
Press number keys to start saving frames for a gesture, S to stop, Q to quit.
"""
import cv2
import mediapipe as mp
import os

GESTURES = {
    "1": "hello",
    "2": "yes",
    "3": "no",
    "4": "thanks",
    "5": "iloveyou",
    # --- Add new gestures below ---
    "6": "please",
    "7": "sorry",
    "8": "help",
    "9": "goodbye",
    "10":"how are you?",
    "11":"what is you name?",
    "12":"are you good?",
}

SAVE_ROOT = "dataset"
HAND_PADDING = 30
FRAMES_PER_GESTURE = 200  # target frames per gesture

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Create folders
for split in ["train", "val"]:
    for gesture in GESTURES.values():
        os.makedirs(os.path.join(SAVE_ROOT, split, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)
recording = False
current_gesture = None
frame_count = 0

print("=== Gesture Data Collector ===")
print("Keys: 1=hello  2=yes  3=no  4=thanks  5=iloveyou")
print("Press a number key to START recording that gesture.")
print("Press S to STOP recording. Press Q to QUIT.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_crop = None
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hl.landmark]
            ys = [int(lm.y * h) for lm in hl.landmark]
            pad = HAND_PADDING
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w, max(xs) + pad)
            y2 = min(h, max(ys) + pad)
            hand_crop = frame[y1:y2, x1:x2]

    # Display status
    status = f"Recording: {current_gesture} ({frame_count})" if recording else "Not recording"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2)
    cv2.putText(frame, "1-5: gesture | S: stop | Q: quit", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if recording and hand_crop is not None and hand_crop.size > 0:
        # 80% train, 20% val split
        split = "train" if frame_count % 5 != 0 else "val"
        path = os.path.join(SAVE_ROOT, split, current_gesture, f"{frame_count:04d}.jpg")
        cv2.imwrite(path, hand_crop)
        frame_count += 1

    if chr(key) in GESTURES:
        current_gesture = GESTURES[chr(key)]
        recording = True
        frame_count = 0
        print(f"Recording '{current_gesture}' — show the gesture to your webcam...")
    elif key == ord("s") or key == ord("S"):
        recording = False
        print(f"Stopped. Saved {frame_count} frames for '{current_gesture}'.")
    elif key == ord("q") or key == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done! Now train with: yolo classify train model=yolov8n-cls.pt data=dataset epochs=30 imgsz=64")

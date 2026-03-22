import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model (pretrained)
yolo_model = YOLO("yolov8n.pt")

# Load trained classifier
classifier = load_model("best_model.keras")

# Gesture labels
labels = ["hello", "yes", "no", "thanks", "iloveyou"]

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Crop hand region
            hand_img = frame[y1:y2, x1:x2]

            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            prediction = classifier.predict(hand_img)
            label = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            if confidence > 0.7:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0,255,0),
                            2)

    cv2.imshow("YOLO Sign Language", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

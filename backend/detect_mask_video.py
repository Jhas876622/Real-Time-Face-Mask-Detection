import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ✅ Updated model path
model = load_model("backend/model/mask_detector.h5")

# ✅ Paths remain same (already correct)
faceNet = cv2.dnn.readNet(
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "face_detector/deploy.prototxt"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Starting mask detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        face_confidence = detections[0, 0, i, 2]

        if face_confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float32")
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face, verbose=0)[0]
            mask_prob = preds[0]
            no_mask_prob = preds[1]

            if mask_prob > no_mask_prob:
                label = "Mask"
                color = (0, 255, 0)
                conf_display = mask_prob * 100
            else:
                label = "No Mask"
                color = (0, 0, 255)
                conf_display = no_mask_prob * 100

            text = f"{label}: {conf_display:.2f}%"
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
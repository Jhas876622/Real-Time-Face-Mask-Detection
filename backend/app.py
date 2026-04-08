import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)

model = None
faceNet = None

def load_models():
    global model, faceNet
    try:
        if model is None:
            print("Building model architecture...")
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None
            )
            x = base_model.output
            x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(2, activation="softmax")(x)
            model = tf.keras.Model(inputs=base_model.input, outputs=x)

            weights_path = os.path.join(BASE, "model", "mask_weights.weights.h5")
            print(f"Weights path: {weights_path}")
            print(f"Weights exist: {os.path.exists(weights_path)}")
            model.load_weights(weights_path)
            print("Model ready!")

        if faceNet is None:
            proto_path = os.path.join(BASE, "face_detector", "deploy.prototxt")
            caffe_path = os.path.join(BASE, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
            faceNet = cv2.dnn.readNet(caffe_path, proto_path)
            print("Face detector loaded!")

        return True

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(traceback.format_exc())
        return False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not load_models():
            return jsonify({"error": "Model loading failed"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (224, 224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = tf.keras.applications.mobilenet_v2.preprocess_input(
                    face.astype("float32")
                )
                face = np.expand_dims(face, axis=0)
                preds = model.predict(face, verbose=0)[0]
                results.append({
                    "mask": float(preds[0]),
                    "no_mask": float(preds[1]),
                    "box": [int(startX), int(startY), int(endX), int(endY)]
                })

        return jsonify({"faces": results})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/debug")
def debug():
    weights_path = os.path.join(BASE, "model", "mask_weights.weights.h5")
    return jsonify({
        "base_dir": BASE,
        "weights_exist": os.path.exists(weights_path),
        "model_files": os.listdir(os.path.join(BASE, "model")),
        "all_files": os.listdir(BASE)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
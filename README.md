<div align="center">

# 🎭 Real-Time Face Mask Detection System

### Deep Learning-Based PPE Compliance Monitoring Using MobileNetV2 Transfer Learning

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Online-00C851?style=for-the-badge)](https://real-time-face-mask-detection.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Accuracy](https://img.shields.io/badge/Val_Accuracy-99.48%25-brightgreen?style=for-the-badge)](https://real-time-face-mask-detection.onrender.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Deployed on](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

<br/>

> A production-deployed, real-time face mask detection system leveraging MobileNetV2 transfer learning with a two-stage detection pipeline — achieving **99.48% validation accuracy** on a balanced binary classification dataset.

<br/>

**[🌐 Try Live Demo](https://real-time-face-mask-detection.onrender.com/) · [📊 View Results](#-results--performance) · [🚀 Quick Start](#-quick-start)**

</div>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Results & Performance](#-results--performance)
- [Tech Stack](#-tech-stack)
- [Dependencies](#-dependencies)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Conclusion](#-conclusion)

---

## 📄 Abstract

This project presents a real-time face mask detection system designed for public health and workplace safety compliance monitoring. The system employs a two-stage deep learning pipeline: (1) a ResNet-10 SSD face detector for robust face localization, followed by (2) a fine-tuned MobileNetV2 classifier for binary mask/no-mask prediction. The complete system is deployed as a production web application accessible via browser with live webcam inference, audio/voice alerts, and a real-time analytics dashboard.

**Key contributions:**
- End-to-end browser-based inference pipeline requiring no client-side installation
- Two-stage detection architecture separating face localization from mask classification
- Production deployment with Flask + Gunicorn backend on cloud infrastructure
- 99.48% validation accuracy on a curated dataset of with/without mask images

---

## 🎯 Problem Statement

The COVID-19 pandemic highlighted the critical need for automated Personal Protective Equipment (PPE) compliance monitoring in high-density public spaces such as hospitals, offices, airports, and educational institutions. Manual monitoring is resource-intensive, error-prone, and unscalable. This system addresses the need for an automated, real-time, browser-accessible solution that can detect mask compliance from live webcam feeds without requiring dedicated hardware or software installation on client devices.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                         │
│                                                                 │
│  📷 Webcam Feed                                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐     face-api.js      ┌──────────────────────┐ │
│  │  Video Feed  │ ──────────────────► │  Face Localization   │ │
│  │  (640×480)   │   TinyFaceDetector  │  (Bounding Box)      │ │
│  └─────────────┘   + SSD Fallback     └──────────┬───────────┘ │
│                                                  │              │
│                                          Crop + Resize          │
│                                          (224×224 px)           │
│                                                  │              │
│                                                  ▼              │
│                                    POST /predict (FormData)     │
└──────────────────────────────────────────────────┼─────────────┘
                                                   │
                              HTTP Request (JPEG blob)
                                                   │
┌──────────────────────────────────────────────────▼─────────────┐
│                        SERVER (Render)                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Flask + Gunicorn                       │  │
│  │                                                           │  │
│  │  ┌─────────────────┐      ┌───────────────────────────┐  │  │
│  │  │  ResNet-10 SSD  │      │   MobileNetV2 Classifier  │  │  │
│  │  │  Face Detector  │─────►│   (Fine-tuned, 2 classes) │  │  │
│  │  │  (OpenCV DNN)   │      │   with_mask / no_mask     │  │  │
│  │  └─────────────────┘      └───────────────────────────┘  │  │
│  │                                        │                  │  │
│  │                                        ▼                  │  │
│  │                          {"mask": 0.99, "no_mask": 0.01}  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    JSON Response → Client
                                   │
                    ┌──────────────▼──────────────┐
                    │   UI: SAFE ✓ / ALERT ⚠️     │
                    │   Bounding Box Overlay        │
                    │   Confidence Score Display    │
                    │   Voice + Audio Alert         │
                    └─────────────────────────────┘
```

---

## 🔬 Methodology

### Stage 1 — Face Detection

The system uses a **ResNet-10 Single Shot Multibox Detector (SSD)** pretrained on face detection datasets, loaded via OpenCV's DNN module. This stage produces bounding box coordinates `(startX, startY, endX, endY)` for all detected faces with confidence > 0.5. On the client side, `face-api.js` running TinyFaceDetector (with SSD fallback) provides an additional lightweight localization pass before sending the cropped face region to the server.

### Stage 2 — Mask Classification

Detected face regions are resized to `224×224` pixels and passed through a **fine-tuned MobileNetV2** classifier. The base MobileNetV2 (pretrained on ImageNet) acts as a frozen feature extractor. A custom classification head is appended:

```
MobileNetV2 (frozen) → AveragePooling2D(7×7) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(2, Softmax)
```

Output: `[P(with_mask), P(without_mask)]`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 (ImageNet weights) |
| Input Shape | (224, 224, 3) |
| Optimizer | Adam (lr=1e-4) |
| Loss Function | Categorical Crossentropy |
| Epochs | 25 |
| Batch Size | 32 |
| Train/Val Split | 80% / 20% |
| Data Augmentation | Rotation ±20°, Zoom 0.2, Shift 0.1, Flip |
| Classes | `with_mask`, `without_mask` |

### Data Augmentation Pipeline

```python
ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
```

---

## 📁 Project Structure

```
Real-Time-Face-Mask-Detection/
│
├── backend/                          # Flask server
│   ├── app.py                        # Main Flask application
│   ├── requirements.txt              # Python dependencies
│   ├── model/
│   │   └── mask_detector_final.keras # Trained model (Keras format)
│   ├── face_detector/
│   │   ├── deploy.prototxt           # ResNet-10 SSD architecture
│   │   └── res10_300x300_ssd_iter_140000.caffemodel  # Pretrained weights
│   └── templates/
│       └── index.html                # Frontend UI
│
├── dataset/
│   ├── with_mask/                    # Training images (masked faces)
│   └── without_mask/                 # Training images (unmasked faces)
│
├── scripts/
│   └── train_mask_detector.py        # Model training script
│
├── render.yaml                       # Render deployment config
└── README.md
```

---

## 📊 Results & Performance

### Training Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 99.08% | **99.48%** |
| Loss | 0.0272 | 0.0246 |
| Epochs | 25 | 25 |

### Detection Performance

| Component | Specification |
|-----------|--------------|
| Face Detection | ResNet-10 SSD @ confidence > 0.5 |
| Classification Threshold | mask_prob > 0.4 |
| Input Resolution | 224 × 224 px |
| Inference Latency | ~150–400ms per frame (server-side) |
| Multi-face Support | ✅ Yes |
| Real-time FPS | 1–4 FPS (free-tier server) |

### Deployment Stats

| Property | Value |
|----------|-------|
| Platform | Render (Free Tier) |
| Server | Gunicorn (1 worker) |
| Uptime | On-demand (spins up on request) |
| Live URL | [real-time-face-mask-detection.onrender.com](https://real-time-face-mask-detection.onrender.com/) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.16 + Keras 3 |
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Face Detection | OpenCV DNN + ResNet-10 SSD |
| Backend | Flask 3.0 + Flask-CORS |
| Production Server | Gunicorn |
| Frontend | Vanilla JS + face-api.js |
| Image Processing | OpenCV, NumPy, Pillow |
| Deployment | Render Cloud |
| Language | Python 3.12 |

---

## 📦 Dependencies

```txt
flask==3.0.0
flask-cors==4.0.0
tensorflow==2.16.1
opencv-python-headless==4.8.1.78
numpy==1.26.4
scikit-learn==1.3.2
Pillow==10.1.0
gunicorn==21.2.0
```

Install all dependencies:

```bash
pip install -r backend/requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Webcam-enabled browser (Chrome recommended)
- Git

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Jhas876622/Real-Time-Face-Mask-Detection.git
cd Real-Time-Face-Mask-Detection

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the Flask server
cd backend
python app.py

# 4. Open in browser
# Navigate to http://localhost:5000
```

### Train From Scratch

```bash
# Prepare dataset
# Place images in:
#   dataset/with_mask/
#   dataset/without_mask/

# Run training (25 epochs, ~15-20 min on CPU)
python scripts/train_mask_detector.py
```

---

## 📡 API Reference

### `POST /predict`

Accepts a face image and returns mask detection results.

**Request:**
```
Content-Type: multipart/form-data
Body: image (JPEG/PNG file)
```

**Response:**
```json
{
  "faces": [
    {
      "mask": 0.9923,
      "no_mask": 0.0077,
      "box": [120, 45, 310, 280]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `mask` | float | Probability of mask present (0–1) |
| `no_mask` | float | Probability of no mask (0–1) |
| `box` | array | Bounding box `[startX, startY, endX, endY]` |

### `GET /health`

Server health check.

```json
{ "status": "ok" }
```

### `GET /debug`

File existence check for deployed model and detector files.

```json
{
  "base_dir": "/opt/render/project/src/backend",
  "model_exists": true,
  "model_files": ["mask_detector_final.keras", "mask_detector.h5"],
  "all_files": ["app.py", "model", "face_detector", "templates"]
}
```

---

## ☁️ Deployment

This project is deployed on **Render** using the following configuration:

```yaml
# render.yaml
services:
  - type: web
    name: real-time-face-mask-detection
    env: python
    rootDir: backend
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 1 --timeout 120 --max-requests 50
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

### Deploy Your Own

1. Fork this repository
2. Create a [Render](https://render.com) account
3. New → Web Service → Connect your fork
4. Render auto-detects `render.yaml` — click **Deploy**

> ⚠️ **Note:** Free tier spins down after 15 minutes of inactivity. First request after sleep takes 30–60 seconds to warm up. Use [UptimeRobot](https://uptimerobot.com) (free) to keep the server warm with periodic pings.

---

## 🔚 Conclusion

This project demonstrates a complete end-to-end deployment of a deep learning-based computer vision system — from dataset curation and model training to cloud deployment and browser-accessible real-time inference. The two-stage pipeline (face localization → mask classification) achieves 99.48% validation accuracy using MobileNetV2 transfer learning, with the full system accessible via any webcam-enabled browser without client-side installation. Future improvements could include multi-class PPE detection (gloves, face shields), edge deployment via TensorFlow Lite, and integration with CCTV infrastructure for large-scale compliance monitoring.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ | Deployed Live at [real-time-face-mask-detection.onrender.com](https://real-time-face-mask-detection.onrender.com/)**

⭐ Star this repo if you found it useful!

</div>

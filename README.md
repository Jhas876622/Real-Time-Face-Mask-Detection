# 😷 Real-Time Face Mask Detection System (Industry + IEEE Level)

![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Tech](https://img.shields.io/badge/Tech-TensorFlow%20%7C%20OpenCV%20%7C%20Flask-orange)

---

## 📌 Abstract

This project presents a real-time face mask detection system using deep learning and computer vision. The system utilizes a pre-trained MobileNetV2 model combined with OpenCV’s DNN-based face detector to classify whether a person is wearing a mask or not. It is designed to be scalable and deployable in real-world environments such as surveillance systems, offices, and public transport monitoring.

---

## 🎯 Problem Statement

Manual monitoring of mask compliance is inefficient, error-prone, and not scalable. This project aims to automate mask detection in real-time video streams using AI to improve safety and compliance.

---

## 🏗️ System Architecture

```
Input (Webcam / Video Stream)
        ↓
Face Detection (OpenCV DNN)
        ↓
Face Preprocessing (Resize, Normalize)
        ↓
Mask Detection Model (MobileNetV2)
        ↓
Prediction (Mask / No Mask)
        ↓
Output Display (Bounding Box + Label)
```

---

## 📁 Project Structure

```
Face-Mask-Detection/
│
├── backend/
│   ├── model/
│   │   └── mask_detector.h5
│   ├── static/
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   ├── detect_mask_video.py
│   ├── fix_model.py
│   └── convert_model.py
│
├── dataset/
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── web/
│   ├── static/
│   │   └── alert.mp3
│   ├── templates/
│   │   └── index.html
│
├── scripts/
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Flask (Backend API)
* HTML, CSS, JavaScript (Frontend)
* NumPy, Scikit-learn, Pillow

---

## 📦 Dependencies

(From your requirements.txt)

```txt
flask==3.0.0
flask-cors==4.0.0
tensorflow==2.12.0
tf-keras==2.12.0
opencv-python-headless==4.8.1.78
numpy==1.23.5
scikit-learn==1.3.2
Pillow==10.1.0
gunicorn==21.2.0
```

---

## 🧠 Methodology

1. Detect faces using OpenCV DNN
2. Extract ROI (Region of Interest)
3. Preprocess image for model input (224x224)
4. Use MobileNetV2 for feature extraction
5. Classify using fully connected layers
6. Display results in real-time

---

## 📊 Dataset

* Source: Kaggle Face Mask Dataset
* Classes: Mask / No Mask
* Data augmentation applied for better generalization

---

## 📈 Results & Performance

* Accuracy: ~96%
* Real-time performance achieved
* Works under moderate lighting and multiple faces

---

## 🎥 Demo

### 🔴 Real-Time Detection

(Add your demo GIF here)

---

## 🚀 Installation & Usage

```bash
# Clone repo
git clone https://github.com/Jhas876622/Real-Time-Face-Mask-Detection.git

# Navigate to project
cd Real-Time-Face-Mask-Detection

# Install dependencies
pip install -r requirements.txt

# Run detection
python backend/detect_mask_video.py
```

---

## 🌐 Deployment

* Backend: Flask + Gunicorn
* Frontend: HTML + JS
* Deployment file: render.yaml

⚠️ Current Status: Deployment attempted but facing errors (to be fixed)

---

## 🧩 Applications

* Smart CCTV Surveillance
* Office Entry Monitoring
* Airport & Railway Safety Systems
* Public Compliance Monitoring

---

## 🔮 Future Improvements

* Deploy on cloud (Render / AWS / GCP)
* Add alert system (sound / notification)
* Multi-face tracking with analytics dashboard
* Edge AI deployment using TensorFlow Lite

---

## 📚 Conclusion

This project demonstrates how deep learning can be effectively used for real-time safety monitoring. The system is scalable, efficient, and ready for industry-level enhancements.

---

## 👨‍💻 Author

**Satyam Jha**

---

## 📄 License

MIT License

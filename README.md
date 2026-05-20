# Attention Monitoring System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

Real-time **attention detection system** using **deep learning** and **eye-tracking** with a **Cyberpunk 2077-inspired interface**.

---

## Key Features

### Real-time Eye Tracking
- **MediaPipe Face Mesh** - 468 facial landmarks
- **Eye Aspect Ratio (EAR)** - Accurate eye detection
- **Gaze Direction** - Head/eye movement tracking
- **30+ FPS** - Real-time processing on Jetson

### Machine Learning Classification
- **Random Forest Classifier** - 200 trees, 18 features
- **92%+ Accuracy** - High reliability
- **3-State Detection**:
  - 🟢 **ATTENTIVE** - Looking at screen
  - 🔴 **DISTRACTED** - Looking away
  - 🔵 **SLEEPY** - Eyes closed

### Live Dashboard
- **Real-time Camera Feed** - With eye keypoints
- **Confidence Meter** - Model certainty indicator
- **System Metrics** - FPS, frames, status
- **Beautiful UI** - Cyberpunk 2077 aesthetic

---

## Quick Start

### Requirements
```
- Python 3.8+
- NVIDIA Jetson (or Linux with GPU)
- Webcam/USB Camera
- 4GB+ RAM
```

### Installation

```bash
# 1. Install dependencies
pip install --break-system-packages -r requirements.txt

# 2. Run the system
python3 attention_monitor.py

# 3. Open browser
http://localhost:8080
```

---

## 📁 Project Structure

```
attention_monitor_v2/
├── attention_monitor.py              # Backend server
├── attention_monitor_cyberpunk.html  # Frontend UI
├── attention_model_trained.pkl       # Trained classifier (92% accuracy)
├── attention_scaler_trained.pkl      # Feature scaler
├── demo_v2.avi                       # Demo video
├── README.md                         # This file
├── SETUP.md                          # Setup guide
└── requirements.txt                  # Python dependencies
```

---

## How It Works

### Processing Pipeline
```
Camera Frame
    ↓
MediaPipe Face Mesh (468 landmarks)
    ↓
Extract 18-dimensional feature vector
    ↓
StandardScaler normalization
    ↓
RandomForest Classification
    ↓
Output: State + Confidence Score
    ↓
Draw overlays on frame
    ↓
Stream to browser via HTTP
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 92% |
| **Precision** | 91% |
| **Recall** | 92% |
| **F1 Score** | 91% |
| **FPS** | 30+ |
| **Latency** | 40-50ms |

---

## User Interface

### Layout
- **Left Panel** - Neural Assessment (state, confidence, metrics)
- **Center** - Live Camera Feed with eye keypoints
- **Right Panel** - Telemetry (FPS, frames, status)
- **Bottom** - Status bar with real-time data

### Color Coding
- 🟢 **ATTENTIVE** (Cyan) - Focused on task
- 🔴 **DISTRACTED** (Magenta) - Looking away
- 🔵 **SLEEPY** (Red) - Eyes closed

---

## API Endpoints

```
GET  /                    → HTML interface
GET  /api/data           → JSON response
GET  /stream.jpg         → JPEG stream
```

---

## Feature Vector (18D)

Eye Aspect Ratios, Gaze Direction, Face Position, Eye Metrics, Nose Position, Inter-eye Distance

---

## Training Data

- 150 ATTENTIVE samples
- 150 DISTRACTED samples
- 150 SLEEPY samples
- Total: 450 samples from real data

---

## Use Cases

✅ Driver drowsiness detection
✅ Student engagement monitoring
✅ Call fatigue detection
✅ UI/UX research
✅ Safety systems

---

## Troubleshooting

### No Camera?
```bash
ls /dev/video*
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Port In Use?
```bash
lsof -i :8080
kill -9 <PID>
```

---

## Tech Stack

- **Eye Tracking**: MediaPipe Face Mesh
- **ML**: scikit-learn RandomForest
- **Video**: OpenCV
- **Backend**: Python HTTP Server
- **Frontend**: HTML5 + JavaScript
- **Data**: NumPy, SciPy

---

## Privacy

✅ Local processing only
✅ No cloud upload
✅ No identification
✅ Ethical use

---

## Documentation

- **[SETUP.md](SETUP.md)** - Installation guide
- **[requirements.txt](requirements.txt)** - Dependencies

---

## License

MIT License

---

**Status: ✅ PRODUCTION READY**

Made with ❤️ for real-time attention monitoring

⭐ Star if helpful!

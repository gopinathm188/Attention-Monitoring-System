# 📋 SETUP GUIDE

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install --break-system-packages -r requirements.txt
```

### 2. Run the System

```bash
python3 attention_monitor.py
```

### 3. Open Browser

```
http://localhost:8080
```

---

## 📦 Requirements

- Python 3.8+
- NVIDIA Jetson (or Linux with GPU)
- Webcam/USB Camera
- 4GB+ RAM
- Dependencies from requirements.txt

---

## ✅ Verify Installation

```bash
# Check Python version
python3 --version

# Check dependencies installed
pip list | grep -E "opencv|mediapipe|scikit-learn"

# Test camera
python3 -c "import cv2; print('✅ OpenCV OK' if cv2.VideoCapture(0).isOpened() else '❌ Camera not found')"
```

---

## 🔧 Running the System

```bash
# Make sure you're in the right directory
cd /ai_dev/attention_monitor_v2

# Run the server
python3 attention_monitor.py
```

Expected output:
```
============================================================
ATTENTION MONITOR BACKEND SERVER
============================================================

🌐 Open in browser:
   http://localhost:8080

📡 Stream: http://localhost:8080/stream.jpg
📊 API: http://localhost:8080/api/data

⚡ Press Ctrl+C to stop
```

---

## 🌐 Access the UI

### Local Access
```
http://localhost:8080
```

### From Another Machine
```
http://<jetson-ip>:8080
```

Find Jetson IP:
```bash
hostname -I
```

---

## 📱 What You'll See

✅ Live camera feed in center
✅ Attention state (ATTENTIVE/DISTRACTED/SLEEPY)
✅ Confidence percentage
✅ Eye metrics
✅ System status
✅ FPS counter

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>
```

### No Camera Found
```bash
# List camera devices
ls /dev/video*

# Test OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'No camera')"
```

### Low FPS
- Close other applications
- Reduce screen resolution
- Enable GPU acceleration if available

### Connection Failed
- Make sure server is running
- Check firewall settings
- Verify port 8080 is not blocked

---

## 📊 Files in This Directory

```
attention_monitor.py              # Backend server
attention_monitor_cyberpunk.html  # Frontend UI
attention_model_trained.pkl       # ML model
attention_scaler_trained.pkl      # Feature scaler
demo_v2.avi                       # Demo video
requirements.txt                  # Dependencies
README.md                         # Documentation
SETUP.md                          # This file
```

---

## ✨ Features

- 🔍 Real-time eye tracking
- 🧠 92% accuracy attention detection
- 📊 Live dashboard with metrics
- 🎮 Cyberpunk 2077 UI
- 📹 Live camera streaming
- 🚀 30+ FPS performance

---

## 🔧 Configuration

No configuration needed! The system works out of the box with:
- Port: 8080
- Camera: /dev/video0 (first camera)
- Model: RandomForest (200 trees)
- Features: 18-dimensional

---

## 📈 Performance

Expected performance:
- FPS: 30-45
- Latency: 40-50ms
- Accuracy: 92%+
- Memory: ~500MB

---

## ❓ FAQ

**Q: Can I use a different camera?**
A: The system defaults to /dev/video0. Edit attention_monitor.py to change.

**Q: Can I change the port?**
A: Yes, modify the port in attention_monitor.py (default: 8080)

**Q: Is GPU required?**
A: GPU is recommended but not required.

**Q: Can I train my own model?**
A: Yes, data collection and training scripts are available.

---

## 📞 Support

For issues:
1. Check troubleshooting section above
2. Verify all dependencies installed
3. Check camera is accessible
4. Ensure port 8080 is free

---

## 🎉 Ready!

Your Attention Monitoring System is ready to use!

Run:
```bash
python3 attention_monitor.py
```

Then open:
```
http://localhost:8080
```

Enjoy! 🚀

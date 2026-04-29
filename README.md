# Attention Monitoring System

Real-time attention detection using computer vision on NVIDIA Jetson devices.

[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-92%25-blue)]()
[![FPS](https://img.shields.io/badge/Performance-20--25%20FPS-brightgreen)]()

## Preliminary Results

### 1. Running Model
- Model operational and processing live video
- Real-time inference on Jetson Nano

### 2. Weights Loaded
- `attention_model_classifier.pkl` - Logistic Regression model
- `attention_model_scaler.pkl` - Feature scaler

### 3. Inference
- Real-time inference pipeline working
- 40-50ms per-frame processing

### 4. Predictions
- Accurate attention classification
- Output: ATTENTIVE / DISTRACTED

### 5. Speed
- **20-25 FPS** on Jetson Nano
- **40-50ms** latency per frame
- CPU-only (no GPU needed)

### 6. Metrics Used
- **Accuracy:** 92.0%
- **Precision:** 87.5%
- **Recall:** 98.0%
- **F1 Score:** 92.5%

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run monitor
python3 src/attention_monitor_v2.py --model models/attention_model

# Train model
python3 src/train_attention_model_simple.py --interactive
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Usage Guide](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Preliminary Results](docs/PRELIMINARY_RESULTS.md)

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 92% |
| FPS | 20-25 |
| Latency | 40-50ms |
| GPU Required | No |

## Files

- `src/` - Source code
- `models/` - Trained models
- `web/` - Web dashboard
- `docs/` - Documentation

## License

MIT License

## Repository

https://github.com/gopinathm188/Attention-Monitoring-System

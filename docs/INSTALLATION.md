# Installation Guide

## Requirements
- NVIDIA Jetson (Nano, Xavier, Orin)
- Python 3.6+
- USB Camera or onboard camera

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/gopinathm188/Attention-Monitoring-System.git
cd Attention-Monitoring-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Monitor
```bash
python3 src/attention_monitor_v2.py --model models/attention_model
```

### 4. Train New Model (Optional)
```bash
python3 src/train_attention_model_simple.py --interactive
```

## Output
- **GREEN box** = ATTENTIVE (looking at screen)
- **RED box** = DISTRACTED (looking away)
- Press **'q'** to quit

## Troubleshooting

If camera not detected:
```bash
# Check camera
ls /dev/video*

# Specify different camera
python3 src/attention_monitor_v2.py --model models/attention_model --camera 1
```

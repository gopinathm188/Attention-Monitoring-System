# Usage Guide

## Run Real-time Monitoring

### Basic Usage
```bash
python3 src/attention_monitor_v2.py --model models/attention_model
```

### Save to Video
```bash
python3 src/attention_monitor_v2.py --model models/attention_model --output results.mp4
```

### Use Different Camera
```bash
python3 src/attention_monitor_v2.py --model models/attention_model --camera 1
```

## Train Model

### Interactive Training
```bash
python3 src/train_attention_model_simple.py --interactive
```

This will:
1. Collect 50 ATTENTIVE samples (look at screen, press SPACEBAR)
2. Collect 50 DISTRACTED samples (look away, press SPACEBAR)
3. Train classifier automatically
4. Save model to `models/attention_model`

## Output Interpretation

- **ATTENTIVE (Green)**: Face centered, looking forward
- **DISTRACTED (Red)**: Face off-center, head turned
- **Confidence**: 0.0 (low) to 1.0 (high)

## Keyboard Controls
- `q` - Quit monitoring
- `SPACEBAR` - Capture training sample
- `ESC` - Skip to next phase

## Dashboard

Run the Flask dashboard:
```bash
python3 src/app_backend.py
```

Then open: http://localhost:5000

# System Architecture

## Pipeline Flow
┌──────────────┐
│ Video Input  │
│  (Camera)    │
└──────┬───────┘
│
▼
┌──────────────────────┐
│  Face Detection      │
│  (Haar Cascade)      │
│  5-10ms              │
└──────┬───────────────┘
│
▼
┌──────────────────────┐
│ Feature Extraction   │
│ (10 dimensions)      │
│ 2-3ms                │
└──────┬───────────────┘
│
▼
┌──────────────────────┐
│ Feature Scaling      │
│ (StandardScaler)     │
└──────┬───────────────┘
│
▼
┌──────────────────────┐
│ Classification       │
│ (Logistic Regression)│
│ 1-2ms                │
└──────┬───────────────┘
│
▼
┌──────────────────────┐
│ Output               │
│ Attention State      │
│ + Confidence         │
└──────────────────────┘

## Components

### 1. Face Detection
- **Method**: OpenCV Haar Cascade Classifier
- **Speed**: 5-10ms per frame
- **Accuracy**: 95%+ for frontal faces
- **GPU**: Not required

### 2. Feature Extraction
- **Dimensions**: 10
- **Features**:
  - Face position (X, Y normalized)
  - Face aspect ratio (width/height)
  - Face size ratio (area/frame_area)
  - Margins (left, right, top, bottom)

### 3. Classifier
- **Algorithm**: Logistic Regression
- **Classes**: 
  - 1 = ATTENTIVE
  - 0 = DISTRACTED
- **Training Data**: 100 samples (50 per class)
- **Feature Scaling**: StandardScaler

## Performance

| Metric | Value |
|--------|-------|
| FPS | 20-25 |
| Latency | 40-50ms |
| Accuracy | 92% |
| Precision | 87.5% |
| Recall | 98% |
| F1 Score | 92.5% |
| Memory | 400-500MB |
| GPU | Not needed |

## Training Process

1. **Data Collection**: Interactive from webcam
2. **Feature Scaling**: StandardScaler fit
3. **Model Training**: Logistic Regression
4. **Serialization**: joblib (pickle format)

## Files

- `attention_model_classifier.pkl` - Trained classifier
- `attention_model_scaler.pkl` - Feature scaler
- `training_data.pkl` - Training samples

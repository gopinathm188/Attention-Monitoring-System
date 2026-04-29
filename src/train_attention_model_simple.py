"""
Training for Simple Attention Monitor
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from attention_monitor_simple import SimpleAttentionMonitor


class SimpleDataCollector:
    """Data collection"""
    
    def __init__(self, monitor, save_dir='training_data'):
        self.monitor = monitor
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.samples = []
        self.labels = []
    
    def collect_samples(self, label, num_samples=50):
        """Collect samples"""
        cap = cv2.VideoCapture(self.monitor.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        label_code = 1 if label == 'attentive' else 0
        collected = 0
        
        print(f"\n{'='*60}")
        print(f"Collecting '{label.upper()}' samples ({num_samples} needed)")
        print(f"{'='*60}")
        print(f"Position yourself for '{label}' pose")
        print(f"Press SPACEBAR to capture | ESC when done\n")
        
        try:
            while collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                faces = self.monitor.detect_face(frame)
                
                if len(faces) > 0:
                    result = self.monitor.extract_features(frame, faces)
                    if result is not None:
                        features, face_rect = result
                        
                        annotated = frame.copy()
                        h, w = frame.shape[:2]
                        x, y, fw, fh = face_rect
                        cv2.rectangle(annotated, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                        
                        cv2.putText(annotated, f"Collecting: {label.upper()}", 
                                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        cv2.putText(annotated, f"Progress: {collected}/{num_samples}", 
                                  (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(annotated, "SPACE to capture | ESC done", 
                                  (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        
                        cv2.imshow('Data Collection', annotated)
                        
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == 32:  # SPACEBAR
                            self.samples.append(features)
                            self.labels.append(label_code)
                            collected += 1
                            print(f"  ✓ Sample {collected} captured")
                        elif key == 27:  # ESC
                            break
                
                else:
                    annotated = frame.copy()
                    cv2.putText(annotated, "No face detected!", 
                              (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Data Collection', annotated)
                    cv2.waitKey(30)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n✓ Collected {collected} '{label}' samples")
    
    def save_data(self):
        """Save"""
        if len(self.samples) == 0:
            print("ERROR: No samples!")
            return False
        
        data = {
            'samples': np.array(self.samples),
            'labels': np.array(self.labels)
        }
        
        save_path = self.save_dir / 'training_data.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Data saved: {save_path}")
        print(f"  Total samples: {len(self.samples)}")
        return True


def train_model(X, y, model_path='attention_model'):
    """Train"""
    monitor = SimpleAttentionMonitor()
    
    print("\n" + "="*60)
    print("TRAINING CLASSIFIER")
    print("="*60)
    
    monitor.train_classifier(X, y)
    monitor.save_model(model_path)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    predictions = monitor.classifier.predict(X_scaled)
    
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    
    print("\nTRAINING METRICS:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print("="*60 + "\n")


def interactive_training():
    """Interactive"""
    print("\n" + "="*60)
    print("ATTENTION MONITOR - SIMPLE VERSION TRAINING")
    print("="*60 + "\n")
    
    monitor = SimpleAttentionMonitor()
    collector = SimpleDataCollector(monitor)
    
    print("\nPhase 1: Collect ATTENTIVE samples")
    print("-" * 60)
    collector.collect_samples('attentive', num_samples=50)
    
    print("\nPhase 2: Collect DISTRACTED samples")
    print("-" * 60)
    collector.collect_samples('distracted', num_samples=50)
    
    print("\nPhase 3: Save training data")
    print("-" * 60)
    collector.save_data()
    
    print("\nPhase 4: Train classifier")
    print("-" * 60)
    X = np.array(collector.samples)
    y = np.array(collector.labels)
    
    train_model(X, y, 'attention_model')
    
    print("\n✓ Training complete!")
    print("Run: python3 attention_monitor_simple.py --model attention_model\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_training()
    else:
        print("Use: python3 train_attention_model_simple.py --interactive")


if __name__ == '__main__':
    main()

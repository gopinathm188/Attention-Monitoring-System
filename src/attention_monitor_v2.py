"""
Attention Monitor - Version 2 (FIXED)
Simple version using face detection
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque
import time
import joblib
import sys

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
except ImportError:
    print("ERROR: scikit-learn not installed")
    sys.exit(1)


class SimpleAttentionMonitor:
    """Simple attention monitoring"""
    
    def __init__(self, model_path=None, camera_id=0):
        self.camera_id = camera_id
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        
        # Try to load model if path provided
        if model_path:
            self.load_model(model_path)
        
        self.frame_count = 0
        self.fps_values = deque(maxlen=30)
        self.last_time = time.time()
        
        self.colors = {
            'attentive': (0, 255, 0),
            'distracted': (0, 0, 255),
            'unknown': (255, 165, 0)
        }
    
    def load_model(self, model_path):
        """Load model with detailed error handling"""
        print(f"\nLoading model from: {model_path}")
        
        classifier_file = f"{model_path}_classifier.pkl"
        scaler_file = f"{model_path}_scaler.pkl"
        
        print(f"Looking for: {classifier_file}")
        print(f"Looking for: {scaler_file}")
        
        # Check if files exist
        if not Path(classifier_file).exists():
            print(f"✗ File not found: {classifier_file}")
            return
        
        if not Path(scaler_file).exists():
            print(f"✗ File not found: {scaler_file}")
            return
        
        # Try to load
        try:
            print("Loading classifier...")
            self.classifier = joblib.load(classifier_file)
            print("✓ Classifier loaded")
            
            print("Loading scaler...")
            self.scaler = joblib.load(scaler_file)
            print("✓ Scaler loaded")
            
            self.is_trained = True
            print("✓ Model ready for predictions!\n")
        
        except Exception as e:
            print(f"✗ Error loading model: {e}\n")
            self.classifier = None
            self.scaler = None
            self.is_trained = False
    
    def detect_face(self, frame):
        """Detect face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def extract_features(self, frame, faces):
        """Extract features"""
        if len(faces) == 0:
            return None
        
        h, w = frame.shape[:2]
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = face
        
        face_center_x = (x + fw/2) / w
        face_center_y = (y + fh/2) / h
        face_aspect_ratio = fw / fh
        face_size_ratio = (fw * fh) / (w * h)
        left_margin = x / w
        right_margin = (w - x - fw) / w
        top_margin = y / h
        bottom_margin = (h - y - fh) / h
        
        features = [
            face_center_x,
            face_center_y,
            face_aspect_ratio,
            face_size_ratio,
            left_margin,
            right_margin,
            top_margin,
            bottom_margin,
            fw / w,
            fh / h
        ]
        
        return np.array(features), face
    
    def detect_attention(self, frame):
        """Detect attention"""
        faces = self.detect_face(frame)
        
        attention_state = 'unknown'
        confidence = 0.0
        face_rect = None
        
        if len(faces) > 0:
            result = self.extract_features(frame, faces)
            if result is not None:
                features, face_rect = result
                
                if self.is_trained and self.classifier and self.scaler:
                    try:
                        features_normalized = self.scaler.transform(features.reshape(1, -1))
                        prediction = self.classifier.predict(features_normalized)[0]
                        probability = self.classifier.predict_proba(features_normalized)[0]
                        
                        attention_state = 'attentive' if prediction == 1 else 'distracted'
                        confidence = max(probability)
                    except Exception as e:
                        print(f"Prediction error: {e}")
        
        return attention_state, confidence, face_rect
    
    def draw_results(self, frame, attention_state, confidence, face_rect):
        """Draw results"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        if face_rect is not None:
            x, y, fw, fh = face_rect
            color = self.colors.get(attention_state, self.colors['unknown'])
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 2)
        
        color = self.colors.get(attention_state, self.colors['unknown'])
        status_text = f"{attention_state.upper()} ({confidence:.2f})"
        
        cv2.rectangle(annotated, (10, 30), (400, 80), color, -1)
        cv2.putText(annotated, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        if self.fps_values:
            avg_fps = np.mean(self.fps_values)
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return annotated
    
    def train_classifier(self, training_data, training_labels):
        """Train"""
        X = np.array(training_data)
        y = np.array(training_labels)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(X_scaled, y)
        
        self.is_trained = True
        print("✓ Classifier trained successfully")
    
    def save_model(self, model_path):
        """Save"""
        joblib.dump(self.classifier, f"{model_path}_classifier.pkl")
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        print(f"✓ Model saved to {model_path}")
    
    def run(self, output_path=None, headless=False):
        """Run"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        print("\n" + "="*60)
        if self.is_trained:
            print("✓ Model loaded - Running predictions")
        else:
            print("⚠ No model loaded - Face detection only")
        print("="*60)
        print("Starting attention monitor... Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.fps_values.append(fps)
                self.last_time = current_time
                
                attention_state, confidence, face_rect = self.detect_attention(frame)
                annotated_frame = self.draw_results(frame, attention_state, confidence, face_rect)
                
                if video_writer:
                    video_writer.write(annotated_frame)
                
                if not headless:
                    cv2.imshow('Attention Monitor', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if not headless:
                cv2.destroyAllWindows()
            
            print(f"\n✓ Processed {self.frame_count} frames")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--output', type=str, help='Output video')
    parser.add_argument('--headless', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ATTENTION MONITOR - V2")
    print("="*60)
    
    monitor = SimpleAttentionMonitor(model_path=args.model)
    monitor.run(output_path=args.output, headless=args.headless)


if __name__ == '__main__':
    main()

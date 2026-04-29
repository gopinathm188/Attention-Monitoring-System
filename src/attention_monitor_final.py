import cv2
import numpy as np
from pathlib import Path
from collections import deque
import time
import joblib

class SimpleAttentionMonitor:
    def __init__(self, model_path=None, camera_id=0):
        self.camera_id = camera_id
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
        
        self.frame_count = 0
        self.fps_values = deque(maxlen=30)
        self.last_time = time.time()
        self.colors = {'attentive': (0, 255, 0), 'distracted': (0, 0, 255)}
    
    def load_model(self, model_path):
        try:
            self.classifier = joblib.load(f"{model_path}_classifier.pkl")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            self.is_trained = True
            print("✓ Model loaded!")
            return True
        except:
            return False
    
    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5)
    
    def extract_features(self, frame, faces):
        if len(faces) == 0:
            return None
        h, w = frame.shape[:2]
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = face
        features = [(x + fw/2) / w, (y + fh/2) / h, fw / fh, (fw * fh) / (w * h),
                    x / w, (w - x - fw) / w, y / h, (h - y - fh) / h, fw / w, fh / h]
        return np.array(features), face
    
    def detect_attention(self, frame):
        faces = self.detect_face(frame)
        attention_state = 'distracted'
        confidence = 0.5
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
                    except:
                        pass
        return attention_state, confidence, face_rect
    
    def draw_results(self, frame, attention_state, confidence, face_rect):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        if face_rect is not None:
            x, y, fw, fh = face_rect
            color = self.colors.get(attention_state, (0, 0, 255))
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 3)
        color = self.colors.get(attention_state, (0, 0, 255))
        status_text = f"{attention_state.upper()} ({confidence:.2f})"
        cv2.rectangle(annotated, (10, 30), (450, 90), color, -1)
        cv2.putText(annotated, status_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        if self.fps_values:
            avg_fps = np.mean(self.fps_values)
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (w-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated
    
    def run(self, output_path=None, headless=False):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("ERROR: Camera not available")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        print("\nStarting monitor... Press q to quit\n")
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
            print("\nStopped")
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if not headless:
                cv2.destroyAllWindows()
            print(f"✓ Processed {self.frame_count} frames")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    monitor = SimpleAttentionMonitor(model_path=args.model)
    monitor.run(output_path=args.output, headless=args.headless)

if __name__ == '__main__':
    main()

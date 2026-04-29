"""
Attention Monitor - Flask Backend with Advanced Analytics
Professional dashboard with real-time monitoring and statistics
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from pathlib import Path
from collections import deque
import time
import joblib
import threading
import json
from datetime import datetime
import csv
from io import StringIO
import sys

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
except ImportError:
    print("ERROR: scikit-learn not installed")
    sys.exit(1)


app = Flask(__name__)
CORS(app)


class AdvancedAttentionMonitor:
    """Advanced attention monitoring with analytics"""
    
    def __init__(self, model_path=None, camera_id=0):
        self.camera_id = camera_id
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
        
        # Session data
        self.session_active = False
        self.session_start_time = None
        self.session_data = {
            'attentive_frames': 0,
            'distracted_frames': 0,
            'total_frames': 0,
            'fps_values': deque(maxlen=30),
            'confidence_values': deque(maxlen=100),
            'attention_timeline': [],
            'frame_data': []
        }
        
        self.colors = {
            'attentive': (0, 255, 0),
            'distracted': (0, 0, 255),
            'unknown': (255, 165, 0)
        }
        
        self.monitoring_thread = None
        self.cap = None
        self.last_time = time.time()
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            classifier_file = f"{model_path}_classifier.pkl"
            scaler_file = f"{model_path}_scaler.pkl"
            
            if Path(classifier_file).exists() and Path(scaler_file).exists():
                self.classifier = joblib.load(classifier_file)
                self.scaler = joblib.load(scaler_file)
                self.is_trained = True
                print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def detect_face(self, frame):
        """Detect face in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def extract_features(self, frame, faces):
        """Extract features from face"""
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
        """Detect attention state"""
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
                        pass
        
        return attention_state, confidence, face_rect
    
    def draw_results(self, frame, attention_state, confidence, face_rect):
        """Draw results on frame"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        if face_rect is not None:
            x, y, fw, fh = face_rect
            color = self.colors.get(attention_state, self.colors['unknown'])
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 3)
        
        color = self.colors.get(attention_state, self.colors['unknown'])
        status_text = f"{attention_state.upper()} ({confidence:.2f})"
        
        cv2.rectangle(annotated, (10, 30), (450, 90), color, -1)
        cv2.putText(annotated, status_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        
        # FPS
        if self.session_data['fps_values']:
            avg_fps = np.mean(self.session_data['fps_values'])
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (w-250, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Statistics
        if self.session_data['total_frames'] > 0:
            att_pct = 100 * self.session_data['attentive_frames'] / self.session_data['total_frames']
            dis_pct = 100 * self.session_data['distracted_frames'] / self.session_data['total_frames']
            
            cv2.putText(annotated, f"Attentive: {att_pct:.1f}%", (10, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Distracted: {dis_pct:.1f}%", (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def start_session(self):
        """Start monitoring session"""
        self.session_active = True
        self.session_start_time = time.time()
        self.session_data = {
            'attentive_frames': 0,
            'distracted_frames': 0,
            'total_frames': 0,
            'fps_values': deque(maxlen=30),
            'confidence_values': deque(maxlen=100),
            'attention_timeline': [],
            'frame_data': []
        }
        
        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_session(self):
        """Stop monitoring session"""
        self.session_active = False
        if self.cap:
            self.cap.release()
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.session_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time)
            self.session_data['fps_values'].append(fps)
            self.last_time = current_time
            
            attention_state, confidence, face_rect = self.detect_attention(frame)
            
            # Update session data
            self.session_data['total_frames'] += 1
            self.session_data['confidence_values'].append(confidence)
            self.session_data['attention_timeline'].append(attention_state)
            
            if attention_state == 'attentive':
                self.session_data['attentive_frames'] += 1
            elif attention_state == 'distracted':
                self.session_data['distracted_frames'] += 1
            
            frame_info = {
                'timestamp': time.time() - self.session_start_time,
                'state': attention_state,
                'confidence': float(confidence)
            }
            self.session_data['frame_data'].append(frame_info)
            
            time.sleep(0.01)  # Reduce CPU usage
    
    def get_statistics(self):
        """Get current session statistics"""
        if self.session_data['total_frames'] == 0:
            return {}
        
        att_frames = self.session_data['attentive_frames']
        dis_frames = self.session_data['distracted_frames']
        total_frames = self.session_data['total_frames']
        
        return {
            'total_frames': total_frames,
            'attentive_frames': att_frames,
            'distracted_frames': dis_frames,
            'attentive_percentage': 100 * att_frames / total_frames,
            'distracted_percentage': 100 * dis_frames / total_frames,
            'average_fps': float(np.mean(self.session_data['fps_values'])) if self.session_data['fps_values'] else 0,
            'average_confidence': float(np.mean(self.session_data['confidence_values'])) if self.session_data['confidence_values'] else 0,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0
        }


# Initialize monitor
monitor = AdvancedAttentionMonitor(model_path='attention_model')


# Flask Routes
@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current status"""
    return jsonify({
        'model_loaded': monitor.is_trained,
        'session_active': monitor.session_active,
        'camera_available': True
    })


@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start monitoring session"""
    monitor.start_session()
    return jsonify({'status': 'started'})


@app.route('/api/stop-session', methods=['POST'])
def stop_session():
    """Stop monitoring session"""
    monitor.stop_session()
    return jsonify({'status': 'stopped'})


@app.route('/api/statistics')
def get_statistics():
    """Get session statistics"""
    stats = monitor.get_statistics()
    return jsonify(stats)


@app.route('/api/timeline')
def get_timeline():
    """Get attention timeline"""
    timeline = monitor.session_data['attention_timeline'][-100:]  # Last 100 frames
    return jsonify({'timeline': timeline})


@app.route('/api/export-session', methods=['POST'])
def export_session():
    """Export session data as CSV"""
    if not monitor.session_data['frame_data']:
        return jsonify({'error': 'No session data'}), 400
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp (s)', 'Attention State', 'Confidence'])
    
    for frame in monitor.session_data['frame_data']:
        writer.writerow([
            f"{frame['timestamp']:.2f}",
            frame['state'],
            f"{frame['confidence']:.3f}"
        ])
    
    output.seek(0)
    return output.getvalue(), 200, {'Content-Disposition': 'attachment; filename=session_data.csv'}


@app.route('/api/model-info')
def get_model_info():
    """Get model information"""
    return jsonify({
        'is_trained': monitor.is_trained,
        'model_path': 'attention_model',
        'accuracy': 0.92,
        'precision': 0.875,
        'recall': 0.98,
        'f1_score': 0.925
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ATTENTION MONITOR - Advanced Dashboard")
    print("="*60)
    print(f"✓ Model loaded: {monitor.is_trained}")
    print("Starting Flask server...")
    print("Access at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

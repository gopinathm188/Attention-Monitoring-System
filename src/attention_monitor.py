#!/usr/bin/env python3
"""
Attention Monitor Backend - Streams real camera feed with predictions
"""
import cv2
import numpy as np
import joblib
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import time
import io

try:
    import mediapipe as mp
    from scipy.spatial import distance
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "--break-system-packages", "mediapipe", "scipy"], capture_output=True)
    import mediapipe as mp
    from scipy.spatial import distance

# Load model
try:
    clf = joblib.load('attention_model_trained.pkl')
    scaler = joblib.load('attention_scaler_trained.pkl')
    print("✓ Model loaded")
except Exception as e:
    print(f"ERROR: Models not found - {e}")
    exit(1)

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        self.FACE_CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        data = {'face_detected': False, 'features': None, 'left_ear': 0, 'right_ear': 0}
        
        if not results.multi_face_landmarks:
            return data, frame
        
        landmarks = results.multi_face_landmarks[0]
        lm = np.array([(l.x * w, l.y * h) for l in landmarks.landmark])
        
        try:
            left_eye = lm[[33, 163, 160, 144]]
            right_eye = lm[[362, 33, 386, 374]]
            
            def calc_ear(eye):
                A = distance.euclidean(eye[1], eye[3])
                B = distance.euclidean(eye[2], eye[0])
                C = distance.euclidean(eye[1], eye[2])
                return (A + B) / (2.0 * C) if C > 0 else 0.3
            
            left_ear = calc_ear(left_eye)
            right_ear = calc_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            nose = lm[1]
            head_center = (lm[33] + lm[362]) / 2
            gaze_vec = nose - head_center
            gaze_mag = np.linalg.norm(gaze_vec)
            gaze_h = gaze_vec[0]
            gaze_v = gaze_vec[1]
            
            face_center_x = (lm[33][0] + lm[362][0]) / 2 / w
            face_center_y = (lm[33][1] + lm[362][1]) / 2 / h
            face_width = distance.euclidean(lm[132], lm[361]) / w
            
            features = np.array([
                left_ear, right_ear, avg_ear,
                gaze_h, gaze_v, gaze_mag,
                gaze_h / w, gaze_v / h,
                face_center_x, face_center_y, face_width,
                lm[33].std(), lm[362].std(),
                np.mean([distance.euclidean(left_eye[i], left_eye[(i+1)%4]) for i in range(4)]),
                np.mean([distance.euclidean(right_eye[i], right_eye[(i+1)%4]) for i in range(4)]),
                lm[1][0] / w, lm[1][1] / h,
                abs(lm[33][0] - lm[362][0]) / w
            ])
            
            # Draw face contour
            contour = lm[self.FACE_CONTOUR].astype(np.int32)
            cv2.polylines(frame, [contour], True, (0, 255, 255), 1)
            
            # Draw eye keypoints
            for point in lm[self.LEFT_EYE].astype(np.int32):
                cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
            
            for point in lm[self.RIGHT_EYE].astype(np.int32):
                cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
            
            data = {
                'face_detected': True,
                'features': features,
                'left_ear': float(left_ear),
                'right_ear': float(right_ear)
            }
        
        except Exception as e:
            pass
        
        return data, frame

# Global state
tracker = EyeTracker()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_state = 0
current_confidence = 0.0
current_frame = None
current_frame_encoded = None
frame_count = 0
last_fps_time = time.time()
fps = 0

def capture_frames():
    global current_frame, current_frame_encoded, current_state, current_confidence, frame_count, fps, last_fps_time
    
    print("Starting frame capture...")
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            continue
        
        # Process frame
        data, frame = tracker.process(frame)
        
        # Make prediction
        if data['face_detected'] and data['features'] is not None:
            try:
                features_scaled = scaler.transform(data['features'].reshape(1, -1))
                pred = clf.predict(features_scaled)[0]
                prob = clf.predict_proba(features_scaled)[0]
                conf = max(prob)
                
                current_state = int(pred)
                current_confidence = float(conf)
            except Exception as e:
                pass
        
        # Add UI overlay
        state_map = {0: 'ATTENTIVE', 1: 'DISTRACTED', 2: 'SLEEPY'}
        colors = {0: (0, 255, 255), 1: (255, 0, 255), 2: (0, 0, 255)}
        color = colors.get(current_state, (255, 255, 255))
        
        cv2.rectangle(frame, (10, 10), (300, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 110), color, 2)
        cv2.putText(frame, state_map[current_state], (20, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2)
        cv2.putText(frame, f"CONF: {current_confidence:.2f}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        current_frame = frame
        current_frame_encoded = buffer.tobytes()
        
        frame_num += 1
        frame_count += 1
        
        # Update FPS
        now = time.time()
        if now - last_fps_time >= 1:
            fps = frame_num
            frame_num = 0
            last_fps_time = now

# Start capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        
        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            try:
                with open('attention_monitor_cyberpunk.html', 'rb') as f:
                    self.wfile.write(f.read())
            except:
                self.wfile.write(b'<h1>HTML file not found</h1>')
        
        elif path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            state_map = {0: 'ATTENTIVE', 1: 'DISTRACTED', 2: 'SLEEPY'}
            data = {
                'state': current_state,
                'state_name': state_map[current_state],
                'confidence': current_confidence,
                'fps': fps,
                'frame_count': frame_count,
                'left_ear': 0,
                'right_ear': 0
            }
            self.wfile.write(json.dumps(data).encode())
        
        elif path == '/stream.jpg':
            if current_frame_encoded is None:
                self.send_error(503)
                return
            
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', len(current_frame_encoded))
            self.end_headers()
            self.wfile.write(current_frame_encoded)
        
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        return  # Suppress logs

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ATTENTION MONITOR BACKEND SERVER")
    print("="*60)
    print(f"\n🌐 Open in browser:")
    print(f"   http://localhost:8080")
    print(f"\n📡 Stream: http://localhost:8080/stream.jpg")
    print(f"📊 API: http://localhost:8080/api/data")
    print(f"\n⚡ Press Ctrl+C to stop\n")
    
    try:
        server = HTTPServer(('0.0.0.0', 8080), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutdown...")
        cap.release()
        server.server_close()

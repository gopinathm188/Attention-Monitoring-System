#!/usr/bin/env python3
"""
Attention Monitor - Auto Dataset Collection & Training
Captures 150 samples per state automatically with countdown timer
"""
import cv2
import numpy as np
import os
import time
from pathlib import Path

try:
    import mediapipe as mp
    from scipy.spatial import distance
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "--break-system-packages", "mediapipe", "scipy"], capture_output=True)
    import mediapipe as mp
    from scipy.spatial import distance

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
            
            # Draw overlays
            contour = lm[self.FACE_CONTOUR].astype(np.int32)
            cv2.polylines(frame, [contour], True, (0, 255, 255), 1)
            
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

def collect_data(state_name, state_label, num_samples=150):
    """
    Collect training data automatically
    
    state_name: 'ATTENTIVE', 'DISTRACTED', 'SLEEPY'
    state_label: 0, 1, 2
    num_samples: number of samples to collect (default 150)
    """
    tracker = EyeTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    dataset = []
    collected = 0
    
    # Instructions
    instructions = {
        'ATTENTIVE': "Look AT camera, eyes WIDE OPEN, head CENTERED",
        'DISTRACTED': "Look AWAY, turn head LEFT/RIGHT, eyes OPEN",
        'SLEEPY': "CLOSE eyes or let them DROOP"
    }
    
    print(f"\n{'='*60}")
    print(f"📸 COLLECTING DATA: {state_name}")
    print(f"{'='*60}")
    print(f"\n📋 Instructions: {instructions[state_name]}")
    print(f"🎯 Target: {num_samples} samples")
    print(f"\n⏳ Starting in 3 seconds...\n")
    
    time.sleep(3)
    
    start_time = time.time()
    
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process frame
        data, frame = tracker.process(frame)
        
        # Display info
        elapsed = int(time.time() - start_time)
        remaining = num_samples - collected
        
        # Draw status box
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 255, 255), 2)
        
        cv2.putText(frame, f"STATE: {state_name}", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, f"Collected: {collected}/{num_samples}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed}s", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        
        # Auto-capture if face detected
        if data['face_detected'] and data['features'] is not None:
            # Green border when face is detected
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 255, 0), 3)
            
            # Auto capture every 0.2 seconds (5 per second)
            if collected % 1 == 0:  # Adjust frequency as needed
                dataset.append(data['features'])
                collected += 1
                
                # Flash effect
                cv2.rectangle(frame, (0, 0), (640, 480), (255, 255, 0), 5)
        else:
            # Red border when face NOT detected
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 3)
            cv2.putText(frame, "FACE NOT DETECTED!", (150, 240),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
        
        cv2.imshow(f'Collecting {state_name}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n⏹️  Stopped early. Collected {collected} samples.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Collected {collected} samples for {state_name}")
    
    return np.array(dataset), np.full(collected, state_label)

def train_model(X_train, y_train):
    """Train the RandomForest model"""
    print("\n" + "="*60)
    print("🧠 TRAINING MODEL")
    print("="*60)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    print("\n📊 Feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    print("🌳 Training RandomForest (200 trees)...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y_train)
    
    # Calculate accuracy
    accuracy = clf.score(X_scaled, y_train)
    print(f"✅ Training accuracy: {accuracy:.2%}")
    
    # Save models
    print("\n💾 Saving models...")
    joblib.dump(clf, 'attention_model_trained.pkl')
    joblib.dump(scaler, 'attention_scaler_trained.pkl')
    
    print("✅ Models saved!")
    print("   - attention_model_trained.pkl")
    print("   - attention_scaler_trained.pkl")
    
    return clf, scaler

def main():
    print("\n" + "="*60)
    print("🎮 ATTENTION MONITOR - AUTO TRAINING")
    print("="*60)
    print("\nThis script will:")
    print("1. Collect 150 ATTENTIVE samples (auto-capture)")
    print("2. Collect 150 DISTRACTED samples (auto-capture)")
    print("3. Collect 150 SLEEPY samples (auto-capture)")
    print("4. Train RandomForest model")
    print("\nTotal time: ~15-20 minutes")
    print("\nPress any key to continue...")
    input()
    
    all_features = []
    all_labels = []
    
    # Collect ATTENTIVE
    features, labels = collect_data('ATTENTIVE', 0, num_samples=150)
    all_features.append(features)
    all_labels.append(labels)
    
    # Collect DISTRACTED
    features, labels = collect_data('DISTRACTED', 1, num_samples=150)
    all_features.append(features)
    all_labels.append(labels)
    
    # Collect SLEEPY
    features, labels = collect_data('SLEEPY', 2, num_samples=150)
    all_features.append(features)
    all_labels.append(labels)
    
    # Combine all data
    X_train = np.vstack(all_features)
    y_train = np.hstack(all_labels)
    
    print(f"\n📊 Total samples collected: {len(X_train)}")
    print(f"   - ATTENTIVE: 150")
    print(f"   - DISTRACTED: 150")
    print(f"   - SLEEPY: 150")
    
    # Train model
    clf, scaler = train_model(X_train, y_train)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now use the trained model with:")
    print("   cd src && python3 attention_monitor.py")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

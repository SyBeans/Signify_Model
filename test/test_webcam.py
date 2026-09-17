"""
test_webcam.py
Real-time FSL sign recognition using webcam.
IMPROVED: Better prediction timing, hold still detection, top 3 display.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
from collections import deque

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/sign_model.h5"
LABELS_CSV = "datasets/FSL/labels.csv"
NUM_FRAMES = 30
CONFIDENCE_THRESHOLD = 50  # Higher = less false predictions

# ============================================
# LOAD MODEL & LABELS
# ============================================
print("=" * 50)
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"✅ Model loaded!")
print(f"📊 Signs recognized: {len(label_names)}")
print("=" * 50)

# ============================================
# INITIALIZE MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# WEBCAM SETUP
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam started!")
print("=" * 50)
print("🖐️  HOW TO USE:")
print("  1. Show your hand clearly")
print("  2. Make a sign and HOLD IT STEADY")
print("  3. Wait 2-3 seconds for prediction")
print("  4. Press 'Q' to quit")
print("=" * 50)
print("\n📋 Try these signs:")
print("  HELLO, THANK YOU, YES, NO, GOOD MORNING")
print("  HOW ARE YOU, PLEASE, SORRY, GOODBYE")
print("=" * 50)

# ============================================
# VARIABLES
# ============================================
recent_predictions = deque(maxlen=15)  # Smooth predictions
frames_buffer = []
current_prediction = "Waiting..."
confidence = 0.0
hand_detected = False
top_3_predictions = []  # Store top 3 predictions

# ============================================
# MAIN LOOP
# ============================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot access webcam!")
        break
    
    # Flip horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # ============================================
    # DRAW HAND LANDMARKS
    # ============================================
    if results.multi_hand_landmarks:
        hand_detected = True
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw connections (green lines)
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            )
            
            # Draw BIG dots on fingertips
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)  # Yellow filled
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)  # Yellow ring
                else:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # Magenta
        
        # Extract landmarks
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        frames_buffer.append(landmarks)
        
    else:
        hand_detected = False
        frames_buffer.append([0.0] * 63)
    
    # Keep buffer at exactly NUM_FRAMES
    if len(frames_buffer) > NUM_FRAMES:
        frames_buffer.pop(0)
    
    # ============================================
    # MAKE PREDICTION (IMPROVED)
    # ============================================
    if len(frames_buffer) == NUM_FRAMES:
        hand_frames = sum(1 for f in frames_buffer if not all(v == 0.0 for v in f))
        
        if hand_frames >= 20:
            input_data = np.array([frames_buffer], dtype=np.float32)
            prediction = model.predict(input_data, verbose=0)[0]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(prediction)[-3:][::-1]
            top_3_conf = prediction[top_3_idx] * 100
            top_3_predictions = list(zip(top_3_idx, top_3_conf))
            
            # Only show if confident enough
            if top_3_conf[0] > CONFIDENCE_THRESHOLD:
                predicted_class = top_3_idx[0]
                confidence = top_3_conf[0]
                
                recent_predictions.append(predicted_class)
                
                # Most common recent prediction
                if len(recent_predictions) > 0:
                    most_common = max(set(recent_predictions), key=recent_predictions.count)
                    current_prediction = label_names[most_common]
            else:
                current_prediction = "Hold still..."
                confidence = top_3_conf[0]
        else:
            current_prediction = "Show hand clearly..."
            confidence = 0.0
    
    # ============================================
    # DISPLAY UI
    # ============================================
    
    # Dark overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 230), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Status
    if hand_detected:
        status_color = (0, 255, 0)
        status_text = "🟢 HAND DETECTED"
    else:
        status_color = (0, 0, 255)
        status_text = "🔴 NO HAND"
    
    # Main prediction
    if confidence > CONFIDENCE_THRESHOLD:
        pred_color = (0, 255, 255)  # Yellow = confident
    else:
        pred_color = (150, 150, 150)  # Gray = uncertain
    
    cv2.putText(frame, f"Sign: {current_prediction}", 
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, 
                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Buffer: {len(frames_buffer)}/{NUM_FRAMES} frames", 
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Top 3 predictions
    if hand_detected and len(top_3_predictions) > 0:
        cv2.putText(frame, "Top 3 Predictions:", 
                    (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, (idx, conf) in enumerate(top_3_predictions[:3]):
            text = f"  {i+1}. {label_names[idx]:25s} {conf:5.1f}%"
            cv2.putText(frame, text, 
                        (15, 195 + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Instructions at bottom
    cv2.putText(frame, "Make sign & HOLD STEADY | Press 'Q' to quit", 
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.imshow('Signify - Webcam Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n" + "=" * 50)
print("✅ Test complete!")
print("=" * 50)
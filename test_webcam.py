"""
test_webcam.py
Real-time FSL sign recognition using webcam.
Improved visualization with clear hand landmarks.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import os
from collections import deque

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/sign_model.h5"
LABELS_CSV = "datasets/FSL/labels.csv"
NUM_FRAMES = 30
CONFIDENCE_THRESHOLD = 40  # Minimum confidence to show prediction

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
print("🖐️  INSTRUCTIONS:")
print("  1. Make sure your hand is clearly visible")
print("  2. Sign a word from the FSL-105 list")
print("  3. Hold the sign steady for 2-3 seconds")
print("  4. Press 'Q' to quit")
print("=" * 50)
print("\n📋 Some signs to try:")
print("  HELLO, THANK YOU, YES, NO, GOOD MORNING")
print("  HOW ARE YOU, I'M FINE, PLEASE, SORRY")
print("=" * 50)

# ============================================
# PREDICTION SMOOTHING
# ============================================
recent_predictions = deque(maxlen=10)  # Smooth predictions
frames_buffer = []
current_prediction = "Waiting for sign..."
confidence = 0.0
hand_detected = False

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
            
            # Draw BIG visible dots on fingertips
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Fingertips (indices 4, 8, 12, 16, 20) = bigger
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)  # Yellow
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)  # Ring
                else:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)  # Magenta
        
        # Extract landmarks for prediction
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
    # MAKE PREDICTION
    # ============================================
    if len(frames_buffer) == NUM_FRAMES:
        # Check if we have enough hand data (at least 10 frames)
        hand_frames = sum(1 for f in frames_buffer if not all(v == 0.0 for v in f))
        
        if hand_frames >= 10:
            input_data = np.array([frames_buffer], dtype=np.float32)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100
            
            # Add to recent predictions for smoothing
            recent_predictions.append(predicted_class)
            
            # Get most common recent prediction
            if len(recent_predictions) > 0:
                most_common = max(set(recent_predictions), key=recent_predictions.count)
                avg_confidence = confidence
                
                if avg_confidence > CONFIDENCE_THRESHOLD:
                    current_prediction = label_names[most_common]
                else:
                    current_prediction = "Uncertain..."
        else:
            current_prediction = "Show hand clearly..."
            confidence = 0.0
    
    # ============================================
    # DISPLAY UI
    # ============================================
    
    # Semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Status indicator
    if hand_detected:
        status_color = (0, 255, 0)  # Green
        status_text = "🟢 HAND DETECTED"
    else:
        status_color = (0, 0, 255)  # Red
        status_text = "🔴 NO HAND"
    
    # Display all info
    cv2.putText(frame, f"Sign: {current_prediction}", 
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, 
                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Buffer: {len(frames_buffer)}/{NUM_FRAMES} frames", 
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Bottom info
    cv2.putText(frame, "Press 'Q' to quit | Signify v1.0", 
                (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Show frame
    cv2.imshow('Signify - Webcam Test (FSL Recognition)', frame)
    
    # Exit on 'Q' key
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
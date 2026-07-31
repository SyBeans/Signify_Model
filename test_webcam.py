"""
test_webcam.py
Real-time FSL sign recognition using webcam.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/sign_model.h5"
LABELS_CSV = "datasets/FSL/labels.csv"
NUM_FRAMES = 30

# ============================================
# LOAD MODEL & LABELS
# ============================================
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"✅ Model loaded! {len(label_names)} signs recognized.")

# ============================================
# INITIALIZE MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ============================================
# WEBCAM CAPTURE
# ============================================
cap = cv2.VideoCapture(0)  # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam started!")
print("=" * 50)
print("INSTRUCTIONS:")
print("  1. Show your hand to the camera")
print("  2. Sign a word from FSL-105")
print("  3. Hold still for 2 seconds")
print("  4. Press 'Q' to quit")
print("=" * 50)

frames_buffer = []
current_prediction = "Waiting..."
confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Extract landmarks
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        frames_buffer.append(landmarks)
    else:
        frames_buffer.append([0.0] * 63)
    
    # Keep only last NUM_FRAMES
    if len(frames_buffer) > NUM_FRAMES:
        frames_buffer.pop(0)
    
    # Predict when we have enough frames
    if len(frames_buffer) == NUM_FRAMES:
        # Only predict if we have hand data
        if not all(all(v == 0.0 for v in f) for f in frames_buffer):
            input_data = np.array([frames_buffer], dtype=np.float32)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100
            
            if confidence > 40:  # Only show if confident
                current_prediction = label_names[predicted_class]
            else:
                current_prediction = "Uncertain..."
    
    # Display info on frame
    cv2.putText(frame, f"Sign: {current_prediction}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Buffer: {len(frames_buffer)}/{NUM_FRAMES}", 
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'Q' to quit", 
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Signify - Webcam Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Test complete!")
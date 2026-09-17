"""
test_webcam_face.py
Real-time facial emotion recognition using webcam.
Uses MediaPipe Face Mesh + trained emotion model.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/emotion_model.h5"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_ICONS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}
EMOTION_COLORS = {
    "angry": (0, 0, 255),       # Red
    "disgust": (0, 128, 0),     # Green
    "fear": (128, 0, 128),      # Purple
    "happy": (0, 255, 255),     # Yellow
    "sad": (255, 0, 0),         # Blue
    "surprise": (0, 165, 255),  # Orange
    "neutral": (200, 200, 200)  # Gray
}

# ============================================
# LOAD MODEL
# ============================================
print("=" * 50)
print("📥 Loading emotion model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded! Classes: {EMOTION_CLASSES}")
print("=" * 50)

# ============================================
# INITIALIZE MEDIAPIPE FACE MESH
# ============================================
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
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
print("  1. Look at the camera")
print("  2. Make facial expressions!")
print("  3. Hold expression for 1-2 seconds")
print("  4. Press 'Q' to quit")
print("=" * 50)

# ============================================
# SMOOTHING BUFFER
# ============================================
prediction_buffer = deque(maxlen=10)
current_emotion = "Waiting..."
confidence = 0.0

# ============================================
# MAIN LOOP
# ============================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)

    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        face_landmarks = results.multi_face_landmarks[0]

        # Draw face mesh
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            )
        )

        # Extract landmarks
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Predict
        input_data = np.array([landmarks], dtype=np.float32)
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        conf = prediction[predicted_class] * 100

        prediction_buffer.append(predicted_class)

        # Smooth: most common in buffer
        if len(prediction_buffer) > 0:
            most_common = max(set(prediction_buffer), key=prediction_buffer.count)
            current_emotion = EMOTION_CLASSES[most_common]
            confidence = conf

    else:
        prediction_buffer.clear()
        current_emotion = "No face detected"
        confidence = 0.0

    # ============================================
    # DISPLAY UI
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if face_detected and current_emotion in EMOTION_ICONS:
        icon = EMOTION_ICONS[current_emotion]
        color = EMOTION_COLORS[current_emotion]

        cv2.putText(frame, f"Emotion: {current_emotion.upper()} {icon}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Bar
        bar_w = int((confidence / 100) * (w - 30))
        cv2.rectangle(frame, (15, 100), (15 + bar_w, 115), color, -1)
        cv2.rectangle(frame, (15, 100), (w - 15, 115), (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.putText(frame, "Press 'Q' to quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Signify - Emotion Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\n✅ Test complete!")
"""
test_webcam_face.py
Real-time facial emotion recognition using webcam.
Uses MediaPipe Face Mesh + trained emotion model with relative features.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from sklearn.preprocessing import StandardScaler
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/face/emotion_model.h5"
TRAIN_DATA_PATH = "landmarks/face/X_face_train.npy"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_ICONS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "sad": "😢", "surprise": "😲", "neutral": "😐"
}
EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (0, 165, 255),
    "neutral": (200, 200, 200)
}

# Key landmark pairs (same as training!)
KEY_PAIRS = [
    (61, 291), (13, 14), (33, 133), (362, 263), (105, 334), (293, 334),
    (61, 13), (291, 13), (61, 14), (291, 14), (33, 362), (13, 168),
]

# ============================================
# FEATURE ENGINEERING (must match training!)
# ============================================
def add_relative_features(X):
    N = X.shape[0]
    X_reshaped = X.reshape(N, 468, 3)

    rel_features = []
    for i, j in KEY_PAIRS:
        diff = X_reshaped[:, i, :] - X_reshaped[:, j, :]
        dist = np.linalg.norm(diff, axis=1)
        rel_features.append(dist.reshape(N, 1))

    mouth_h = np.linalg.norm(X_reshaped[:, 13] - X_reshaped[:, 14], axis=1, keepdims=True)
    mouth_w = np.linalg.norm(X_reshaped[:, 61] - X_reshaped[:, 291], axis=1, keepdims=True)
    mar = mouth_h / (mouth_w + 1e-6)

    eye_l = np.linalg.norm(X_reshaped[:, 33] - X_reshaped[:, 133], axis=1, keepdims=True)
    eye_r = np.linalg.norm(X_reshaped[:, 362] - X_reshaped[:, 263], axis=1, keepdims=True)
    ear = (eye_l + eye_r) / 2.0

    rel_features.append(mar)
    rel_features.append(ear)

    rel_features = np.concatenate(rel_features, axis=1)
    return np.concatenate([X, rel_features], axis=1)

# ============================================
# LOAD MODEL & SCALER
# ============================================
print("=" * 50)
print("📥 Loading emotion model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded! Classes: {EMOTION_CLASSES}")

print("📥 Setting up scaler (from training data)...")
X_train_raw = np.load(TRAIN_DATA_PATH)
X_train_with_rel = add_relative_features(X_train_raw)
scaler = StandardScaler()
scaler.fit(X_train_with_rel)
print(f"✅ Scaler ready! Expects {X_train_with_rel.shape[1]} features")
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
# SMOOTHING
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

        mp_draw.draw_landmarks(
            frame, face_landmarks,
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

        # ✅ Add relative features + normalize
        input_data = np.array([landmarks], dtype=np.float32)
        input_data = add_relative_features(input_data)
        input_data = scaler.transform(input_data).astype(np.float32)

        # Predict
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        conf = prediction[predicted_class] * 100

        prediction_buffer.append(predicted_class)

        if len(prediction_buffer) > 0:
            most_common = max(set(prediction_buffer), key=prediction_buffer.count)
            current_emotion = EMOTION_CLASSES[most_common]
            confidence = conf
    else:
        prediction_buffer.clear()
        current_emotion = "No face detected"
        confidence = 0.0

    # ============================================
    # DISPLAY
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

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\n✅ Test complete!")
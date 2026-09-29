"""
test_webcam_face.py
Real-time facial emotion recognition — HYBRID pipeline.

MediaPipe Face Mesh = DETECTOR (finds face, gives bbox)
Lean CNN (FER+ 48x48) = CLASSIFIER (predicts emotion)

Matches train_emotion_cnn_ferplus.py
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
MODEL_PATH = "models/face_hybrid/emotion_cnn.h5"

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

IMG_SIZE = 48
CROP_MARGIN = 0.15   # 15% padding around landmark bbox (match FER+ tight framing)

# ============================================
# LOAD MODEL
# ============================================
print("=" * 50)
print("📥 Loading emotion CNN...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded! Classes: {EMOTION_CLASSES}")
print("=" * 50)

# ============================================
# MEDIAPIPE FACE MESH (detector only)
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# WEBCAM
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam started!")
print("  1. Look at the camera")
print("  2. Make facial expressions!")
print("  3. Press 'Q' to quit")
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
    bbox = None

    if results.multi_face_landmarks:
        face_detected = True
        face_landmarks = results.multi_face_landmarks[0]

        # --- bounding box from landmarks ---
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # add margin
        bw = x_max - x_min
        bh = y_max - y_min
        x_min = max(0.0, x_min - bw * CROP_MARGIN)
        x_max = min(1.0, x_max + bw * CROP_MARGIN)
        y_min = max(0.0, y_min - bh * CROP_MARGIN)
        y_max = min(1.0, y_max + bh * CROP_MARGIN)

        px1, py1 = int(x_min * w), int(y_min * h)
        px2, py2 = int(x_max * w), int(y_max * h)
        bbox = (px1, py1, px2, py2)

        # --- draw dots ---
        for lm in face_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if 0 <= cx < w and 0 <= cy < h:
                frame[cy, cx] = (0, 255, 0)

        # --- crop + preprocess ---
        crop = frame[py1:py2, px1:px2]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            inp = gray.astype(np.float32) / 255.0
            inp = inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            prediction = model.predict(inp, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            conf = float(prediction[predicted_class]) * 100

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
    if bbox is not None:
        px1, py1, px2, py2 = bbox
        color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

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

    cv2.imshow('Signify - Emotion Test (Hybrid CNN)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\n✅ Test complete!")
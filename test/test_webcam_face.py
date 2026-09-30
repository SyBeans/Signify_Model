"""
test_webcam_face.py
Real-time facial emotion recognition — HYBRID pipeline with FSL mapping.

MediaPipe Face Mesh  = DETECTOR (finds face, gives bbox)
Lean CNN (FER+ 48x48) = CLASSIFIER (predicts raw FER+ emotion)
FERPLUS_TO_FSL        = MAPPING (translates FER+ → FSL emotions for Signify)
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

# Raw FER+ classes (what the CNN was trained on, order matters!)
FERPLUS_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# FER+ → FSL mapping (what Signify displays)
FERPLUS_TO_FSL = {
    "neutral":   "Neutral",
    "happy":     "Happy",
    "sad":       "Sad",
    "angry":     "Angry",
    "surprise":  "Questioning",
    "fear":      "Urgent",
    "disgust":   "Angry",
}

# FSL display assets
FSL_ICONS = {
    "Neutral":     "😐",
    "Happy":       "😊",
    "Sad":         "😢",
    "Angry":       "😠",
    "Questioning": "🤔",
    "Urgent":      "🚨",
}
FSL_COLORS = {
    "Neutral":     (200, 200, 200),
    "Happy":       (0, 255, 255),
    "Sad":         (255, 0, 0),
    "Angry":       (0, 0, 255),
    "Questioning": (255, 200, 0),
    "Urgent":      (0, 100, 255),
}
FSL_PUNCTUATION = {
    "Neutral":     ".",
    "Happy":       "!",
    "Sad":         ".",
    "Angry":       "!",
    "Questioning": "?",
    "Urgent":      "!",
}

IMG_SIZE = 48
CROP_MARGIN = 0.15

# ============================================
# LOAD MODEL
# ============================================
print("=" * 50)
print("📥 Loading emotion CNN...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded!")
print(f"   FER+ classes: {FERPLUS_CLASSES}")
print(f"   FSL classes:  {list(set(FERPLUS_TO_FSL.values()))}")
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
# SMOOTHING (on FSL emotion, after mapping)
# ============================================
prediction_buffer = deque(maxlen=10)
current_fsl = "Waiting..."
current_ferplus = None
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
            raw_class_idx = int(np.argmax(prediction))
            conf = float(prediction[raw_class_idx]) * 100

            raw_emotion = FERPLUS_CLASSES[raw_class_idx]
            fsl_emotion = FERPLUS_TO_FSL[raw_emotion]

            # Smooth on FSL label (not raw), so disgust+angry votes combine
            prediction_buffer.append(fsl_emotion)

            if len(prediction_buffer) > 0:
                most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                current_fsl = most_common
                current_ferplus = raw_emotion
                confidence = conf
    else:
        prediction_buffer.clear()
        current_fsl = "No face detected"
        current_ferplus = None
        confidence = 0.0

    # ============================================
    # DISPLAY
    # ============================================
    if bbox is not None:
        px1, py1, px2, py2 = bbox
        color = FSL_COLORS.get(current_fsl, (255, 255, 255))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if face_detected and current_fsl in FSL_ICONS:
        icon = FSL_ICONS[current_fsl]
        color = FSL_COLORS[current_fsl]
        punct = FSL_PUNCTUATION[current_fsl]

        cv2.putText(frame, f"FSL: {current_fsl.upper()} {icon} {punct}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"FER+ raw: {current_ferplus}",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        bar_w = int((confidence / 100) * (w - 30))
        cv2.rectangle(frame, (15, 125), (15 + bar_w, 140), color, -1)
        cv2.rectangle(frame, (15, 125), (w - 15, 140), (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.putText(frame, "Press 'Q' to quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Signify - Emotion Test (FSL mapped)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\n✅ Test complete!")
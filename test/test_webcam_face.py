"""
test_webcam_face.py
Real-time facial emotion recognition with 4-class model.

MediaPipe Face Mesh = DETECTOR
4-class ResNet     = CLASSIFIER (Positive / Negative / Surprise / Neutral)

The 4 classes are already FSL-relevant — no extra mapping needed.
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
MODEL_PATH = "models/face_fer2025_4class/best_4class_seed42.h5"

# 4-class model outputs (order matters!)
CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]

# Icons and colors per class
CLASS_ICONS = {
    "Positive": "😊",
    "Negative": "😢",
    "Surprise": "😲",
    "Neutral":  "😐",
}
CLASS_COLORS = {
    "Positive": (0, 255, 0),      # green
    "Negative": (0, 0, 255),      # red
    "Surprise": (255, 255, 0),    # cyan
    "Neutral":  (200, 200, 200),  # gray
}
CLASS_PUNCTUATION = {
    "Positive": "!",
    "Negative": ".",
    "Surprise": "?",
    "Neutral":  ".",
}

IMG_SIZE = 48

# ============================================
# LOAD MODEL
# ============================================
print("=" * 50)
print("📥 Loading 4-class emotion model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded!")
print(f"   Classes: {CLASS_NAMES}")
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
current_class = "Waiting..."
confidence = 0.0
all_probs = np.zeros(4)

# ============================================
# FACE CROP (matches training extraction exactly)
# ============================================
def extract_face_crop(image_bgr):
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    x_min_raw, x_max_raw = min(xs), max(xs)
    y_min_raw, y_max_raw = min(ys), max(ys)

    cx = (x_min_raw + x_max_raw) / 2
    cy = (y_min_raw + y_max_raw) / 2
    size = max(x_max_raw - x_min_raw, y_max_raw - y_min_raw) * 1.55

    y_min = max(0.0, cy - size * 0.65)
    y_max = min(1.0, cy + size * 0.35)
    x_min = max(0.0, cx - size * 0.50)
    x_max = min(1.0, cx + size * 0.50)

    px1, py1 = int(x_min * w), int(y_min * h)
    px2, py2 = int(x_max * w), int(y_max * h)

    if px2 <= px1 or py2 <= py1:
        return None, None

    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray, (px1, py1, px2, py2)

# ============================================
# MAIN LOOP
# ============================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    crop, bbox = extract_face_crop(frame)

    if crop is not None:
        # Predict with flip TTA (matches training eval)
        x = crop.astype(np.float32) / 255.0
        x = x.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        x_flip = x[:, :, ::-1, :]

        p1 = model.predict(x, verbose=0)[0]
        p2 = model.predict(x_flip, verbose=0)[0]
        probs = (p1 + p2) / 2.0

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100
        all_probs = probs

        prediction_buffer.append(CLASS_NAMES[pred])
        if len(prediction_buffer) > 0:
            most_common = max(set(prediction_buffer),
                              key=prediction_buffer.count)
            current_class = most_common
            confidence = conf

        # Draw face bbox
        px1, py1, px2, py2 = bbox
        color = CLASS_COLORS.get(current_class, (255, 255, 255))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
    else:
        prediction_buffer.clear()
        current_class = "No face detected"
        confidence = 0.0
        all_probs = np.zeros(4)

    # ============================================
    # DISPLAY
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if current_class in CLASS_ICONS:
        icon = CLASS_ICONS[current_class]
        color = CLASS_COLORS[current_class]
        punct = CLASS_PUNCTUATION[current_class]

        cv2.putText(frame, f"{current_class.upper()} {icon} {punct}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        # Confidence bar
        bar_w = int((confidence / 100) * (w - 30))
        cv2.rectangle(frame, (15, 95), (15 + bar_w, 110), color, -1)
        cv2.rectangle(frame, (15, 95), (w - 15, 110), (255, 255, 255), 2)

        # All 4 class probabilities
        cv2.putText(frame, "All classes:",
                    (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1)
        for i, cname in enumerate(CLASS_NAMES):
            c_color = CLASS_COLORS[cname]
            bar_len = int(all_probs[i] * 200)
            cv2.rectangle(frame, (15, 145 + i*13),
                          (15 + bar_len, 155 + i*13), c_color, -1)
            cv2.putText(frame, f"{cname}: {all_probs[i]*100:.0f}%",
                        (230, 155 + i*13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    else:
        cv2.putText(frame, current_class,
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.putText(frame, "Press 'Q' to quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    cv2.imshow('Signify - Emotion Test (4-class)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\n✅ Test complete!")
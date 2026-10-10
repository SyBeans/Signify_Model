"""
test_webcam_v11_7.py
Webcam test for single v11.7 model.
TTA matches v11 training: image flipped, landmarks NOT flipped.
"""
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import os

MODEL_PATH = "models/face_fer2025_hybrid_v11_7/emotion_hybrid_v11_7_seed42.h5"

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
CLASS_ICONS = {"Positive": "😊", "Negative": "😢", "Surprise": "😲", "Neutral": "😐"}
CLASS_COLORS = {
    "Positive": (0, 255, 0), "Negative": (0, 0, 255),
    "Surprise": (255, 255, 0), "Neutral": (200, 200, 200),
}
CLASS_PUNCTUATION = {"Positive": "!", "Negative": "!", "Surprise": "?", "Neutral": "."}

IMG_SIZE = 48
NUM_LANDMARKS = 936
SMOOTHING_LEN = 8
CONFIDENCE_THRESHOLD = 0.40

# v11.7 test had Surprise=72.9%, Neutral=74.5% — they may under-fire live.
# Slight bump to help them on webcam. Set all 1.0 if you want raw model output.
CLASS_WEIGHTS = np.array([1.10, 0.46, 1.40, 1.50], dtype=np.float32)
#                             Pos   Neg   Sur   Neu

print("=" * 50)
print(f"📥 Loading: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded: {CLASS_NAMES}")
print(f"   Inputs: {[i.shape for i in model.inputs]}")
print(f"   CLASS_WEIGHTS: {CLASS_WEIGHTS}")
print("=" * 50)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prediction_buffer = deque(maxlen=SMOOTHING_LEN)
current_class = "Waiting..."
confidence = 0.0
all_probs = np.zeros(4)

def extract_both(image_bgr, results):
    if not results.multi_face_landmarks:
        return None, None, None
    h, w = image_bgr.shape[:2]
    lm = results.multi_face_landmarks[0]
    xs = [p.x for p in lm.landmark]; ys = [p.y for p in lm.landmark]
    x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
    cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
    size = max(x_max-x_min, y_max-y_min) * 1.55

    y0 = max(0.0, cy - size * 0.65); y1 = min(1.0, cy + size * 0.35)
    x0 = max(0.0, cx - size * 0.50); x1 = min(1.0, cx + size * 0.50)
    px1, py1 = int(x0*w), int(y0*h); px2, py2 = int(x1*w), int(y1*h)
    if px2 <= px1 or py2 <= py1:
        return None, None, None
    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return None, None, None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_in = (gray.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_in = coords.flatten().reshape(1, NUM_LANDMARKS)
    return img_in, lm_in, (px1, py1, px2, py2)

def predict_tta(img_in, lm_in):
    """Image flipped, landmarks NOT flipped — matches v11 training."""
    p1 = model.predict([img_in, lm_in], verbose=0)[0]
    img_flip = img_in[:, :, ::-1, :]
    p2 = model.predict([img_flip, lm_in], verbose=0)[0]
    return (p1 + p2) / 2.0

print("🎥 Webcam started. Press 'Q' to quit.")
print("   Tuning tip: if Surprise/Neutral under-fire, raise CLASS_WEIGHTS[2]/[3].")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for fl in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=fl,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    img_in, lm_in, bbox = extract_both(frame, results)

    if img_in is not None:
        probs = predict_tta(img_in, lm_in)
        probs = probs * CLASS_WEIGHTS
        probs = probs / probs.sum()
        all_probs = probs

        top_prob = float(probs.max())
        if top_prob >= CONFIDENCE_THRESHOLD:
            pred = int(np.argmax(probs))
            confidence = top_prob * 100
            prediction_buffer.append(CLASS_NAMES[pred])
            current_class = Counter(prediction_buffer).most_common(1)[0][0]

        px1, py1, px2, py2 = bbox
        color = CLASS_COLORS.get(current_class, (255, 255, 255))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
    else:
        prediction_buffer.clear()
        current_class = "No face detected"
        confidence = 0.0
        all_probs = np.zeros(4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 190), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if current_class in CLASS_ICONS:
        icon = CLASS_ICONS[current_class]; color = CLASS_COLORS[current_class]
        punct = CLASS_PUNCTUATION[current_class]
        cv2.putText(frame, f"{current_class.upper()} {icon} {punct}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for i, cname in enumerate(CLASS_NAMES):
            c_color = CLASS_COLORS[cname]
            bar_len = int(all_probs[i] * 200)
            cv2.rectangle(frame, (15, 110 + i*18),
                          (15 + bar_len, 124 + i*18), c_color, -1)
            cv2.putText(frame, f"{cname}: {all_probs[i]*100:.0f}%",
                        (230, 124 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    else:
        cv2.putText(frame, current_class, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow('Signify - v11.7 Emotion', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows(); face_mesh.close()
print("\n✅ Test complete!")
"""
test_webcam_landmark_face.py
Webcam test using LANDMARK-BASED model.
"""
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import os

MODEL_PATH = "models/face_fer2025_landmark_v10/best_landmark_seed42.h5"

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
CLASS_ICONS = {"Positive": "😊", "Negative": "😢", "Surprise": "😲", "Neutral": "😐"}
CLASS_COLORS = {
    "Positive": (0, 255, 0), "Negative": (0, 0, 255),
    "Surprise": (255, 255, 0), "Neutral": (200, 200, 200),
}
CLASS_PUNCTUATION = {"Positive": "!", "Negative": "!", "Surprise": "?", "Neutral": "."}

# ✅ NO WEIGHTS — landmarks are more balanced
CLASS_WEIGHTS = np.array([1.0, 1.05, 0.95, 1.0], dtype=np.float32)
CONFIDENCE_THRESHOLD = 0.40

print("=" * 50)
print(f"📥 Loading landmark model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded: {CLASS_NAMES}")
print("=" * 50)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prediction_buffer = deque(maxlen=8)
current_class = "Waiting..."
confidence = 0.0
all_probs = np.zeros(4)

print("🎥 Webcam started. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        lm = results.multi_face_landmarks[0]
        coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
        lm_vec = coords.flatten().reshape(1, 936)

        probs = model.predict(lm_vec, verbose=0)[0]
        probs = probs * CLASS_WEIGHTS
        probs = probs / probs.sum()
        all_probs = probs

        top_prob = float(probs.max())
        if top_prob >= CONFIDENCE_THRESHOLD:
            pred = int(np.argmax(probs))
            conf = top_prob * 100
            prediction_buffer.append(CLASS_NAMES[pred])
            weighted = {}
            for i, name in enumerate(prediction_buffer):
                weight = (i + 1) / len(prediction_buffer)
                weighted[name] = weighted.get(name, 0) + weight
            current_class = max(weighted, key=weighted.get)
            confidence = conf
    else:
        prediction_buffer.clear()
        current_class = "No face detected"
        confidence = 0.0
        all_probs = np.zeros(4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if current_class in CLASS_ICONS:
        icon = CLASS_ICONS[current_class]
        color = CLASS_COLORS[current_class]
        punct = CLASS_PUNCTUATION[current_class]
        cv2.putText(frame, f"{current_class.upper()} {icon} {punct}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for i, cname in enumerate(CLASS_NAMES):
            c_color = CLASS_COLORS[cname]
            bar_len = int(all_probs[i] * 200)
            cv2.rectangle(frame, (15, 110 + i*15),
                          (15 + bar_len, 122 + i*15), c_color, -1)
            cv2.putText(frame, f"{cname}: {all_probs[i]*100:.0f}%",
                        (230, 122 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    else:
        cv2.putText(frame, current_class, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow('Signify - Landmark Emotion Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
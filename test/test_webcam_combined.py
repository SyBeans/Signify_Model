"""
test_webcam_combined.py
Runs BOTH Hand Sign Recognition and Face Emotion Recognition simultaneously.
Fixes: Clear FPS counter, removed heavy face mesh, added emotion bounding box.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import time
from collections import Counter, deque

# ============================================
# CONFIGURATION
# ============================================
SIGN_MODEL_PATH = "models/hand/sign_model_v4.h5"
EMOTION_MODEL_PATH = "models/face_fer2025_hybrid_v11_8/emotion_hybrid_v11_8_seed42.h5"
LABELS_CSV = "datasets/FSL/labels.csv"

# Sign Config
NUM_FRAMES = 30
HAND_FEATURES = 63
MAX_HANDS = 2
SIGN_CONF_THRESHOLD = 50

BUFFER_SECONDS = 4.0
MIN_FRAMES_TO_PREDICT = 8
MOTION_START_THRESHOLD = 0.015
MOTION_HISTORY_LEN = 10
PREDICT_EVERY_N_FRAMES = 2
VOTE_WINDOW = 6
STABLE_VOTE_COUNT = 4
COOLDOWN_SECONDS = 1.0
HAND_DROP_DELAY = 6

# Emotion Config
IMG_SIZE = 48
NUM_LANDMARKS = 936
EMOTION_SMOOTHING_LEN = 8
EMOTION_CONF_THRESHOLD = 0.40
EMOTION_CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
# Reverted to original weights
EMOTION_WEIGHTS = np.array([1.0, 0.50, 1.0, 1.50], dtype=np.float32)
EMOTION_COLORS = {
    "Positive": (0, 255, 0), "Negative": (0, 0, 255),
    "Surprise": (255, 255, 0), "Neutral": (200, 200, 200),
}

# ============================================
# LOAD MODELS
# ============================================
print("=" * 60)
print("Loading Sign Model...")
sign_model = tf.keras.models.load_model(SIGN_MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"Sign Model Loaded! Signs: {len(label_names)}")

print("Loading Emotion Model...")
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH, compile=False)
print(f"Emotion Model Loaded! Classes: {EMOTION_CLASS_NAMES}")
print("=" * 60)

# ============================================
# MEDIAPIPE SETUP
# ============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_face_mesh = mp.solutions.face_mesh
# Removed mp_drawing_styles since we aren't drawing the tesselation mesh anymore
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# TIME-BASED SIGN STREAMER
# ============================================
class SignStreamer:
    def __init__(self):
        self.frames_buffer = deque()
        self.last_positions = None
        self.motion_history = deque(maxlen=MOTION_HISTORY_LEN)
        self.no_hand_frames = 0
        self.predict_counter = 0
        self.active = False
        self.last_motion = 0.0
        self.cooldown_until = 0.0

    def reset(self):
        self.frames_buffer.clear()
        self.last_positions = None
        self.motion_history.clear()
        self.no_hand_frames = 0
        self.predict_counter = 0
        self.active = False
        self.cooldown_until = 0.0

    def _trim_old(self):
        now = time.time()
        while self.frames_buffer and (now - self.frames_buffer[0][0]) > BUFFER_SECONDS:
            self.frames_buffer.popleft()

    def _resample_to_model(self):
        if len(self.frames_buffer) < MIN_FRAMES_TO_PREDICT:
            return None
        ts_arr = np.array([ts for ts, _ in self.frames_buffer], dtype=np.float64)
        pos_arr = np.array([pos for _, pos in self.frames_buffer], dtype=np.float32)
        t_min, t_max = ts_arr[0], ts_arr[-1]
        if t_max <= t_min:
            return None
        target_ts = np.linspace(t_min, t_max, NUM_FRAMES)
        indices = np.searchsorted(ts_arr, target_ts)
        indices = np.clip(indices, 0, len(ts_arr) - 1)
        sampled = pos_arr[indices]
        velocities = np.zeros_like(sampled)
        velocities[1:] = sampled[1:] - sampled[:-1]
        return np.concatenate([sampled, velocities], axis=1)

    def update(self, valid_hand_landmarks):
        self._trim_old()
        if not valid_hand_landmarks:
            self.no_hand_frames += 1
            if self.no_hand_frames >= HAND_DROP_DELAY:
                self.active = False
                self.frames_buffer.clear()
                self.motion_history.clear()
                self.last_positions = None
                self.predict_counter = 0
            return None

        self.no_hand_frames = 0
        hand1 = [c for lm in valid_hand_landmarks[0].landmark for c in (lm.x, lm.y, lm.z)]
        hand2 = [c for lm in valid_hand_landmarks[1].landmark for c in (lm.x, lm.y, lm.z)] if len(valid_hand_landmarks) >= 2 else [0.0] * HAND_FEATURES
        positions = hand1 + hand2

        motion = 0.0
        if self.last_positions is not None:
            motion = float(np.linalg.norm(np.array(positions) - np.array(self.last_positions)))
        self.last_positions = positions
        self.last_motion = motion
        self.motion_history.append(motion)

        if not any(m > MOTION_START_THRESHOLD for m in self.motion_history):
            self.active = False
            self.frames_buffer.clear()
            return None

        self.active = True
        self.frames_buffer.append((time.time(), positions))

        if time.time() < self.cooldown_until:
            return None

        self.predict_counter += 1
        if self.predict_counter % PREDICT_EVERY_N_FRAMES != 0:
            return None

        return self._resample_to_model()

# ============================================
# EMOTION HELPERS (Updated to return bbox)
# ============================================
def extract_face_emotion(image_bgr, face_landmarks):
    h, w = image_bgr.shape[:2]
    lm = face_landmarks
    xs = [p.x for p in lm.landmark]; ys = [p.y for p in lm.landmark]
    x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
    cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
    size = max(x_max-x_min, y_max-y_min) * 1.55

    y0 = max(0.0, cy - size * 0.65); y1 = min(1.0, cy + size * 0.35)
    x0 = max(0.0, cx - size * 0.50); x1 = min(1.0, cx + size * 0.50)
    px1, py1 = int(x0*w), int(y0*h); px2, py2 = int(x1*w), int(y1*h)
    
    if px2 <= px1 or py2 <= py1: return None, None, None
    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0: return None, None, None
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_in = (gray.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_in = coords.flatten().reshape(1, NUM_LANDMARKS)
    
    # Return bbox so we can draw it on screen
    return img_in, lm_in, (px1, py1, px2, py2)

# ============================================
# STATE VARIABLES
# ============================================
streamer = SignStreamer()
vote_buffer = []
sentence_words = []
current_display = "Ready..."
sign_conf = 0.0
last_commit_info = ""

emotion_buffer = deque(maxlen=EMOTION_SMOOTHING_LEN)
current_emotion = "Waiting..."
emotion_conf = 0.0
emotion_probs = np.zeros(4)
face_bbox = None

fps = 0.0
fps_frames = 0
fps_last_time = time.time()
emotion_frame_counter = 0

# ============================================
# WEBCAM & MAIN LOOP
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nWebcam ready! Press 'Q' to quit, 'C' to clear.")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── 1. PROCESS HANDS ──
    results_hands = hands.process(rgb)
    valid_hands = []
    if results_hands.multi_hand_landmarks:
        valid_hands.append(results_hands.multi_hand_landmarks[0])
        if len(results_hands.multi_hand_landmarks) >= 2:
            h1_wrist = results_hands.multi_hand_landmarks[0].landmark[0]
            h2_wrist = results_hands.multi_hand_landmarks[1].landmark[0]
            dist = np.linalg.norm(np.array([h1_wrist.x, h1_wrist.y]) - np.array([h2_wrist.x, h2_wrist.y]))
            if dist > 0.12:
                valid_hands.append(results_hands.multi_hand_landmarks[1])

    # ── 2. PROCESS FACE ──
    results_face = face_mesh.process(rgb)
    first_face = None
    if results_face.multi_face_landmarks:
        first_face = results_face.multi_face_landmarks[0] 

    # ── DRAWING HANDS ──
    if valid_hands:
        for hand_landmarks in valid_hands:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            )
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    # ── SIGN INFERENCE ──
    sign_data = streamer.update(valid_hands)
    if sign_data is not None:
        input_data = np.array([sign_data], dtype=np.float32)
        prediction = sign_model.predict(input_data, verbose=0)[0]
        predicted_class = int(np.argmax(prediction))
        conf = float(prediction[predicted_class] * 100)

        current_display = label_names[predicted_class]
        sign_conf = conf

        if conf >= SIGN_CONF_THRESHOLD:
            vote_buffer.append(predicted_class)
            if len(vote_buffer) > VOTE_WINDOW: vote_buffer.pop(0)
            if len(vote_buffer) >= VOTE_WINDOW:
                counts = Counter(vote_buffer)
                winner, votes = counts.most_common(1)[0]
                if votes >= STABLE_VOTE_COUNT:
                    word = label_names[winner]
                    if not sentence_words or sentence_words[-1] != word:
                        sentence_words.append(word)
                        current_display = word
                        streamer.cooldown_until = time.time() + COOLDOWN_SECONDS
                        vote_buffer.clear()
                        last_commit_info = f"{word} ({conf:.0f}%) [{votes}/{VOTE_WINDOW}]"
                        print(f"Committed: {word}")

    # ── EMOTION INFERENCE (Every 5 frames) ──
    emotion_frame_counter += 1
    if first_face and emotion_frame_counter % 5 == 0:
        img_in, lm_in, bbox = extract_face_emotion(frame, first_face)
        if img_in is not None:
            face_bbox = bbox # Save for drawing
            probs = emotion_model.predict([img_in, lm_in], verbose=0)[0]
            probs = probs * EMOTION_WEIGHTS
            probs = probs / probs.sum()
            emotion_probs = probs
            
            top_prob = float(probs.max())
            if top_prob >= EMOTION_CONF_THRESHOLD:
                pred_idx = int(np.argmax(probs))
                emotion_conf = top_prob * 100
                emotion_buffer.append(EMOTION_CLASS_NAMES[pred_idx])
                current_emotion = Counter(emotion_buffer).most_common(1)[0][0]

    # ── DRAW FACE BOX (Replaces the heavy mesh) ──
    if face_bbox is not None:
        px1, py1, px2, py2 = face_bbox
        color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, "Face Crop", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── FPS ──
    fps_frames += 1
    now = time.time()
    if now - fps_last_time >= 0.5:
        fps = fps_frames / (now - fps_last_time)
        fps_frames = 0
        fps_last_time = now

    # ============================================
    # UI OVERLAY (Reduced from 180 to 130 height)
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # LEFT SIDE: SIGN
    status_text = "SIGNING" if streamer.active else "READY"
    status_color = (0, 255, 0) if streamer.active else (0, 200, 255)
    if not valid_hands: status_text, status_color = "NO HAND", (0, 0, 255)

    cv2.putText(frame, f"SIGN: {current_display}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Conf: {sign_conf:.1f}%  Status: {status_text}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    cv2.putText(frame, f"Buffer: {len(streamer.frames_buffer)}  Votes: {len(vote_buffer)}/{VOTE_WINDOW}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if last_commit_info:
        cv2.putText(frame, f"Last: {last_commit_info}", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # RIGHT SIDE: EMOTION (Moved up slightly to fit FPS below it)
    if current_emotion in EMOTION_CLASS_NAMES:
        emo_color = EMOTION_COLORS[current_emotion]
        cv2.putText(frame, f"EMOTION: {current_emotion.upper()}", (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emo_color, 2)
        cv2.putText(frame, f"Conf: {emotion_conf:.1f}%", (w - 250, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw probability bars
        for i, cname in enumerate(EMOTION_CLASS_NAMES):
            bar_len = int(emotion_probs[i] * 80)
            cv2.rectangle(frame, (w - 250, 80 + i*12), (w - 250 + bar_len, 90 + i*12), EMOTION_COLORS[cname], -1)
            cv2.putText(frame, f"{cname[:3]}: {emotion_probs[i]*100:.0f}%", (w - 160, 90 + i*12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "EMOTION: WAITING", (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # ── FPS COUNTER (Moved to Top Center, Large & Green) ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (w//2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # SENTENCE BOTTOM
    if sentence_words:
        sentence_str = " ".join(sentence_words[-8:])
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"Sentence: {sentence_str}", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Signify - Combined Sign + Emotion', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        sentence_words = []
        vote_buffer = []
        current_display = "Cleared"
        last_commit_info = ""

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()
print("\nTest complete!")
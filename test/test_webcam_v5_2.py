"""
test_webcam_v5_2.py (ORIGINAL LOGIC + IMPROVED UI)
106-class v5 model (FSL-105 + REST).
Keeps original safe behavior: 1.0s cooldown, 4/6 votes, no bypass.
UI improvements only: 640x480, large resizable window, face visible.
"""
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
import time
from collections import Counter, deque

# ============================================
# CONFIGURATION
# ============================================
SIGN_MODEL_PATH = "models/hand_v5/sign_model_v5.h5"
LABELS_JSON = "models/hand_v5/labels_v5.json"
EMOTION_MODEL_PATH = "models/face_fer2025_hybrid_v11_8/emotion_hybrid_v11_8_seed42.h5"

# Sign Config (ORIGINAL — unchanged)
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

# REST class
REST_CLASS_ID = 105
REST_CLASS_NAME = "REST"

# Emotion Config
IMG_SIZE = 48
NUM_LANDMARKS = 936
EMOTION_SMOOTHING_LEN = 8
EMOTION_CONF_THRESHOLD = 0.40
EMOTION_CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
EMOTION_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
EMOTION_COLORS = {
    "Positive": (0, 255, 0), "Negative": (0, 0, 255),
    "Surprise": (255, 255, 0), "Neutral": (200, 200, 200),
}
EMOTION_EVERY_N_FRAMES = 10

# Window (🆕 LARGE resizable)
WINDOW_NAME = 'Signify v5 — Original + Big Window'
WINDOW_W = 1280
WINDOW_H = 960

# ============================================
# LOAD MODELS
# ============================================
print("=" * 60)
print("Loading Sign Model (v5, 106 classes)...")
sign_model = tf.keras.models.load_model(SIGN_MODEL_PATH)

with open(LABELS_JSON) as f:
    label_names = json.load(f)

print(f"✅ Sign Model Loaded! Classes: {len(label_names)}")
print(f"   Last 3 labels: {label_names[-3:]}")
print("Loading Emotion Model (v11.8)...")
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH, compile=False)
print(f"✅ Emotion Model Loaded!")
print("=" * 60)

# ============================================
# MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# SIGN STREAMER (ORIGINAL — no sample-after-commit)
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
# EMOTION HELPERS
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

    if px2 <= px1 or py2 <= py1: return None, None
    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0: return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_in = (gray.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_in = coords.flatten().reshape(1, NUM_LANDMARKS)

    return img_in, lm_in

# ============================================
# STATE
# ============================================
streamer = SignStreamer()
vote_buffer = []
sentence_words = []
current_display = "Ready..."
sign_conf = 0.0
last_commit_info = ""
rest_blocked_count = 0

emotion_buffer = deque(maxlen=EMOTION_SMOOTHING_LEN)
current_emotion = "Waiting..."
emotion_conf = 0.0
emotion_probs = np.zeros(4)

fps = 0.0
fps_frames = 0
fps_last_time = time.time()

frame_count = 0
cached_valid_hands = []
cached_first_face = None
emotion_frame_counter = 0
rest_confidence = 0.0

# ============================================
# WEBCAM
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 Camera resolution: {actual_w} × {actual_h}")

if actual_w < 640 or actual_h < 480:
    print(f"⚠️  Trying 1280×720 fallback...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Now: {actual_w} × {actual_h}")

# 🆕 Large resizable window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

print("\n🎥 Webcam ready! Press 'Q' to quit, 'C' to clear.")
print(f"   Model: {len(label_names)} classes")
print(f"   Mode: ORIGINAL (1.0s cooldown, 4/6 votes, no bypass)")
print(f"   Window: {WINDOW_W}×{WINDOW_H} (resizable)")
print("=" * 60)

# ============================================
# MAIN LOOP
# ============================================
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_count += 1

    # ── ALTERNATING MEDIAPIPE ──
    if frame_count % 2 == 0:
        results_hands = hands.process(rgb)
        cached_valid_hands = []
        if results_hands.multi_hand_landmarks:
            cached_valid_hands.append(results_hands.multi_hand_landmarks[0])
            if len(results_hands.multi_hand_landmarks) >= 2:
                h1_wrist = results_hands.multi_hand_landmarks[0].landmark[0]
                h2_wrist = results_hands.multi_hand_landmarks[1].landmark[0]
                dist = np.linalg.norm(np.array([h1_wrist.x, h1_wrist.y]) - np.array([h2_wrist.x, h2_wrist.y]))
                if dist > 0.12:
                    cached_valid_hands.append(results_hands.multi_hand_landmarks[1])
    else:
        results_face = face_mesh.process(rgb)
        if results_face.multi_face_landmarks:
            cached_first_face = results_face.multi_face_landmarks[0]
        else:
            cached_first_face = None

    # ── DRAW HANDS ──
    if cached_valid_hands:
        for hand_landmarks in cached_valid_hands:
            for idx, lm in enumerate(hand_landmarks.landmark):
                if idx in [0, 4, 8, 12, 16, 20]:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

    # ── DRAW FACE DOTS ──
    if cached_first_face:
        color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))
        for pt_idx in [10, 234, 152, 454, 1]:
            pt = cached_first_face.landmark[pt_idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)

    # ── SIGN INFERENCE (ORIGINAL — no bypass) ──
    sign_data = streamer.update(cached_valid_hands)
    if sign_data is not None:
        input_data = np.array([sign_data], dtype=np.float32)
        prediction = sign_model.predict(input_data, verbose=0)[0]
        predicted_class = int(np.argmax(prediction))
        conf = float(prediction[predicted_class] * 100)

        rest_confidence = float(prediction[REST_CLASS_ID] * 100) if len(prediction) > REST_CLASS_ID else 0.0

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

                    # REST BLOCK
                    if winner == REST_CLASS_ID or word == REST_CLASS_NAME:
                        vote_buffer.clear()
                        streamer.cooldown_until = time.time() + COOLDOWN_SECONDS
                        rest_blocked_count += 1
                        last_commit_info = f"🛑 REST BLOCKED ({rest_blocked_count}x)"
                        print(f"🛑 REST blocked (total: {rest_blocked_count})")
                    elif not sentence_words or sentence_words[-1] != word:
                        sentence_words.append(word)
                        current_display = word
                        streamer.cooldown_until = time.time() + COOLDOWN_SECONDS
                        vote_buffer.clear()
                        last_commit_info = f"{word} ({conf:.0f}%) [{votes}/{VOTE_WINDOW}]"
                        print(f"✅ Committed: {word} ({conf:.0f}%)")

    # ── EMOTION INFERENCE ──
    emotion_frame_counter += 1
    if cached_first_face and emotion_frame_counter % EMOTION_EVERY_N_FRAMES == 0:
        img_in, lm_in = extract_face_emotion(frame, cached_first_face)
        if img_in is not None:
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

    # ── FPS ──
    fps_frames += 1
    now = time.time()
    if now - fps_last_time >= 0.5:
        fps = fps_frames / (now - fps_last_time)
        fps_frames = 0
        fps_last_time = now

    # ============================================
    # UI OVERLAY (same as before)
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # LEFT: SIGN
    status_text = "SIGNING" if streamer.active else "READY"
    status_color = (0, 255, 0) if streamer.active else (0, 200, 255)
    if not cached_valid_hands: status_text, status_color = "NO HAND", (0, 0, 255)

    display_color = (0, 0, 255) if current_display == REST_CLASS_NAME else (0, 255, 255)

    cv2.putText(frame, f"SIGN: {current_display}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, display_color, 2)
    cv2.putText(frame, f"Conf: {sign_conf:.0f}%  {status_text}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    cv2.putText(frame, f"Buf: {len(streamer.frames_buffer)}  Votes: {len(vote_buffer)}/{VOTE_WINDOW}  REST: {rest_confidence:.0f}%", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    if last_commit_info:
        cv2.putText(frame, last_commit_info, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1)

    # RIGHT: EMOTION
    if current_emotion in EMOTION_CLASS_NAMES:
        emo_color = EMOTION_COLORS[current_emotion]
        cv2.putText(frame, f"EMO: {current_emotion.upper()}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, emo_color, 2)
        cv2.putText(frame, f"{emotion_conf:.0f}%", (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for i, cname in enumerate(EMOTION_CLASS_NAMES):
            bar_len = int(emotion_probs[i] * 50)
            cv2.rectangle(frame, (w - 200, 68 + i*10), (w - 200 + bar_len, 75 + i*10), EMOTION_COLORS[cname], -1)
            cv2.putText(frame, f"{cname[:3]} {emotion_probs[i]*100:.0f}", (w - 145, 76 + i*10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "EMO: WAIT", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w // 2 - 30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # SENTENCE
    if sentence_words:
        sentence_str = " ".join(sentence_words[-8:])
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0)
        cv2.putText(frame, f"Sent: {sentence_str}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        sentence_words = []
        vote_buffer = []
        current_display = "Cleared"
        last_commit_info = ""
        rest_blocked_count = 0

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()

print(f"\n✅ Test complete!")
print(f"   Total REST blocks: {rest_blocked_count}")
print(f"   Final sentence: {' '.join(sentence_words)}")
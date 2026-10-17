"""
test_webcam_natural.py (CONTINUOUS + TIME-BASED BUFFER)
Fixes: Emojis removed for OpenCV compatibility, time-based cooldown.
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
MODEL_PATH = "models/hand/sign_model_v4.h5"
LABELS_CSV = "datasets/FSL/labels.csv"

# Model input
NUM_FRAMES = 30
HAND_FEATURES = 63
MAX_HANDS = 2
CONFIDENCE_THRESHOLD = 50

# Time-based buffer
BUFFER_SECONDS = 4.0
MIN_FRAMES_TO_PREDICT = 8

# Motion gate
MOTION_START_THRESHOLD = 0.015
MOTION_HISTORY_LEN = 10

# Inference cadence
PREDICT_EVERY_N_FRAMES = 2

# Vote window
VOTE_WINDOW = 6
STABLE_VOTE_COUNT = 4

# Time-based cooldown (seconds)
COOLDOWN_SECONDS = 1.0

HAND_DROP_DELAY = 6

# ============================================
# LOAD MODEL & LABELS
# ============================================
print("=" * 60)
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"Model loaded! Signs: {len(label_names)}")
print("=" * 60)

# ============================================
# MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# TIME-BASED STREAMING DETECTOR
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

    def update(self, hand_landmarks_list):
        self._trim_old()

        if not hand_landmarks_list:
            self.no_hand_frames += 1
            if self.no_hand_frames >= HAND_DROP_DELAY:
                self.active = False
                self.frames_buffer.clear()
                self.motion_history.clear()
                self.last_positions = None
                self.predict_counter = 0
            return None

        self.no_hand_frames = 0

        hand1 = []
        for lm in hand_landmarks_list[0].landmark:
            hand1.extend([lm.x, lm.y, lm.z])
        if len(hand_landmarks_list) >= 2:
            hand2 = []
            for lm in hand_landmarks_list[1].landmark:
                hand2.extend([lm.x, lm.y, lm.z])
        else:
            hand2 = [0.0] * HAND_FEATURES
        positions = hand1 + hand2

        motion = 0.0
        if self.last_positions is not None:
            motion = float(np.linalg.norm(np.array(positions) - np.array(self.last_positions)))
        self.last_positions = positions
        self.last_motion = motion
        self.motion_history.append(motion)

        any_motion = any(m > MOTION_START_THRESHOLD for m in self.motion_history)
        if not any_motion:
            self.active = False
            self.frames_buffer.clear()
            return None

        self.active = True
        self.frames_buffer.append((time.time(), positions))

        # Time-based cooldown
        if time.time() < self.cooldown_until:
            return None

        self.predict_counter += 1
        if self.predict_counter % PREDICT_EVERY_N_FRAMES != 0:
            return None

        return self._resample_to_model()


streamer = SignStreamer()
vote_buffer = []
sentence_words = []
current_display = "Ready..."
confidence_display = 0.0
flash_frames = 0
last_commit_info = ""

fps = 0.0
fps_frames = 0
fps_last_time = time.time()

# ============================================
# WEBCAM
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nWebcam ready!")
print("=" * 60)
print("HOW TO USE:")
print("  1. Sign naturally - no pause needed")
print("  2. Buffer holds last 4 seconds of frames (matches training)")
print("  3. At predict time, resampled to 30 frames evenly")
print("  4. Words commit when 4 of last 6 predictions agree")
print("  5. Q=Quit | C=Clear")
print("=" * 60)

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
    results = hands.process(frame_rgb)

    hand_detected = bool(results.multi_hand_landmarks)

    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            )
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    fps_frames += 1
    now = time.time()
    if now - fps_last_time >= 0.5:
        fps = fps_frames / (now - fps_last_time)
        fps_frames = 0
        fps_last_time = now

    sign_data = streamer.update(results.multi_hand_landmarks)

    if sign_data is not None:
        input_data = np.array([sign_data], dtype=np.float32)
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class = int(np.argmax(prediction))
        conf = float(prediction[predicted_class] * 100)

        current_display = label_names[predicted_class]
        confidence_display = conf

        if conf >= CONFIDENCE_THRESHOLD:
            vote_buffer.append(predicted_class)
            if len(vote_buffer) > VOTE_WINDOW:
                vote_buffer.pop(0)

            if len(vote_buffer) >= VOTE_WINDOW:
                counts = Counter(vote_buffer)
                winner, votes = counts.most_common(1)[0]

                if votes >= STABLE_VOTE_COUNT:
                    word = label_names[winner]
                    if not sentence_words or sentence_words[-1] != word:
                        sentence_words.append(word)
                        current_display = word
                        flash_frames = 15
                        # Set time-based cooldown
                        streamer.cooldown_until = time.time() + COOLDOWN_SECONDS
                        vote_buffer.clear()
                        last_commit_info = f"{word} ({conf:.0f}%) [{votes}/{VOTE_WINDOW}]"
                        print(f"Committed: {word} ({conf:.1f}%) [{votes}/{VOTE_WINDOW}]")

    # ============================================
    # DISPLAY (Emojis removed for cv2)
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 380), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if hand_detected:
        status_color = (0, 255, 0) if streamer.active else (0, 200, 255)
        status_text = "SIGNING" if streamer.active else "HAND (still)"
    else:
        status_color = (0, 0, 255)
        status_text = "READY - Show hand"

    if flash_frames > 0:
        flash_frames -= 1
        cv2.rectangle(frame, (0, 0), (w, 380), (0, 255, 0), 4)

    cv2.putText(frame, f"Current: {current_display}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Conf: {confidence_display:.1f}%   Status: {status_text}",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    buf_n = len(streamer.frames_buffer)
    if buf_n >= 2:
        oldest_ts = streamer.frames_buffer[0][0]
        newest_ts = streamer.frames_buffer[-1][0]
        buf_span = newest_ts - oldest_ts
    else:
        buf_span = 0.0

    eff_rate = (buf_n / buf_span) if buf_span > 0 else 0.0
    resample_rate = (NUM_FRAMES / buf_span) if buf_span > 0 else 0.0

    cv2.putText(frame, f"FPS: {fps:.1f}   Buffer: {buf_n} frames / {buf_span:.2f}s (target {BUFFER_SECONDS:.1f}s)",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Eff. rate: {eff_rate:.1f} Hz   Resample -> {NUM_FRAMES} frames @ {resample_rate:.1f} Hz",
                (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    motion_val = streamer.last_motion
    motion_indicator = "[MOVING]" if motion_val > MOTION_START_THRESHOLD else "[STILL]"
    cv2.putText(frame, f"Motion: {motion_val:.3f} {motion_indicator} (thr {MOTION_START_THRESHOLD})",
                (15, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cd_remaining = max(0.0, streamer.cooldown_until - time.time())
    cv2.putText(frame, f"Votes: {len(vote_buffer)}/{VOTE_WINDOW}   Cooldown: {cd_remaining:.1f}s",
                (15, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if vote_buffer:
        vote_counts = Counter(vote_buffer)
        vote_strs = [f"{label_names[k][:10]}:{v}" for k, v in vote_counts.most_common(3)]
        cv2.putText(frame, "  " + "  ".join(vote_strs),
                    (15, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)

    if last_commit_info:
        cv2.putText(frame, f"Last commit: {last_commit_info}",
                    (15, 213), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

    sentence_str = " ".join(sentence_words[-8:])
    if sentence_str:
        cv2.putText(frame, "Sentence:", (15, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, sentence_str, (15, 273), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, "Q=Quit | C=Clear | Time-based buffer",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Signify - Continuous Signing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence_words = []
        vote_buffer = []
        current_display = "Cleared"
        last_commit_info = ""
        print("Sentence cleared")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nTest complete!")
if sentence_words:
    print(f"Final sentence: {' '.join(sentence_words)}")
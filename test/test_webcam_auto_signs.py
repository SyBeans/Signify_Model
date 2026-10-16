"""
test_webcam_natural.py (CONTINUOUS NATURAL SIGNING + DEBUG OVERLAY)
Predicts continuously while the user is actively signing.
Motion gate only suppresses STILL hands — no pause required.

Debug overlay shows: FPS, buffer fill, motion value, vote progress, cooldown.
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

NUM_FRAMES = 30
KEEP_FRAMES = 40
HAND_FEATURES = 63
MAX_HANDS = 2
CONFIDENCE_THRESHOLD = 50

MOTION_START_THRESHOLD = 0.015
MOTION_HISTORY_LEN = 10

PREDICT_EVERY_N_FRAMES = 3

VOTE_WINDOW = 6
STABLE_VOTE_COUNT = 4
COOLDOWN_FRAMES = 8

HAND_DROP_DELAY = 6

# ============================================
# LOAD MODEL & LABELS
# ============================================
print("=" * 60)
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"✅ Model loaded! Signs: {len(label_names)}")
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
# CONTINUOUS STREAMING DETECTOR
# ============================================
class SignStreamer:
    def __init__(self):
        self.frames_buffer = []
        self.last_positions = None
        self.motion_history = deque(maxlen=MOTION_HISTORY_LEN)
        self.no_hand_frames = 0
        self.predict_counter = 0
        self.cooldown = 0
        self.active = False
        self.last_motion = 0.0

    def reset(self):
        self.frames_buffer = []
        self.last_positions = None
        self.motion_history.clear()
        self.no_hand_frames = 0
        self.predict_counter = 0
        self.active = False

    def _prepare_prediction(self):
        frames = list(self.frames_buffer[-NUM_FRAMES:])
        positions = np.array(frames, dtype=np.float32)
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        return np.concatenate([positions, velocities], axis=1)

    def update(self, hand_landmarks_list):
        # ── Hand absent ──
        if not hand_landmarks_list:
            self.no_hand_frames += 1
            if self.no_hand_frames >= HAND_DROP_DELAY:
                self.active = False
                self.frames_buffer = []
                self.motion_history.clear()
                self.last_positions = None
                self.predict_counter = 0
            return None

        self.no_hand_frames = 0

        # ── Build 126-feature frame ──
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

        # ── Motion ──
        motion = 0.0
        if self.last_positions is not None:
            motion = float(np.linalg.norm(
                np.array(positions) - np.array(self.last_positions)
            ))
        self.last_positions = positions
        self.last_motion = motion

        self.motion_history.append(motion)

        any_motion = any(m > MOTION_START_THRESHOLD
                         for m in self.motion_history)
        if not any_motion:
            self.active = False
            self.frames_buffer = []
            return None

        self.active = True

        self.frames_buffer.append(positions)
        if len(self.frames_buffer) > KEEP_FRAMES:
            self.frames_buffer.pop(0)

        if len(self.frames_buffer) < NUM_FRAMES:
            return None

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        self.predict_counter += 1
        if self.predict_counter % PREDICT_EVERY_N_FRAMES != 0:
            return None

        return self._prepare_prediction()


streamer = SignStreamer()
vote_buffer = []
sentence_words = []
current_display = "Ready..."
confidence_display = 0.0
flash_frames = 0
last_commit_info = ""

# ── FPS tracking ──
fps = 0.0
fps_frames = 0
fps_last_time = time.time()

# ============================================
# WEBCAM
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam ready!")
print("=" * 60)
print("🖐️  HOW TO USE:")
print("  1. Sign naturally and continuously — no pause needed")
print("  2. While your hand is moving, predictions run")
print("  3. Words commit when 4 of the last 6 predictions agree")
print("  4. Rest your hand to stop predicting")
print("  5. Q=Quit | C=Clear sentence")
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
                mp_draw.DrawingSpec(color=(0, 255, 0),
                                    thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255),
                                    thickness=3, circle_radius=4)
            )
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    # ── FPS ──
    fps_frames += 1
    now = time.time()
    if now - fps_last_time >= 0.5:
        fps = fps_frames / (now - fps_last_time)
        fps_frames = 0
        fps_last_time = now

    # ============================================
    # CONTINUOUS INFERENCE
    # ============================================
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
                        streamer.cooldown = COOLDOWN_FRAMES
                        vote_buffer.clear()
                        last_commit_info = f"{word} ({conf:.0f}%) [{votes}/{VOTE_WINDOW}]"
                        print(f"✅ Committed: {word} ({conf:.1f}%) "
                              f"[{votes}/{VOTE_WINDOW} votes]")

    # ============================================
    # DISPLAY
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 340), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Status
    if hand_detected:
        status_color = (0, 255, 0) if streamer.active else (0, 200, 255)
        status_text = "🟢 SIGNING" if streamer.active else "🟡 HAND (still)"
    else:
        status_color = (0, 0, 255)
        status_text = "🔴 READY - Show hand"

    if flash_frames > 0:
        flash_frames -= 1
        cv2.rectangle(frame, (0, 0), (w, 340), (0, 255, 0), 4)

    # Main info
    cv2.putText(frame, f"Current: {current_display}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Conf: {confidence_display:.1f}%   "
                       f"Status: {status_text}",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, status_color, 2)

    # ── DEBUG: FPS + buffer + motion + votes ──
    buf_fill = len(streamer.frames_buffer)
    buf_pct = int(buf_fill / KEEP_FRAMES * 100)
    motion_val = streamer.last_motion
    motion_str = f"{motion_val:.3f}"
    motion_indicator = "🟢" if motion_val > MOTION_START_THRESHOLD else "⚪"

    cv2.putText(frame, f"FPS: {fps:.1f}   Buffer: {buf_fill}/{KEEP_FRAMES} ({buf_pct}%)",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Motion: {motion_str} {motion_indicator}  "
                       f"(thr {MOTION_START_THRESHOLD})",
                (15, 118), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Votes: {len(vote_buffer)}/{VOTE_WINDOW}   "
                       f"Cooldown: {streamer.cooldown}",
                (15, 141), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    # Show what votes are for (top labels in current window)
    if vote_buffer:
        vote_counts = Counter(vote_buffer)
        vote_strs = [f"{label_names[k][:10]}:{v}"
                     for k, v in vote_counts.most_common(3)]
        cv2.putText(frame, "  " + "  ".join(vote_strs),
                    (15, 164), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 220), 1)

    # Last commit
    if last_commit_info:
        cv2.putText(frame, f"Last commit: {last_commit_info}",
                    (15, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (100, 255, 100), 1)

    # Sentence
    sentence_str = " ".join(sentence_words[-8:])
    if sentence_str:
        cv2.putText(frame, "Sentence:",
                    (15, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        cv2.putText(frame, sentence_str,
                    (15, 248), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    cv2.putText(frame, "Q=Quit | C=Clear | Continuous natural",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    cv2.imshow('Signify - Continuous Signing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence_words = []
        vote_buffer = []
        current_display = "Cleared"
        last_commit_info = ""
        print("🧹 Sentence cleared")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Test complete!")
if sentence_words:
    print(f"📝 Final sentence: {' '.join(sentence_words)}")
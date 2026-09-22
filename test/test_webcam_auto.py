"""
test_webcam_auto.py
Real-time FSL sign recognition with AUTO-DETECTION.
- No holding required
- Detects when sign starts (hand appears)
- Detects when sign ends (hand drops)
- Auto-prints text
- Supports continuous sentence building
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
from collections import deque
import time

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/hand/sign_model_v4.h5"
LABELS_CSV = "datasets/FSL/labels.csv"

NUM_FRAMES = 30
MAX_HANDS = 2
HAND_FEATURES = 63
CONFIDENCE_THRESHOLD = 50

# Auto-detection settings
MIN_FRAMES_FOR_SIGN = 20      # Need at least 20 frames to predict
HAND_DROP_DELAY = 5           # Frames without hand = sign complete
MOTION_THRESHOLD = 0.005      # Below this = stable
STABLE_FRAMES = 8             # Frames stable before considering complete
COOLDOWN_FRAMES = 30          # Frames to wait after prediction (avoid duplicates)

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
# MEDIAPIPE (2 HANDS)
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
# WEBCAM
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam ready!")
print("=" * 60)
print("🖐️  HOW TO USE:")
print("  1. Sign naturally (no need to hold!)")
print("  2. Drop your hand when done")
print("  3. Text appears automatically")
print("  4. Sign next word")
print("  5. Press 'Q' to quit | 'C' to clear sentence")
print("=" * 60)

# ============================================
# STATE MACHINE
# ============================================
class SignDetector:
    def __init__(self):
        self.state = "IDLE"           # IDLE, SIGNING, WAITING
        self.frames_buffer = []       # Current sign frames
        self.last_positions = None
        self.motion_history = []
        self.no_hand_frames = 0
        self.stable_frames = 0
        self.cooldown = 0
    
    def reset(self):
        self.frames_buffer = []
        self.last_positions = None
        self.motion_history = []
        self.no_hand_frames = 0
        self.stable_frames = 0
    
    def update(self, hand_landmarks_list):
        """
        Process one frame.
        Returns: (prediction_ready, sign_data)
        """
        # Cooldown after prediction
        if self.cooldown > 0:
            self.cooldown -= 1
            return (False, None)
        
        if hand_landmarks_list:
            # Hand detected
            self.no_hand_frames = 0
            self.state = "SIGNING"
            
            # Extract 126 features (2 hands × 63)
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
            
            # Compute motion
            if self.last_positions is not None:
                motion = np.linalg.norm(
                    np.array(positions) - np.array(self.last_positions)
                )
                self.motion_history.append(motion)
                if len(self.motion_history) > 10:
                    self.motion_history.pop(0)
            
            self.last_positions = positions
            self.frames_buffer.append(positions)
            
            # Keep max 60 frames (2 sec @ 30fps)
            if len(self.frames_buffer) > 60:
                self.frames_buffer.pop(0)
            
            # Check if hand is stable (sign may be complete)
            if len(self.motion_history) >= 5:
                recent_motion = np.mean(self.motion_history[-5:])
                if recent_motion < MOTION_THRESHOLD:
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                
                # Stable for a while → sign complete
                if self.stable_frames >= STABLE_FRAMES:
                    if len(self.frames_buffer) >= MIN_FRAMES_FOR_SIGN:
                        result = self._prepare_prediction()
                        self.reset()
                        self.cooldown = COOLDOWN_FRAMES
                        return (True, result)
                    else:
                        self.stable_frames = 0
        else:
            # No hand detected
            self.no_hand_frames += 1
            
            # Hand was signing, now dropped → sign complete!
            if self.state == "SIGNING" and self.no_hand_frames >= HAND_DROP_DELAY:
                if len(self.frames_buffer) >= MIN_FRAMES_FOR_SIGN:
                    result = self._prepare_prediction()
                    self.reset()
                    self.state = "IDLE"
                    self.cooldown = COOLDOWN_FRAMES
                    return (True, result)
                else:
                    self.reset()
                    self.state = "IDLE"
        
        return (False, None)
    
    def _prepare_prediction(self):
        """Prepare (30, 252) input from buffer."""
        frames = self.frames_buffer[-NUM_FRAMES:] if len(self.frames_buffer) >= NUM_FRAMES else self.frames_buffer
        
        # Pad to 30 frames
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1] if frames else [0.0] * 126)
        
        positions = np.array(frames, dtype=np.float32)
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        combined = np.concatenate([positions, velocities], axis=1)
        return combined


detector = SignDetector()

# ============================================
# SENTENCE HISTORY
# ============================================
sentence_words = []       # List of predicted signs
current_display = "Ready..."
confidence_display = 0.0
last_prediction = ""
last_prediction_time = 0
flash_frames = 0          # For visual feedback

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

    # ============================================
    # DRAW HAND LANDMARKS
    # ============================================
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

    # ============================================
    # AUTO-DETECTION
    # ============================================
    ready, sign_data = detector.update(results.multi_hand_landmarks)

    if ready and sign_data is not None:
        input_data = np.array([sign_data], dtype=np.float32)
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        conf = prediction[predicted_class] * 100

        if conf > CONFIDENCE_THRESHOLD:
            predicted_word = label_names[predicted_class]
            sentence_words.append(predicted_word)
            current_display = predicted_word
            confidence_display = conf
            last_prediction = predicted_word
            last_prediction_time = time.time()
            flash_frames = 15  # Flash green for 0.5 sec
            print(f"✅ Detected: {predicted_word} ({conf:.1f}%)")
        else:
            current_display = f"Uncertain ({conf:.1f}%)"
            confidence_display = conf

    # ============================================
    # DISPLAY UI
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Status
    if hand_detected:
        status_color = (0, 255, 0)
        status_text = "🟢 HAND DETECTED - Signing..."
    else:
        status_color = (0, 0, 255)
        status_text = "🔴 READY - Show hand"

    # Flash green if just detected
    if flash_frames > 0:
        flash_frames -= 1
        cv2.rectangle(frame, (0, 0), (w, 280), (0, 255, 0), 4)

    cv2.putText(frame, f"Current: {current_display}",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence_display:.1f}%",
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text,
                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Sentence builder
    sentence_str = " ".join(sentence_words[-8:])  # last 8 words
    if sentence_str:
        cv2.putText(frame, "Sentence:",
                    (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, sentence_str,
                    (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Q=Quit | C=Clear sentence | Natural signing",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Signify - Auto Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence_words = []
        current_display = "Cleared"
        print("🧹 Sentence cleared")

# ============================================
# CLEANUP
# ============================================
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Test complete!")
if sentence_words:
    print(f"📝 Final sentence: {' '.join(sentence_words)}")

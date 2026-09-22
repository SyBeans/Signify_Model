"""
test_diagnose.py
Diagnostic with PREDICTION SMOOTHING.
Collects multiple predictions during sign, uses majority vote.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
from collections import deque, Counter

MODEL_PATH = "models/hand/sign_model_v4.h5"
LABELS_CSV = "datasets/FSL/labels.csv"

NUM_FRAMES = 30
MAX_HANDS = 2
HAND_FEATURES = 63

MOTION_THRESHOLD = 0.05
STABLE_FRAMES = 5
COOLDOWN = 30

# Smoothing settings
PREDICT_EVERY_N_FRAMES = 5   # Predict every 5 frames while stable
MIN_PREDICTIONS = 3          # Need at least 3 predictions
SMOOTHING_WINDOW = 6         # Keep last 6 predictions

print("=" * 60)
print("🔬 DIAGNOSTIC WITH SMOOTHING")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"✅ Model loaded! Signs: {len(label_names)}")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nINSTRUCTIONS:")
print("  1. Sign and HOLD STEADY for 2 seconds")
print("  2. System collects multiple predictions")
print("  3. Majority vote = final answer")
print("  4. Press 'Q' to quit")
print("=" * 60)

# State
frames_buffer = []
last_positions = None
motion_history = []
stable_counter = 0
frames_since_last_pred = 0

recent_predictions = deque(maxlen=SMOOTHING_WINDOW)  # for smoothing

current_prediction = "Waiting..."
confidence = 0.0
top_5_display = []
hand_detected = False
sign_count = 0
debug_motion = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
       break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True

        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            )

        hand1 = []
        for lm in results.multi_hand_landmarks[0].landmark:
            hand1.extend([lm.x, lm.y, lm.z])
        if len(results.multi_hand_landmarks) >= 2:
            hand2 = []
            for lm in results.multi_hand_landmarks[1].landmark:
                hand2.extend([lm.x, lm.y, lm.z])
        else:
            hand2 = [0.0] * HAND_FEATURES

        current_positions = hand1 + hand2

        if last_positions is not None:
            motion = np.linalg.norm(
                np.array(current_positions) - np.array(last_positions)
            )
            debug_motion = motion
            motion_history.append(motion)
            if len(motion_history) > 10:
                motion_history.pop(0)

            recent_motion = np.mean(motion_history[-5:]) if len(motion_history) >= 5 else 999

            if recent_motion < MOTION_THRESHOLD:
                stable_counter += 1
            else:
                stable_counter = 0
                recent_predictions.clear()  # reset on movement

        last_positions = current_positions
        frames_buffer.append(current_positions)
        if len(frames_buffer) > NUM_FRAMES:
            frames_buffer.pop(0)

        # ============================================
        # PREDICT EVERY N FRAMES WHILE STABLE
        # ============================================
        if stable_counter >= STABLE_FRAMES:
            frames_since_last_pred += 1

            if frames_since_last_pred >= PREDICT_EVERY_N_FRAMES:
                frames_since_last_pred = 0

                if len(frames_buffer) >= 20:
                    buf = list(frames_buffer[-NUM_FRAMES:])
                    while len(buf) < NUM_FRAMES:
                        buf.append(buf[-1] if buf else [0.0] * (HAND_FEATURES * 2))

                    positions = np.array(buf, dtype=np.float32)
                    velocities = np.zeros_like(positions)
                    velocities[1:] = positions[1:] - positions[:-1]
                    combined = np.concatenate([positions, velocities], axis=1)

                    pred = model.predict(np.array([combined]), verbose=0)[0]
                    top_idx = int(np.argmax(pred))
                    top_conf = pred[top_idx] * 100

                    recent_predictions.append(top_idx)

                    # Show current prediction in UI
                    current_prediction = label_names[top_idx]
                    confidence = top_conf

                    # Top 5
                    top5_idx = np.argsort(pred)[-5:][::-1]
                    top5_conf = pred[top5_idx] * 100
                    top_5_display = list(zip(top5_idx, top5_conf))

            # ============================================
            # AFTER ENOUGH PREDICTIONS → FINAL ANSWER
            # ============================================
            if len(recent_predictions) >= MIN_PREDICTIONS:
                # Majority vote
                counter = Counter(recent_predictions)
                final_idx, final_count = counter.most_common(1)[0]
                final_name = label_names[final_idx]
                agreement = final_count / len(recent_predictions) * 100

                # Print when we have strong consensus
                if agreement >= 50:  # At least 50% agreement
                    sign_count += 1
                    print("\n" + "=" * 60)
                    print(f"🎯 SIGN #{sign_count} → {final_name}")
                    print(f"   Agreement: {agreement:.0f}% ({final_count}/{len(recent_predictions)} predictions)")
                    print("=" * 60)
                    print(f"All predictions in window:")
                    for idx, cnt in counter.most_common():
                        print(f"  {label_names[idx]:25s} × {cnt}")
                    print("=" * 60)

                    # Reset
                    recent_predictions.clear()
                    stable_counter = 0
                    frames_buffer = []
                    motion_history = []
                    last_positions = None
                    current_prediction = final_name
                    #break  # exit after one strong detection
    else:
        hand_detected = False
        stable_counter = 0
        last_positions = None
        motion_history = []
        recent_predictions.clear()

    # ============================================
    # DISPLAY
    # ============================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 320), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if hand_detected:
        if stable_counter >= STABLE_FRAMES:
            status_color = (0, 255, 0)
            status_text = f"🟢 STABLE ({stable_counter})"
        elif stable_counter > 0:
            status_color = (0, 255, 255)
            status_text = f"🟡 Stabilizing ({stable_counter}/{STABLE_FRAMES})"
        else:
            status_color = (0, 165, 255)
            status_text = "🟠 Moving"
    else:
        status_color = (0, 0, 255)
        status_text = "🔴 READY"

    cv2.putText(frame, f"Current: {current_prediction}",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text,
                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Predictions collected: {len(recent_predictions)}/{MIN_PREDICTIONS}",
                (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if top_5_display:
        y_pos = 175
        cv2.putText(frame, "Top 5 (latest):",
                    (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, (idx, conf) in enumerate(top_5_display):
            text = f"  {i+1}. {label_names[idx]:22s} {conf:5.1f}%"
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (15, y_pos + 22 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    cv2.putText(frame, "Sign + HOLD STEADY 2sec | Q=Quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Signify - Smoothing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Diagnostic complete!")
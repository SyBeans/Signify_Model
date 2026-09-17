"""
extract_landmarks_v6_twohand.py
Extracts BOTH hands (max_num_hands=2) with position + velocity features.
Saves 252 features per frame (126 per hand).

Features breakdown:
  Hand 1: 21 landmarks × 3 (x,y,z) = 63 position + 63 velocity = 126
  Hand 2: 21 landmarks × 3 (x,y,z) = 63 position + 63 velocity = 126
  Total: 252 features per frame
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = "datasets/FSL"
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
LABELS_CSV = os.path.join(DATASET_PATH, "labels.csv")
LANDMARKS_PATH = "landmarks/hand"

NUM_FRAMES = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3
HAND_FEATURES = NUM_LANDMARKS * NUM_COORDS  # 63 per hand
MIN_HAND_FRAMES = 15
MAX_HANDS = 2

# ============================================
# INITIALIZE MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,    # ← Detect up to 2 hands
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


def extract_hand_features(hand_landmarks):
    """Extract 63 features (21 landmarks × 3 coords) from one hand."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


def extract_landmarks_from_video(video_path):
    """
    Scan ALL frames, collect frames with hands.
    For each frame: extract up to 2 hands (pad missing hand with zeros).
    Returns (30, 252) with position + velocity features.
    """
    cap = cv2.VideoCapture(video_path)
    hand_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            num_detected = len(results.multi_hand_landmarks)

            # Hand 1
            hand1 = extract_hand_features(results.multi_hand_landmarks[0])

            # Hand 2 (or zeros if only 1 hand)
            if num_detected >= 2:
                hand2 = extract_hand_features(results.multi_hand_landmarks[1])
            else:
                hand2 = [0.0] * HAND_FEATURES  # Pad with zeros

            # Combined: 126 features (63 + 63)
            hand_frames.append(hand1 + hand2)

    cap.release()

    if len(hand_frames) < MIN_HAND_FRAMES:
        return None

    # Take 30 evenly spaced frames
    if len(hand_frames) >= NUM_FRAMES:
        indices = np.linspace(0, len(hand_frames) - 1, NUM_FRAMES, dtype=int)
        selected = [hand_frames[i] for i in indices]
    else:
        selected = hand_frames.copy()
        while len(selected) < NUM_FRAMES:
            selected.append(hand_frames[-1])

    positions = np.array(selected, dtype=np.float32)  # (30, 126)

    # Velocity (difference between consecutive frames)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]

    # Combined: (30, 252)
    combined = np.concatenate([positions, velocities], axis=1)

    return combined


def process_split(df, name):
    print(f"\n{'=' * 60}")
    print(f"🔨 PROCESSING {name.upper()}")
    print(f"{'=' * 60}")

    X, y = [], []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=name):
        video_rel_path = row['vid_path'].replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)

        if not os.path.exists(video_path):
            skipped += 1
            continue

        landmarks = extract_landmarks_from_video(video_path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(row['id_label'])
        else:
            skipped += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"\n✅ X_{name}: {X.shape}")
    print(f"✅ y_{name}: {y.shape}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"✅ Valid: {len(X)}/{len(df)} ({len(X)/len(df)*100:.1f}%)")

    return X, y


def main():
    os.makedirs(LANDMARKS_PATH, exist_ok=True)

    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\n📊 Total Signs: {len(labels_df)}")
    print(f"📋 Max hands: {MAX_HANDS}")
    print(f"📋 Features per frame: 252 (126 per hand × 2 hands)")
    print(f"📋 Model input shape: (30, 252)")

    train_df = pd.read_csv(TRAIN_CSV)
    X_train, y_train = process_split(train_df, "train")

    test_df = pd.read_csv(TEST_CSV)
    X_test, y_test = process_split(test_df, "test")

    np.save(os.path.join(LANDMARKS_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_test.npy"), y_test)

    print(f"\n✅ Saved to {LANDMARKS_PATH}/")
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ X_test:  {X_test.shape}")
    print(f"\n✅ DONE! Next: train_model_v6_twohand.py")


if __name__ == "__main__":
    main()
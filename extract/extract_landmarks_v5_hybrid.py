"""
extract_landmarks_v5_hybrid.py
BEST OF BOTH: V3 (frames with hands) + V4 (position + velocity)
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

DATASET_PATH = "datasets/FSL"
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
LABELS_CSV = os.path.join(DATASET_PATH, "labels.csv")
LANDMARKS_PATH = "landmarks/hand"

NUM_FRAMES = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3
MIN_HAND_FRAMES = 15  # Need at least 15 frames with hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


def extract_landmarks_from_video(video_path):
    """
    Step 1: Scan ALL frames, collect only frames WITH hands (like V3)
    Step 2: Take 30 best frames evenly spaced (like V3)
    Step 3: Compute velocity features (like V4)
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
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            hand_frames.append(landmarks)

    cap.release()

    if len(hand_frames) < MIN_HAND_FRAMES:
        return None

    # Take 30 frames from hand_frames
    if len(hand_frames) >= NUM_FRAMES:
        indices = np.linspace(0, len(hand_frames) - 1, NUM_FRAMES, dtype=int)
        selected = [hand_frames[i] for i in indices]
    else:
        # Pad by repeating last frame
        selected = hand_frames.copy()
        while len(selected) < NUM_FRAMES:
            selected.append(hand_frames[-1])

    positions = np.array(selected, dtype=np.float32)  # (30, 63)

    # Compute velocity
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]

    # Concatenate position + velocity
    combined = np.concatenate([positions, velocities], axis=1)  # (30, 126)

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
    print(f"📋 Features per frame: 126 (63 position + 63 velocity)")

    train_df = pd.read_csv(TRAIN_CSV)
    X_train, y_train = process_split(train_df, "train")

    test_df = pd.read_csv(TEST_CSV)
    X_test, y_test = process_split(test_df, "test")

    np.save(os.path.join(LANDMARKS_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_test.npy"), y_test)

    print(f"\n✅ Saved to {LANDMARKS_PATH}/")
    print(f"✅ DONE!")


if __name__ == "__main__":
    main()
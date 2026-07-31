"""
extract_landmarks_v3.py
Only extracts frames WHERE HANDS ARE DETECTED.
Skips empty frames completely.
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
LANDMARKS_PATH = "landmarks"

NUM_FRAMES = 30
NUM_LANDMARKS = 21
NUM_COORDINATES = 3
MIN_HAND_FRAMES = 15  # Minimum frames with hands required

# ============================================
# INITIALIZE MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


def extract_landmarks_from_video(video_path):
    """
    Scan ALL frames, collect only frames with hands.
    Return the best NUM_FRAMES or None if not enough.
    """
    cap = cv2.VideoCapture(video_path)
    
    hand_frames = []  # Only frames WITH hands
    
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
    
    # If we have enough hand frames
    if len(hand_frames) >= MIN_HAND_FRAMES:
        # Take up to NUM_FRAMES evenly spaced
        if len(hand_frames) >= NUM_FRAMES:
            indices = np.linspace(0, len(hand_frames)-1, NUM_FRAMES, dtype=int)
            selected = [hand_frames[i] for i in indices]
        else:
            # Pad by repeating
            selected = hand_frames.copy()
            while len(selected) < NUM_FRAMES:
                selected.append(hand_frames[-1])  # Repeat last frame
        
        return np.array(selected, dtype=np.float32)
    else:
        return None  # Not enough hand frames


def process_dataset():
    """Process all videos"""
    
    os.makedirs(LANDMARKS_PATH, exist_ok=True)
    
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\n📊 Total Signs: {len(labels_df)}")
    print(f"📋 Min hand frames required: {MIN_HAND_FRAMES}")
    
    # ============================================
    # TRAINING DATA
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 PROCESSING TRAINING DATA")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Training samples: {len(train_df)}")
    
    X_train = []
    y_train = []
    skipped = 0
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training"):
        video_rel_path = row['vid_path'].replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            skipped += 1
            continue
        
        landmarks = extract_landmarks_from_video(video_path)
        
        if landmarks is not None:
            X_train.append(landmarks)
            y_train.append(row['id_label'])
        else:
            skipped += 1
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    
    print(f"\n✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"⚠️  Skipped (not enough hands): {skipped}")
    print(f"✅ Valid samples: {len(X_train)}/{len(train_df)} ({len(X_train)/len(train_df)*100:.1f}%)")
    
    # ============================================
    # TESTING DATA
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 PROCESSING TESTING DATA")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_CSV)
    print(f"Testing samples: {len(test_df)}")
    
    X_test = []
    y_test = []
    skipped_test = 0
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        video_rel_path = row['vid_path'].replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            skipped_test += 1
            continue
        
        landmarks = extract_landmarks_from_video(video_path)
        
        if landmarks is not None:
            X_test.append(landmarks)
            y_test.append(row['id_label'])
        else:
            skipped_test += 1
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)
    
    print(f"\n✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    print(f"⚠️  Skipped: {skipped_test}")
    print(f"✅ Valid samples: {len(X_test)}/{len(test_df)} ({len(X_test)/len(test_df)*100:.1f}%)")
    
    # ============================================
    # SAVE
    # ============================================
    print("\n" + "=" * 60)
    print("💾 SAVING LANDMARKS")
    print("=" * 60)
    
    np.save(os.path.join(LANDMARKS_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_test.npy"), y_test)
    
    total = len(X_train) + len(X_test)
    total_skipped = skipped + skipped_test
    
    print(f"✅ Saved to '{LANDMARKS_PATH}/'")
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Valid samples: {total}")
    print(f"   Skipped: {total_skipped}")
    print(f"   All frames have hand data! (0% zeros)")
    print("\n✅ DONE! Next: train_model.py")


if __name__ == "__main__":
    process_dataset()
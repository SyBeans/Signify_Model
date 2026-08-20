"""
extract_landmarks_v4_motion.py
Extracts hand landmarks WITH MOTION FEATURES (velocity).
Takes frames across the FULL video to capture sign movement.
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
NUM_COORDS = 3  # x, y, z per landmark
MIN_HAND_FRAMES = 10  # Minimum frames with hands required

# Each frame: 63 position + 63 velocity = 126 features
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDS * 2  # 126

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
    Extract landmarks from FULL video (captures motion).
    Returns combined position + velocity features.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < NUM_FRAMES:
        cap.release()
        return None
    
    # Take NUM_FRAMES evenly spaced from the full video
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    all_landmarks = []
    hand_count = 0
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            all_landmarks.append(None)
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(landmarks)
            hand_count += 1
        else:
            all_landmarks.append(None)
    
    cap.release()
    
    # Need at least MIN_HAND_FRAMES with hands
    if hand_count < MIN_HAND_FRAMES:
        return None
    
    # Fill missing frames with interpolation
    for i in range(len(all_landmarks)):
        if all_landmarks[i] is None:
            # Find nearest valid frame
            for j in range(1, NUM_FRAMES):
                if i - j >= 0 and all_landmarks[i - j] is not None:
                    all_landmarks[i] = all_landmarks[i - j].copy()
                    break
                if i + j < NUM_FRAMES and all_landmarks[i + j] is not None:
                    all_landmarks[i] = all_landmarks[i + j].copy()
                    break
            if all_landmarks[i] is None:
                all_landmarks[i] = [0.0] * (NUM_LANDMARKS * NUM_COORDS)
    
    positions = np.array(all_landmarks, dtype=np.float32)  # (30, 63)
    
    # Compute velocity (difference between consecutive frames)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]  # dx, dy, dz
    
    # Concatenate position + velocity
    combined = np.concatenate([positions, velocities], axis=1)  # (30, 126)
    
    return combined


def process_dataset():
    """Process all videos with motion features"""
    
    os.makedirs(LANDMARKS_PATH, exist_ok=True)
    
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\n📊 Total Signs: {len(labels_df)}")
    print(f"📋 Features per frame: {NUM_FEATURES} (63 position + 63 velocity)")
    print(f"📹 Frames per sample: {NUM_FRAMES}")
    
    # ============================================
    # TRAINING DATA
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 PROCESSING TRAINING DATA (Motion-Aware)")
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
    print(f"⚠️  Skipped: {skipped}")
    print(f"✅ Valid: {len(X_train)}/{len(train_df)} ({len(X_train)/len(train_df)*100:.1f}%)")
    
    # ============================================
    # TESTING DATA
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 PROCESSING TESTING DATA (Motion-Aware)")
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
    print(f"✅ Valid: {len(X_test)}/{len(test_df)} ({len(X_test)/len(test_df)*100:.1f}%)")
    
    # ============================================
    # SAVE
    # ============================================
    print("\n" + "=" * 60)
    print("💾 SAVING MOTION LANDMARKS")
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
    print(f"   Features per frame: {NUM_FEATURES} (position + velocity)")
    print(f"\n✅ DONE! Next: train_model_v3_motion.py")


if __name__ == "__main__":
    process_dataset()
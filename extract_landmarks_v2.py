"""
extract_landmarks_v2.py
Uses older MediaPipe API for better hand detection.
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

# ============================================
# INITIALIZE MEDIAPIPE (OLD API - More Reliable)
# ============================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,  # LOWER threshold
    min_tracking_confidence=0.3    # LOWER threshold
)


def extract_landmarks_from_video(video_path):
    """
    Extract hand landmarks using older MediaPipe API.
    """
    cap = cv2.VideoCapture(video_path)
    
    all_landmarks = []
    
    while len(all_landmarks) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(landmarks)
        else:
            all_landmarks.append([0.0] * (NUM_LANDMARKS * NUM_COORDINATES))
    
    cap.release()
    
    # Pad if too short
    while len(all_landmarks) < NUM_FRAMES:
        all_landmarks.append([0.0] * (NUM_LANDMARKS * NUM_COORDINATES))
    
    all_landmarks = all_landmarks[:NUM_FRAMES]
    
    return np.array(all_landmarks, dtype=np.float32)


def process_dataset():
    """Process all videos"""
    
    os.makedirs(LANDMARKS_PATH, exist_ok=True)
    
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\n📊 Total Signs: {len(labels_df)}")
    
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
    errors = 0
    zero_count = 0
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training"):
        video_rel_path = row['vid_path'].replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            errors += 1
            continue
        
        try:
            landmarks = extract_landmarks_from_video(video_path)
            
            # Check if all zeros
            if np.all(landmarks == 0):
                zero_count += 1
            
            X_train.append(landmarks)
            y_train.append(row['id_label'])
        except Exception as e:
            errors += 1
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    
    print(f"\n✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"⚠️  Zero-filled samples: {zero_count}/{len(X_train)} ({zero_count/len(X_train)*100:.1f}%)")
    
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
    zero_count_test = 0
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        video_rel_path = row['vid_path'].replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            errors += 1
            continue
        
        try:
            landmarks = extract_landmarks_from_video(video_path)
            
            if np.all(landmarks == 0):
                zero_count_test += 1
            
            X_test.append(landmarks)
            y_test.append(row['id_label'])
        except Exception as e:
            errors += 1
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)
    
    print(f"\n✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    print(f"⚠️  Zero-filled samples: {zero_count_test}/{len(X_test)} ({zero_count_test/len(X_test)*100:.1f}%)")
    
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
    
    total_zero = zero_count + zero_count_test
    total_samples = len(X_train) + len(X_test)
    
    print(f"✅ Saved to '{LANDMARKS_PATH}/'")
    print(f"\n📊 SUMMARY:")
    print(f"   Total samples: {total_samples}")
    print(f"   Zero-filled: {total_zero} ({total_zero/total_samples*100:.1f}%)")
    print(f"   Valid samples: {total_samples - total_zero} ({(total_samples-total_zero)/total_samples*100:.1f}%)")
    
    if errors:
        print(f"   Errors: {errors}")
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    process_dataset()
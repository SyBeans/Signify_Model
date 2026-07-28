"""
extract_landmarks.py
Extracts hand landmarks from FSL-105 videos using MediaPipe.
Saves as .npy files for training.
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
CLIPS_PATH = os.path.join(DATASET_PATH, "clips")
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
LABELS_CSV = os.path.join(DATASET_PATH, "labels.csv")
LANDMARKS_PATH = "landmarks"

# MediaPipe settings
NUM_HANDS = 1                 # Detect 1 hand (dominant hand)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_FRAMES = 30               # Frames per sample
NUM_LANDMARKS = 21            # MediaPipe hand landmarks
NUM_COORDINATES = 3           # x, y, z

# ============================================
# INITIALIZE MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)


def extract_landmarks_from_video(video_path):
    """
    Extract hand landmarks from a single video.
    Returns a numpy array of shape (NUM_FRAMES, 63)
    63 = 21 landmarks × 3 (x, y, z)
    """
    cap = cv2.VideoCapture(video_path)
    
    all_landmarks = []
    
    while len(all_landmarks) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Extract landmarks if hand detected
        if results.multi_hand_landmarks:
            # Take the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(landmarks)
        else:
            # No hand detected → use zeros
            all_landmarks.append([0.0] * (NUM_LANDMARKS * NUM_COORDINATES))
    
    cap.release()
    
    # Pad with zeros if video is too short
    while len(all_landmarks) < NUM_FRAMES:
        all_landmarks.append([0.0] * (NUM_LANDMARKS * NUM_COORDINATES))
    
    # Take exactly NUM_FRAMES
    all_landmarks = all_landmarks[:NUM_FRAMES]
    
    return np.array(all_landmarks)  # Shape: (30, 63)


def process_dataset():
    """
    Process all videos from train.csv and test.csv,
    extract landmarks, and save as .npy files.
    """
    # Create landmarks folder
    os.makedirs(LANDMARKS_PATH, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\n📊 Total Signs: {len(labels_df)}")
    print(f"Categories: {labels_df['category'].nunique()}")
    
    # ============================================
    # PROCESS TRAINING DATA
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 PROCESSING TRAINING DATA")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Training samples: {len(train_df)}")
    
    X_train = []
    y_train = []
    errors = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training"):
        video_rel_path = row['vid_path']  # e.g., clips\17\6.MOV
        # Convert Windows path to Linux path
        video_rel_path = video_rel_path.replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            errors.append(video_path)
            continue
        
        try:
            landmarks = extract_landmarks_from_video(video_path)
            X_train.append(landmarks)
            y_train.append(row['id_label'])  # Numeric label (0-104)
        except Exception as e:
            errors.append(f"{video_path}: {e}")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)  # Shape: (n_samples, 30, 63)
    y_train = np.array(y_train)  # Shape: (n_samples,)
    
    print(f"\n✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    
    # ============================================
    # PROCESS TESTING DATA
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 PROCESSING TESTING DATA")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_CSV)
    print(f"Testing samples: {len(test_df)}")
    
    X_test = []
    y_test = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        video_rel_path = row['vid_path']
        video_rel_path = video_rel_path.replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            errors.append(video_path)
            continue
        
        try:
            landmarks = extract_landmarks_from_video(video_path)
            X_test.append(landmarks)
            y_test.append(row['id_label'])
        except Exception as e:
            errors.append(f"{video_path}: {e}")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\n✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    
    # ============================================
    # SAVE TO .NPY FILES
    # ============================================
    print("\n" + "=" * 60)
    print("💾 SAVING LANDMARKS")
    print("=" * 60)
    
    np.save(os.path.join(LANDMARKS_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_test.npy"), y_test)
    
    print(f"✅ Saved to '{LANDMARKS_PATH}/' folder:")
    print(f"   - X_train.npy ({X_train.shape})")
    print(f"   - y_train.npy ({y_train.shape})")
    print(f"   - X_test.npy ({X_test.shape})")
    print(f"   - y_test.npy ({y_test.shape})")
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Training samples:   {len(X_train)}")
    print(f"Testing samples:    {len(X_test)}")
    print(f"Total samples:      {len(X_train) + len(X_test)}")
    print(f"Total signs:        {len(labels_df)}")
    print(f"Frames per sample:  {NUM_FRAMES}")
    print(f"Features per frame: {NUM_LANDMARKS * NUM_COORDINATES} (21 × 3)")
    print(f"Features per sample: {NUM_FRAMES * NUM_LANDMARKS * NUM_COORDINATES} (30 × 63)")
    
    if errors:
        print(f"\n⚠️  Errors: {len(errors)} files not found")
        for e in errors[:5]:
            print(f"   - {e}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")
    
    file_size = os.path.getsize(os.path.join(LANDMARKS_PATH, "X_train.npy"))
    file_size += os.path.getsize(os.path.join(LANDMARKS_PATH, "X_test.npy"))
    print(f"\n💾 Total landmarks size: {file_size / 1024 / 1024:.2f} MB")
    
    print("\n✅ DONE! Landmarks extracted successfully!")
    print("Next step: Run 'python train_model.py'")


if __name__ == "__main__":
    process_dataset(
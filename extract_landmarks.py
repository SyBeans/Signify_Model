"""
extract_landmarks.py
Extracts hand landmarks from FSL videos using MediaPipe.
Saves as .npy files for training.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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
MODEL_PATH = "hand_landmarker.task"  # Your hand landmarker file

# Settings
NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_FRAMES = 30
NUM_LANDMARKS = 21
NUM_COORDINATES = 3  # x, y, z

# ============================================
# INITIALIZE MEDIAPIPE (New API)
# ============================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)
detector = vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_video(video_path):
    """
    Extract hand landmarks from a single video.
    Returns a numpy array of shape (NUM_FRAMES, 63)
    """
    cap = cv2.VideoCapture(video_path)
    
    all_landmarks = []
    
    while len(all_landmarks) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        detection_result = detector.detect(mp_image)
        
        # Extract landmarks if hand detected
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]  # First hand
            landmarks = []
            for lm in hand_landmarks:
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
    
    return np.array(all_landmarks)


def process_dataset():
    """Process all videos and extract landmarks"""
    
    os.makedirs(LANDMARKS_PATH, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\n📊 Total Signs: {len(labels_df)}")
    
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
        video_rel_path = row['vid_path'].replace('\\', '/')
        video_path = os.path.join(DATASET_PATH, video_rel_path)
        
        if not os.path.exists(video_path):
            errors.append(video_path)
            continue
        
        try:
            landmarks = extract_landmarks_from_video(video_path)
            X_train.append(landmarks)
            y_train.append(row['id_label'])
        except Exception as e:
            errors.append(f"{video_path}: {e}")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
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
        video_rel_path = row['vid_path'].replace('\\', '/')
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
    # SAVE
    # ============================================
    print("\n" + "=" * 60)
    print("💾 SAVING LANDMARKS")
    print("=" * 60)
    
    np.save(os.path.join(LANDMARKS_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_test.npy"), y_test)
    
    print(f"✅ Saved to '{LANDMARKS_PATH}/'")
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Training samples:   {len(X_train)}")
    print(f"Testing samples:    {len(X_test)}")
    print(f"Total:              {len(X_train) + len(X_test)}")
    print(f"Signs:              {len(labels_df)}")
    print(f"Features/sample:    {NUM_FRAMES * NUM_LANDMARKS * NUM_COORDINATES}")
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors (files not found)")
    
    print("\n✅ DONE! Next: python train_model.py")


if __name__ == "__main__":
    process_dataset()
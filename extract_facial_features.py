"""
extract_facial_features.py
Extracts facial landmarks for emotion recognition.
Uses MediaPipe Face Mesh (OLD API - no .task file needed).
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
LANDMARKS_PATH = "landmarks"

NUM_FRAMES = 30
NUM_FACE_LANDMARKS = 468  # MediaPipe Face Mesh has 468 landmarks (not 478)
NUM_COORDS = 3  # x, y, z

# ============================================
# INITIALIZE FACE MESH (OLD API)
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_face_from_video(video_path):
    """Extract face landmarks from video frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < NUM_FRAMES:
        cap.release()
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    all_faces = []
    face_count = 0
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            all_faces.append(None)
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_faces.append(landmarks)
            face_count += 1
        else:
            all_faces.append(None)
    
    cap.release()
    
    # Need at least 10 frames with faces
    if face_count < 10:
        return None
    
    # Fill missing frames
    for i in range(len(all_faces)):
        if all_faces[i] is None:
            for j in range(1, NUM_FRAMES):
                if i - j >= 0 and all_faces[i - j] is not None:
                    all_faces[i] = all_faces[i - j].copy()
                    break
                if i + j < NUM_FRAMES and all_faces[i + j] is not None:
                    all_faces[i] = all_faces[i + j].copy()
                    break
            if all_faces[i] is None:
                all_faces[i] = [0.0] * (NUM_FACE_LANDMARKS * NUM_COORDS)
    
    return np.array(all_faces, dtype=np.float32)  # (30, 1404)


def process_dataset():
    """Extract face landmarks from all videos."""
    os.makedirs(LANDMARKS_PATH, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🔨 EXTRACTING FACIAL LANDMARKS (Training)")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_CSV)
    X_face_train = []
    skipped = 0
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training"):
        video_path = os.path.join(DATASET_PATH, row['vid_path'].replace('\\', '/'))
        
        if not os.path.exists(video_path):
            skipped += 1
            continue
        
        faces = extract_face_from_video(video_path)
        if faces is not None:
            X_face_train.append(faces)
        else:
            skipped += 1
    
    X_face_train = np.array(X_face_train, dtype=np.float32)
    print(f"\n✅ X_face_train: {X_face_train.shape}")
    print(f"⚠️  Skipped: {skipped}")
    
    # ============================================
    print("\n" + "=" * 60)
    print("🔨 EXTRACTING FACIAL LANDMARKS (Testing)")
    print("=" * 60)
    
    test_df = pd.read_csv(TEST_CSV)
    X_face_test = []
    skipped_test = 0
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        video_path = os.path.join(DATASET_PATH, row['vid_path'].replace('\\', '/'))
        
        if not os.path.exists(video_path):
            skipped_test += 1
            continue
        
        faces = extract_face_from_video(video_path)
        if faces is not None:
            X_face_test.append(faces)
        else:
            skipped_test += 1
    
    X_face_test = np.array(X_face_test, dtype=np.float32)
    print(f"\n✅ X_face_test: {X_face_test.shape}")
    print(f"⚠️  Skipped: {skipped_test}")
    
    # Save
    np.save(os.path.join(LANDMARKS_PATH, "X_face_train.npy"), X_face_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_face_test.npy"), X_face_test)
    
    print("\n✅ Facial landmarks saved!")
    print(f"   - X_face_train.npy ({X_face_train.shape})")
    print(f"   - X_face_test.npy ({X_face_test.shape})")


if __name__ == "__main__":
    process_dataset()
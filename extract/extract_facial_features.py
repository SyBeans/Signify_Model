"""
extract_facial_features.py (FER+ version)
Extracts 468 facial landmarks from FER+ dataset.
Uses FER+ labels (cleaner than FER2013).
7 emotions (drops contempt).
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
FERPLUS_PATH = os.path.expanduser(
    "~/Signify/Signify_Model/datasets/FERPLUS/FER2013Plus"
)
LANDMARKS_PATH = "landmarks/face_ferplus"

NUM_FACE_LANDMARKS = 468
NUM_COORDS = 3
NUM_FEATURES = NUM_FACE_LANDMARKS * NUM_COORDS  # 1404

# 7 emotions (drops "contempt")
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# FER+ label → our emotion mapping
LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

EMOTION_TO_ID = {name: i for i, name in enumerate(EMOTIONS)}

UPSCALE_SIZE = 192

# ============================================
# MEDIAPIPE
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3
)


def extract_face_landmarks(image_path):
    """Extract 468 face landmarks with upscaling."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if w < 100 or h < 100:
        img = cv2.resize(img, (UPSCALE_SIZE, UPSCALE_SIZE),
                         interpolation=cv2.INTER_CUBIC)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = []
    for lm in face_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)


def process_split(split_folder, split_name):
    """Process one split (FER2013Train/FER2013Valid/FER2013Test)."""
    split_path = os.path.join(FERPLUS_PATH, split_folder)
    labels_path = os.path.join(split_path, "labels.csv")

    if not os.path.exists(labels_path):
        print(f"⚠️  Not found: {labels_path}")
        return None, None

    labels_df = pd.read_csv(labels_path)

    X, y = [], []
    skipped_no_face = 0
    skipped_contempt = 0

    print(f"\n{'=' * 60}")
    print(f"🔨 PROCESSING {split_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Total: {len(labels_df)} images")

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                          desc=split_name):
        fname = row["filename"]
        emotion_raw = str(row["emotion"]).lower().strip()

        if emotion_raw not in LABEL_MAP:
            skipped_contempt += 1
            continue

        image_path = os.path.join(split_path, fname)
        if not os.path.exists(image_path):
            skipped_no_face += 1
            continue

        landmarks = extract_face_landmarks(image_path)
        if landmarks is None:
            skipped_no_face += 1
            continue

        X.append(landmarks)
        y.append(EMOTION_TO_ID[LABEL_MAP[emotion_raw]])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n✅ {split_name}: X = {X.shape}, y = {y.shape}")
    print(f"   Skipped (no face): {skipped_no_face}")
    print(f"   Skipped (contempt): {skipped_contempt}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"   Class distribution:")
    for uid, cnt in zip(unique, counts):
        print(f"     {EMOTIONS[uid]:10s}: {cnt}")

    return X, y


def main():
    os.makedirs(LANDMARKS_PATH, exist_ok=True)

    print("=" * 60)
    print("📊 FER+ FACIAL LANDMARK EXTRACTION")
    print("=" * 60)
    print(f"Dataset: {FERPLUS_PATH}")
    print(f"Features: {NUM_FEATURES}")
    print(f"Emotions (7): {EMOTIONS}")
    print(f"Output: {LANDMARKS_PATH}/")

    X_train, y_train = process_split("FER2013Train", "train")
    X_val, y_val = process_split("FER2013Valid", "val")
    X_test, y_test = process_split("FER2013Test", "test")

    print(f"\n{'=' * 60}")
    print("💾 SAVING LANDMARKS")
    print(f"{'=' * 60}")

    np.save(os.path.join(LANDMARKS_PATH, "X_face_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_face_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_face_val.npy"), X_val)
    np.save(os.path.join(LANDMARKS_PATH, "y_face_val.npy"), y_val)
    np.save(os.path.join(LANDMARKS_PATH, "X_face_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_face_test.npy"), y_test)

    print(f"✅ Saved to {LANDMARKS_PATH}/")
    print(f"   X_face_train: {X_train.shape}")
    print(f"   X_face_val:   {X_val.shape}")
    print(f"   X_face_test:  {X_test.shape}")

    print("\n" + "=" * 60)
    print("✅ DONE! Next: train/train_emotion_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
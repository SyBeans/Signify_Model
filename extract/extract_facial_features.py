"""
extract_facial_features.py
Extracts 468 facial landmarks from FER2013 images using MediaPipe Face Mesh.
Saves as .npy files for training the emotion model.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
FER2013_PATH = os.path.expanduser(
    "~/Signify/Signify_Model/datasets/FER2013"
)
LANDMARKS_PATH = "landmarks/face"

NUM_FACE_LANDMARKS = 468
NUM_COORDS = 3
NUM_FEATURES = NUM_FACE_LANDMARKS * NUM_COORDS  # 1404

# Emotion order (must match training script!)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_TO_ID = {name: i for i, name in enumerate(EMOTIONS)}

# ============================================
# INITIALIZE MEDIAPIPE FACE MESH
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,        # Image mode (not video)
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3
)

def extract_face_landmarks(image_path):
    """Extract 468 face landmarks from an image (with upscaling)."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # ✅ UPSCALE small FER2013 images (48x48 → 192x192) for better MediaPipe
    h, w = img.shape[:2]
    if w < 100 or h < 100:
        img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = []
    for lm in face_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)

def process_split(split_name):
    """Process one split (train/test/val)."""
    split_path = os.path.join(FER2013_PATH, split_name)
    X, y = [], []
    skipped_no_face = 0

    print(f"\n{'=' * 60}")
    print(f"🔨 PROCESSING {split_name.upper()}")
    print(f"{'=' * 60}")

    for emotion in EMOTIONS:
        emotion_path = os.path.join(split_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️  Skipping {emotion} (not found)")
            continue

        files = [f for f in os.listdir(emotion_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in tqdm(files, desc=f"{emotion:10s}", leave=False):
            img_path = os.path.join(emotion_path, fname)
            landmarks = extract_face_landmarks(img_path)

            if landmarks is None:
                skipped_no_face += 1
                continue

            X.append(landmarks)
            y.append(EMOTION_TO_ID[emotion])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"✅ {split_name}: X = {X.shape}, y = {y.shape}")
    print(f"   Skipped (no face): {skipped_no_face}")

    return X, y


def main():
    os.makedirs(LANDMARKS_PATH, exist_ok=True)

    print("=" * 60)
    print("📊 FER2013 FACIAL LANDMARK EXTRACTION")
    print("=" * 60)
    print(f"Features per image: {NUM_FEATURES}")
    print(f"Upscale target: 192×192 (for small images)")
    print(f"Emotions: {EMOTIONS}")

    X_train, y_train = process_split("train")
    X_test, y_test = process_split("test")
    X_val, y_val = process_split("val")

    print(f"\n{'=' * 60}")
    print("💾 SAVING LANDMARKS")
    print(f"{'=' * 60}")

    np.save(os.path.join(LANDMARKS_PATH, "X_face_train.npy"), X_train)
    np.save(os.path.join(LANDMARKS_PATH, "y_face_train.npy"), y_train)
    np.save(os.path.join(LANDMARKS_PATH, "X_face_test.npy"), X_test)
    np.save(os.path.join(LANDMARKS_PATH, "y_face_test.npy"), y_test)
    np.save(os.path.join(LANDMARKS_PATH, "X_face_val.npy"), X_val)
    np.save(os.path.join(LANDMARKS_PATH, "y_face_val.npy"), y_val)

    print(f"✅ Saved to {LANDMARKS_PATH}/")
    print(f"   X_face_train.npy: {X_train.shape}")
    print(f"   y_face_train.npy: {y_train.shape}")
    print(f"   X_face_test.npy:  {X_test.shape}")
    print(f"   y_face_test.npy:  {y_test.shape}")
    print(f"   X_face_val.npy:   {X_val.shape}")
    print(f"   y_face_val.npy:   {y_val.shape}")

    print("\n" + "=" * 60)
    print("✅ DONE! Next: train_emotion_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
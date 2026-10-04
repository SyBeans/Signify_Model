"""
extract_rafdb_v12.py
Extract RAF-DB images + landmarks, map 7-class → 4-class.
Input:  datasets/RAF-DB/DATASET/{train,test}/{1,2,3,4,5,6,7}/*.jpg
Output: landmarks/rafdb_v12/
  X_train.npy, L_train.npy, y_train.npy
  X_val.npy,   L_val.npy,   y_val.npy
  X_test.npy,  L_test.npy,  y_test.npy
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'

import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import random

RAF_PATH    = os.path.expanduser("~/Signify/Signify_Model/datasets/RAF-DB/DATASET")
OUTPUT_PATH = os.path.expanduser("~/Signify/Signify_Model/landmarks/rafdb_v12")
IMG_SIZE = 48

# RAF folder → 4-class
RAF_TO_4CLASS = {
    "1": 2,  # surprise  → Surprise
    "2": 1,  # fear      → Negative
    "3": 1,  # disgust   → Negative
    "4": 0,  # happiness → Positive
    "5": 1,  # sadness   → Negative
    "6": 1,  # anger     → Negative
    "7": 3,  # neutral   → Neutral
}

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_PATH, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.3
)

def extract_both(image_bgr):
    """Return (image_48x48, landmarks_936) or (None, None)."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    # Landmarks vector
    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_vec = coords.flatten()

    # Image crop (same 1.55x as FER)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.55

    y_min = max(0.0, cy - size * 0.65)
    y_max = min(1.0, cy + size * 0.35)
    x_min = max(0.0, cx - size * 0.50)
    x_max = min(1.0, cx + size * 0.50)

    px1, py1 = int(x_min * w), int(y_min * h)
    px2, py2 = int(x_max * w), int(y_max * h)

    if px2 <= px1 or py2 <= py1:
        return None, None

    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray, lm_vec

def process_split(split_name):
    print(f"\n📦 RAF-DB {split_name}")
    X_list, L_list, y_list = [], [], []
    skipped = 0
    total = 0

    for folder, class_4 in RAF_TO_4CLASS.items():
        folder_path = os.path.join(RAF_PATH, split_name, folder)
        if not os.path.isdir(folder_path):
            print(f"  ⚠️  Missing: {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        pbar = tqdm(files, desc=f"  Folder {folder} → class {class_4}")
        for fname in pbar:
            total += 1
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue

            crop, lm_vec = extract_both(img)
            if crop is None:
                skipped += 1
                continue

            X_list.append(crop)
            L_list.append(lm_vec)
            y_list.append(class_4)
        pbar.close()

    print(f"✅ {split_name}: {len(X_list)} saved, {skipped} skipped (of {total})")
    return X_list, L_list, y_list

print("=" * 60)
print("📥 EXTRACTING RAF-DB → 4-CLASS")
print("=" * 60)

# Process official RAF split
Xtr, Ltr, ytr = process_split("train")
Xte, Lte, yte = process_split("test")

face_mesh.close()

# Train split → 85% train + 15% val
Xtr = np.array(Xtr, dtype=np.uint8)
Ltr = np.array(Ltr, dtype=np.float32)
ytr = np.array(ytr, dtype=np.int32)

print(f"\nTrain raw: X={Xtr.shape}, L={Ltr.shape}, per-class: {np.bincount(ytr, minlength=4)}")

perm = np.random.permutation(len(Xtr))
Xtr, Ltr, ytr = Xtr[perm], Ltr[perm], ytr[perm]

n_train = int(0.85 * len(Xtr))
X_train, L_train, y_train = Xtr[:n_train], Ltr[:n_train], ytr[:n_train]
X_val,   L_val,   y_val   = Xtr[n_train:], Ltr[n_train:], ytr[n_train:]

# Test (RAF's official test set)
X_test = np.array(Xte, dtype=np.uint8)
L_test = np.array(Lte, dtype=np.float32)
y_test = np.array(yte, dtype=np.int32)

perm = np.random.permutation(len(X_test))
X_test, L_test, y_test = X_test[perm], L_test[perm], y_test[perm]

# Normalize images
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0

print(f"\n✅ Final splits:")
print(f"  Train: X={X_train.shape}, L={L_train.shape}, y={y_train.shape}")
print(f"  Val:   X={X_val.shape}, L={L_val.shape}, y={y_val.shape}")
print(f"  Test:  X={X_test.shape}, L={L_test.shape}, y={y_test.shape}")
print(f"  Train per-class: {np.bincount(y_train, minlength=4)}")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_train); np.save(f"{OUTPUT_PATH}/L_train.npy", L_train); np.save(f"{OUTPUT_PATH}/y_train.npy", y_train)
np.save(f"{OUTPUT_PATH}/X_val.npy",   X_val);   np.save(f"{OUTPUT_PATH}/L_val.npy",   L_val);   np.save(f"{OUTPUT_PATH}/y_val.npy", y_val)
np.save(f"{OUTPUT_PATH}/X_test.npy",  X_test);  np.save(f"{OUTPUT_PATH}/L_test.npy",  L_test);  np.save(f"{OUTPUT_PATH}/y_test.npy", y_test)

print(f"\n✅ Saved to {OUTPUT_PATH}")
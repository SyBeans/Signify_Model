"""
extract_fer2025_subset.py  (v3 — wider forehead crop)

Reads FER2025 TAR files, takes 6k images per class, extracts MediaPipe
Face Mesh landmarks, crops faces with EXTENDED forehead margin (1.55x)
and ASPECT-PRESERVING square crop, resizes to 48x48 grayscale.

Output: landmarks/fer2025/
  X_train.npy  (33600, 48, 48, 1)
  y_train.npy  (33600,)
  X_val.npy    (4200, 48, 48, 1)
  y_val.npy    (4200,)
  X_test.npy   (4200, 48, 48, 1)
  y_test.npy   (4200,)
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import random
import tarfile

# ============================================
# CONFIGURATION
# ============================================
FER2025_PATH = os.path.expanduser("~/Signify/Signify_Model/datasets/FER2025")
OUTPUT_PATH = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025")

IMG_SIZE = 48
IMAGES_PER_CLASS = 6000
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# MEDIAPIPE FACE MESH
# ============================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3
)

# ============================================
# ✅ v3 IMPROVED FACE CROP (wider forehead)
# ============================================
def extract_face_crop(image_bgr):
    """Extract face crop with:
    - SQUARE bounding box (no aspect distortion)
    - Wider margin (1.55x)
    - Extra forehead room (0.65 top vs 0.35 bottom)
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    x_min_raw, x_max_raw = min(xs), max(xs)
    y_min_raw, y_max_raw = min(ys), max(ys)

    # Center + square extent
    cx = (x_min_raw + x_max_raw) / 2
    cy = (y_min_raw + y_max_raw) / 2
    size = max(x_max_raw - x_min_raw, y_max_raw - y_min_raw) * 1.55   # ← was 1.35

    # Extra forehead (top 65%) vs chin (bottom 35%)   ← was 55/45
    y_min = max(0.0, cy - size * 0.65)
    y_max = min(1.0, cy + size * 0.35)
    x_min = max(0.0, cx - size * 0.50)
    x_max = min(1.0, cx + size * 0.50)

    px1, py1 = int(x_min * w), int(y_min * h)
    px2, py2 = int(x_max * w), int(y_max * h)

    if px2 <= px1 or py2 <= py1:
        return None

    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray

# ============================================
# READ ONE TAR
# ============================================
def process_tar(tar_path, class_name, max_images):
    print(f"\n📦 Processing {class_name}: {tar_path}")
    X_list, y_list = [], []
    skipped_no_face = 0
    processed = 0
    label = CLASS_TO_ID[class_name]

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.lower().endswith(
                       ('.jpg', '.jpeg', '.png'))]
        random.shuffle(members)

        pbar = tqdm(total=max_images, desc=class_name)
        for member in members:
            if processed >= max_images:
                break

            f = tar.extractfile(member)
            if f is None:
                continue

            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue

            crop = extract_face_crop(img)
            if crop is None:
                skipped_no_face += 1
                continue

            X_list.append(crop)
            y_list.append(label)
            processed += 1
            pbar.update(1)

        pbar.close()

    print(f"✅ {class_name}: {len(X_list)} saved, {skipped_no_face} no-face")
    return X_list, y_list

# ============================================
# MAIN
# ============================================
print("=" * 60)
print("📥 EXTRACTING FER2025 (6k/class, v3 wider crop)")
print("=" * 60)

all_X, all_y = [], []

for class_name in CLASSES:
    tar_path = os.path.join(FER2025_PATH, f"{class_name}.tar")
    if not os.path.exists(tar_path):
        print(f"❌ Missing: {tar_path}")
        continue

    X_c, y_c = process_tar(tar_path, class_name, IMAGES_PER_CLASS)
    all_X.extend(X_c)
    all_y.extend(y_c)

face_mesh.close()

# ============================================
# SPLIT + SAVE
# ============================================
print("\n" + "=" * 60)
print("✂️  SPLITTING TRAIN / VAL / TEST")
print("=" * 60)

X_all = np.array(all_X, dtype=np.uint8)
y_all = np.array(all_y, dtype=np.int32)

print(f"Total: X={X_all.shape}, y={y_all.shape}")
print(f"Per class: {np.bincount(y_all)}")

X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []
X_test_list, y_test_list = [], []

for class_id in range(len(CLASSES)):
    idx = np.where(y_all == class_id)[0]
    np.random.shuffle(idx)

    n = len(idx)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    X_train_list.append(X_all[idx[:n_train]])
    y_train_list.append(y_all[idx[:n_train]])
    X_val_list.append(X_all[idx[n_train:n_train + n_val]])
    y_val_list.append(y_all[idx[n_train:n_train + n_val]])
    X_test_list.append(X_all[idx[n_train + n_val:]])
    y_test_list.append(y_all[idx[n_train + n_val:]])

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_val = np.concatenate(X_val_list, axis=0)
y_val = np.concatenate(y_val_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
    perm = np.random.permutation(len(X))
    X[:] = X[perm]
    y[:] = y[perm]

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0

print(f"\n✅ Final splits:")
print(f"  Train: X={X_train.shape}, y={y_train.shape}")
print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_PATH, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

print(f"\n✅ Saved to {OUTPUT_PATH}")
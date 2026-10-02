"""
extract_fer2025_v5.py
- MediaPipe as detector + generous crop (1.8x margin)
- CLAHE contrast enhancement
- Same 6k/class, same splits
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'

import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import random
import tarfile

FER2025_PATH = os.path.expanduser("~/Signify/Signify_Model/datasets/FER2025")
OUTPUT_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v5")

IMG_SIZE = 48
IMAGES_PER_CLASS = 6000
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
random.seed(SEED); np.random.seed(SEED)

os.makedirs(OUTPUT_PATH, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.3
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_face_crop(image_bgr):
    """MediaPipe crop (1.8x generous) + CLAHE."""
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

    cx = (x_min_raw + x_max_raw) / 2
    cy = (y_min_raw + y_max_raw) / 2
    # ✅ v5: generous 1.8x margin
    size = max(x_max_raw - x_min_raw, y_max_raw - y_min_raw) * 1.8

    # ✅ v5: extended forehead (60%) + chin (40%)
    y_min = max(0.0, cy - size * 0.60)
    y_max = min(1.0, cy + size * 0.40)
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
    gray = clahe.apply(gray)   # ✅ CLAHE from v4
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray

def process_tar(tar_path, class_name, max_images):
    print(f"\n📦 {class_name}: {tar_path}")
    X_list, y_list = [], []
    skipped = 0
    label = CLASS_TO_ID[class_name]

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(members)

        pbar = tqdm(total=max_images, desc=class_name)
        for member in members:
            if len(X_list) >= max_images:
                break
            f = tar.extractfile(member)
            if f is None: continue
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None: continue

            crop = extract_face_crop(img)
            if crop is None:
                skipped += 1
                continue

            X_list.append(crop); y_list.append(label)
            pbar.update(1)
        pbar.close()

    print(f"✅ {class_name}: {len(X_list)} saved, {skipped} skipped")
    return X_list, y_list

print("=" * 60)
print("📥 EXTRACTING FER2025 v5 (generous crop + CLAHE)")
print("=" * 60)

all_X, all_y = [], []
for cls in CLASSES:
    tp = os.path.join(FER2025_PATH, f"{cls}.tar")
    if not os.path.exists(tp):
        print(f"❌ Missing: {tp}"); continue
    X_c, y_c = process_tar(tp, cls, IMAGES_PER_CLASS)
    all_X.extend(X_c); all_y.extend(y_c)

face_mesh.close()

print("\n" + "=" * 60); print("✂️  SPLITTING"); print("=" * 60)

X_all = np.array(all_X, dtype=np.uint8)
y_all = np.array(all_y, dtype=np.int32)
print(f"Total: {X_all.shape}, per-class: {np.bincount(y_all)}")

Xtr, ytr = [], []
Xv, yv = [], []
Xte, yte = [], []

for cid in range(len(CLASSES)):
    idx = np.where(y_all == cid)[0]
    np.random.shuffle(idx)
    n = len(idx)
    n_tr = int(n * TRAIN_RATIO)
    n_v = int(n * VAL_RATIO)

    Xtr.append(X_all[idx[:n_tr]]);       ytr.append(y_all[idx[:n_tr]])
    Xv.append(X_all[idx[n_tr:n_tr+n_v]]); yv.append(y_all[idx[n_tr:n_tr+n_v]])
    Xte.append(X_all[idx[n_tr+n_v:]]);    yte.append(y_all[idx[n_tr+n_v:]])

X_train = np.concatenate(Xtr); y_train = np.concatenate(ytr)
X_val   = np.concatenate(Xv);  y_val   = np.concatenate(yv)
X_test  = np.concatenate(Xte); y_test  = np.concatenate(yte)

for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
    p = np.random.permutation(len(X))
    X[:] = X[p]; y[:] = y[p]

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0

print(f"\n✅ Splits:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_train); np.save(f"{OUTPUT_PATH}/y_train.npy", y_train)
np.save(f"{OUTPUT_PATH}/X_val.npy",   X_val);   np.save(f"{OUTPUT_PATH}/y_val.npy", y_val)
np.save(f"{OUTPUT_PATH}/X_test.npy",  X_test);  np.save(f"{OUTPUT_PATH}/y_test.npy", y_test)

print(f"\n✅ Saved to {OUTPUT_PATH}")
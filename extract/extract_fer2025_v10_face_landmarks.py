"""
extract_fer2025_v10_landmarks.py
Extract 468 MediaPipe Face Mesh landmarks per image.
Save as (936,) feature vectors — 468 landmarks × (x, y).

Output: landmarks/fer2025_v10/
  L_train.npy (61600, 936)
  y_train.npy (61600,)
  L_val.npy   (7700, 936)
  y_val.npy   (7700,)
  L_test.npy  (7700, 936)
  y_test.npy  (7700,)

Same 7-class extraction; 4-class grouping at training time.
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
OUTPUT_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v10")

# Same class counts as v9
IMAGES_PER_CLASS = {
    "Angry":    8000,
    "Disgust":  8000,
    "Fear":     8000,
    "Happy":    15000,
    "Neutral":  15000,
    "Sad":      8000,
    "Surprise": 15000,
}

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
random.seed(SEED); np.random.seed(SEED)

os.makedirs(OUTPUT_PATH, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,   # 468 points (no iris)
    min_detection_confidence=0.3
)

# Quality thresholds (same as v8/v9)
MIN_LANDMARKS = 450
MIN_FACE_SIZE = 0.15
MAX_FACE_SIZE = 0.90
MIN_ASPECT    = 0.6
MAX_ASPECT    = 1.6
MIN_BLUR_VAR  = 30
MIN_BRIGHT    = 0.15
MAX_BRIGHT    = 0.85

def passes_quality(gray_full, face_w, face_h, num_landmarks):
    if num_landmarks < MIN_LANDMARKS: return False
    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE: return False
    if face_w > MAX_FACE_SIZE or face_h > MAX_FACE_SIZE: return False
    aspect = face_w / max(face_h, 1e-6)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT: return False
    if gray_full.size > 0:
        lap_var = cv2.Laplacian(gray_full, cv2.CV_64F).var()
        if lap_var < MIN_BLUR_VAR: return False
    mean_bright = gray_full.mean() / 255.0
    if mean_bright < MIN_BRIGHT or mean_bright > MAX_BRIGHT: return False
    return True

def extract_landmarks(image_bgr):
    """Return flat (936,) landmark vector or None."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0]
    num_landmarks = len(lm.landmark)

    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)

    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if not passes_quality(gray_full, face_w, face_h, num_landmarks):
        return None

    # 468 landmarks × (x, y) = 936 features
    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    return coords.flatten()   # (936,)

def process_tar(tar_path, class_name, max_images):
    print(f"\n📦 {class_name} (target: {max_images})")
    L_list, y_list = [], []
    skipped = 0
    label = CLASS_TO_ID[class_name]

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(members)

        pbar = tqdm(total=max_images, desc=class_name)
        for member in members:
            if len(L_list) >= max_images:
                break
            f = tar.extractfile(member)
            if f is None: continue
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None: continue

            lm_vec = extract_landmarks(img)
            if lm_vec is None:
                skipped += 1
                continue

            L_list.append(lm_vec); y_list.append(label)
            pbar.update(1)
        pbar.close()

    print(f"✅ {class_name}: {len(L_list)} saved, {skipped} rejected")
    return L_list, y_list

print("=" * 60)
print("📥 EXTRACTING LANDMARKS v10")
print("=" * 60)

all_L, all_y = [], []
for cls in CLASSES:
    tp = os.path.join(FER2025_PATH, f"{cls}.tar")
    if not os.path.exists(tp):
        print(f"❌ Missing: {tp}"); continue
    L_c, y_c = process_tar(tp, cls, IMAGES_PER_CLASS[cls])
    all_L.extend(L_c); all_y.extend(y_c)

face_mesh.close()

print("\n" + "=" * 60); print("✂️  SPLITTING"); print("=" * 60)

L_all = np.array(all_L, dtype=np.float32)
y_all = np.array(all_y, dtype=np.int32)
print(f"Total: {L_all.shape}, per-class: {np.bincount(y_all)}")

Ltr, ytr = [], []
Lv, yv = [], []
Lte, yte = [], []

for cid in range(len(CLASSES)):
    idx = np.where(y_all == cid)[0]
    np.random.shuffle(idx)
    n = len(idx)
    n_tr = int(n * TRAIN_RATIO)
    n_v = int(n * VAL_RATIO)

    Ltr.append(L_all[idx[:n_tr]]);       ytr.append(y_all[idx[:n_tr]])
    Lv.append(L_all[idx[n_tr:n_tr+n_v]]); yv.append(y_all[idx[n_tr:n_tr+n_v]])
    Lte.append(L_all[idx[n_tr+n_v:]]);    yte.append(y_all[idx[n_tr+n_v:]])

L_train = np.concatenate(Ltr); y_train = np.concatenate(ytr)
L_val   = np.concatenate(Lv);  y_val   = np.concatenate(yv)
L_test  = np.concatenate(Lte); y_test  = np.concatenate(yte)

for L, y in [(L_train, y_train), (L_val, y_val), (L_test, y_test)]:
    p = np.random.permutation(len(L))
    L[:] = L[p]; y[:] = y[p]

print(f"\n✅ Splits:")
print(f"  Train: {L_train.shape}")
print(f"  Val:   {L_val.shape}")
print(f"  Test:  {L_test.shape}")

np.save(f"{OUTPUT_PATH}/L_train.npy", L_train); np.save(f"{OUTPUT_PATH}/y_train.npy", y_train)
np.save(f"{OUTPUT_PATH}/L_val.npy",   L_val);   np.save(f"{OUTPUT_PATH}/y_val.npy", y_val)
np.save(f"{OUTPUT_PATH}/L_test.npy",  L_test);  np.save(f"{OUTPUT_PATH}/y_test.npy", y_test)

print(f"\n✅ Saved to {OUTPUT_PATH}")
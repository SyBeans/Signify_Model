"""
extract_fer2025_v11_hybrid.py
Save BOTH images AND landmarks for hybrid training.
Output: landmarks/fer2025_v11/
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
import tarfile

FER2025_PATH = os.path.expanduser("~/Signify/Signify_Model/datasets/FER2025")
OUTPUT_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")

IMG_SIZE = 48
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
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.3
)

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
        if cv2.Laplacian(gray_full, cv2.CV_64F).var() < MIN_BLUR_VAR: return False
    mb = gray_full.mean() / 255.0
    if mb < MIN_BRIGHT or mb > MAX_BRIGHT: return False
    return True

def extract_both(image_bgr):
    """Return (image_48x48, landmarks_936) or (None, None)."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0]
    num_landmarks = len(lm.landmark)
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if not passes_quality(gray_full, face_w, face_h, num_landmarks):
        return None, None

    # Landmarks (936)
    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_vec = coords.flatten()

    # Image crop (48x48)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    size = max(face_w, face_h) * 1.55
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

def process_tar(tar_path, class_name, max_images):
    print(f"\n📦 {class_name} (target: {max_images})")
    X_list, L_list, y_list = [], [], []
    skipped = 0
    label = CLASS_TO_ID[class_name]

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(members)
        pbar = tqdm(total=max_images, desc=class_name)
        for member in members:
            if len(X_list) >= max_images: break
            f = tar.extractfile(member)
            if f is None: continue
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None: continue

            crop, lm_vec = extract_both(img)
            if crop is None:
                skipped += 1
                continue

            X_list.append(crop); L_list.append(lm_vec); y_list.append(label)
            pbar.update(1)
        pbar.close()
    print(f"✅ {class_name}: {len(X_list)} saved, {skipped} rejected")
    return X_list, L_list, y_list

print("=" * 60)
print("📥 EXTRACTING FER2025 v11 (images + landmarks)")
print("=" * 60)

all_X, all_L, all_y = [], [], []
for cls in CLASSES:
    tp = os.path.join(FER2025_PATH, f"{cls}.tar")
    if not os.path.exists(tp):
        print(f"❌ Missing: {tp}"); continue
    X_c, L_c, y_c = process_tar(tp, cls, IMAGES_PER_CLASS[cls])
    all_X.extend(X_c); all_L.extend(L_c); all_y.extend(y_c)

face_mesh.close()

print("\n" + "=" * 60); print("✂️  SPLITTING"); print("=" * 60)

X_all = np.array(all_X, dtype=np.uint8)
L_all = np.array(all_L, dtype=np.float32)
y_all = np.array(all_y, dtype=np.int32)
print(f"Total: X={X_all.shape}, L={L_all.shape}, y={y_all.shape}")

Xtr, Ltr, ytr = [], [], []
Xv, Lv, yv = [], [], []
Xte, Lte, yte = [], [], []

for cid in range(len(CLASSES)):
    idx = np.where(y_all == cid)[0]
    np.random.shuffle(idx)
    n = len(idx)
    n_tr = int(n * TRAIN_RATIO)
    n_v = int(n * VAL_RATIO)
    tr, v, te = idx[:n_tr], idx[n_tr:n_tr+n_v], idx[n_tr+n_v:]
    Xtr.append(X_all[tr]); Ltr.append(L_all[tr]); ytr.append(y_all[tr])
    Xv.append(X_all[v]);   Lv.append(L_all[v]);   yv.append(y_all[v])
    Xte.append(X_all[te]); Lte.append(L_all[te]); yte.append(y_all[te])

X_train = np.concatenate(Xtr); L_train = np.concatenate(Ltr); y_train = np.concatenate(ytr)
X_val   = np.concatenate(Xv);  L_val   = np.concatenate(Lv);  y_val   = np.concatenate(yv)
X_test  = np.concatenate(Xte); L_test  = np.concatenate(Lte); y_test  = np.concatenate(yte)

for idx_perm in [
    (X_train, L_train, y_train),
    (X_val,   L_val,   y_val),
    (X_test,  L_test,  y_test),
]:
    p = np.random.permutation(len(idx_perm[0]))
    idx_perm[0][:] = idx_perm[0][p]
    idx_perm[1][:] = idx_perm[1][p]
    idx_perm[2][:] = idx_perm[2][p]

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_val   = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0

print(f"\n✅ Splits:")
print(f"  Train: X={X_train.shape}, L={L_train.shape}, y={y_train.shape}")
print(f"  Val:   X={X_val.shape}, L={L_val.shape}, y={y_val.shape}")
print(f"  Test:  X={X_test.shape}, L={L_test.shape}, y={y_test.shape}")

np.save(f"{OUTPUT_PATH}/X_train.npy", X_train); np.save(f"{OUTPUT_PATH}/L_train.npy", L_train); np.save(f"{OUTPUT_PATH}/y_train.npy", y_train)
np.save(f"{OUTPUT_PATH}/X_val.npy",   X_val);   np.save(f"{OUTPUT_PATH}/L_val.npy",   L_val);   np.save(f"{OUTPUT_PATH}/y_val.npy", y_val)
np.save(f"{OUTPUT_PATH}/X_test.npy",  X_test);  np.save(f"{OUTPUT_PATH}/L_test.npy",  L_test);  np.save(f"{OUTPUT_PATH}/y_test.npy", y_test)

print(f"\n✅ Saved to {OUTPUT_PATH}")
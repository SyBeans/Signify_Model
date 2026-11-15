"""
extract_fer2025_v12_balanced.py
Weighted Negative: Angry + Sad get MORE data (more common in FSL).
Disgust + Fear get less (rare in real-world FSL emotion).

4-class is still perfectly balanced at 20k each:
  Negative = Angry(8k) + Sad(8k) + Disgust(2k) + Fear(2k) = 20k
  Positive = Happy(20k)      = 20k
  Surprise = Surprise(20k)   = 20k
  Neutral  = Neutral(20k)    = 20k
  Total    = 80k samples

Rationale:
  - Angry and Sad are commonly expressed in FSL context (medical, emergency)
  - Disgust and Fear are rare → less training data needed
  - When Disgust appears, it may lean toward Angry (both → Negative, OK)
  - When Fear appears, it may lean toward Surprise (4-class issue — see notes)
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

# ============================================
# CONFIG — WEIGHTED NEGATIVE
# ============================================
FER2025_PATH = os.path.expanduser("~/Signify/Signify_Model/datasets/FER2025")
OUTPUT_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v12_balanced")

IMG_SIZE = 48
IMAGES_PER_CLASS = {
    "Angry":    8000,    # 🆕 UP from 5000 — common in FSL
    "Disgust":  2000,    # 🆕 DOWN from 5000 — rare in FSL
    "Fear":     2000,    # 🆕 DOWN from 5000 — rare in FSL
    "Happy":    20000,
    "Neutral":  20000,
    "Sad":      8000,    # 🆕 UP from 5000 — common in FSL
    "Surprise": 20000,
}

print("=" * 60)
print(f"📊 WEIGHTED NEGATIVE EXTRACTION PLAN")
print("=" * 60)
print(f"  4-class balance (all equal at 20k):")
print(f"    Positive: 20,000  (Happy)")
print(f"    Negative: 20,000  (Angry 8k + Sad 8k + Disgust 2k + Fear 2k)")
print(f"    Surprise: 20,000")
print(f"    Neutral : 20,000")
print(f"  ─────────────────────────────────")
print(f"  Total:    80,000 samples")
print(f"")
print(f"  Negative internal weights:")
print(f"    Angry    : 8,000  (40% of Negative)")
print(f"    Sad      : 8,000  (40% of Negative)")
print(f"    Disgust  : 2,000  (10% of Negative) — rare")
print(f"    Fear     : 2,000  (10% of Negative) — rare")
print("=" * 60)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

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

    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_vec = coords.flatten()

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

    if len(X_list) < max_images:
        print(f"⚠️  {class_name}: only got {len(X_list)}/{max_images} — dataset may be limited")
    print(f"✅ {class_name}: {len(X_list)} saved, {skipped} rejected")
    return X_list, L_list, y_list

print("\n" + "=" * 60)
print("📥 EXTRACTING WEIGHTED FER2025")
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

print(f"\n📊 Achieved 7-class counts:")
for i, cls in enumerate(CLASSES):
    n = int(np.sum(y_all == i))
    print(f"  {cls:<10}: {n:,}")

print(f"\n📊 Achieved 4-class counts:")
y_4class = GROUP_MAP[y_all]
for i, name in enumerate(["Positive", "Negative", "Surprise", "Neutral"]):
    n = int(np.sum(y_4class == i))
    bar = "█" * int(n / 500)
    print(f"  {name:<10}: {n:,}  {bar}")

counts_4 = np.bincount(y_4class, minlength=4)
imbalance = counts_4.max() / max(counts_4.min(), 1)
print(f"\n  Imbalance ratio: {imbalance:.2f}x  (1.00 = perfectly balanced)")

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
print(f"\n⚠️  NOTE: y_train.npy still has 7-class labels (0-6)")
print(f"   train_hybrid_v11_13.py handles 7→4 grouping via GROUP_MAP")
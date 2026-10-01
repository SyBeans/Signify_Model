"""
test_crop_v3.py — Quick 2k/class test to preview new crop.
Does NOT overwrite landmarks/fer2025/*.npy.
"""
import os, numpy as np, cv2, mediapipe as mp, random, tarfile
from tqdm import tqdm
import matplotlib.pyplot as plt

FER2025_PATH = os.path.expanduser("~/Signify/Signify_Model/datasets/FER2025")
IMG_SIZE = 48
IMAGES_PER_CLASS = 2000

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

random.seed(42); np.random.seed(42)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.3
)

def extract_face_crop(image_bgr):
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
    size = max(x_max_raw - x_min_raw, y_max_raw - y_min_raw) * 1.55
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
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

def process_tar(tar_path, class_name, max_images):
    X_list, y_list = [], []
    label = CLASS_TO_ID[class_name]
    processed = 0
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(members)
        pbar = tqdm(total=max_images, desc=class_name)
        for member in members:
            if processed >= max_images:
                break
            f = tar.extractfile(member)
            if f is None: continue
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None: continue
            crop = extract_face_crop(img)
            if crop is None: continue
            X_list.append(crop); y_list.append(label)
            processed += 1; pbar.update(1)
        pbar.close()
    return X_list, y_list

print("=" * 60)
print(f"🧪 TEST EXTRACTION (v3 crop, {IMAGES_PER_CLASS}/class)")
print("=" * 60)

all_X, all_y = [], []
for class_name in CLASSES:
    tar_path = os.path.join(FER2025_PATH, f"{class_name}.tar")
    if not os.path.exists(tar_path):
        print(f"❌ Missing: {tar_path}"); continue
    X_c, y_c = process_tar(tar_path, class_name, IMAGES_PER_CLASS)
    all_X.extend(X_c); all_y.extend(y_c)

face_mesh.close()

X = np.array(all_X, dtype=np.uint8)
y = np.array(all_y, dtype=np.int32)
print(f"\nTotal: {X.shape}, per-class: {np.bincount(y)}")

np.save("test_crop_v3_X.npy", X)
np.save("test_crop_v3_y.npy", y)

EMO = CLASSES
fig, axes = plt.subplots(3, 7, figsize=(14, 6))
for cls in range(7):
    idx = np.where(y == cls)[0][:3]
    for row, k in enumerate(idx):
        axes[row, cls].imshow(X[k], cmap='gray', vmin=0, vmax=255)
        axes[row, cls].set_title(EMO[cls], fontsize=9)
        axes[row, cls].axis('off')
plt.tight_layout()
plt.savefig("sample_check_v3_test.png", dpi=100)
print("✅ Saved sample_check_v3_test.png")
"""
eval_v11_models.py — evaluate existing v11.8/9/10/11 checkpoints on test set
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras

keras.config.enable_unsafe_deserialization()

DATA_PATH = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")
NUM_LANDMARKS = 936
CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

MODELS = {
    "v11.8": "models/face_fer2025_hybrid_v11_8/emotion_hybrid_v11_8_seed42.h5",
    "v11.9": "models/face_fer2025_hybrid_v11_9/emotion_hybrid_v11_9.h5",
    "v11.10": "models/face_fer2025_hybrid_v11_10/emotion_hybrid_v11_10.h5",
    "v11.11": "models/face_fer2025_hybrid_v11_11/emotion_hybrid_v11_11.h5",
}

print("=" * 60)
print("📥 Loading test data")
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test = GROUP_MAP[np.load(f"{DATA_PATH}/y_test.npy")]
print(f"Test: X={X_test.shape}, L={L_test.shape}, per-class={np.bincount(y_test)}")
print("=" * 60)

results = {}
for name, path in MODELS.items():
    full_path = os.path.expanduser(f"~/Signify/Signify_Model/{path}")
    print(f"\n🔍 {name}: {full_path}")
    if not os.path.exists(full_path):
        print(f"   ❌ Not found — skipping")
        continue
    try:
        model = keras.models.load_model(full_path, compile=False, safe_mode=False)
    except Exception as e:
        print(f"   ❌ Load failed: {e}")
        continue

    # Single pass
    p1 = model.predict([X_test, L_test], batch_size=256, verbose=0)
    pred1 = np.argmax(p1, axis=1)
    acc_single = (pred1 == y_test).mean() * 100

    # Correct TTA (image flip only)
    Xf = X_test[:, :, ::-1, :]
    p2 = model.predict([Xf, L_test], batch_size=256, verbose=0)
    avg = (p1 + p2) / 2.0
    pred_tta = np.argmax(avg, axis=1)
    acc_tta = (pred_tta == y_test).mean() * 100

    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        m = y_test == i
        per_class[cname] = (pred_tta[m] == i).mean() * 100

    results[name] = {"single": acc_single, "tta": acc_tta, "per_class": per_class}
    print(f"   Single-pass: {acc_single:.2f}%")
    print(f"   TTA:         {acc_tta:.2f}%")
    for c in CLASS_NAMES:
        print(f"     {c:<10} → {per_class[c]:.2f}%")

# Summary
print("\n" + "=" * 80)
print("📊 FINAL COMPARISON")
print("=" * 80)
print(f"{'Model':<8} | {'Overall':<8} | {'Positive':<9} | {'Negative':<9} | {'Surprise':<9} | {'Neutral':<8}")
print("-" * 80)
for name, r in results.items():
    pc = r["per_class"]
    print(f"{name:<8} | {r['tta']:>6.2f}% | {pc['Positive']:>7.2f}% | {pc['Negative']:>7.2f}% | {pc['Surprise']:>7.2f}% | {pc['Neutral']:>6.2f}%")
print("=" * 80)
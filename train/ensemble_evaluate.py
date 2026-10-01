"""Average predictions from all trained seeds → final accuracy."""
import os, glob
import numpy as np
from sklearn.metrics import classification_report

DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025")
MODELS_PATH = "models/face_fer2025"
EMOTION_CLASSES = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

y_test = np.load(f"{DATA_PATH}/y_test.npy")

prob_files = sorted(glob.glob(f"{MODELS_PATH}/test_probs_v2_seed*.npy"))
print(f"Found {len(prob_files)} models:")
for f in prob_files: print(f"  {os.path.basename(f)}")

if len(prob_files) == 0:
    print("❌ No models found. Train first.")
    exit()

all_probs = np.stack([np.load(f) for f in prob_files], axis=0)
avg_probs = all_probs.mean(axis=0)

y_pred = np.argmax(avg_probs, axis=1)
acc = (y_pred == y_test).mean()

print(f"\n{'='*60}")
print(f"🏆 ENSEMBLE ACCURACY ({len(prob_files)} models): {acc*100:.2f}%")
print(f"{'='*60}\n")
print(classification_report(y_test, y_pred, target_names=EMOTION_CLASSES, digits=4))

np.save(f"{MODELS_PATH}/ensemble_probs.npy", avg_probs)
print(f"✅ Saved ensemble_probs.npy")
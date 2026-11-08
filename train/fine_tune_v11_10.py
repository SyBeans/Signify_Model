"""
fine_tune_v11_10.py
Goal: Push ALL classes (Positive, Negative, Surprise, Neutral) above 80%.
Uses the balanced dataset (15k per class) but adjusts weights to boost Negative & Neutral.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import random

# ============================================
# CONFIG
# ============================================
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")
MODELS_PATH = "models/face_fer2025_hybrid_v11_10"
BASE_MODEL  = "models/face_fer2025_hybrid_v11_8/emotion_hybrid_v11_8_seed42.h5" # Start from the best
IMG_SIZE    = 48
NUM_LANDMARKS = 936
BATCH_SIZE  = 256
EPOCHS      = 20        
BASE_LR     = 2e-4      # Slightly higher LR to give it room to move
SEED        = 42

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD DATA
# ============================================
print("=" * 60); print("📥 LOADING & BALANCING DATA"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val   = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]

# Balance to 15,000 per class
TARGET_SAMPLES = 15000 
new_X, new_L, new_y = [], [], []
for cid in range(NUM_CLASSES):
    idx = np.where(y_train == cid)[0]
    np.random.shuffle(idx)
    if len(idx) > TARGET_SAMPLES:
        idx = idx[:TARGET_SAMPLES]
    new_X.append(X_train[idx])
    new_L.append(L_train[idx])
    new_y.append(y_train[idx])

X_train_bal = np.concatenate(new_X)
L_train_bal = np.concatenate(new_L)
y_train_bal = np.concatenate(new_y)

p = np.random.permutation(len(X_train_bal))
X_train_bal = X_train_bal[p]
L_train_bal = L_train_bal[p]
y_train_bal = y_train_bal[p]

y_train_cat = keras.utils.to_categorical(y_train_bal, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)

def build_ds(X, L, y, bs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L), y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(X_train_bal, L_train_bal, y_train_cat, BATCH_SIZE, True)
val_ds   = build_ds(X_val,       L_val,       y_val_cat,     BATCH_SIZE, False)

# ============================================
# LOAD v11.8 MODEL & FINE-TUNE
# ============================================
print("\n" + "=" * 60); print("🏗️  LOADING v11.8 MODEL"); print("=" * 60)
model = keras.models.load_model(BASE_MODEL, compile=False)
model._name = "emotion_hybrid_v11_10"

# NEW WEIGHTS: 
# Positive (0): 0.85 (Already at 81.4%, can afford to drop)
# Negative (1): 1.15 (Needs a boost from 76% to 80%)
# Surprise (2): 0.95 (Already at 81.7%, keep stable)
# Neutral  (3): 1.15 (Needs a boost from 78.5% to 80%)
class_weight_dict = {0: 0.85, 1: 1.15, 2: 0.95, 3: 1.15}
print(f"Using class weights: {class_weight_dict}")

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=BASE_LR, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

# ============================================
# TRAIN
# ============================================
ckpt_path = os.path.join(MODELS_PATH, "best_v11_10.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              mode='max', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                            patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                patience=2, min_lr=1e-6, verbose=1)
]

print("\n" + "=" * 60); print("🚀 FINE-TUNING v11.10 (Optimized Weights)"); print("=" * 60)
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict, verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING v11.10"); print("=" * 60)

X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test  = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")
y_test = GROUP_MAP[y_test_orig]

def predict_tta(model, X, L, batch=128):
    p1 = model.predict([X, L], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L], batch_size=batch, verbose=0)
    return (p1 + p2) / 2.0

y_prob = predict_tta(model, X_test, L_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test).mean()

print(f"\n✅ Test Accuracy (TTA): {acc*100:.2f}%")
print("\n📊 Per-Class Recognition Rate:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = (y_pred[mask] == i).mean()
    verdict = "✅" if class_acc >= 0.80 else ("⚠️" if class_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {class_acc*100:.2f}%")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'emotion_hybrid_v11_10.h5'))
print(f"\n✅ Saved: {MODELS_PATH}/emotion_hybrid_v11_10.h5")
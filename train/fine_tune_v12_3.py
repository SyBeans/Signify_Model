"""
fine_tune_v12_3.py
v12.3 = v12.2 (same architecture) fine-tuned with boosted Negative weight.

Strategy:
  - Start from v12.2's best checkpoint (loads, doesn't modify)
  - Low LR (5e-5) — preserve v12.2's features
  - 12 epochs — short nudge, not full retrain
  - Negative 1.05 → 1.45 (the fix)
  - Neutral  1.10 → 1.0  (undo over-weight, clean baseline)
  - Same architecture, data, TTA

Saves to: models/face_fer2025_hybrid_v12_3/   ← NEW FOLDER
Does NOT touch: models/face_fer2025_hybrid_v12_2/   ← SAFE

Expected outcome:
  Negative:   77.33% → ~79-81%
  Positive:   88.12% → ~87-88%
  Surprise:   88.85% → ~87-88%
  Neutral:    87.08% → ~84-86%
  Overall:    85.34% → ~84.8-85.2%
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS']      = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

keras.config.enable_unsafe_deserialization()

# ============================================
# CONFIG
# ============================================
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v13_balanced")
BASE_MODEL  = "models/face_fer2025_hybrid_v12_2/emotion_v12_2_best.keras"   # read-only
MODELS_PATH = "models/face_fer2025_hybrid_v12_3"                            # NEW folder — v12.2 untouched

IMG_SIZE = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2
ENGINEERED_DIM = 26

BATCH_SIZE = 256
EPOCHS     = 12          # short nudge
BASE_LR    = 5e-5        # very low — preserve v12.2's features
SEED       = 42

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
GROUP_MAP   = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# SAFETY: confirm v12.2 exists and is untouched
# ============================================
print("=" * 60); print("🛡️  SAFETY CHECK"); print("=" * 60)
print(f"  Source (read-only):  {BASE_MODEL}")
print(f"  Target (new folder): {MODELS_PATH}/")

if not os.path.exists(BASE_MODEL):
    raise FileNotFoundError(f"❌ v12.2 base not found: {BASE_MODEL}")

if os.path.abspath(BASE_MODEL).startswith(os.path.abspath(MODELS_PATH)):
    raise SystemExit("❌ DANGER: BASE_MODEL is inside MODELS_PATH — would overwrite! Aborting.")

print("  ✅ v12.2 will NOT be modified (different folder)")
print("=" * 60)

# ============================================
# LOAD DATA
# ============================================
print("\n📥 LOADING v13 data")

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
E_train = np.load(f"{DATA_PATH}/E_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
E_val = np.load(f"{DATA_PATH}/E_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
E_test = np.load(f"{DATA_PATH}/E_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]
y_test  = GROUP_MAP[y_test_orig]

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"  Train 7-class: {np.bincount(y_train_orig, minlength=7)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)

# ============================================
# WEIGHTS — boosted Negative, clean Neutral
# ============================================
class_weight_dict = {0: 1.0, 1: 1.45, 2: 1.05, 3: 1.0}
print(f"\n📊 Class weights: {class_weight_dict}")
print(f"   Positive 1.0  (unchanged)")
print(f"   Negative 1.45 (+0.40 — main boost, was 1.05)")
print(f"   Surprise 1.05 (unchanged — preserve 88.85%)")
print(f"   Neutral  1.0  (−0.10 — undo over-weight, was 1.10)")

# ============================================
# DATASET (no augmentation — fine-tune on real data)
# ============================================
def build_ds(X, L, E, y, bs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L, E), y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = build_ds(X_train, L_train, E_train, y_train_cat, BATCH_SIZE, shuffle=True)
val_ds   = build_ds(X_val,   L_val,   E_val,   y_val_cat,   BATCH_SIZE, shuffle=False)

# ============================================
# BUILD v12.2 ARCHITECTURE + LOAD WEIGHTS
# ============================================
def se_block(x, ratio=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // ratio, 4), activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])

def residual_conv_block(x, filters, stride=1, drop=0.0):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding='same', use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    return x

def residual_mlp_block(x, units, drop=0.3):
    h = layers.Dense(units, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    h = layers.BatchNormalization()(h); h = layers.Activation('relu')(h); h = layers.Dropout(drop)(h)
    h = layers.Dense(units, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(h)
    h = layers.BatchNormalization()(h)
    if x.shape[-1] != units:
        x = layers.Dense(units, use_bias=False)(x); x = layers.BatchNormalization()(x)
    x = layers.Add()([x, h])
    x = layers.Activation('relu')(x)
    return x

def build_v12_2():
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    xi = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
    xi = residual_conv_block(xi, 32,  drop=0.10)
    xi = residual_conv_block(xi, 64,  stride=2, drop=0.15)
    xi = residual_conv_block(xi, 128, stride=2, drop=0.20)
    xi = residual_conv_block(xi, 256, stride=2, drop=0.25)
    xi = layers.GlobalAveragePooling2D()(xi)
    xi = layers.Dense(128, use_bias=False)(xi)
    xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
    xi = layers.Dropout(0.4)(xi)

    lm_input  = layers.Input(shape=(RAW_LANDMARK_DIM,), name='landmarks')
    eng_input = layers.Input(shape=(ENGINEERED_DIM,),   name='engineered')

    lm = layers.Reshape((NUM_LANDMARKS, 2), name='lm_reshape')(lm_input)
    center = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        output_shape=(1, 2), name='lm_center')(lm)
    centered = layers.Subtract(name='lm_center_sub')([lm, center])
    max_d = layers.Lambda(
        lambda t: tf.reduce_max(tf.norm(t, axis=-1), axis=1, keepdims=True)[..., None] + 1e-6,
        output_shape=(1, 1), name='lm_maxd')(centered)
    normed = layers.Lambda(
        lambda i: i[0] / i[1],
        output_shape=(NUM_LANDMARKS, 2), name='lm_norm')([centered, max_d])
    flat = layers.Reshape((RAW_LANDMARK_DIM,), name='lm_flat')(normed)

    lm_combined = layers.Concatenate(name='lm_plus_eng')([flat, eng_input])

    xl = residual_mlp_block(lm_combined, 512, drop=0.30)
    xl = residual_mlp_block(xl,           256, drop=0.30)
    xl = residual_mlp_block(xl,           128, drop=0.25)

    combined = layers.Concatenate()([xi, xl])
    combined = layers.Dense(128, use_bias=False)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

    return models.Model(inputs=[img_input, lm_input, eng_input],
                        outputs=out, name='emotion_v12_3')

print("\n" + "=" * 60); print("🔨 Building v12.2 architecture + loading v12.2 weights"); print("=" * 60)
model = build_v12_2()
print(f"📥 Loading weights from: {BASE_MODEL}")
model.load_weights(BASE_MODEL)
print("✅ Weights loaded — starting point is v12.2 (85.34% TTA)")

# ============================================
# COMPILE — low LR, no schedule (fine-tune)
# ============================================
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=BASE_LR, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'],
)

# ============================================
# CALLBACKS — save to v12_3 folder only
# ============================================
ckpt_best = f"{MODELS_PATH}/emotion_v12_3_best.keras"
ckpt_last = f"{MODELS_PATH}/emotion_v12_3_last.keras"

callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_best, monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1),
    callbacks.ModelCheckpoint(ckpt_last, save_best_only=False, verbose=0),
    callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                            patience=4, restore_best_weights=True, verbose=1),
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60); print(f"🚀 TRAINING v12.3 (LR={BASE_LR}, {EPOCHS} epochs)"); print("=" * 60)
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1,
)

# ============================================
# EVAL
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING v12.3"); print("=" * 60)

def predict_tta(X, L, E, batch=128):
    p1 = model.predict([X, L, E], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L, E], batch_size=batch, verbose=0)
    return (p1 + p2) / 2.0

# Single pass
p1 = model.predict([X_test, L_test, E_test], batch_size=256, verbose=0)
pred_single = np.argmax(p1, axis=1)
acc_single = (pred_single == y_test).mean() * 100

# TTA
y_prob = predict_tta(X_test, L_test, E_test)
y_pred = np.argmax(y_prob, axis=1)
acc_tta = (y_pred == y_test).mean() * 100

print(f"\n✅ Single-pass: {acc_single:.2f}%")
print(f"✅ TTA:         {acc_tta:.2f}%")

print("\n📋 Classification Report (TTA):")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("📊 Per-Class (TTA):")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    cls_acc = (y_pred[mask] == i).mean() * 100
    verdict = "✅" if cls_acc >= 0.80 else ("⚠️" if cls_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {cls_acc:.2f}%")

# Sub-emotion breakdown for Negative
NEG_SUBS = {0: "Angry", 1: "Disgust", 2: "Fear", 5: "Sad"}
print("\n📊 Negative sub-emotion breakdown:")
for sub_id, sub_name in NEG_SUBS.items():
    mask = (y_test_orig == sub_id)
    n = int(mask.sum())
    if n == 0: continue
    correct = int((y_pred[mask] == 1).sum())
    acc = correct / n * 100
    verdict = "✅" if acc >= 0.80 else ("⚠️" if acc >= 0.65 else "❌")
    print(f"  {verdict} {sub_name:<10} → {acc:.2f}%  ({correct}/{n})")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'v12.3  |  TTA acc={acc_tta:.2f}%')
plt.tight_layout()
plt.savefig(f"{MODELS_PATH}/cm_v12_3.png", dpi=150)
plt.close()

# ============================================
# SAVE
# ============================================
model.save(f"{MODELS_PATH}/emotion_v12_3.keras")
np.save(f"{MODELS_PATH}/test_probs_v12_3.npy", y_prob)
print(f"\n✅ Saved: {MODELS_PATH}/emotion_v12_3.keras")
print(f"✅ Single-pass: {acc_single:.2f}%")
print(f"✅ TTA:         {acc_tta:.2f}%")

# ============================================
# COMPARISON
# ============================================
neg_acc = (y_pred[y_test==1]==1).mean()*100
print("\n" + "=" * 60)
print("📊 COMPARISON")
print("=" * 60)
print(f"  {'Model':<8} | {'Overall TTA':<12} | {'Negative':<10} | {'Deploy?'}")
print("-" * 60)
print(f"  {'v12.2':<8} | {'85.34%':<12} | {'77.33%':<10} | {'if overall matters'}")
print(f"  {'v12.3':<8} | {f'{acc_tta:.2f}%':<12} | {f'{neg_acc:.2f}%':<10} | {'if Negative ≥80%'}")
print("=" * 60)
print(f"\n📁 Both models preserved:")
print(f"   v12.2: {BASE_MODEL}")
print(f"   v12.3: {MODELS_PATH}/emotion_v12_3.keras")
print(f"\n📌 Next: pick the better model, then run export to TFLite.")
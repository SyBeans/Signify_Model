"""
diagnose_v12_tta.py — v3 (rebuild + load_weights)

Why v2 failed:
  Keras 3 cannot reload Lambda layers that lack `output_shape`.
  safe_mode=False fixes safety, but not shape inference.

Fix:
  Rebuild the exact v12-fast architecture from code (same as training).
  Then load_weights() from the .keras file — weights-only, no config parsing.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

MODEL_PATH = "models/face_fer2025_hybrid_v12/emotion_hybrid_v12_fast_best.keras"
DATA_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v12_balanced")

IMG_SIZE = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2
ENGINEERED_DIM = 26
NUM_CLASSES = 4
CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

# ============================================================
# ARCHITECTURE — exact copy of train_hybrid_v12.py
# (WITH output_shape added to Lambdas so reload works in future)
# ============================================================
def se_block(x, ratio=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // ratio, 4), activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])

def residual_mlp_block(x, units, drop=0.3):
    h = layers.Dense(units, use_bias=False)(x)
    h = layers.BatchNormalization()(h); h = layers.Activation('relu')(h); h = layers.Dropout(drop)(h)
    h = layers.Dense(units, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    if x.shape[-1] != units:
        x = layers.Dense(units, use_bias=False)(x); x = layers.BatchNormalization()(x)
    x = layers.Add()([x, h])
    x = layers.Activation('relu')(x)
    return x

def build_v12_fast():
    # IMAGE BRANCH (simple 4-conv, matches v12 fast)
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='image')
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x); x = se_block(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x); x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x); x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x); x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x); x = layers.Dropout(0.4)(x)

    # LANDMARK BRANCH
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

    l = residual_mlp_block(lm_combined, 512, drop=0.30)
    l = residual_mlp_block(l,           256, drop=0.30)
    l = residual_mlp_block(l,           128, drop=0.30)

    # FUSION
    z = layers.Concatenate()([x, l])
    z = layers.Dense(128, use_bias=False)(z)
    z = layers.BatchNormalization()(z); z = layers.Activation('relu')(z); z = layers.Dropout(0.30)(z)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(z)

    return models.Model(inputs=[img_input, lm_input, eng_input], outputs=out, name='v12_fast')

# ============================================================
# LOAD
# ============================================================
print("=" * 60)
print(f"📥 Building model architecture (v12-fast copy)...")
model = build_v12_fast()
print(f"📥 Loading weights from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ {MODEL_PATH} not found")

try:
    model.load_weights(MODEL_PATH)
    print("✅ Weights loaded successfully")
except Exception as e:
    print(f"❌ load_weights failed: {e}")
    print(f"\nTrying alternate: load as legacy .h5...")
    try:
        keras.config.enable_unsafe_deserialization()
        model_alt = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        model.set_weights(model_alt.get_weights())
        print("✅ Weights loaded via alternate path")
    except Exception as e2:
        print(f"❌ Alternate also failed: {e2}")
        raise

print(f"   Inputs:  {[(i.name, i.shape) for i in model.inputs]}")
print(f"   Outputs: {[(o.name, o.shape) for o in model.outputs]}")
print("=" * 60)

# ============================================================
# LOAD TEST DATA
# ============================================================
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
E_test = np.load(f"{DATA_PATH}/E_test.npy").astype(np.float32)
y_test = GROUP_MAP[np.load(f"{DATA_PATH}/y_test.npy")]

print(f"Test data: X={X_test.shape}, L={L_test.shape}, E={E_test.shape}, y={y_test.shape}")
print(f"Test per-class: {np.bincount(y_test, minlength=4)}")

def per_class_acc(preds, y, header=""):
    if header:
        print(f"    {header}")
    acc = (preds == y).mean() * 100
    print(f"    Overall: {acc:.2f}%")
    for i, n in enumerate(CLASS_NAMES):
        m = y == i
        if m.sum() > 0:
            ca = (preds[m] == i).mean() * 100
            print(f"    {n:10s} → {ca:.2f}%  ({m.sum()} samples)")
    return acc

# ============================================================
# 1. SINGLE PASS
# ============================================================
print("\n" + "=" * 60)
print("1️⃣  SINGLE PASS (no TTA) — what the app will do")
print("=" * 60)
p1 = model.predict([X_test, L_test, E_test], batch_size=256, verbose=0)
pred1 = np.argmax(p1, axis=1)
acc1 = per_class_acc(pred1, y_test)

# ============================================================
# 2. BROKEN TTA (flips both)
# ============================================================
print("\n" + "=" * 60)
print("2️⃣  BROKEN TTA (flips image + landmarks) — gave 50.65%")
print("=" * 60)
Xf = X_test[:, :, ::-1, :]
Lf = L_test.reshape(-1, NUM_LANDMARKS, 2).copy()
Lf[:, :, 0] = 1.0 - Lf[:, :, 0]
Lf = Lf.reshape(-1, NUM_LANDMARKS * 2)
p2_broken = model.predict([Xf, Lf, E_test], batch_size=256, verbose=0)
avg_broken = (p1 + p2_broken) / 2.0
pred_broken = np.argmax(avg_broken, axis=1)
acc_broken = per_class_acc(pred_broken, y_test)

# ============================================================
# 3. CORRECT TTA (flips image only)
# ============================================================
print("\n" + "=" * 60)
print("3️⃣  CORRECT TTA (flips image only, landmarks unchanged)")
print("=" * 60)
p2_correct = model.predict([Xf, L_test, E_test], batch_size=256, verbose=0)
avg_correct = (p1 + p2_correct) / 2.0
pred_correct = np.argmax(avg_correct, axis=1)
acc_correct = per_class_acc(pred_correct, y_test)

# ============================================================
# 4. INDIVIDUAL FLIPPED PASSES (to prove landmark flip is garbage)
# ============================================================
print("\n" + "=" * 60)
print("4️⃣a  FLIPPED PASS ONLY (image flip + landmark flip, no avg)")
print("=" * 60)
pred_p2_broken = np.argmax(p2_broken, axis=1)
acc_p2_broken = per_class_acc(pred_p2_broken, y_test)

print("\n" + "=" * 60)
print("4️⃣b  FLIPPED PASS ONLY (image flip, landmarks UNCHANGED, no avg)")
print("=" * 60)
pred_p2_correct = np.argmax(p2_correct, axis=1)
acc_p2_correct = per_class_acc(pred_p2_correct, y_test)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("📊 SUMMARY — v12 fast")
print("=" * 60)
print(f"  1. Single-pass (app will use):        {acc1:.2f}%")
print(f"  2. Broken TTA (flip both):            {acc_broken:.2f}%   ← log printed ~50.65%")
print(f"  3. Correct TTA (flip image only):     {acc_correct:.2f}%   ← v11.8-style fix")
print(f"  4a. Flipped pass alone (broken lm):   {acc_p2_broken:.2f}%")
print(f"  4b. Flipped pass alone (correct lm):  {acc_p2_correct:.2f}%")
print()
print(f"  Val accuracy (epoch 95, from log):    79.18%")
print()
print(f"  → Single-pass ≈ 79%? Model is FINE, TTA was the bug ✅")
print(f"  → Single-pass ≈ 50%? Model genuinely broken ❌")
print()
print(f"  Corrected v12 fast accuracy: {max(acc1, acc_correct):.2f}%")
print("=" * 60)
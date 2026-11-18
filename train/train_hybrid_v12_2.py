"""
train_hybrid_v12_2.py
v12.2 = v12.1 architecture (v11.8 image + v12 landmarks)
      + v13 BALANCED data (40k per 4-class, 10k each sub-emotion, 160k total)
      + mild class weights (data is fully balanced at both levels)

Key change vs v12.1:
  DATA_PATH → fer2025_v13_balanced (160k samples, doubly balanced)
  CLASS WEIGHTS → {1.0, 1.05, 1.05, 1.10}

Why milder weights:
  v12.1 used {1.0, 1.05, 1.05, 1.25} to fight Neutral-vs-Negative clash caused
  by v12's internal imbalance (Angry/Sad 8k vs Disgust/Fear 2k).
  v13 balances all 7 sub-emotions (10k each) → no more clash → mild weights
  are enough.

Expected: ~82-84% overall, Negative ~78-82%, Neutral ~81-83%
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

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

keras.config.enable_unsafe_deserialization()

# ============================================
# CONFIG
# ============================================
DATA_PATH     = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v13_balanced")
MODELS_PATH   = "models/face_fer2025_hybrid_v12_2"
IMG_SIZE      = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2
ENGINEERED_DIM   = 26

BATCH_SIZE = 256
EPOCHS     = 80
BASE_LR    = 5e-4
SEED       = 42

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
GROUP_MAP   = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD
# ============================================
print("=" * 60); print(f"📥 LOADING v12.2 (v13 balanced, seed={SEED})"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
E_train = np.load(f"{DATA_PATH}/E_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val   = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
E_val   = np.load(f"{DATA_PATH}/E_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test  = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
E_test  = np.load(f"{DATA_PATH}/E_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]
y_test  = GROUP_MAP[y_test_orig]

print(f"Train: X={X_train.shape}, L={L_train.shape}, E={E_train.shape}")
print(f"Val:   X={X_val.shape},   L={L_val.shape},   E={E_val.shape}")
print(f"Test:  X={X_test.shape},  L={L_test.shape},  E={E_test.shape}")
print(f"Train per-class (7): {np.bincount(y_train_orig, minlength=7)}")
print(f"Train per-class (4): {np.bincount(y_train, minlength=4)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# ============================================
# CLASS WEIGHTS — mild (v13 data is fully balanced)
# ============================================
class_weight_dict = {0: 1.0, 1: 1.05, 2: 1.05, 3: 1.10}
print(f"\nClass weights: {class_weight_dict}")
print(f"  Mild across the board — v13 data is balanced at both levels")
print(f"  Negative 1.05 — recovered by Disgust/Fear getting 6.25× more data")
print(f"  Neutral  1.10 — small nudge")

# ============================================
# AUGMENTATION (v11.8 exact)
# ============================================
def augment_pair(image, landmarks_936, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    image = tf.clip_by_value(image, 0.0, 1.0)

    coords = tf.reshape(landmarks_936, [NUM_LANDMARKS, 2])
    noise  = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.005)
    coords = coords + noise

    if tf.random.uniform([]) < 0.5:
        angle = tf.random.uniform([], -0.10, 0.10)
        cos_a, sin_a = tf.cos(angle), tf.sin(angle)
        cx = tf.reduce_mean(coords[:, 0]); cy = tf.reduce_mean(coords[:, 1])
        x_c = coords[:, 0] - cx; y_c = coords[:, 1] - cy
        coords = tf.stack([cos_a*x_c - sin_a*y_c + cx,
                           sin_a*x_c + cos_a*y_c + cy], axis=1)

    landmarks_936 = tf.reshape(coords, [RAW_LANDMARK_DIM])
    return (image, landmarks_936), label

def build_ds(X, L, E, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L, E), y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        def _aug(inputs, label):
            (img, lm_936, eng) = inputs
            (img, lm_936), label = augment_pair(img, lm_936, label)
            return (img, lm_936, eng), label
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = build_ds(X_train, L_train, E_train, y_train_cat, BATCH_SIZE, True,  True)
val_ds   = build_ds(X_val,   L_val,   E_val,   y_val_cat,   BATCH_SIZE, False, False)

# ============================================
# BLOCKS
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
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
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

print("\n" + "=" * 60); print("🏗️  BUILDING v12.2"); print("=" * 60)

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

model = models.Model(
    inputs=[img_input, lm_input, eng_input],
    outputs=out,
    name=f'emotion_v12_2_seed{SEED}',
)
model.summary()

# ============================================
# LR (v11.8 exact)
# ============================================
steps_per_epoch = len(X_train) // BATCH_SIZE
total_steps     = EPOCHS * steps_per_epoch
warmup_steps    = 2 * steps_per_epoch

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup, total, min_lr=1e-5):
        super().__init__()
        self.base_lr = float(base_lr); self.warmup = float(warmup)
        self.total = float(total);     self.min_lr = float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup, tf.float32)
        total  = tf.cast(self.total,  tf.float32)
        warmup_lr = self.base_lr * (step / warmup)
        progress  = tf.clip_by_value((step - warmup) / (total - warmup), 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                    (1 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, warmup_lr, cosine_lr)
    def get_config(self):
        return {"base_lr": self.base_lr, "warmup": self.warmup,
                "total": self.total, "min_lr": self.min_lr}

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=WarmupCosine(BASE_LR, warmup_steps, total_steps),
        weight_decay=1e-4,
    ),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'],
)

# ============================================
# CALLBACKS + RESUME
# ============================================
ckpt_best  = f"{MODELS_PATH}/emotion_v12_2_best.keras"
ckpt_last  = f"{MODELS_PATH}/emotion_v12_2_last.keras"
STATE_FILE = f"{MODELS_PATH}/training_state.json"

start_epoch = 0
resume_from_last = False
if os.path.exists(STATE_FILE) and os.path.exists(ckpt_last):
    try:
        with open(STATE_FILE) as f:
            _state = json.load(f)
        start_epoch = int(_state.get("last_epoch", -1)) + 1
        resume_from_last = True
        print(f"🔄 RESUMING from epoch {start_epoch} (best: {_state.get('best_val_acc', 0):.4f})")
    except Exception as e:
        print(f"⚠️  Resume state unreadable ({e}) — starting fresh")

class StateTracker(keras.callbacks.Callback):
    def __init__(self, path, starting_best=0.0):
        super().__init__(); self.path = path; self.best = starting_best
    def on_epoch_end(self, epoch, logs=None):
        va = float(logs.get("val_accuracy", 0))
        if va > self.best: self.best = va
        with open(self.path, "w") as f:
            json.dump({"last_epoch": int(epoch), "best_val_acc": self.best}, f)

_initial_best = 0.0
if resume_from_last:
    try:
        with open(STATE_FILE) as f:
            _initial_best = float(json.load(f).get("best_val_acc", 0.0))
    except Exception:
        pass

callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_best, monitor='val_accuracy',
                              save_best_only=True, mode='max', verbose=1),
    callbacks.ModelCheckpoint(ckpt_last, save_best_only=False, verbose=0),
    StateTracker(STATE_FILE, starting_best=_initial_best),
    callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                            patience=15, restore_best_weights=True, verbose=1),
]

print("\n" + "=" * 60); print(f"🚀 TRAINING v12.2 (seed={SEED}, EPOCHS={EPOCHS})"); print("=" * 60)

if resume_from_last and start_epoch < EPOCHS:
    model = keras.models.load_model(ckpt_last, safe_mode=False)
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, initial_epoch=start_epoch,
        class_weight=class_weight_dict, callbacks=callbacks_list, verbose=1,
    )
else:
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict, callbacks=callbacks_list, verbose=1,
    )

# ============================================
# EVAL
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING v12.2"); print("=" * 60)

def predict_tta(X, L, E, batch=128):
    p1 = model.predict([X, L, E], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L, E], batch_size=batch, verbose=0)
    return (p1 + p2) / 2.0

print("\n--- SINGLE PASS ---")
y_prob_single = model.predict([X_test, L_test, E_test], batch_size=256, verbose=0)
y_pred_single = np.argmax(y_prob_single, axis=1)
acc_single = (y_pred_single == y_test).mean()
print(f"Single-pass: {acc_single*100:.2f}%")

print("\n--- TTA ---")
y_prob = predict_tta(X_test, L_test, E_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test).mean()
print(f"TTA: {acc*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("📊 Per-Class:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = (y_pred[mask] == i).mean()
    verdict = "✅" if class_acc >= 0.80 else ("⚠️" if class_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {class_acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'v12.2 TTA acc={acc*100:.2f}%')
plt.tight_layout()
plt.savefig(f"{MODELS_PATH}/cm_v12_2.png", dpi=150)
plt.close()

model.save(f"{MODELS_PATH}/emotion_v12_2.keras")
np.save(f"{MODELS_PATH}/test_probs_v12_2.npy", y_prob)
print(f"\n✅ Saved: {MODELS_PATH}/emotion_v12_2.keras")
print(f"✅ Single-pass: {acc_single*100:.2f}%  (app uses this)")
print(f"✅ TTA: {acc*100:.2f}%")
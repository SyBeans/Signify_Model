"""
train_hybrid_v12.py  —  v12 FAST (v11.8-skeleton + engineered features)
+ AUTO-RESUME support (survives crashes, reboots, power loss)

Changes vs original v12:
  1. 26 engineered features are PRECOMPUTED in NumPy (E_*.npy) → model gets 3rd input
  2. Landmark branch = 5-op normalize → concat with E → 3 residual MLP blocks
  3. Image branch = v11.8 ResNet+SE (fast, proven)
  4. tf.data uses AUTOTUNE everywhere
  5. Auto-resume: saves last.keras + training_state.json every epoch

4 classes: Positive, Negative, Surprise, Neutral
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

# ============================================
# CONFIG
# ============================================
DATA_PATH     = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v12_balanced")
MODELS_PATH   = "models/face_fer2025_hybrid_v12"
IMG_SIZE      = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2          # 936
ENGINEERED_DIM   = 26

BATCH_SIZE = 256
EPOCHS     = 100
BASE_LR    = 5e-4
SEED       = 42

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
GROUP_MAP   = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)   # 7→4

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD DATA
# ============================================
print("=" * 60); print(f"📥 LOADING hybrid v12 FAST (seed={SEED})"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val   = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test  = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")

# ── Precomputed engineered features (26) ──
E_train = np.load(f"{DATA_PATH}/E_train.npy").astype(np.float32)
E_val   = np.load(f"{DATA_PATH}/E_val.npy").astype(np.float32)
E_test  = np.load(f"{DATA_PATH}/E_test.npy").astype(np.float32)

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]
y_test  = GROUP_MAP[y_test_orig]

print(f"Train: X={X_train.shape}, L={L_train.shape}, E={E_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape},   L={L_val.shape},   E={E_val.shape},   y={y_val.shape}")
print(f"Test:  X={X_test.shape},  L={L_test.shape},  E={E_test.shape},  y={y_test.shape}")
print(f"\n7-class (raw):   {np.bincount(y_train_orig, minlength=7)}")
print(f"4-class (train): {np.bincount(y_train, minlength=4)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# ============================================
# AUGMENTATION (only image + landmarks)
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
        x_c = coords[:, 0] - cx;            y_c = coords[:, 1] - cy
        coords = tf.stack([cos_a*x_c - sin_a*y_c + cx,
                           sin_a*x_c + cos_a*y_c + cy], axis=1)

    s = tf.random.uniform([], 0.95, 1.05)
    coords = coords * s
    tx = tf.random.uniform([], -0.01, 0.01)
    ty = tf.random.uniform([], -0.01, 0.01)
    coords = coords + tf.stack([tx, ty])

    landmarks_936 = tf.reshape(coords, [RAW_LANDMARK_DIM])
    return (image, landmarks_936), label

# ============================================
# DATASET  (3-tuple: X, L, E)
# ============================================
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
# MODEL BUILDERS
# ============================================
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

print("\n" + "=" * 60); print("🏗️  BUILDING HYBRID v12 FAST"); print("=" * 60)

# ---------- IMAGE BRANCH (v11.8 ResNet+SE) ----------
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

# ---------- LANDMARK BRANCH (5-op normalize → concat E → 3 residual MLPs) ----------
lm_input  = layers.Input(shape=(RAW_LANDMARK_DIM,), name='landmarks')
eng_input = layers.Input(shape=(ENGINEERED_DIM,),    name='engineered')

lm       = layers.Reshape((NUM_LANDMARKS, 2), name='lm_reshape')(lm_input)
center   = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True), name='lm_center')(lm)
centered = layers.Subtract(name='lm_center_sub')([lm, center])
max_d    = layers.Lambda(
    lambda t: tf.reduce_max(tf.norm(t, axis=-1), axis=1, keepdims=True)[..., None] + 1e-6,
    name='lm_maxd')(centered)
normed = layers.Lambda(lambda i: i[0] / i[1], name='lm_norm')([centered, max_d])
flat   = layers.Reshape((RAW_LANDMARK_DIM,), name='lm_flat')(normed)

lm_combined = layers.Concatenate(name='lm_plus_eng')([flat, eng_input])     # 962-dim

l = residual_mlp_block(lm_combined, 512, drop=0.30)
l = residual_mlp_block(l,           256, drop=0.30)
l = residual_mlp_block(l,           128, drop=0.30)

# ---------- FUSION ----------
z = layers.Concatenate()([x, l])
z = layers.Dense(128, use_bias=False)(z)
z = layers.BatchNormalization()(z); z = layers.Activation('relu')(z); z = layers.Dropout(0.30)(z)
out = layers.Dense(NUM_CLASSES, activation='softmax')(z)

model = models.Model(
    inputs=[img_input, lm_input, eng_input],
    outputs=out,
    name='emotion_hybrid_v12_fast',
)
model.summary()

# ============================================
# COMPILE
# ============================================
class_weight_dict = {0: 1.0, 1: 1.05, 2: 1.25, 3: 1.2}
print(f"\nClass weights: {class_weight_dict}")

steps_per_epoch = len(X_train) // BATCH_SIZE
warmup_epochs   = 5

def lr_schedule(epoch, lr):
    if epoch < warmup_epochs:
        return BASE_LR * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
    return BASE_LR * 0.5 * (1.0 + np.cos(np.pi * progress))

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=BASE_LR, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'],
)

# ============================================
# CALLBACKS + RESUME STATE
# ============================================
ckpt_path  = f"{MODELS_PATH}/emotion_hybrid_v12_fast_best.keras"
last_ckpt  = f"{MODELS_PATH}/emotion_hybrid_v12_fast_last.keras"
STATE_FILE = f"{MODELS_PATH}/training_state.json"

# --- Resume detection ---
start_epoch = 0
resume_from_last = False
if os.path.exists(STATE_FILE) and os.path.exists(last_ckpt):
    try:
        with open(STATE_FILE) as f:
            _state = json.load(f)
        start_epoch = int(_state.get("last_epoch", -1)) + 1
        resume_from_last = True
        print(f"🔄 RESUMING from epoch {start_epoch} "
              f"(best val_acc so far: {_state.get('best_val_acc', 0):.4f})")
    except Exception as e:
        print(f"⚠️  Could not read resume state ({e}) — starting fresh")

class StateTracker(keras.callbacks.Callback):
    def __init__(self, path, starting_best=0.0):
        super().__init__()
        self.path = path
        self.best = starting_best
    def on_epoch_end(self, epoch, logs=None):
        va = float(logs.get("val_accuracy", 0))
        if va > self.best:
            self.best = va
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
    callbacks.LearningRateScheduler(lr_schedule, verbose=1),
    callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
    ),
    # Save EVERY epoch (for resume). Not best-only.
    callbacks.ModelCheckpoint(
        last_ckpt,
        save_best_only=False,
        verbose=0,
    ),
    StateTracker(STATE_FILE, starting_best=_initial_best),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max',
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ============================================
# TRAIN  (auto-resume if last.keras exists)
# ============================================
print("\n" + "=" * 60); print(f"🚀 TRAINING v12 FAST (seed={SEED})"); print("=" * 60)

if resume_from_last and start_epoch < EPOCHS:
    print(f"🔁 Loading weights from {last_ckpt}")
    model = keras.models.load_model(last_ckpt)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1,
    )
else:
    if resume_from_last:
        print(f"ℹ️  training_state.json says epoch {start_epoch} ≥ EPOCHS={EPOCHS} — "
              f"starting fresh")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1,
    )

# ============================================
# TTA EVALUATION on TEST
# ============================================
print("\n" + "=" * 60); print("🧪 TTA EVALUATION (test set)"); print("=" * 60)

def predict_tta(X, L, E, batch=256):
    preds = model.predict([X, L, E], batch_size=batch, verbose=0)
    Xf = X[:, :, ::-1, :]
    Lf = L.reshape(-1, NUM_LANDMARKS, 2).copy()
    Lf[:, :, 0] = 1.0 - Lf[:, :, 0]
    Lf = Lf.reshape(-1, RAW_LANDMARK_DIM)
    Ef = E   # TTA on E is approximate — matches v11.8 behavior
    preds = (preds + model.predict([Xf, Lf, Ef], batch_size=batch, verbose=0)) / 2.0
    return preds

test_preds = predict_tta(X_test, L_test, E_test)
test_pred_labels = np.argmax(test_preds, axis=1)

test_acc = (test_pred_labels == y_test).mean()
print(f"\nTest Accuracy (TTA): {test_acc*100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, test_pred_labels, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_test, test_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'v12-fast TTA  |  test acc = {test_acc*100:.2f}%')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f"{MODELS_PATH}/confusion_matrix_v12_fast.png", dpi=120)
plt.close()

# ============================================
# SAVE FINAL
# ============================================
final_path = f"{MODELS_PATH}/emotion_hybrid_v12_fast.keras"
model.save(final_path)
print(f"\n✅ Saved best:  {ckpt_path}")
print(f"✅ Saved last:  {last_ckpt}")
print(f"✅ Saved final: {final_path}")

print("\nPer-class accuracy:")
for c, name in enumerate(CLASS_NAMES):
    mask = (y_test == c)
    if mask.sum() > 0:
        acc = (test_pred_labels[mask] == c).mean() * 100
        print(f"  {name:10s} → {acc:.2f}%  ({mask.sum()} samples)")
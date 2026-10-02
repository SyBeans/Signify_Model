"""
train_emotion_4class_v6.py
4-class training — stable version.
- No mixup (was causing oscillation)
- Batch 128 (more stable BN)
- LR 1e-3 (was 2e-3)
- Warmup 3 epochs (was 1)
- Label smoothing 0.05 (was 0.1)
- Simple augmentation (flip + brightness only)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, regularizers
import random

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# ============================================
# CONFIG
# ============================================
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v6")
MODELS_PATH = "models/face_fer2025_4class_v6"
IMG_SIZE    = 48
BATCH_SIZE  = 128        # ↑ more stable BN
EPOCHS      = 50
BASE_LR     = 1e-3       # ↓ reduced
SEED        = int(os.environ.get("SEED", 42))

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4

GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD
# ============================================
print("=" * 60); print(f"📥 LOADING v6 (seed={SEED})"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")
X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
y_val_orig   = np.load(f"{DATA_PATH}/y_val.npy")
X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
y_test_orig  = np.load(f"{DATA_PATH}/y_test.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]
y_test  = GROUP_MAP[y_test_orig]

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Train per-class (4): {np.bincount(y_train, minlength=4)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# ============================================
# CLASS WEIGHTS
# ============================================
print("\n" + "=" * 60); print("⚖️  CLASS WEIGHTS"); print("=" * 60)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: float(w) for i, w in enumerate(cw)}
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:10s} → {class_weight_dict[i]:.4f}")

# ============================================
# AUGMENTATION — SIMPLE
# ============================================
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    return tf.clip_by_value(image, 0.0, 1.0), label

AUTOTUNE = tf.data.AUTOTUNE

def build_ds(X, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=2)
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(X_train, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,   y_val_cat,   BATCH_SIZE, False, False)
test_ds  = build_ds(X_test,  y_test_cat,  BATCH_SIZE, False, False)

# ============================================
# BLOCKS
# ============================================
def se_block(x, ratio=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])

def residual_block(x, filters, stride=1, drop=0.0):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding='same', use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
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

# ============================================
# BUILD
# ============================================
print("\n" + "=" * 60); print("🏗️  BUILDING MINI-RESNET (4-class v6)"); print("=" * 60)

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = residual_block(x, 32, drop=0.10)
x = residual_block(x, 64, stride=2, drop=0.15)
x = residual_block(x, 128, stride=2, drop=0.20)
x = residual_block(x, 256, stride=2, drop=0.25)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.4)(x)   # ↓ from 0.5
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs, name=f"emotion_4class_v6_seed{SEED}")

# ============================================
# LR SCHEDULE
# ============================================
steps_per_epoch = len(train_ds)
total_steps  = EPOCHS * steps_per_epoch
warmup_steps = 3 * steps_per_epoch   # ↑ 3 epochs

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup, total, min_lr=1e-5):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup  = float(warmup)
        self.total   = float(total)
        self.min_lr  = float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup, tf.float32)
        total = tf.cast(self.total, tf.float32)
        warmup_lr = self.base_lr * (step / warmup)
        progress = (step - warmup) / (total - warmup)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                    (1 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, warmup_lr, cosine_lr)
    def get_config(self):
        return {"base_lr": self.base_lr, "warmup": self.warmup,
                "total": self.total, "min_lr": self.min_lr}

lr_schedule = WarmupCosine(BASE_LR, warmup_steps, total_steps)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.summary()

# ============================================
# CALLBACKS
# ============================================
ckpt_path = os.path.join(MODELS_PATH, f"best_v6_seed{SEED}.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=12,
                            restore_best_weights=True, verbose=1),
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60); print(f"🚀 TRAINING v6 (seed={SEED})"); print("=" * 60)

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING (with flip TTA)"); print("=" * 60)

def predict_with_tta(model, X, batch=128):
    preds = model.predict(X, batch_size=batch, verbose=0)
    preds_flip = model.predict(X[:, :, ::-1, :], batch_size=batch, verbose=0)
    return (preds + preds_flip) / 2.0

y_prob = predict_with_tta(model, X_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test).mean()
print(f"\n✅ Test Accuracy (TTA): {acc*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("\n📊 Per-Class Recognition Rate:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = (y_pred[mask] == i).mean()
    verdict = "✅" if class_acc >= 0.80 else ("⚠️" if class_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {class_acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'4-Class v6 (TTA acc={acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_v6_seed{SEED}.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, f'emotion_4class_v6_seed{SEED}.h5'))
np.save(os.path.join(MODELS_PATH, f'test_probs_v6_seed{SEED}.npy'), y_prob)
np.save(os.path.join(MODELS_PATH, f'val_probs_v6_seed{SEED}.npy'),
        predict_with_tta(model, X_val))

print(f"\n✅ Saved: {MODELS_PATH}/emotion_4class_v6_seed{SEED}.h5")
print(f"✅ Test acc: {acc*100:.2f}%")
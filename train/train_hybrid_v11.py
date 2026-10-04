"""
train_hybrid_v11.py
Two-branch model:
  - Branch 1: ResNet CNN on 48x48 pixel crops
  - Branch 2: MLP on 936 landmark coords
  - Fusion: concatenate + dense
Expected: ~82% test, ~75% webcam
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
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
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")
MODELS_PATH = "models/face_fer2025_hybrid_v11"
IMG_SIZE    = 48
NUM_LANDMARKS = 936
BATCH_SIZE  = 192
EPOCHS      = 50
BASE_LR     = 1e-3
SEED        = int(os.environ.get("SEED", 42))

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4

GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD
# ============================================
print("=" * 60); print(f"📥 LOADING hybrid v11 (seed={SEED})"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val   = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test  = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]
y_test  = GROUP_MAP[y_test_orig]

print(f"Train: X={X_train.shape}, L={L_train.shape}")
print(f"Val:   X={X_val.shape}, L={L_val.shape}")
print(f"Test:  X={X_test.shape}, L={L_test.shape}")
print(f"Train per-class (4): {np.bincount(y_train, minlength=4)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

class_weight_dict = {0: 1.0, 1: 0.4, 2: 1.0, 3: 1.0}

# ============================================
# AUGMENTATION (image + landmark separately)
# ============================================
def augment_pair(image, landmarks, label):
    # Image aug
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Landmark aug (NO flip — breaks indices)
    coords = tf.reshape(landmarks, [468, 2])
    noise = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.005)
    coords = coords + noise
    if tf.random.uniform([]) < 0.5:
        angle = tf.random.uniform([], -0.10, 0.10)
        cos_a = tf.cos(angle); sin_a = tf.sin(angle)
        cx = tf.reduce_mean(coords[:, 0]); cy = tf.reduce_mean(coords[:, 1])
        x_c = coords[:, 0] - cx; y_c = coords[:, 1] - cy
        x_r = cos_a * x_c - sin_a * y_c + cx
        y_r = sin_a * x_c + cos_a * y_c + cy
        coords = tf.stack([x_r, y_r], axis=1)
    landmarks = tf.reshape(coords, [936])

    return (image, landmarks), label

AUTOTUNE = tf.data.AUTOTUNE

def build_ds(X, L, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L), y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        ds = ds.map(augment_pair, num_parallel_calls=2)
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(X_train, L_train, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,   L_val,   y_val_cat,   BATCH_SIZE, False, False)
test_ds  = build_ds(X_test,  L_test,  y_test_cat,  BATCH_SIZE, False, False)

# ============================================
# IMAGE BRANCH (ResNet-style)
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
# BUILD HYBRID MODEL
# ============================================
print("\n" + "=" * 60); print("🏗️  BUILDING HYBRID MODEL"); print("=" * 60)

# --- Image branch ---
img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
xi = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
xi = layers.BatchNormalization()(xi)
xi = layers.Activation('relu')(xi)
xi = residual_block(xi, 32, drop=0.10)
xi = residual_block(xi, 64, stride=2, drop=0.15)
xi = residual_block(xi, 128, stride=2, drop=0.20)
xi = residual_block(xi, 256, stride=2, drop=0.25)
xi = layers.GlobalAveragePooling2D()(xi)
xi = layers.Dense(128, use_bias=False)(xi)
xi = layers.BatchNormalization()(xi)
xi = layers.Activation('relu')(xi)
xi = layers.Dropout(0.4)(xi)

# --- Landmark branch ---
lm_input = layers.Input(shape=(NUM_LANDMARKS,), name="landmarks")
xl = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(lm_input)
xl = layers.BatchNormalization()(xl)
xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.3)(xl)
xl = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(xl)
xl = layers.BatchNormalization()(xl)
xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.3)(xl)
xl = layers.Dense(128)(xl)
xl = layers.BatchNormalization()(xl)
xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.25)(xl)

# --- Fusion ---
combined = layers.Concatenate()([xi, xl])
combined = layers.Dense(128, use_bias=False)(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.Activation('relu')(combined)
combined = layers.Dropout(0.4)(combined)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

model = models.Model(inputs=[img_input, lm_input], outputs=outputs,
                     name=f"emotion_hybrid_v11_seed{SEED}")

# ============================================
# LR SCHEDULE
# ============================================
steps_per_epoch = len(train_ds)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = 2 * steps_per_epoch

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
ckpt_path = os.path.join(MODELS_PATH, f"best_hybrid_seed{SEED}.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=12,
                            restore_best_weights=True, verbose=1),
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60); print(f"🚀 TRAINING HYBRID v11 (seed={SEED})"); print("=" * 60)

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict, verbose=1
)

# ============================================
# EVALUATE WITH FLIP TTA
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING (with flip TTA)"); print("=" * 60)

# Note: flip TTA on image only, landmarks unchanged
def predict_hybrid_tta(model, X, L, batch=128):
    p1 = model.predict([X, L], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L], batch_size=batch, verbose=0)
    return (p1 + p2) / 2.0

y_prob = predict_hybrid_tta(model, X_test, L_test)
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
plt.title(f'Hybrid v11 (TTA acc={acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_hybrid_seed{SEED}.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, f'emotion_hybrid_seed{SEED}.h5'))
np.save(os.path.join(MODELS_PATH, f'test_probs_hybrid_seed{SEED}.npy'), y_prob)

print(f"\n✅ Saved: {MODELS_PATH}/emotion_hybrid_seed{SEED}.h5")
print(f"✅ Test acc: {acc*100:.2f}%")
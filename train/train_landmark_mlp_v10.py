"""
train_landmark_mlp_v10.py  (FIXED)
Pure landmark-based emotion classifier.
- Removed horizontal flip (breaks landmark indices)
- Reduced LR to 5e-4
- Increased patience to 20
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, regularizers
import random

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# ============================================
# CONFIG
# ============================================
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v10")
MODELS_PATH = "models/face_fer2025_landmark_v10"
BATCH_SIZE  = 256
EPOCHS      = 60
BASE_LR     = 5e-4       # ✅ reduced from 1e-3
SEED        = int(os.environ.get("SEED", 42))

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
NUM_LANDMARKS = 936   # 468 × 2

GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD
# ============================================
print("=" * 60); print(f"📥 LOADING landmarks v10 (seed={SEED})"); print("=" * 60)

L_train = np.load(f"{DATA_PATH}/L_train.npy")
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")
L_val   = np.load(f"{DATA_PATH}/L_val.npy")
y_val_orig   = np.load(f"{DATA_PATH}/y_val.npy")
L_test  = np.load(f"{DATA_PATH}/L_test.npy")
y_test_orig  = np.load(f"{DATA_PATH}/y_test.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]
y_test  = GROUP_MAP[y_test_orig]

print(f"Train: {L_train.shape}  Val: {L_val.shape}  Test: {L_test.shape}")
print(f"Train per-class (4): {np.bincount(y_train, minlength=4)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# Class weights
class_weight_dict = {0: 1.0, 1: 0.4, 2: 1.0, 3: 1.0}

# ============================================
# LANDMARK AUGMENTATION (NO FLIP)
# ============================================
def augment_landmarks(landmarks, label):
    coords = tf.reshape(landmarks, [468, 2])
    
    # 1. Small noise
    noise = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.005)
    coords = coords + noise
    
    # 2. Small rotation (head tilt)
    if tf.random.uniform([]) < 0.5:
        angle = tf.random.uniform([], -0.12, 0.12)
        cos_a = tf.cos(angle); sin_a = tf.sin(angle)
        cx = tf.reduce_mean(coords[:, 0])
        cy = tf.reduce_mean(coords[:, 1])
        x_centered = coords[:, 0] - cx
        y_centered = coords[:, 1] - cy
        x_rot = cos_a * x_centered - sin_a * y_centered + cx
        y_rot = sin_a * x_centered + cos_a * y_centered + cy
        coords = tf.stack([x_rot, y_rot], axis=1)
    
    # 3. Small scale
    if tf.random.uniform([]) < 0.5:
        scale = tf.random.uniform([], 0.95, 1.05)
        cx = tf.reduce_mean(coords[:, 0])
        cy = tf.reduce_mean(coords[:, 1])
        coords = (coords - [cx, cy]) * scale + [cx, cy]
    
    # ❌ REMOVED: Horizontal flip (breaks landmark index semantics)
    
    return tf.reshape(coords, [936]), label

AUTOTUNE = tf.data.AUTOTUNE

def build_ds(L, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((L, y))
    if shuffle:
        ds = ds.shuffle(min(len(L), 10000), seed=SEED)
    if augment_data:
        ds = ds.map(augment_landmarks, num_parallel_calls=2)
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(L_train, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(L_val,   y_val_cat,   BATCH_SIZE, False, False)
test_ds  = build_ds(L_test,  y_test_cat,  BATCH_SIZE, False, False)

# ============================================
# BUILD MLP
# ============================================
print("\n" + "=" * 60); print("🏗️  BUILDING LANDMARK MLP"); print("=" * 60)

inputs = layers.Input(shape=(NUM_LANDMARKS,), name="landmarks")

x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.25)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs, name=f"emotion_landmark_mlp_seed{SEED}")

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
        self.warmup = float(warmup)
        self.total = float(total)
        self.min_lr = float(min_lr)
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
ckpt_path = os.path.join(MODELS_PATH, f"best_landmark_seed{SEED}.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=20,  # ✅ increased
                            restore_best_weights=True, verbose=1),
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60); print(f"🚀 TRAINING LANDMARK MLP (seed={SEED})"); print("=" * 60)

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict, verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING"); print("=" * 60)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

y_prob = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
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
plt.title(f'Landmark MLP (acc={test_acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_landmark_seed{SEED}.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, f'emotion_landmark_seed{SEED}.h5'))
np.save(os.path.join(MODELS_PATH, f'test_probs_seed{SEED}.npy'), y_prob)

print(f"\n✅ Saved: {MODELS_PATH}/emotion_landmark_seed{SEED}.h5")
print(f"✅ Test acc: {test_acc*100:.2f}%")
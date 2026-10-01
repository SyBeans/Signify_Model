"""
train_emotion_resnet.py  (CPU-OPTIMIZED)
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
# CONFIG (CPU-OPTIMIZED)
# ============================================
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025")
MODELS_PATH = "models/face_fer2025"
IMG_SIZE    = 48
BATCH_SIZE  = 64
EPOCHS      = 60
BASE_LR     = 2e-3
SEED        = int(os.environ.get("SEED", 42))

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = 7

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD
# ============================================
print("=" * 60); print(f"📥 LOADING (seed={SEED})"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
y_train = np.load(f"{DATA_PATH}/y_train.npy")
X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
y_val   = np.load(f"{DATA_PATH}/y_val.npy")
X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
y_test  = np.load(f"{DATA_PATH}/y_test.npy")

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Train per-class: {np.bincount(y_train)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# ============================================
# AUGMENTATION (CPU-friendly)
# ============================================
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.10)
    return tf.clip_by_value(image, 0.0, 1.0), label

AUTOTUNE = tf.data.AUTOTUNE

def build_ds(X, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 5000), seed=SEED)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=2)
    ds = ds.batch(bs).prefetch(2)
    return ds

train_ds = build_ds(X_train, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,   y_val_cat,   BATCH_SIZE, False, False)
test_ds  = build_ds(X_test,  y_test_cat,  BATCH_SIZE, False, False)

# ============================================
# SE BLOCK + RESIDUAL BLOCK
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
# BUILD MODEL (smaller)
# ============================================
print("\n" + "=" * 60); print("🏗️  BUILDING MINI-RESNET (CPU)"); print("=" * 60)

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
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs, name=f"emotion_resnet_seed{SEED}")

# ============================================
# COSINE LR WITH WARMUP
# ============================================
steps_per_epoch = len(train_ds)
total_steps  = EPOCHS * steps_per_epoch
warmup_steps = 1 * steps_per_epoch

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup, total, min_lr=1e-5):
        super().__init__()
        self.base_lr = base_lr; self.warmup = warmup
        self.total = total; self.min_lr = min_lr
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

lr_schedule = WarmupCosine(BASE_LR, warmup_steps, total_steps)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
model.summary()

# ============================================
# CALLBACKS
# ============================================
ckpt_path = os.path.join(MODELS_PATH, f"best_resnet_seed{SEED}.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                            restore_best_weights=True, verbose=1),
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60); print(f"🚀 TRAINING (seed={SEED})"); print("=" * 60)

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list, verbose=1
)

# ============================================
# EVALUATE WITH TTA
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
print(classification_report(y_test, y_pred, target_names=EMOTION_CLASSES, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
plt.title(f'Confusion Matrix (seed={SEED}, TTA acc={acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_seed{SEED}.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, f'emotion_resnet_seed{SEED}.h5'))
np.save(os.path.join(MODELS_PATH, f'test_probs_seed{SEED}.npy'), y_prob)
np.save(os.path.join(MODELS_PATH, f'val_probs_seed{SEED}.npy'),
        predict_with_tta(model, X_val))

print(f"\n✅ Saved seed {SEED} → {MODELS_PATH}/emotion_resnet_seed{SEED}.h5")
print(f"✅ Test acc: {acc*100:.2f}%")
"""
train_hybrid_v11_7.py
v11 core + class weights tuned to raise Negative AND keep Neutral/Surprise >=80.
Changes vs v11:
  class_weight_dict: {1.0, 0.4, 1.0, 1.0} -> {1.0, 1.3, 1.05, 1.05}
  Eval TTA corrected: image flipped, landmarks NOT flipped (matches v11 training).
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

DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")
MODELS_PATH = "models/face_fer2025_hybrid_v11_7"          # <-- changed to _v11_7
IMG_SIZE    = 48
NUM_LANDMARKS = 936
BATCH_SIZE  = 256
EPOCHS      = 65
BASE_LR     = 5e-4
SEED        = int(os.environ.get("SEED", 42))

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ---- LOAD ----
print("=" * 60); print(f"📥 LOADING hybrid v11.7 (seed={SEED})"); print("=" * 60)   # <-- v11.7

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

print(f"Train per-class (4): {np.bincount(y_train, minlength=4)}")
assert X_train.shape[1] == IMG_SIZE

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# ---- CLASS WEIGHTS ----
class_weight_dict = {0: 1.0, 1: 1.3, 2: 1.05, 3: 1.05}
print(f"Class weights: {class_weight_dict}")

# ---- AUGMENTATION (v11-style: image flips, landmarks don't) ----
def augment_pair(image, landmarks, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    image = tf.clip_by_value(image, 0.0, 1.0)
    coords = tf.reshape(landmarks, [468, 2])
    noise = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.005)
    coords = coords + noise
    if tf.random.uniform([]) < 0.5:
        angle = tf.random.uniform([], -0.10, 0.10)
        cos_a = tf.cos(angle); sin_a = tf.sin(angle)
        cx = tf.reduce_mean(coords[:, 0]); cy = tf.reduce_mean(coords[:, 1])
        x_c = coords[:, 0] - cx; y_c = coords[:, 1] - cy
        coords = tf.stack([cos_a*x_c - sin_a*y_c + cx,
                           sin_a*x_c + cos_a*y_c + cy], axis=1)
    return (image, tf.reshape(coords, [936])), label

def build_ds(X, L, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L), y))
    if shuffle: ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        ds = ds.map(lambda inputs, label: augment_pair(inputs[0], inputs[1], label),
                    num_parallel_calls=2)
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(X_train, L_train, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,   L_val,   y_val_cat,   BATCH_SIZE, False, False)

# ---- MODEL (identical to v11) ----
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
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    if drop > 0: x = layers.Dropout(drop)(x)
    return x

print("\n🏗️  BUILDING v11.7")                                       # <-- v11.7
img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
xi = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
xi = residual_block(xi, 32, drop=0.10)
xi = residual_block(xi, 64, stride=2, drop=0.15)
xi = residual_block(xi, 128, stride=2, drop=0.20)
xi = residual_block(xi, 256, stride=2, drop=0.25)
xi = layers.GlobalAveragePooling2D()(xi)
xi = layers.Dense(128, use_bias=False)(xi)
xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
xi = layers.Dropout(0.4)(xi)

lm_input = layers.Input(shape=(NUM_LANDMARKS,), name="landmarks")
xl = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(lm_input)
xl = layers.BatchNormalization()(xl); xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.3)(xl)
xl = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(xl)
xl = layers.BatchNormalization()(xl); xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.3)(xl)
xl = layers.Dense(128)(xl)
xl = layers.BatchNormalization()(xl); xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.25)(xl)

combined = layers.Concatenate()([xi, xl])
combined = layers.Dense(128, use_bias=False)(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.Activation('relu')(combined)
combined = layers.Dropout(0.4)(combined)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

model = models.Model(inputs=[img_input, lm_input], outputs=outputs,
                     name=f"emotion_hybrid_v11_7_seed{SEED}")   # <-- v11_7

# ---- LR SCHEDULE ----
steps_per_epoch = len(train_ds)
total_steps  = EPOCHS * steps_per_epoch
warmup_steps = 2 * steps_per_epoch

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup, total, min_lr=1e-5):
        super().__init__()
        self.base_lr = float(base_lr); self.warmup = float(warmup)
        self.total = float(total); self.min_lr = float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup, tf.float32)
        total = tf.cast(self.total, tf.float32)
        warmup_lr = self.base_lr * (step / warmup)
        progress = tf.clip_by_value((step - warmup) / (total - warmup), 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, warmup_lr, cosine_lr)
    def get_config(self):
        return {"base_lr": self.base_lr, "warmup": self.warmup,
                "total": self.total, "min_lr": self.min_lr}

lr_schedule = WarmupCosine(BASE_LR, warmup_steps, total_steps)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'])
model.summary()

# ---- TRAIN ----
ckpt_path = os.path.join(MODELS_PATH, f"best_hybrid_v11_7_seed{SEED}.h5")   # <-- v11_7
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', mode='max',
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                            patience=15, restore_best_weights=True, verbose=1),
]
print(f"\n🚀 TRAINING v11.7 (seed={SEED})")                          # <-- v11.7
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=callbacks_list, class_weight=class_weight_dict,
                    verbose=1)

# ---- EVALUATE (TTA matching v11 training) ----
print("\n📊 EVALUATING v11.7 (TTA matches training)")               # <-- v11.7
def predict_tta(model, X, L, batch=128):
    p1 = model.predict([X, L], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L], batch_size=batch, verbose=0)   # landmarks NOT flipped
    return (p1 + p2) / 2.0

y_prob = predict_tta(model, X_test, L_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test).mean()
print(f"\n✅ Test Accuracy (TTA): {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("📊 Per-Class:")
for i, name in enumerate(CLASS_NAMES):
    ca = (y_pred[y_test == i] == i).mean()
    v = "✅" if ca >= 0.80 else ("⚠️" if ca >= 0.65 else "❌")
    print(f"  {v} {name:<10} → {ca*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'v11.7 (TTA acc={acc*100:.2f}%)')                        # <-- v11.7
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_v11_7_seed{SEED}.png'), dpi=150)   # <-- v11_7

model.save(os.path.join(MODELS_PATH, f'emotion_hybrid_v11_7_seed{SEED}.h5'))  # <-- v11_7
np.save(os.path.join(MODELS_PATH, f'test_probs_v11_7_seed{SEED}.npy'), y_prob)  # <-- v11_7
print(f"\n✅ Saved: {MODELS_PATH}/emotion_hybrid_v11_7_seed{SEED}.h5")        # <-- v11_7
print(f"✅ Test acc: {acc*100:.2f}%")
"""
train_hybrid_v11_5.py
v11 + aux 7-class head. Reuses v11 48px landmarks — no re-extraction.
Only change vs v11: 7-class aux output forces trunk to separate
Angry/Disgust/Fear/Sad inside the "Negative" group.
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
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")   # reuse v11
MODELS_PATH = "models/face_fer2025_hybrid_v11_5"                                    # new dir
IMG_SIZE    = 48
NUM_LANDMARKS = 936
BATCH_SIZE  = 256          # CPU: larger = fewer steps = faster
EPOCHS      = 50           # early stopping will cut sooner
BASE_LR     = 5e-4
SEED        = int(os.environ.get("SEED", 42))

CLASS_NAMES_4 = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES_4 = 4
NUM_CLASSES_7 = 7

GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)   # 7 -> 4

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD
# ============================================
print("=" * 60); print(f"📥 LOADING hybrid v11.5 (seed={SEED})"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val   = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test  = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")

y_train4 = GROUP_MAP[y_train_orig]
y_val4   = GROUP_MAP[y_val_orig]
y_test4  = GROUP_MAP[y_test_orig]

print(f"Train: X={X_train.shape}, L={L_train.shape}")
print(f"Val:   X={X_val.shape}, L={L_val.shape}")
print(f"Test:  X={X_test.shape}, L={L_test.shape}")
print(f"Train per-class (4): {np.bincount(y_train4, minlength=4)}")
print(f"Train per-class (7): {np.bincount(y_train_orig, minlength=7)}")

assert X_train.shape[1] == IMG_SIZE, \
    f"IMG_SIZE mismatch: npy has {X_train.shape[1]}, config has {IMG_SIZE}"

y_train_c4 = keras.utils.to_categorical(y_train4, NUM_CLASSES_4).astype(np.float32)
y_val_c4   = keras.utils.to_categorical(y_val4,   NUM_CLASSES_4).astype(np.float32)
y_test_c4  = keras.utils.to_categorical(y_test4,  NUM_CLASSES_4).astype(np.float32)

y_train_c7 = keras.utils.to_categorical(y_train_orig, NUM_CLASSES_7).astype(np.float32)
y_val_c7   = keras.utils.to_categorical(y_val_orig,   NUM_CLASSES_7).astype(np.float32)

# ---- sqrt inverse-freq class weights (fixes Negative down-weight bug from v11) ----
counts4 = np.bincount(y_train4, minlength=NUM_CLASSES_4).astype(np.float32)
w4 = np.sqrt(counts4.max() / counts4)
cw4 = {i: float(w4[i]) for i in range(NUM_CLASSES_4)}
print(f"4-class counts: {counts4}")
print(f"4-class weights: {cw4}")

# ============================================
# AUGMENTATION  (consistent flip on image AND landmarks)
# ============================================
def augment_pair(image, landmarks, label4, label7):
    do_flip = tf.random.uniform([]) < 0.5

    image = tf.cond(do_flip,
                    lambda: tf.image.flip_left_right(image),
                    lambda: image)

    coords = tf.reshape(landmarks, [468, 2])
    coords = tf.cond(
        do_flip,
        lambda: tf.stack([1.0 - coords[:, 0], coords[:, 1]], axis=1),
        lambda: coords,
    )

    # Scale jitter (webcam robustness)
    scale = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.round(IMG_SIZE * scale), tf.int32)
    image = tf.image.resize(image, [new_size, new_size])
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)

    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Landmark noise + small rotation
    noise = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.003)
    coords = coords + noise
    if tf.random.uniform([]) < 0.5:
        angle = tf.random.uniform([], -0.10, 0.10)
        cos_a, sin_a = tf.cos(angle), tf.sin(angle)
        cx = tf.reduce_mean(coords[:, 0]); cy = tf.reduce_mean(coords[:, 1])
        x_c, y_c = coords[:, 0] - cx, coords[:, 1] - cy
        coords = tf.stack(
            [cos_a*x_c - sin_a*y_c + cx, sin_a*x_c + cos_a*y_c + cy], axis=1
        )

    return (image, tf.reshape(coords, [NUM_LANDMARKS])), (label4, label7)

def build_ds(X, L, y4, y7=None, bs=BATCH_SIZE, augment_data=False, shuffle=False):
    if y7 is None:
        ds = tf.data.Dataset.from_tensor_slices(((X, L), y4))
    else:
        ds = tf.data.Dataset.from_tensor_slices(((X, L), (y4, y7)))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        ds = ds.map(
            lambda inputs, labels: augment_pair(inputs[0], inputs[1],
                                                labels[0], labels[1]),
            num_parallel_calls=2,
        )
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(X_train, L_train, y_train_c4, y_train_c7, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,   L_val,   y_val_c4,   y_val_c7,   BATCH_SIZE, False, False)

# ============================================
# MODEL (identical trunk to v11; adds aux head)
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

print("\n" + "=" * 60); print("🏗️  BUILDING HYBRID v11.5 (dual head)"); print("=" * 60)

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

out4 = layers.Dense(NUM_CLASSES_4, activation='softmax', name='cls4')(combined)
aux = layers.Dense(64, activation='relu')(combined)
out7 = layers.Dense(NUM_CLASSES_7, activation='softmax', name='cls7')(aux)

model = models.Model(inputs=[img_input, lm_input],
                     outputs=[out4, out7],
                     name=f"emotion_hybrid_v11_5_seed{SEED}")

# ============================================
# LR SCHEDULE
# ============================================
steps_per_epoch = len(train_ds)
total_steps  = EPOCHS * steps_per_epoch
warmup_steps = 3 * steps_per_epoch

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup, total, min_lr=1e-5):
        super().__init__()
        self.base_lr = float(base_lr); self.warmup = float(warmup)
        self.total = float(total);     self.min_lr = float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup, tf.float32)
        total = tf.cast(self.total, tf.float32)
        warmup_lr = self.base_lr * (step / warmup)
        progress = tf.clip_by_value((step - warmup) / (total - warmup), 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                    (1 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, warmup_lr, cosine_lr)
    def get_config(self):
        return {"base_lr": self.base_lr, "warmup": self.warmup,
                "total": self.total, "min_lr": self.min_lr}

lr_schedule = WarmupCosine(BASE_LR, warmup_steps, total_steps)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss={
        'cls4': keras.losses.CategoricalFocalCrossentropy(
                    gamma=2.0, alpha=0.25, label_smoothing=0.05),
        'cls7': keras.losses.CategoricalFocalCrossentropy(
                    gamma=2.0, alpha=0.25, label_smoothing=0.05),
    },
    loss_weights={'cls4': 1.0, 'cls7': 0.4},
    metrics={'cls4': 'accuracy', 'cls7': 'accuracy'},
)
model.summary()

# ============================================
# TRAIN
# ============================================
ckpt_path = os.path.join(MODELS_PATH, f"best_hybrid_v11_5_seed{SEED}.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_cls4_accuracy',
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_cls4_accuracy', patience=12,
                            restore_best_weights=True, verbose=1),
]

class_weight_dict = {'cls4': cw4}

print("\n" + "=" * 60); print(f"🚀 TRAINING v11.5 (seed={SEED})"); print("=" * 60)
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict, verbose=1
)

# ============================================
# EVALUATE (consistent flip TTA)
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING v11.5 (consistent flip TTA)"); print("=" * 60)

def predict_tta_4class(model, X, L, batch=128):
    p1_full = model.predict([X, L], batch_size=batch, verbose=0)
    p1 = p1_full[0] if isinstance(p1_full, list) else p1_full

    X_flip = X[:, :, ::-1, :]
    Lr = L.reshape(-1, 468, 2).copy()
    Lr[:, :, 0] = 1.0 - Lr[:, :, 0]
    L_flip = Lr.reshape(-1, NUM_LANDMARKS)

    p2_full = model.predict([X_flip, L_flip], batch_size=batch, verbose=0)
    p2 = p2_full[0] if isinstance(p2_full, list) else p2_full
    return (p1 + p2) / 2.0

y_prob = predict_tta_4class(model, X_test, L_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test4).mean()
print(f"\n✅ Test Accuracy (TTA): {acc*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test4, y_pred, target_names=CLASS_NAMES_4, digits=4))

print("\n📊 Per-Class Recognition Rate:")
for i, name in enumerate(CLASS_NAMES_4):
    mask = y_test4 == i
    class_acc = (y_pred[mask] == i).mean()
    verdict = "✅" if class_acc >= 0.80 else ("⚠️" if class_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {class_acc*100:.2f}%")

cm = confusion_matrix(y_test4, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES_4, yticklabels=CLASS_NAMES_4)
plt.title(f'Hybrid v11.5 (TTA acc={acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_hybrid_v11_5_seed{SEED}.png'), dpi=150)

# ============================================
# SAVE — strip aux head for deployment
# ============================================
print("\n🧹 Stripping 7-class aux head for deployment...")
inference_model = models.Model(
    inputs=model.input,
    outputs=model.get_layer('cls4').output,
    name=f"emotion_hybrid_v11_5_infer_seed{SEED}"
)
inference_model.save(os.path.join(MODELS_PATH, f'emotion_hybrid_v11_5_seed{SEED}.h5'))
np.save(os.path.join(MODELS_PATH, f'test_probs_v11_5_seed{SEED}.npy'), y_prob)

print(f"\n✅ Saved inference model: {MODELS_PATH}/emotion_hybrid_v11_5_seed{SEED}.h5")
print(f"✅ Test acc: {acc*100:.2f}%")
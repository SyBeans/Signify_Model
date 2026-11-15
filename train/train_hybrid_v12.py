"""
train_hybrid_v12.py
Hybrid emotion model v12 — trained on BALANCED 4-class data.

Improvements over v11.8:
1. Balanced 4-class data (v12_balanced, 20k each)
2. Landmark branch: per-sample normalization + 26 engineered geometric features
3. Residual MLP blocks in landmark branch
4. Preprocessing baked INSIDE model → TFLite compatible (app sends 936 raw)
5. TTA evaluation

Negative internal distribution (weighted):
  Angry 8k + Sad 8k + Disgust 2k + Fear 2k = 20k
  (Angry/Sad prioritized as more common in FSL context)

4 classes: Positive, Negative, Surprise, Neutral
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
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v12_balanced")
MODELS_PATH = "models/face_fer2025_hybrid_v12"
IMG_SIZE    = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2  # 936
ENGINEERED_DIM = 26
USE_ENGINEERED = True

BATCH_SIZE  = 256
EPOCHS      = 100
BASE_LR     = 5e-4
SEED        = 42

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
NUM_CLASSES = 4
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)

os.makedirs(MODELS_PATH, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# ============================================
# LOAD DATA
# ============================================
print("=" * 60); print(f"📥 LOADING hybrid v12 (seed={SEED})"); print("=" * 60)

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

print(f"Train: X={X_train.shape}, L={L_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape}, L={L_val.shape}, y={y_val.shape}")
print(f"Test:  X={X_test.shape}, L={L_test.shape}, y={y_test.shape}")
print(f"\n7-class (raw):   {np.bincount(y_train_orig, minlength=7)}")
print(f"4-class (train): {np.bincount(y_train, minlength=4)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  NUM_CLASSES).astype(np.float32)

# ============================================
# PREPROCESSING LAYERS (FIXED: shapes + no duplicate names)
# ============================================
def build_preprocessing_layers():
    def normalize(lm_936):
        lm = layers.Reshape((NUM_LANDMARKS, 2))(lm_936)
        center = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
            name='lm_center'
        )(lm)
        centered = layers.Subtract(name='lm_subtract')([lm, center])
        # ✅ FIX: [..., None] → shape (batch, 1, 1) for proper broadcasting
        max_dist = layers.Lambda(
            lambda x: tf.reduce_max(tf.norm(x, axis=-1), axis=1, keepdims=True)[..., None] + 1e-6,
            name='lm_max_dist'
        )(centered)
        normalized = layers.Lambda(
            lambda inp: inp[0] / inp[1],
            name='lm_divide'
        )([centered, max_dist])
        flat = layers.Reshape((RAW_LANDMARK_DIM,), name='lm_flatten')(normalized)
        return normalized, flat

    def engineered(normalized_468x2):
        # Pre-extract unique landmarks ONCE to prevent duplicate layer names
        unique_indices = [
            159, 145, 33, 133, 386, 374, 362, 263,   # eyes
            13, 14, 61, 291, 105, 334, 107, 336,     # mouth + brows
            10, 152, 234, 454, 1, 172, 397, 93, 323  # face + jaw
        ]

        lm_xy = {}
        lm_y = {}
        for idx in unique_indices:
            lm_xy[idx] = layers.Lambda(
                lambda x, i=idx: x[:, i, :],
                name=f'lmxy_{idx}'
            )(normalized_468x2)
            lm_y[idx] = layers.Lambda(
                lambda x: x[:, 1:2],
                name=f'lmy_{idx}'
            )(lm_xy[idx])

        dist_cache = {}

        def dist(a_idx, b_idx):
            key = (a_idx, b_idx)
            if key in dist_cache:
                return dist_cache[key]
            diff = layers.Subtract(name=f'lmsub_{a_idx}_{b_idx}')(
                [lm_xy[a_idx], lm_xy[b_idx]]
            )
            result = layers.Lambda(
                lambda x: tf.norm(x, axis=-1, keepdims=True) + 1e-6,
                name=f'lmdist_{a_idx}_{b_idx}'
            )(diff)
            dist_cache[key] = result
            return result

        def y_coord(idx):
            return lm_y[idx]

        features = []

        # 1-2. Eye Aspect Ratios
        left_ear_num = dist(159, 145)
        left_ear_den = dist(33, 133)
        left_ear = layers.Lambda(
            lambda x: x[0] / (x[1] + 1e-6), name='left_ear'
        )([left_ear_num, left_ear_den])
        features.append(left_ear)

        right_ear_num = dist(386, 374)
        right_ear_den = dist(362, 263)
        right_ear = layers.Lambda(
            lambda x: x[0] / (x[1] + 1e-6), name='right_ear'
        )([right_ear_num, right_ear_den])
        features.append(right_ear)

        # 3. Mouth Aspect Ratio
        mouth_h = dist(13, 14)
        mouth_w = dist(61, 291)
        mar = layers.Lambda(
            lambda x: x[0] / (x[1] + 1e-6), name='mar'
        )([mouth_h, mouth_w])
        features.append(mar)

        # 4. Mouth curve
        l_cy = y_coord(61)
        r_cy = y_coord(291)
        c_up = y_coord(13)
        c_lo = y_coord(14)
        mouth_center_y = layers.Lambda(
            lambda x: (x[0] + x[1]) / 2.0, name='mouth_center_y'
        )([c_up, c_lo])
        corners_avg = layers.Lambda(
            lambda x: (x[0] + x[1]) / 2.0, name='corners_avg'
        )([l_cy, r_cy])
        mouth_curve = layers.Subtract(name='mouth_curve')(
            [mouth_center_y, corners_avg]
        )
        features.append(mouth_curve)

        # 5-9. Distances
        features.append(dist(61, 133))
        features.append(dist(291, 362))
        features.append(dist(105, 159))
        features.append(dist(334, 386))
        features.append(dist(107, 336))

        # 10-11. Brow heights
        lb_above = layers.Subtract(name='lb_above')(
            [y_coord(159), y_coord(105)]
        )
        features.append(lb_above)
        rb_above = layers.Subtract(name='rb_above')(
            [y_coord(386), y_coord(334)]
        )
        features.append(rb_above)

        # 12. Face aspect
        face_h = dist(10, 152)
        face_w = dist(234, 454)
        face_aspect = layers.Lambda(
            lambda x: x[0] / (x[1] + 1e-6), name='face_aspect'
        )([face_h, face_w])
        features.append(face_aspect)

        # 13-16. More geometry
        features.append(dist(1, 13))
        features.append(dist(152, 14))
        features.append(dist(172, 397))
        features.append(dist(93, 323))

        # 17. Nose vs eyes
        leye_y = y_coord(159)
        reye_y = y_coord(386)
        nose_y = y_coord(1)
        eye_avg_y = layers.Lambda(
            lambda x: (x[0] + x[1]) / 2.0, name='eye_avg_y'
        )([leye_y, reye_y])
        nose_vs_eyes = layers.Subtract(name='nose_vs_eyes')(
            [nose_y, eye_avg_y]
        )
        features.append(nose_vs_eyes)

        # 18-20. Asymmetries
        features.append(layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]), name='ear_diff'
        )([left_ear, right_ear]))
        features.append(layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]), name='corner_diff'
        )([l_cy, r_cy]))
        features.append(layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]), name='brow_asym'
        )([lb_above, rb_above]))

        # 21. Eye width ratio (uses cached dist results)
        left_eye_w = dist(33, 133)
        right_eye_w = dist(362, 263)
        eye_w_ratio = layers.Lambda(
            lambda x: x[0] / (x[1] + 1e-6), name='eye_w_ratio'
        )([left_eye_w, right_eye_w])
        features.append(eye_w_ratio)

        # 22. Mouth below nose
        corners_avg2 = layers.Lambda(
            lambda x: (x[0] + x[1]) / 2.0, name='corners_avg2'
        )([l_cy, r_cy])
        mouth_below_nose = layers.Subtract(name='mouth_below_nose')(
            [corners_avg2, nose_y]
        )
        features.append(mouth_below_nose)

        # 23. Brow to nose
        bn_l = dist(107, 1)
        bn_r = dist(336, 1)
        brow_nose_avg = layers.Lambda(
            lambda x: (x[0] + x[1]) / 2.0, name='brow_nose_avg'
        )([bn_l, bn_r])
        features.append(brow_nose_avg)

        # 24-26. Raw dims (reuse cached distances)
        features.append(mouth_w)
        features.append(mouth_h)
        features.append(left_eye_w)

        return layers.Concatenate(name='engineered_concat')(features)

    def preprocessing(input_936):
        normalized, flat = normalize(input_936)
        if USE_ENGINEERED:
            eng = engineered(normalized)
            return layers.Concatenate(name='full_preprocessed')([flat, eng])
        return flat

    return preprocessing

# ============================================
# DATA AUGMENTATION
# ============================================
def augment_pair(image, landmarks_936, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 4, IMG_SIZE + 4)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    image = tf.clip_by_value(image, 0.0, 1.0)

    coords = tf.reshape(landmarks_936, [NUM_LANDMARKS, 2])
    noise = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.005)
    coords = coords + noise

    if tf.random.uniform([]) < 0.5:
        angle = tf.random.uniform([], -0.10, 0.10)
        cos_a = tf.cos(angle); sin_a = tf.sin(angle)
        cx = tf.reduce_mean(coords[:, 0]); cy = tf.reduce_mean(coords[:, 1])
        x_c = coords[:, 0] - cx; y_c = coords[:, 1] - cy
        coords = tf.stack([cos_a*x_c - sin_a*y_c + cx,
                           sin_a*x_c + cos_a*y_c + cy], axis=1)

    s = tf.random.uniform([], 0.95, 1.05)
    coords = coords * s
    tx = tf.random.uniform([], -0.01, 0.01)
    ty = tf.random.uniform([], -0.01, 0.01)
    coords = coords + tf.stack([tx, ty])

    landmarks_936 = tf.reshape(coords, [RAW_LANDMARK_DIM])
    return (image, landmarks_936), label

def build_ds(X, L, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L), y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        ds = ds.map(
            lambda inputs, label: augment_pair(inputs[0], inputs[1], label),
            num_parallel_calls=2,
        )
    return ds.batch(bs).prefetch(2)

train_ds = build_ds(X_train, L_train, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,   L_val,   y_val_cat,   BATCH_SIZE, False, False)

# ============================================
# MODEL BUILDERS
# ============================================
def se_block(x, ratio=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
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
    h = layers.Dense(units, use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation('relu')(h)
    h = layers.Dropout(drop)(h)
    h = layers.Dense(units, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    if x.shape[-1] != units:
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Add()([x, h])
    x = layers.Activation('relu')(x)
    return x

print("\n" + "=" * 60); print("🏗️  BUILDING HYBRID v12"); print("=" * 60)

# --- IMAGE BRANCH ---
img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
xi = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
xi = residual_conv_block(xi, 32, drop=0.10)
xi = residual_conv_block(xi, 64, stride=2, drop=0.15)
xi = residual_conv_block(xi, 128, stride=2, drop=0.20)
xi = residual_conv_block(xi, 256, stride=2, drop=0.25)
xi = layers.GlobalAveragePooling2D()(xi)
xi = layers.Dense(128, use_bias=False)(xi)
xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
xi = layers.Dropout(0.4)(xi)

# --- LANDMARK BRANCH ---
lm_input = layers.Input(shape=(RAW_LANDMARK_DIM,), name="landmarks")
lm_preprocessed = build_preprocessing_layers()(lm_input)

xl = layers.Dense(512, use_bias=False)(lm_preprocessed)
xl = layers.BatchNormalization()(xl)
xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.3)(xl)
xl = residual_mlp_block(xl, 512, drop=0.3)
xl = residual_mlp_block(xl, 256, drop=0.3)
xl = residual_mlp_block(xl, 128, drop=0.3)
xl = layers.Dense(128, use_bias=False)(xl)
xl = layers.BatchNormalization()(xl); xl = layers.Activation('relu')(xl)
xl = layers.Dropout(0.25)(xl)

# --- FUSION ---
combined = layers.Concatenate()([xi, xl])
combined = layers.Dense(128, use_bias=False)(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.Activation('relu')(combined)
combined = layers.Dropout(0.4)(combined)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

model = models.Model(inputs=[img_input, lm_input], outputs=outputs,
                     name=f"emotion_hybrid_v12_seed{SEED}")
model.summary()

# ============================================
# LR SCHEDULE
# ============================================
steps_per_epoch = len(train_ds)
total_steps  = EPOCHS * steps_per_epoch
warmup_steps = 2 * steps_per_epoch

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
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'])

class_weight_dict = {0: 1.0, 1: 1.05, 2: 1.25, 3: 1.20}
print(f"\nClass weights: {class_weight_dict}")

# ============================================
# TRAIN
# ============================================
ckpt_path = os.path.join(MODELS_PATH, f"best_v12_seed{SEED}.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              mode='max', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                            patience=20, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=6, min_lr=1e-6, verbose=1),
]

print("\n" + "=" * 60); print(f"🚀 TRAINING v12 (seed={SEED})"); print("=" * 60)
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict, verbose=1
)

# ============================================
# EVALUATE (TTA)
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING v12 (TTA)"); print("=" * 60)

def predict_tta(model, X, L, batch=128):
    p1 = model.predict([X, L], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L], batch_size=batch, verbose=0)
    return (p1 + p2) / 2.0

y_prob = predict_tta(model, X_test, L_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test).mean()

print(f"\n✅ Test Accuracy (TTA): {acc*100:.2f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print(f"\n📊 Per-Class Recognition Rate:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = (y_pred[mask] == i).mean()
    verdict = "✅" if class_acc >= 0.80 else ("⚠️" if class_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {class_acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Hybrid v12 Balanced (TTA acc={acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, f'cm_v12_seed{SEED}.png'), dpi=150)

print(f"\n{'='*60}")
print(f"COMPARISON vs v11.8")
print(f"{'='*60}")
print(f"  v11.8 (unbalanced):  80.12%")
print(f"  v12   (balanced):    {acc*100:.2f}%")
print(f"  Δ:                   {acc*100 - 80.12:+.2f}%")

model.save(os.path.join(MODELS_PATH, f'emotion_hybrid_v12_seed{SEED}.h5'))
print(f"\n✅ Saved: {MODELS_PATH}/emotion_hybrid_v12_seed{SEED}.h5")
print(f"\n⚠️  NEXT: Test TFLite export")
"""
fine_tune_v11_12.py
Final Polish: Aggressive Data Augmentation to make the model robust to 
REAL-WORLD faces (angles, zoom, lighting) instead of just dataset actors.
Goal: Generalize to "All People".
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import callbacks, layers
import random

# ============================================
# CONFIG
# ============================================
DATA_PATH   = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")
MODELS_PATH = "models/face_fer2025_hybrid_v11_12"
BASE_MODEL  = "models/face_fer2025_hybrid_v11_8/emotion_hybrid_v11_8_seed42.h5" 
IMG_SIZE    = 48
NUM_LANDMARKS = 936
BATCH_SIZE  = 256
EPOCHS      = 20        
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
print("=" * 60); print("📥 LOADING & BALANCING DATA"); print("=" * 60)

X_train = np.load(f"{DATA_PATH}/X_train.npy").astype(np.float32)
L_train = np.load(f"{DATA_PATH}/L_train.npy").astype(np.float32)
y_train_orig = np.load(f"{DATA_PATH}/y_train.npy")

X_val   = np.load(f"{DATA_PATH}/X_val.npy").astype(np.float32)
L_val   = np.load(f"{DATA_PATH}/L_val.npy").astype(np.float32)
y_val_orig = np.load(f"{DATA_PATH}/y_val.npy")

y_train = GROUP_MAP[y_train_orig]
y_val   = GROUP_MAP[y_val_orig]

# Balance to 15,000 per class
TARGET_SAMPLES = 15000 
new_X, new_L, new_y = [], [], []
for cid in range(NUM_CLASSES):
    idx = np.where(y_train == cid)[0]
    np.random.shuffle(idx)
    if len(idx) > TARGET_SAMPLES:
        idx = idx[:TARGET_SAMPLES]
    new_X.append(X_train[idx])
    new_L.append(L_train[idx])
    new_y.append(y_train[idx])

X_train_bal = np.concatenate(new_X)
L_train_bal = np.concatenate(new_L)
y_train_bal = np.concatenate(new_y)

p = np.random.permutation(len(X_train_bal))
X_train_bal = X_train_bal[p]
L_train_bal = L_train_bal[p]
y_train_bal = y_train_bal[p]

y_train_cat = keras.utils.to_categorical(y_train_bal, NUM_CLASSES).astype(np.float32)
y_val_cat   = keras.utils.to_categorical(y_val,   NUM_CLASSES).astype(np.float32)

# ============================================
# AGGRESSIVE AUGMENTATION (Simulating Real World)
# ============================================
def augment_pair(image, landmarks, label):
    # 1. Random Flip
    if tf.random.uniform([]) < 0.5:
        image = tf.image.flip_left_right(image)
        # Note: We don't flip landmarks because the app doesn't flip them either

    # 2. Random Brightness & Contrast (Lighting variations)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 3. Random Zoom / Crop (Distance variations)
    # Sometimes crop tighter, sometimes pad wider
    if tf.random.uniform([]) < 0.5:
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
        image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    else:
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE - 4, IMG_SIZE - 4)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # 4. Landmark Noise & Rotation (Angle variations)
    coords = tf.reshape(landmarks, [468, 2])
    noise = tf.random.normal(tf.shape(coords), mean=0.0, stddev=0.01) # More noise than before
    coords = coords + noise
    
    # Random Rotation up to 15 degrees
    angle = tf.random.uniform([], -0.26, 0.26) # ~15 degrees in radians
    cos_a = tf.cos(angle); sin_a = tf.sin(angle)
    cx = tf.reduce_mean(coords[:, 0]); cy = tf.reduce_mean(coords[:, 1])
    x_c = coords[:, 0] - cx; y_c = coords[:, 1] - cy
    coords = tf.stack([cos_a*x_c - sin_a*y_c + cx,
                       sin_a*x_c + cos_a*y_c + cy], axis=1)
    landmarks = tf.reshape(coords, [NUM_LANDMARKS])

    return (image, landmarks), label

def build_ds(X, L, y, bs, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(((X, L), y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 8000), seed=SEED)
    if augment_data:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = build_ds(X_train_bal, L_train_bal, y_train_cat, BATCH_SIZE, True, True)
val_ds   = build_ds(X_val,       L_val,       y_val_cat,     BATCH_SIZE, False, False)

# ============================================
# LOAD MODEL & INCREASE DROPOUT (Anti-Memorization)
# ============================================
print("\n" + "=" * 60); print("🏗️  LOADING v11.8 MODEL"); print("=" * 60)
model = keras.models.load_model(BASE_MODEL, compile=False)
model._name = "emotion_hybrid_v11_12"

# We drop the last layer and add a stronger Dropout before it
# This forces the model to learn robust features instead of memorizing exact pixel patterns
x = model.layers[-2].output
x = layers.Dropout(0.5)(x) # Increased from 0.4
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='final_output')(x)
model = keras.models.Model(inputs=model.inputs, outputs=outputs)

# We use uniform weights to let the model balance itself with the augmented data
class_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=BASE_LR, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), # Increased smoothing
    metrics=['accuracy']
)

# ============================================
# TRAIN
# ============================================
ckpt_path = os.path.join(MODELS_PATH, "best_v11_12.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                              mode='max', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                            patience=5, restore_best_weights=True, verbose=1),
]

print("\n" + "=" * 60); print("🚀 FINE-TUNING v11.12 (Aggressive Augmentation)"); print("=" * 60)
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks_list,
    class_weight=class_weight_dict, verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60); print("📊 EVALUATING v11.12"); print("=" * 60)

X_test  = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test  = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")
y_test = GROUP_MAP[y_test_orig]

def predict_tta(model, X, L, batch=128):
    p1 = model.predict([X, L], batch_size=batch, verbose=0)
    X_flip = X[:, :, ::-1, :]
    p2 = model.predict([X_flip, L], batch_size=batch, verbose=0)
    return (p1 + p2) / 2.0

y_prob = predict_tta(model, X_test, L_test)
y_pred = np.argmax(y_prob, axis=1)
acc = (y_pred == y_test).mean()

print(f"\n✅ Test Accuracy (TTA): {acc*100:.2f}%")
print("\n📊 Per-Class Recognition Rate:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = (y_pred[mask] == i).mean()
    verdict = "✅" if class_acc >= 0.80 else ("⚠️" if class_acc >= 0.65 else "❌")
    print(f"  {verdict} {name:<10} → {class_acc*100:.2f}%")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'emotion_hybrid_v11_12.h5'))
print(f"\n✅ Saved: {MODELS_PATH}/emotion_hybrid_v11_12.h5")
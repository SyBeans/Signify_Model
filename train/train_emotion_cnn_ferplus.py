"""
train_emotion_cnn_ferplus.py (v2.1 - 80%+ target)

Fixes applied vs v1:
  - BatchNorm after every Conv2D (biggest accuracy win)
  - 4th conv block + wider Dense (more capacity)
  - Stronger augmentation: zoom + random erasing
  - Cosine LR schedule (smoother convergence)
  - Longer patience (25) so LR decay can do its job

v2.1 fix:
  - augment() now runs BEFORE batch() so random_crop sees (48,48,1) not (128,48,48,1)

Target: 80%+ val accuracy
"""

# ============================================
# FORCE THREADING (before TF import)
# ============================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# Force TF threading
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

AUTOTUNE = tf.data.AUTOTUNE

# ============================================
# CONFIGURATION
# ============================================
FERPLUS_PATH = os.path.expanduser(
    "~/Signify/Signify_Model/datasets/FERPLUS/FER2013Plus"
)
MODELS_PATH = "models/face_hybrid"

IMG_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 120          # more epochs since LR decays smoothly now
LEARNING_RATE = 0.002 # slightly higher start for cosine schedule

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7

LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
EMOTION_TO_ID = {name: i for i, name in enumerate(EMOTION_CLASSES)}


# ============================================
# LOAD DATA
# ============================================
def load_split(split_folder, split_name):
    split_path = os.path.join(FERPLUS_PATH, split_folder)
    labels_path = os.path.join(split_path, "labels.csv")

    if not os.path.exists(labels_path):
        print(f"❌ Not found: {labels_path}")
        return None, None

    labels_df = pd.read_csv(labels_path)
    X, y = [], []
    skipped_contempt = 0

    print(f"\n📥 Loading {split_name}: {len(labels_df)} images")

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                          desc=split_name):
        fname = row["filename"]
        emotion_raw = str(row["emotion"]).lower().strip()

        if emotion_raw not in LABEL_MAP:
            skipped_contempt += 1
            continue

        image_path = os.path.join(split_path, fname)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(EMOTION_TO_ID[LABEL_MAP[emotion_raw]])

    X = np.array(X, dtype=np.float32) / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y, dtype=np.int32)

    print(f"✅ {split_name}: X = {X.shape}, y = {y.shape}")
    print(f"   Skipped (contempt): {skipped_contempt}")

    return X, y


print("=" * 60)
print("📥 LOADING FER+ PIXEL DATA")
print("=" * 60)

X_train, y_train = load_split("FER2013Train", "train")
X_val, y_val = load_split("FER2013Valid", "val")
X_test, y_test = load_split("FER2013Test", "test")


# ============================================
# CLASS WEIGHTS
# ============================================
print("\n" + "=" * 60)
print("⚖️  COMPUTING CLASS WEIGHTS")
print("=" * 60)

raw_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
soft_weights = np.sqrt(raw_weights)
class_weight_dict = dict(enumerate(soft_weights))

for i, w in class_weight_dict.items():
    print(f"  {EMOTION_CLASSES[i]:10s}: {w:.3f}")


# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)


# ============================================
# AUGMENTATION (flip + brightness + contrast + zoom + erasing)
# Runs on a SINGLE image (48, 48, 1) — must be applied BEFORE batch()
# ============================================
def augment(image, label):
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Brightness + contrast
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    # Random zoom (pad then crop back to IMG_SIZE)
    zoom_pad = 6
    image = tf.image.resize_with_crop_or_pad(
        image, IMG_SIZE + zoom_pad, IMG_SIZE + zoom_pad
    )
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])

    # Random erasing (cutout) - 30% chance
    if tf.random.uniform([]) < 0.30:
        eh = tf.random.uniform([], 6, 12, dtype=tf.int32)
        ew = tf.random.uniform([], 6, 12, dtype=tf.int32)
        ey = tf.random.uniform([], 0, IMG_SIZE - eh, dtype=tf.int32)
        ex = tf.random.uniform([], 0, IMG_SIZE - ew, dtype=tf.int32)

        # Build a mask that is 1 inside the erase region
        mask = tf.pad(
            tf.ones([eh, ew, 1]),
            [[ey, IMG_SIZE - eh - ey], [ex, IMG_SIZE - ew - ex], [0, 0]]
        )
        image = image * (1.0 - mask)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_dataset(X, y_cat, batch_size, augment_data=False, shuffle=False):
    """
    IMPORTANT: augment runs BEFORE batch.
    Per-image ops (random_crop) need a single (H,W,C) tensor,
    not a batched (N,H,W,C) tensor.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y_cat))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10000))
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)   # ← per-image
    ds = ds.batch(batch_size)                                # ← batch after
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = build_dataset(X_train, y_train_cat, BATCH_SIZE, True, True)
val_ds = build_dataset(X_val, y_val_cat, BATCH_SIZE, False, False)
test_ds = build_dataset(X_test, y_test_cat, BATCH_SIZE, False, False)


# ============================================
# BUILD VGG-STYLE CNN (with BatchNorm)
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING CNN (BatchNorm + 4 blocks)")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    # Block 4 (new)
    layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    # Classifier
    layers.Flatten(),
    layers.Dense(512, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ============================================
# CALLBACKS (cosine LR + longer patience)
# ============================================
os.makedirs(MODELS_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=25,
        restore_best_weights=True, verbose=1
    ),
    callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS)),
        verbose=0
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_PATH, 'best_emotion_cnn.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]


# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING CNN")
print("=" * 60)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list,
    class_weight=class_weight_dict,
    verbose=1
)


# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60)
print("📊 EVALUATING ON TEST SET")
print("=" * 60)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

print("\n📋 Classification Report:")
print(classification_report(
    y_true_classes, y_pred_classes,
    target_names=EMOTION_CLASSES,
    zero_division=0
))

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_CLASSES,
            yticklabels=EMOTION_CLASSES)
plt.title('Emotion Confusion Matrix (CNN v2)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'emotion_cnn_confusion_matrix.png'), dpi=150)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy (CNN v2)')
ax1.legend(); ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss (CNN v2)')
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history_cnn.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, 'emotion_cnn.h5'))
print(f"\n✅ Saved to {MODELS_PATH}/emotion_cnn.h5")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
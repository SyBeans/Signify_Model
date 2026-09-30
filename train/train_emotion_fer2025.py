"""
train_emotion_fer2025.py

Train CNN on balanced FER2025 subset (6k/class, 42k total).
No class weights needed — data is perfectly balanced.
Same architecture as v2.1 (BatchNorm + 4 blocks).
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
from keras import layers, models, callbacks

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)
AUTOTUNE = tf.data.AUTOTUNE

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025")
MODELS_PATH = "models/face_fer2025"

IMG_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 0.001

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_CLASSES = 7

os.makedirs(MODELS_PATH, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING FER2025 CROPS")
print("=" * 60)

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train per-class: {np.bincount(y_train)}")

y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ============================================
# AUGMENTATION
# ============================================
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    zoom_pad = 6
    image = tf.image.resize_with_crop_or_pad(
        image, IMG_SIZE + zoom_pad, IMG_SIZE + zoom_pad
    )
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])

    if tf.random.uniform([]) < 0.30:
        eh = tf.random.uniform([], 6, 12, dtype=tf.int32)
        ew = tf.random.uniform([], 6, 12, dtype=tf.int32)
        ey = tf.random.uniform([], 0, IMG_SIZE - eh, dtype=tf.int32)
        ex = tf.random.uniform([], 0, IMG_SIZE - ew, dtype=tf.int32)
        mask = tf.pad(
            tf.ones([eh, ew, 1]),
            [[ey, IMG_SIZE - eh - ey], [ex, IMG_SIZE - ew - ex], [0, 0]]
        )
        image = image * (1.0 - mask)

    return tf.clip_by_value(image, 0.0, 1.0), label


def build_dataset(X, y_cat, batch_size, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y_cat))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10000))
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = build_dataset(X_train, y_train_cat, BATCH_SIZE, True, True)
val_ds = build_dataset(X_val, y_val_cat, BATCH_SIZE, False, False)
test_ds = build_dataset(X_test, y_test_cat, BATCH_SIZE, False, False)

# ============================================
# BUILD CNN
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING CNN (BatchNorm + 4 blocks)")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

    layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),

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
# CALLBACKS
# ============================================
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=6, min_lr=1e-7, verbose=1
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
print("🚀 TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list,
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
plt.title('Emotion Confusion Matrix (FER2025)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'confusion_matrix.png'), dpi=150)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.legend(); ax1.grid(True)
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, 'emotion_cnn.h5'))
print(f"\n✅ Saved to {MODELS_PATH}/emotion_cnn.h5")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
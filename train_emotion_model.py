"""
train_emotion_model.py
Trains a CNN model for facial expression/emotion recognition.
Uses MediaPipe Face Mesh landmarks (468 points).
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks"
MODELS_PATH = "models"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

NUM_FRAMES = 30
NUM_FACE_LANDMARKS = 468
NUM_COORDS = 3
NUM_FEATURES = NUM_FACE_LANDMARKS * NUM_COORDS  # 1404

# Emotion classes (7)
EMOTION_CLASSES = ['neutral', 'happy', 'sad', 'confused', 'urgent', 'questioning', 'angry']

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING FACIAL LANDMARK DATA")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_face_train.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_face_test.npy"))

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Features per frame: {NUM_FEATURES}")

# ============================================
# CREATE EMOTION LABELS (Placeholder - you need to label your data!)
# ============================================
# NOTE: FSL-105 videos don't have emotion labels.
# You need to manually label or use a pre-labeled emotion dataset.
# For now, we'll create dummy labels for demonstration.

print("\n⚠️  IMPORTANT:")
print("FSL-105 dataset doesn't have emotion labels.")
print("You need to either:")
print("  1. Manually label videos with emotions")
print("  2. Use a separate emotion dataset (FER2013, AffectNet, etc.)")
print("  3. Create your own labeled dataset")
print("=" * 60)

# Dummy labels for now (to be replaced with real labels)
num_train = len(X_train)
num_test = len(X_test)

# For demonstration, assign random labels (REPLACE THIS!)
y_train = np.random.randint(0, 7, num_train)
y_test = np.random.randint(0, 7, num_test)

print(f"\ny_train (dummy): {y_train.shape}")
print(f"y_test (dummy):  {y_test.shape}")
print("⚠️  These are DUMMY labels - replace with real emotion labels!")

# ============================================
# ONE-HOT ENCODE
# ============================================
NUM_CLASSES = len(EMOTION_CLASSES)
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ============================================
# CLASS WEIGHTS
# ============================================
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# ============================================
# BUILD CNN MODEL FOR EMOTION
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING CNN EMOTION MODEL")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(NUM_FRAMES, NUM_FEATURES)),
    
    # 1D CNN layers for temporal face landmark patterns
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # LSTM for temporal emotion sequence
    layers.LSTM(128, return_sequences=False),
    layers.Dropout(0.3),
    
    # Dense layers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    
    # Output
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
os.makedirs(MODELS_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy', patience=25,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8,
        min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_PATH, 'best_emotion_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING EMOTION MODEL")
print("=" * 60)

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=callbacks_list,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60)
print("📊 EVALUATING EMOTION MODEL")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ============================================
# PREDICTIONS
# ============================================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

print("\n📋 Classification Report:")
print(classification_report(
    y_true_classes, y_pred_classes,
    target_names=EMOTION_CLASSES,
    zero_division=0
))

# ============================================
# TRAINING PLOTS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Emotion Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Emotion Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'emotion_training_history.png'), dpi=150)
print(f"\n✅ Saved plot to {MODELS_PATH}/emotion_training_history.png")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'emotion_model.h5'))
print(f"✅ Saved model to {MODELS_PATH}/emotion_model.h5")

print("\n" + "=" * 60)
print("✅ EMOTION MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Model: {MODELS_PATH}/emotion_model.h5")
print("\n⚠️  REMINDER: This used DUMMY labels!")
print("For real emotion recognition, you need labeled data:")
print("  1. FER2013 dataset (7 emotions)")
print("  2. AffectNet dataset")
print("  3. Your own FSL emotion-labeled videos")
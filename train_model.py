"""
train_model.py (IMPROVED)
Trains an LSTM model with better settings for FSL-105
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks"
MODELS_PATH = "models"
LABELS_CSV = "datasets/FSL/labels.csv"

# Training settings - IMPROVED
BATCH_SIZE = 16          # Smaller batch
EPOCHS = 100             # More epochs
LEARNING_RATE = 0.0005   # Lower learning rate
TEST_SIZE = 0.2

NUM_FRAMES = 30
NUM_FEATURES = 63
NUM_CLASSES = 105

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING DATA")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_train.npy"))
y_train = np.load(os.path.join(LANDMARKS_PATH, "y_train.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_test.npy"))
y_test = np.load(os.path.join(LANDMARKS_PATH, "y_test.npy"))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# ============================================
# NORMALIZE DATA (IMPORTANT!)
# ============================================
print("\n" + "=" * 60)
print("🔧 NORMALIZING DATA")
print("=" * 60)

# Flatten, normalize, reshape back
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Simple min-max normalization
X_train_norm = (X_train_flat - X_train_flat.min()) / (X_train_flat.max() - X_train_flat.min() + 1e-8)
X_test_norm = (X_test_flat - X_test_flat.min()) / (X_test_flat.max() - X_test_flat.min() + 1e-8)

X_train = X_train_norm.reshape(X_train.shape)
X_test = X_test_norm.reshape(X_test.shape)

print("✅ Data normalized (0-1 range)")

# ============================================
# LOAD LABELS
# ============================================
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"\n📊 Classes: {len(label_names)}")

# ============================================
# COMPUTE CLASS WEIGHTS (Fixes imbalance)
# ============================================
print("\n" + "=" * 60)
print("⚖️  COMPUTING CLASS WEIGHTS")
print("=" * 60)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Class weights computed (range: {min(class_weights):.2f} - {max(class_weights):.2f})")

# ============================================
# ENCODE LABELS
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

# ============================================
# BUILD IMPROVED MODEL
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING IMPROVED MODEL")
print("=" * 60)

model = models.Sequential([
    # Input layer
    layers.Input(shape=(NUM_FRAMES, NUM_FEATURES)),
    
    # LSTM 1
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.4),
    
    # LSTM 2
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.4),
    
    # LSTM 3
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.4),
    
    # Dense layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    
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
        monitor='val_accuracy',
        patience=25,          # More patience
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING MODEL (Improved)")
print("=" * 60)

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=TEST_SIZE,
    callbacks=callbacks_list,
    class_weight=class_weight_dict,  # Use class weights!
    verbose=1
)

# ============================================
# EVALUATE
# ============================================
print("\n" + "=" * 60)
print("📊 EVALUATING ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ============================================
# PREDICTIONS
# ============================================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"\n🎯 Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Per-class accuracy
print("\n📋 Top 10 Signs by Accuracy:")
class_accuracies = []
for i in range(NUM_CLASSES):
    mask = y_true_classes == i
    if mask.sum() > 0:
        acc = accuracy_score(y_true_classes[mask], y_pred_classes[mask])
        class_accuracies.append((i, label_names[i], acc, mask.sum()))

class_accuracies.sort(key=lambda x: x[2], reverse=True)
for i, (cls_id, name, acc, count) in enumerate(class_accuracies[:10]):
    print(f"  {i+1}. {name:20s} → {acc*100:5.1f}% ({count} samples)")

# ============================================
# TRAINING PLOTS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history.png'), dpi=150)
print(f"\n✅ Saved to {MODELS_PATH}/training_history.png")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'sign_model.h5'))
print(f"✅ Saved to {MODELS_PATH}/sign_model.h5")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Next step: python convert_to_tflite.py")
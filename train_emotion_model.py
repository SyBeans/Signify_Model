"""
train_emotion_model.py
Trains a CNN/Dense model for facial expression recognition
using FER2013 facial landmarks (468 points per image).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks/face"
MODELS_PATH = "models"

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

NUM_FEATURES = 468 * 3  # 1404

# FER2013 emotion classes (must match extraction order!)
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTION_CLASSES)

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING FER2013 FACIAL LANDMARK DATA")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_face_train.npy"))
y_train = np.load(os.path.join(LANDMARKS_PATH, "y_face_train.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_face_test.npy"))
y_test = np.load(os.path.join(LANDMARKS_PATH, "y_face_test.npy"))
X_val = np.load(os.path.join(LANDMARKS_PATH, "X_face_val.npy"))
y_val = np.load(os.path.join(LANDMARKS_PATH, "y_face_val.npy"))

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")
print(f"X_val:   {X_val.shape}")
print(f"y_val:   {y_val.shape}")
print(f"Features per image: {NUM_FEATURES}")
print(f"Classes: {EMOTION_CLASSES}")

# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)

# ============================================
# CLASS WEIGHTS (FER2013 is imbalanced)
# ============================================
print("\n" + "=" * 60)
print("⚖️  COMPUTING CLASS WEIGHTS")
print("=" * 60)

class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Range: {min(class_weights):.2f} - {max(class_weights):.2f}")
for i, w in class_weight_dict.items():
    print(f"   {EMOTION_CLASSES[i]:10s}: {w:.3f}")

# ============================================
# BUILD MODEL (Dense MLP for single images)
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING EMOTION MODEL (MLP)")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(NUM_FEATURES,)),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

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
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_PATH, 'best_emotion_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# ============================================
# TRAIN (use val split as validation)
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING EMOTION MODEL")
print("=" * 60)

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val_cat),
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

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ============================================
# PREDICTIONS
# ============================================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

overall_acc = accuracy_score(y_true_classes, y_pred_classes)
print(f"\n🎯 Overall Accuracy: {overall_acc*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(
    y_true_classes, y_pred_classes,
    target_names=EMOTION_CLASSES,
    zero_division=0
))

# ============================================
# CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_CLASSES,
            yticklabels=EMOTION_CLASSES)
plt.title('Emotion Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'emotion_confusion_matrix.png'), dpi=150)
print(f"✅ Saved to {MODELS_PATH}/emotion_confusion_matrix.png")

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
print(f"✅ Saved plot to {MODELS_PATH}/emotion_training_history.png")

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
print("Next step: convert_to_tflite.py")
"""
train_model_v3_motion.py
Trains a Bidirectional LSTM model on motion features (position + velocity).
Best for motion-aware sign recognition.
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
LABELS_CSV = "datasets/FSL/labels.csv"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

NUM_FRAMES = 30
NUM_FEATURES = 126  # 63 position + 63 velocity
NUM_CLASSES = 105

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING MOTION DATA")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_train.npy"))
y_train = np.load(os.path.join(LANDMARKS_PATH, "y_train.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_test.npy"))
y_test = np.load(os.path.join(LANDMARKS_PATH, "y_test.npy"))

print(f"X_train: {X_train.shape}, range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"y_train: {y_train.shape}, unique: {len(np.unique(y_train))}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

# Load labels
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"📊 Classes: {len(label_names)}")

# ============================================
# CLASS WEIGHTS
# ============================================
print("\n" + "=" * 60)
print("⚖️  COMPUTING CLASS WEIGHTS")
print("=" * 60)

class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Range: {min(class_weights):.2f} - {max(class_weights):.2f}")

# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ============================================
# BUILD BiLSTM MODEL (Motion-Aware)
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING BiLSTM MODEL (Motion-Aware)")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(NUM_FRAMES, NUM_FEATURES)),
    
    # TimeDistributed Dense to process each frame's features
    layers.TimeDistributed(layers.Dense(64, activation='relu')),
    layers.Dropout(0.3),
    
    # Bidirectional LSTM 1
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Bidirectional LSTM 2
    layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Dense layers
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    
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
        os.path.join(MODELS_PATH, 'best_model_motion.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING BiLSTM MODEL (Motion-Aware)")
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

# Top 10 signs
class_accs = []
for i in range(NUM_CLASSES):
    mask = y_true_classes == i
    if mask.sum() > 0:
        acc = accuracy_score(y_true_classes[mask], y_pred_classes[mask])
        class_accs.append((i, label_names[i], acc, mask.sum()))

class_accs.sort(key=lambda x: x[2], reverse=True)
print("\n📋 Top 10 Signs by Accuracy:")
for i, (cid, name, acc, cnt) in enumerate(class_accs[:10]):
    print(f"  {i+1}. {name:20s} → {acc*100:5.1f}% ({cnt} samples)")

# Bottom 10 signs
print("\n📋 Bottom 10 Signs (Most Confused):")
for i, (cid, name, acc, cnt) in enumerate(class_accs[-10:]):
    print(f"  {name:20s} → {acc*100:5.1f}% ({cnt} samples)")

# ============================================
# TRAINING PLOTS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy (Motion-Aware)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss (Motion-Aware)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history_motion.png'), dpi=150)
print(f"\n✅ Saved plot to {MODELS_PATH}/training_history_motion.png")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'sign_model_motion.h5'))
print(f"✅ Saved model to {MODELS_PATH}/sign_model_motion.h5")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Model: {MODELS_PATH}/sign_model_motion.h5")
print(f"Next step: python convert_to_tflite.py")
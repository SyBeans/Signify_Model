"""
train_model.py
Trains an LSTM model on extracted hand landmarks.
Saves the trained model to models/sign_model.h5
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks"
MODELS_PATH = "models"
LABELS_CSV = "datasets/FSL/labels.csv"

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SIZE = 0.2  # Validation split

# Model settings
NUM_FRAMES = 30
NUM_FEATURES = 63  # 21 landmarks × 3 coordinates
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
# LOAD LABEL MAPPING
# ============================================
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()  # English labels
print(f"\n📊 Classes: {len(label_names)}")
print(f"First 5 labels: {label_names[:5]}")

# ============================================
# ENCODE LABELS (0-104 → One-Hot)
# ============================================
print("\n" + "=" * 60)
print("🔢 ENCODING LABELS")
print("=" * 60)

# Labels are already 0-104, just convert to categorical
y_train_cat = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"y_train_cat shape: {y_train_cat.shape}")
print(f"y_test_cat shape:  {y_test_cat.shape}")

# ============================================
# BUILD LSTM MODEL
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING MODEL")
print("=" * 60)

model = models.Sequential([
    # LSTM Layer 1
    layers.LSTM(128, return_sequences=True, input_shape=(NUM_FRAMES, NUM_FEATURES)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # LSTM Layer 2
    layers.LSTM(128, return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # LSTM Layer 3
    layers.LSTM(64, return_sequences=False),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Dense Layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    # Output Layer
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile
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
    # Early stopping if no improvement
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateau
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    # Save best model
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
print("🚀 TRAINING MODEL")
print("=" * 60)

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=TEST_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

# ============================================
# EVALUATE ON TEST SET
# ============================================
print("\n" + "=" * 60)
print("📊 EVALUATING ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ============================================
# PREDICTIONS & METRICS
# ============================================
print("\n" + "=" * 60)
print("📈 DETAILED METRICS")
print("=" * 60)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

# Accuracy per class
print("\n📋 Classification Report (Top 10 signs):")
# Get unique classes in test set
unique_classes = np.unique(y_true_classes)
top_classes = unique_classes[:10]
target_names = [label_names[i] for i in top_classes]

# Filter for top classes
mask = np.isin(y_true_classes, top_classes)
print(classification_report(
    y_true_classes[mask], 
    y_pred_classes[mask], 
    labels=top_classes,
    target_names=target_names,
    zero_division=0
))

# Overall accuracy
overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# ============================================
# CONFUSION MATRIX (SIMPLIFIED)
# ============================================
print("\n" + "=" * 60)
print("📊 CONFUSION MATRIX (Top 10 Signs)")
print("=" * 60)

cm = confusion_matrix(y_true_classes[mask], y_pred_classes[mask], labels=top_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix - Top 10 Signs')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'confusion_matrix.png'), dpi=150)
print(f"✅ Saved to {MODELS_PATH}/confusion_matrix.png")

# ============================================
# TRAINING HISTORY PLOTS
# ============================================
print("\n" + "=" * 60)
print("📈 TRAINING HISTORY")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history.png'), dpi=150)
print(f"✅ Saved to {MODELS_PATH}/training_history.png")

# ============================================
# SAVE FINAL MODEL
# ============================================
print("\n" + "=" * 60)
print("💾 SAVING MODEL")
print("=" * 60)

model.save(os.path.join(MODELS_PATH, 'sign_model.h5'))
print(f"✅ Saved to {MODELS_PATH}/sign_model.h5")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Model: {MODELS_PATH}/sign_model.h5")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Total Parameters: {model.count_params():,}")
print(f"\nNext step: python convert_to_tflite.py")
"""
train_model_v2.py - FIXED VERSION
Skips normalization (landmarks already 0-1), simpler model.
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
EPOCHS = 150
LEARNING_RATE = 0.001

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

print(f"X_train: {X_train.shape}, range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"y_train: {y_train.shape}, unique: {len(np.unique(y_train))}")

# Load labels
labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()

# ============================================
# CLASS WEIGHTS
# ============================================
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ============================================
# SIMPLER MODEL
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING MODEL")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(NUM_FRAMES, NUM_FEATURES)),
    
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    
    layers.LSTM(128, return_sequences=False),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    
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
        monitor='val_accuracy', patience=30,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_PATH, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
]

# ============================================
# TRAIN
# ============================================
print("\n🚀 TRAINING...")
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
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

print(f"🎯 Overall: {accuracy_score(y_true_classes, y_pred_classes)*100:.2f}%")

# Top 10 signs
class_accs = []
for i in range(NUM_CLASSES):
    mask = y_true_classes == i
    if mask.sum() > 0:
        class_accs.append((i, label_names[i], 
                          accuracy_score(y_true_classes[mask], y_pred_classes[mask]), 
                          mask.sum()))
class_accs.sort(key=lambda x: x[2], reverse=True)
print("\n📋 Top 10 Signs:")
for i, (cid, name, acc, cnt) in enumerate(class_accs[:10]):
    print(f"  {i+1}. {name:20s} → {acc*100:5.1f}% ({cnt})")

# Save
model.save(os.path.join(MODELS_PATH, 'sign_model.h5'))
print(f"\n✅ Saved to {MODELS_PATH}/sign_model.h5")
print("✅ DONE!")
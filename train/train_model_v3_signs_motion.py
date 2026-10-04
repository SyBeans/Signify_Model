"""
train_model_v3_motion.py (UPGRADED)
Trains a Bidirectional LSTM with:
- Data augmentation (4x training data)
- Attention mechanism
- Longer training (200 epochs)
- Better learning rate schedule
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
LANDMARKS_PATH = "landmarks/hand"
MODELS_PATH = "models/hand"
LABELS_CSV = "datasets/FSL/labels.csv"

BATCH_SIZE = 32
EPOCHS = 200                    # ← Increased from 100
LEARNING_RATE = 0.001

NUM_FRAMES = 30
NUM_FEATURES = 252
NUM_CLASSES = 105

USE_AUGMENTATION = True         # ← Toggle augmentation

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

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"📊 Classes: {len(label_names)}")

# ============================================
# DATA AUGMENTATION ✅
# ============================================
if USE_AUGMENTATION:
    print("\n" + "=" * 60)
    print("🔧 AUGMENTING TRAINING DATA (4x)")
    print("=" * 60)

    def augment_sample(sample, label):
        """Create 3 augmented copies of a sample."""
        augmented = [(sample, label)]  # original

        # 1. Small noise
        noise = np.random.normal(0, 0.01, sample.shape).astype(np.float32)
        augmented.append((sample + noise, label))

        # 2. Time shift
        shift = np.random.randint(-3, 4)
        augmented.append((np.roll(sample, shift, axis=0), label))

        # 3. Scale
        scale = np.random.uniform(0.95, 1.05)
        augmented.append((sample * scale, label))

        return augmented

    np.random.seed(42)
    X_aug, y_aug = [], []
    for i in range(len(X_train)):
        for s, l in augment_sample(X_train[i], y_train[i]):
            X_aug.append(s)
            y_aug.append(l)

    X_train = np.array(X_aug, dtype=np.float32)
    y_train = np.array(y_aug, dtype=np.int32)

    print(f"✅ Original: 1,703 → Augmented: {X_train.shape[0]}")

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
# BUILD MODEL WITH ATTENTION ✅
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING BiLSTM + ATTENTION MODEL")
print("=" * 60)

inputs = layers.Input(shape=(NUM_FRAMES, NUM_FEATURES))

# TimeDistributed Dense
x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inputs)
x = layers.Dropout(0.3)(x)

# BiLSTM 1
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# BiLSTM 2 (keep sequences for attention)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# ✅ Self-Attention Mechanism
# Score each timestep
attention_scores = layers.Dense(1, activation='tanh')(x)       # (batch, 30, 1)
attention_weights = layers.Softmax(axis=1)(attention_scores)   # (batch, 30, 1)
weighted = layers.Multiply()([x, attention_weights])           # (batch, 30, 256)
x = layers.GlobalAveragePooling1D()(weighted) # (batch, 256)

# Dense layers
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# CALLBACKS (Longer patience for 200 epochs)
# ============================================
os.makedirs(MODELS_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=35,                    # ← Increased for longer training
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        os.path.join(MODELS_PATH, 'best_model_motion.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING (Augmented + Attention)")
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

# Top/Bottom signs
class_accs = []
for i in range(NUM_CLASSES):
    mask = y_true_classes == i
    if mask.sum() > 0:
        acc = accuracy_score(y_true_classes[mask], y_pred_classes[mask])
        class_accs.append((i, label_names[i], acc, mask.sum()))

class_accs.sort(key=lambda x: x[2], reverse=True)
print("\n📋 Top 10 Signs:")
for i, (cid, name, acc, cnt) in enumerate(class_accs[:10]):
    print(f"  {i+1}. {name:20s} → {acc*100:5.1f}% ({cnt})")

print("\n📋 Bottom 10 Signs:")
for name, acc, cnt in [(n, a, c) for _, n, a, c in class_accs[-10:]]:
    print(f"  {name:20s} → {acc*100:5.1f}% ({cnt})")

# ============================================
# PLOTS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy (Augmented + Attention)')
ax1.legend(); ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss (Augmented + Attention)')
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history_v4.png'), dpi=150)
print(f"✅ Saved plot")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'sign_model_v4.h5'))
print(f"✅ Saved to {MODELS_PATH}/sign_model_v4.h5")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Model: {MODELS_PATH}/sign_model_v4.h5")

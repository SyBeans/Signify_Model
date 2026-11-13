"""
train_model_v5.py
Trains a Bidirectional LSTM with Attention on the MERGED dataset
(FSL-105 + Team Data + REST sign = 106 classes).
Architecture unchanged from v3/v4 (proven 88.5% baseline).
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks/merged"        # ← CHANGED: merged dataset
MODELS_PATH = "models/hand"
LABELS_CSV = "datasets/FSL/labels.csv"     # Original 105 FSL labels
OUTPUT_LABELS_JSON = "models/hand/labels_v5.json"  # New: full 106 labels for app

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001

NUM_FRAMES = 30
NUM_FEATURES = 252
NUM_CLASSES = 106                          # ← CHANGED: 105 FSL + 1 REST

USE_AUGMENTATION = True

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING MOTION DATA (MERGED)")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_train.npy"))
y_train = np.load(os.path.join(LANDMARKS_PATH, "y_train.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_test.npy"))
y_test = np.load(os.path.join(LANDMARKS_PATH, "y_test.npy"))

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")
print(f"Train classes: {len(np.unique(y_train))}")
print(f"Test classes:  {len(np.unique(y_test))}")

# ============================================
# LOAD LABELS (105 FSL + REST)
# ============================================
print("\n" + "=" * 60)
print("🏷️  LOADING LABELS")
print("=" * 60)

labels_df = pd.read_csv(LABELS_CSV)
label_names = labels_df['label'].tolist()
print(f"📊 FSL-105 labels loaded: {len(label_names)}")

# Append REST (ID 105)
assert len(label_names) == 105, f"Expected 105 FSL labels, got {len(label_names)}"
label_names.append("REST")
print(f"✅ Total labels (with REST): {len(label_names)}")

assert len(label_names) == NUM_CLASSES, \
    f"Label count ({len(label_names)}) != NUM_CLASSES ({NUM_CLASSES})"

# Save labels_v5.json for the Flutter app
os.makedirs(MODELS_PATH, exist_ok=True)
with open(OUTPUT_LABELS_JSON, 'w') as f:
    json.dump(label_names, f, indent=2)
print(f"✅ Saved labels for app → {OUTPUT_LABELS_JSON}")

# ============================================
# DATA AUGMENTATION (4x)
# ============================================
if USE_AUGMENTATION:
    print("\n" + "=" * 60)
    print("🔧 AUGMENTING TRAINING DATA (4x)")
    print("=" * 60)
    original_count = len(X_train)

    def augment_sample(sample, label):
        augmented = [(sample, label)]

        noise = np.random.normal(0, 0.01, sample.shape).astype(np.float32)
        augmented.append((sample + noise, label))

        shift = np.random.randint(-3, 4)
        augmented.append((np.roll(sample, shift, axis=0), label))

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

    print(f"✅ Original: {original_count:,} → Augmented: {X_train.shape[0]:,}")

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

# Verify all 106 classes have samples
missing = [i for i in range(NUM_CLASSES) if i not in np.unique(y_train)]
if missing:
    print(f"❌ CRITICAL: Missing classes in training data: {missing}")
    raise SystemExit("Fix dataset before training")
else:
    print(f"✅ All {NUM_CLASSES} classes present in training data")

# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ============================================
# BUILD MODEL (BiLSTM + Attention)
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING BiLSTM + ATTENTION MODEL")
print("=" * 60)

inputs = layers.Input(shape=(NUM_FRAMES, NUM_FEATURES))

x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inputs)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

attention_scores = layers.Dense(1, activation='tanh')(x)
attention_weights = layers.Softmax(axis=1)(attention_scores)
weighted = layers.Multiply()([x, attention_weights])
x = layers.GlobalAveragePooling1D()(weighted)

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
# CALLBACKS
# ============================================
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=35,
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
        os.path.join(MODELS_PATH, 'best_model_v5.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🚀 TRAINING v5 (106 classes, Augmented + Attention)")
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
# REST class — critical check
# ============================================
rest_id = label_names.index("REST") if "REST" in label_names else -1
if rest_id >= 0 and (y_true_classes == rest_id).sum() > 0:
    rest_acc = accuracy_score(
        y_true_classes[y_true_classes == rest_id],
        y_pred_classes[y_true_classes == rest_id]
    )
    print(f"\n🎯 REST (ID {rest_id}) Accuracy: {rest_acc*100:.2f}%")
    if rest_acc >= 0.85:
        print("   ✅ REST detection is strong — ghost signs will be blocked!")
    elif rest_acc >= 0.70:
        print("   ⚠️  REST detection is okay but could be better")
    else:
        print("   ❌ REST detection weak — capture more diverse REST samples")

# ============================================
# PLOTS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy (v5, 106 classes)')
ax1.legend(); ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss (v5, 106 classes)')
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history_v5.png'), dpi=150)
print(f"✅ Saved plot")

# ============================================
# SAVE MODEL
# ============================================
model.save(os.path.join(MODELS_PATH, 'sign_model_v5.h5'))
print(f"✅ Saved to {MODELS_PATH}/sign_model_v5.h5")
print(f"✅ Saved labels to {OUTPUT_LABELS_JSON}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Classes: {NUM_CLASSES} (105 FSL + 1 REST)")
print(f"Model: {MODELS_PATH}/sign_model_v5.h5")
print(f"Labels: {OUTPUT_LABELS_JSON}")
print(f"\nNEXT: Export to TFLite with convert_v5_to_tflite.py")
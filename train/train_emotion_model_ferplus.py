"""
train_emotion_model_ferplus.py
Trains emotion model on FER+ landmarks (7 classes).

Features:
- Class weights (sqrt balancing for imbalanced data)
- Data augmentation (4x training data)
- MLP + BatchNorm + Attention gating
- Uses separate val set (X_face_val.npy)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow import keras
from keras import layers, models, callbacks, Model

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks/face_ferplus"
MODELS_PATH = "models/face_ferplus"

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0005

NUM_FACE_LANDMARKS = 468
NUM_COORDS = 3
BASE_FEATURES = NUM_FACE_LANDMARKS * NUM_COORDS  # 1404

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7

USE_AUGMENTATION = True

# Key MediaPipe landmark indices
KEY_PAIRS = [
    (61, 291), (13, 14), (33, 133), (362, 263),
    (105, 334), (293, 334), (61, 13), (291, 13),
    (61, 14), (291, 14), (33, 362), (13, 168),
]

# ============================================
# FEATURE ENGINEERING
# ============================================
def add_relative_features(X):
    """Add relative distance features between key landmarks."""
    N = X.shape[0]
    X_reshaped = X.reshape(N, 468, 3)

    rel_features = []
    for i, j in KEY_PAIRS:
        diff = X_reshaped[:, i, :] - X_reshaped[:, j, :]
        dist = np.linalg.norm(diff, axis=1)
        rel_features.append(dist.reshape(N, 1))

    mouth_h = np.linalg.norm(X_reshaped[:, 13] - X_reshaped[:, 14], axis=1, keepdims=True)
    mouth_w = np.linalg.norm(X_reshaped[:, 61] - X_reshaped[:, 291], axis=1, keepdims=True)
    mar = mouth_h / (mouth_w + 1e-6)

    eye_l = np.linalg.norm(X_reshaped[:, 33] - X_reshaped[:, 133], axis=1, keepdims=True)
    eye_r = np.linalg.norm(X_reshaped[:, 362] - X_reshaped[:, 263], axis=1, keepdims=True)
    ear = (eye_l + eye_r) / 2.0

    rel_features.append(mar)
    rel_features.append(ear)

    rel_features = np.concatenate(rel_features, axis=1)
    return np.concatenate([X, rel_features], axis=1)


# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING FER+ DATA")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_face_train.npy"))
y_train = np.load(os.path.join(LANDMARKS_PATH, "y_face_train.npy"))
X_val = np.load(os.path.join(LANDMARKS_PATH, "X_face_val.npy"))
y_val = np.load(os.path.join(LANDMARKS_PATH, "y_face_val.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_face_test.npy"))
y_test = np.load(os.path.join(LANDMARKS_PATH, "y_face_test.npy"))

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

# ============================================
# ADD RELATIVE FEATURES
# ============================================
print("\n🔧 Adding relative distance features...")
X_train = add_relative_features(X_train)
X_val = add_relative_features(X_val)
X_test = add_relative_features(X_test)

NUM_FEATURES = X_train.shape[1]
print(f"✅ Feature count: {NUM_FEATURES}")

# ============================================
# NORMALIZE
# ============================================
print("\n🔧 Normalizing...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# ============================================
# DATA AUGMENTATION (4x)
# ============================================
if USE_AUGMENTATION:
    print("\n" + "=" * 60)
    print("🔧 AUGMENTING TRAINING DATA (4x)")
    print("=" * 60)

    def augment_sample(sample, label):
        augmented = [(sample, label)]

        # 1. Small noise
        noise = np.random.normal(0, 0.01, sample.shape).astype(np.float32)
        augmented.append((sample + noise, label))

        # 2. Scale
        scale = np.random.uniform(0.95, 1.05)
        augmented.append((sample * scale, label))

        # 3. Time-shift (roll along features)
        shift = np.random.randint(-3, 4)
        augmented.append((np.roll(sample, shift), label))

        return augmented

    np.random.seed(42)
    X_aug, y_aug = [], []
    for i in range(len(X_train)):
        for s, l in augment_sample(X_train[i], y_train[i]):
            X_aug.append(s)
            y_aug.append(l)

    X_train = np.array(X_aug, dtype=np.float32)
    y_train = np.array(y_aug, dtype=np.int32)

    print(f"✅ Original: 27,246 → Augmented: {X_train.shape[0]}")

# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ============================================
# CLASS WEIGHTS (sqrt balancing)
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
# BUILD MODEL
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING ATTENTION MLP MODEL")
print("=" * 60)

inputs = layers.Input(shape=(NUM_FEATURES,))

x = layers.Dense(512, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# Attention gating
attention = layers.Dense(256, activation='sigmoid')(x)
x = layers.Multiply()([x, attention])

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

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
        monitor='val_loss', factor=0.5,
        patience=10, min_lr=1e-7, verbose=1
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
print("🚀 TRAINING ON FER+")
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
# CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_CLASSES,
            yticklabels=EMOTION_CLASSES)
plt.title('Emotion Confusion Matrix (FER+)')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'emotion_confusion_matrix.png'), dpi=150)
print(f"✅ Saved confusion matrix")

# ============================================
# TRAINING PLOTS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy (FER+)')
ax1.legend(); ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss (FER+)')
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'training_history.png'), dpi=150)
print(f"✅ Saved training plot")

# ============================================
# SAVE
# ============================================
model.save(os.path.join(MODELS_PATH, 'emotion_model.h5'))
print(f"\n✅ Saved to {MODELS_PATH}/emotion_model.h5")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
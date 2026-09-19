"""
train_emotion_model.py
Enhanced MLP with relative distance features + attention.
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
LANDMARKS_PATH = "landmarks/face"
MODELS_PATH = "models/face"

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0005

NUM_FACE_LANDMARKS = 468
NUM_COORDS = 3
BASE_FEATURES = NUM_FACE_LANDMARKS * NUM_COORDS  # 1404

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7

# Key MediaPipe landmark indices
# Mouth corners: 61, 291 | Upper lip: 13 | Lower lip: 14
# Left eye: 33, 133 | Right eye: 362, 263
# Left eyebrow: 105, 334 | Right eyebrow: 293, 334
KEY_PAIRS = [
    (61, 291),   # Mouth width
    (13, 14),    # Mouth height
    (33, 133),   # Left eye
    (362, 263),  # Right eye
    (105, 334),  # Left eyebrow
    (293, 334),  # Right eyebrow
    (61, 13),    # Left mouth corner to top
    (291, 13),   # Right mouth corner to top
    (61, 14),    # Left mouth corner to bottom
    (291, 14),   # Right mouth corner to bottom
    (33, 362),   # Eye distance (left to right)
    (13, 168),   # Nose to upper lip
]

# ============================================
# FEATURE ENGINEERING
# ============================================
def add_relative_features(X):
    """
    Add relative distance features between key landmarks.
    Input: (N, 1404) — raw landmarks
    Output: (N, 1404 + 24) — with relative features
    """
    N = X.shape[0]
    X_reshaped = X.reshape(N, 468, 3)  # (N, 468, 3)

    rel_features = []
    for i, j in KEY_PAIRS:
        # Euclidean distance in 3D
        diff = X_reshaped[:, i, :] - X_reshaped[:, j, :]
        dist = np.linalg.norm(diff, axis=1)  # (N,)
        rel_features.append(dist.reshape(N, 1))

    # Also add mouth aspect ratio and eye aspect ratio
    mouth_h = np.linalg.norm(X_reshaped[:, 13] - X_reshaped[:, 14], axis=1, keepdims=True)
    mouth_w = np.linalg.norm(X_reshaped[:, 61] - X_reshaped[:, 291], axis=1, keepdims=True)
    mar = mouth_h / (mouth_w + 1e-6)  # Mouth Aspect Ratio

    eye_l = np.linalg.norm(X_reshaped[:, 33] - X_reshaped[:, 133], axis=1, keepdims=True)
    eye_r = np.linalg.norm(X_reshaped[:, 362] - X_reshaped[:, 263], axis=1, keepdims=True)
    ear = (eye_l + eye_r) / 2.0  # Eye Aspect Ratio

    rel_features.append(mar)
    rel_features.append(ear)

    rel_features = np.concatenate(rel_features, axis=1)  # (N, 14)
    return np.concatenate([X, rel_features], axis=1)  # (N, 1418)


# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("📥 LOADING DATA")
print("=" * 60)

X_train = np.load(os.path.join(LANDMARKS_PATH, "X_face_train.npy"))
y_train = np.load(os.path.join(LANDMARKS_PATH, "y_face_train.npy"))
X_test = np.load(os.path.join(LANDMARKS_PATH, "X_face_test.npy"))
y_test = np.load(os.path.join(LANDMARKS_PATH, "y_face_test.npy"))
X_val = np.load(os.path.join(LANDMARKS_PATH, "X_face_val.npy"))
y_val = np.load(os.path.join(LANDMARKS_PATH, "y_face_val.npy"))

print(f"Original shapes: train={X_train.shape}, test={X_test.shape}, val={X_val.shape}")

# ============================================
# ADD RELATIVE FEATURES ✅
# ============================================
print("\n🔧 Adding relative distance features...")
X_train = add_relative_features(X_train)
X_test = add_relative_features(X_test)
X_val = add_relative_features(X_val)

NUM_FEATURES = X_train.shape[1]
print(f"✅ New feature count: {NUM_FEATURES}")

# ============================================
# NORMALIZE
# ============================================
print("\n🔧 Normalizing...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

# ============================================
# ONE-HOT
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)

# ============================================
# CLASS WEIGHTS
# ============================================
raw_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
soft_weights = np.sqrt(raw_weights)
class_weight_dict = dict(enumerate(soft_weights))
print(f"✅ Class weights: {soft_weights}")

# ============================================
# BUILD ATTENTION MODEL
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING ATTENTION MODEL")
print("=" * 60)

inputs = layers.Input(shape=(NUM_FEATURES,))

# First dense block
x = layers.Dense(512, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# Second dense block
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# Attention-like gating
attention = layers.Dense(256, activation='sigmoid')(x)
x = layers.Multiply()([x, attention])

# Third dense block
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Output
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
print("\n🚀 TRAINING...")
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
print("\n📊 EVALUATING...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(X_test)
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
plt.title('Emotion Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_PATH, 'emotion_confusion_matrix.png'), dpi=150)

model.save(os.path.join(MODELS_PATH, 'emotion_model.h5'))
print(f"\n✅ Saved to {MODELS_PATH}/emotion_model.h5")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
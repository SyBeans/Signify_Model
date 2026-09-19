"""
train_emotion_model.py
Trains an MLP with normalization + softened class weights.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow import keras
from keras import layers, models, callbacks

# ============================================
# CONFIGURATION
# ============================================
LANDMARKS_PATH = "landmarks/face"
MODELS_PATH = "models/face"

BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.0005

NUM_FEATURES = 468 * 3
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = 7

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

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {X_val.shape}")

# ============================================
# NORMALIZE DATA ✅
# ============================================
print("\n🔧 Normalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
print(f"✅ After normalization: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

# ============================================
# ONE-HOT ENCODE
# ============================================
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)

# ============================================
# SOFTENED CLASS WEIGHTS ✅
# ============================================
print("\n⚖️  Computing softened class weights...")
raw_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
# Soften: sqrt makes weights less aggressive
soft_weights = np.sqrt(raw_weights)
class_weight_dict = dict(enumerate(soft_weights))
for i, w in class_weight_dict.items():
    print(f"   {EMOTION_CLASSES[i]:10s}: {w:.3f}")

# ============================================
# BUILD MODEL
# ============================================
print("\n" + "=" * 60)
print("🏗️  BUILDING MODEL")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(NUM_FEATURES,)),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
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
        monitor='val_accuracy', patience=25,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=8, min_lr=1e-7, verbose=1
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
print("\n📊 EVALUATING ON TEST SET")
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

# Confusion matrix
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
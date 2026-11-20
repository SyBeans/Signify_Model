"""
check_negative_subs.py — v2 (rebuild architecture to fix Lambda tf issue)

Same approach as diagnose_v12_tta.py: rebuild the model from code, load weights.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers

keras.config.enable_unsafe_deserialization()

DATA_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v13_balanced")
MODEL_PATH = "models/face_fer2025_hybrid_v12_2/emotion_v12_2.keras"

IMG_SIZE = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2
ENGINEERED_DIM = 26
NUM_CLASSES = 4

# ============================================================
# REBUILD ARCHITECTURE (identical to train_hybrid_v12_2.py)
# ============================================================
def se_block(x, ratio=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // ratio, 4), activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])

def residual_conv_block(x, filters, stride=1, drop=0.0):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding='same', use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    return x

def residual_mlp_block(x, units, drop=0.3):
    h = layers.Dense(units, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    h = layers.BatchNormalization()(h); h = layers.Activation('relu')(h); h = layers.Dropout(drop)(h)
    h = layers.Dense(units, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(h)
    h = layers.BatchNormalization()(h)
    if x.shape[-1] != units:
        x = layers.Dense(units, use_bias=False)(x); x = layers.BatchNormalization()(x)
    x = layers.Add()([x, h])
    x = layers.Activation('relu')(x)
    return x

def build_v12_2():
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    xi = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
    xi = residual_conv_block(xi, 32,  drop=0.10)
    xi = residual_conv_block(xi, 64,  stride=2, drop=0.15)
    xi = residual_conv_block(xi, 128, stride=2, drop=0.20)
    xi = residual_conv_block(xi, 256, stride=2, drop=0.25)
    xi = layers.GlobalAveragePooling2D()(xi)
    xi = layers.Dense(128, use_bias=False)(xi)
    xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
    xi = layers.Dropout(0.4)(xi)

    lm_input  = layers.Input(shape=(RAW_LANDMARK_DIM,), name='landmarks')
    eng_input = layers.Input(shape=(ENGINEERED_DIM,),   name='engineered')

    lm = layers.Reshape((NUM_LANDMARKS, 2), name='lm_reshape')(lm_input)
    center = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        output_shape=(1, 2), name='lm_center')(lm)
    centered = layers.Subtract(name='lm_center_sub')([lm, center])
    max_d = layers.Lambda(
        lambda t: tf.reduce_max(tf.norm(t, axis=-1), axis=1, keepdims=True)[..., None] + 1e-6,
        output_shape=(1, 1), name='lm_maxd')(centered)
    normed = layers.Lambda(
        lambda i: i[0] / i[1],
        output_shape=(NUM_LANDMARKS, 2), name='lm_norm')([centered, max_d])
    flat = layers.Reshape((RAW_LANDMARK_DIM,), name='lm_flat')(normed)

    lm_combined = layers.Concatenate(name='lm_plus_eng')([flat, eng_input])

    xl = residual_mlp_block(lm_combined, 512, drop=0.30)
    xl = residual_mlp_block(xl,           256, drop=0.30)
    xl = residual_mlp_block(xl,           128, drop=0.25)

    combined = layers.Concatenate()([xi, xl])
    combined = layers.Dense(128, use_bias=False)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

    return models.Model(inputs=[img_input, lm_input, eng_input],
                        outputs=out, name='emotion_v12_2')

# ============================================================
# LOAD
# ============================================================
print("=" * 60)
print("🔨 Building v12.2 architecture...")
model = build_v12_2()

print(f"📥 Loading weights from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Not found: {MODEL_PATH}")

model.load_weights(MODEL_PATH)
print("✅ Weights loaded")
print("=" * 60)

# ============================================================
# LOAD TEST DATA
# ============================================================
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
E_test = np.load(f"{DATA_PATH}/E_test.npy").astype(np.float32)
y_test_7 = np.load(f"{DATA_PATH}/y_test.npy")   # 7-class labels

print(f"Test: X={X_test.shape}, L={L_test.shape}, E={E_test.shape}")
print(f"Test 7-class: {np.bincount(y_test_7, minlength=7)}")

# ============================================================
# PREDICT (single pass — what app uses)
# ============================================================
print("\n📊 Predicting...")
p = model.predict([X_test, L_test, E_test], batch_size=256, verbose=0)
pred_4 = np.argmax(p, axis=1)   # 0-3 (Positive, Negative, Surprise, Neutral)

# ============================================================
# NEGATIVE SUB-EMOTION BREAKDOWN
# ============================================================
SUB_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NEG_SUBS = {0: "Angry", 1: "Disgust", 2: "Fear", 5: "Sad"}

print("\n" + "=" * 60)
print("NEGATIVE sub-emotion recall (from 4-class model)")
print("=" * 60)
print(f"  {'Sub-emotion':<12} | {'Test n':>6} | {'Pred Neg':>8} | {'Recall':>7}")
print("-" * 60)

total_neg_correct = 0
total_neg_samples = 0
for sub_id, sub_name in NEG_SUBS.items():
    mask = (y_test_7 == sub_id)
    n = int(mask.sum())
    if n == 0:
        continue
    # Among these sub-emotion test samples, how many did model predict as Negative (class 1)?
    correct = int((pred_4[mask] == 1).sum())
    acc = correct / n * 100
    total_neg_correct += correct
    total_neg_samples += n
    bar = "█" * int(acc / 5)
    print(f"  {sub_name:<12} | {n:>6} | {correct:>8} | {acc:>6.2f}%  {bar}")

print("-" * 60)
overall_neg = total_neg_correct / total_neg_samples * 100
print(f"  {'OVERALL NEG':<12} | {total_neg_samples:>6} | {total_neg_correct:>8} | {overall_neg:>6.2f}%")

# ============================================================
# WHERE DO MISSED NEGATIVE SAMPLES GO?
# ============================================================
print("\n" + "=" * 60)
print("Where do MISSED Negative samples go?")
print("=" * 60)
CLASS_NAMES_4 = ["Positive", "Negative", "Surprise", "Neutral"]

# For all Negative 4-class samples, show predicted distribution
y_test_4 = np.array([1, 1, 1, 0, 3, 1, 2])[y_test_7]   # GROUP_MAP
neg_mask = (y_test_4 == 1)
from collections import Counter
pred_counter = Counter(pred_4[neg_mask])

for i, name in enumerate(CLASS_NAMES_4):
    count = pred_counter.get(i, 0)
    pct = count / neg_mask.sum() * 100
    tag = " ✅ (correct)" if i == 1 else ""
    bar = "█" * int(pct / 2)
    print(f"  → {name:<10}: {count:>5} ({pct:>5.2f}%)  {bar}{tag}")

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
angry_recall = (pred_4[y_test_7 == 0] == 1).mean() * 100
sad_recall = (pred_4[y_test_7 == 5] == 1).mean() * 100

if angry_recall >= 80 and sad_recall >= 80:
    print(f"  ✅ Angry {angry_recall:.1f}% + Sad {sad_recall:.1f}% — both ≥80%")
    print(f"     → App's emergency distress detection WORKS")
    print(f"     → DEPLOY v12.2 immediately")
elif angry_recall >= 75 or sad_recall >= 75:
    print(f"  ⚠️  Angry {angry_recall:.1f}% + Sad {sad_recall:.1f}% — one is 75-80%")
    print(f"     → Still functional for emergencies")
    print(f"     → Deploy, document the weak one")
else:
    print(f"  🔴 Angry {angry_recall:.1f}% + Sad {sad_recall:.1f}% — both <75%")
    print(f"     → Consider fine-tune with Negative boost")
print("=" * 60)
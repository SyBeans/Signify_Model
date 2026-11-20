"""
reexport_emotion_v12_3.py
Rebuild v12.3 emotion architecture from scratch, transfer weights, export TFLite.
Bypasses Keras 3 .keras save/load path. 3-input model: image + landmarks + engineered.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

BASE        = os.path.expanduser("~/Signify/Signify_Model")
CKPT_PATH   = f"{BASE}/models/face_fer2025_hybrid_v12_3/emotion_v12_3_best.keras"
FULL_PATH   = f"{BASE}/models/face_fer2025_hybrid_v12_3/emotion_v12_3.keras"
TFLITE_PATH = f"{BASE}/convert/output/emotion_model_v12_3.tflite"
DATA_PATH   = f"{BASE}/landmarks/fer2025_v13_balanced"

IMG_SIZE         = 48
NUM_LANDMARKS    = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2   # 936
ENGINEERED_DIM   = 26
NUM_CLASSES      = 4


# ── Exact architecture copied from train_hybrid_v12_2.py ────────────────────
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
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
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
    h = layers.Dense(units, use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    h = layers.BatchNormalization()(h); h = layers.Activation('relu')(h)
    h = layers.Dropout(drop)(h)
    h = layers.Dense(units, use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-4))(h)
    h = layers.BatchNormalization()(h)
    if x.shape[-1] != units:
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    x = layers.Add()([x, h])
    x = layers.Activation('relu')(x)
    return x


def build_model():
    # ── Image branch (identical to v11.8) ──
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

    # ── Landmark branch (v12.3 — residual MLP + Lambdas) ──
    lm_input  = layers.Input(shape=(RAW_LANDMARK_DIM,), name='landmarks')
    eng_input = layers.Input(shape=(ENGINEERED_DIM,),   name='engineered')

    lm = layers.Reshape((NUM_LANDMARKS, 2), name='lm_reshape')(lm_input)
    center = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        output_shape=(1, 2), name='lm_center')(lm)
    centered = layers.Subtract(name='lm_center_sub')([lm, center])
    max_d = layers.Lambda(
        lambda t: tf.reduce_max(tf.norm(t, axis=-1), axis=1,
                                keepdims=True)[..., None] + 1e-6,
        output_shape=(1, 1), name='lm_maxd')(centered)
    normed = layers.Lambda(
        lambda i: i[0] / i[1],
        output_shape=(NUM_LANDMARKS, 2), name='lm_norm')([centered, max_d])
    flat = layers.Reshape((RAW_LANDMARK_DIM,), name='lm_flat')(normed)

    lm_combined = layers.Concatenate(name='lm_plus_eng')([flat, eng_input])

    xl = residual_mlp_block(lm_combined, 512, drop=0.30)
    xl = residual_mlp_block(xl,           256, drop=0.30)
    xl = residual_mlp_block(xl,           128, drop=0.25)

    # ── Fusion ──
    combined = layers.Concatenate()([xi, xl])
    combined = layers.Dense(128, use_bias=False)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

    return keras.Model(inputs=[img_input, lm_input, eng_input],
                       outputs=outputs, name="emotion_v12_3_rebuilt")


# ── 1. Build clean model ────────────────────────────────────────────────────
print("=" * 60); print("🔨 Rebuilding v12.3 architecture from scratch"); print("=" * 60)
model = build_model()
print(f"fresh inputs : {[tuple(i.shape) for i in model.inputs]}")
print(f"fresh outputs: {[tuple(o.shape) for o in model.outputs]}")
print(f"params       : {model.count_params():,}")

# ── 2. Load weights from .keras ─────────────────────────────────────────────
print("\n" + "=" * 60); print("📥 Loading v12.3 weights"); print("=" * 60)
loaded = False
for candidate in (CKPT_PATH, FULL_PATH):
    if not os.path.exists(candidate):
        print(f"  missing: {candidate}"); continue
    try:
        print(f"  trying load_weights({os.path.basename(candidate)})")
        model.load_weights(candidate)
        print(f"  ✅ weights loaded from {os.path.basename(candidate)}")
        loaded = True; break
    except Exception as e:
        print(f"  load_weights failed: {e}")
        try:
            print(f"  trying load_model(safe_mode=False) + set_weights")
            old = keras.models.load_model(candidate,
                                          safe_mode=False, compile=False)
            model.set_weights(old.get_weights())
            print(f"  ✅ weights transferred from {os.path.basename(candidate)}")
            loaded = True; break
        except Exception as e2:
            print(f"  set_weights also failed: {e2}")

if not loaded:
    raise RuntimeError("Could not load weights from either .keras file")

# ── 3. Sanity check (3 inputs) ──────────────────────────────────────────────
print("\n" + "=" * 60); print("📊 Sanity check (first 500 test samples)"); print("=" * 60)
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
E_test = np.load(f"{DATA_PATH}/E_test.npy").astype(np.float32)
y_orig = np.load(f"{DATA_PATH}/y_test.npy")
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)
y4 = GROUP_MAP[y_orig]

N = 500
keras_pred = model.predict([X_test[:N], L_test[:N], E_test[:N]],
                           verbose=0).argmax(axis=1)
keras_acc = (keras_pred == y4[:N]).mean()
print(f"Keras accuracy: {keras_acc * 100:.2f}%  (expected ~86% for v12.3)")
if keras_acc < 0.80:
    raise SystemExit("❌ Weight load failed — aborting before conversion")

# ── 4. Freeze + convert (3 inputs) ──────────────────────────────────────────
print("\n" + "=" * 60); print("🔄 Converting to TFLite"); print("=" * 60)

@tf.function
def serve(img, lm, eng):
    return model([img, lm, eng], training=False)

concrete = serve.get_concrete_function(
    tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 1], tf.float32, name='image'),
    tf.TensorSpec([1, RAW_LANDMARK_DIM],   tf.float32, name='landmarks'),
    tf.TensorSpec([1, ENGINEERED_DIM],     tf.float32, name='engineered'),
)

frozen = convert_variables_to_constants_v2(concrete)
converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"✅ saved: {TFLITE_PATH} ({size_mb:.2f} MB)")

# ── 5. Verify signature + cross-check ───────────────────────────────────────
print("\n" + "=" * 60); print("✅ VERIFY TFLITE"); print("=" * 60)
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
in_details = interp.get_input_details()
out_detail = interp.get_output_details()[0]

for d in in_details:
    n_elem = int(np.prod(d['shape']))
    print(f"  in : name={d['name']:<30} shape={tuple(d['shape'])} "
          f"elems={n_elem} dtype={d['dtype'].__name__}")
print(f"  out: name={out_detail['name']:<30} "
      f"shape={tuple(out_detail['shape'])} dtype={out_detail['dtype'].__name__}")

assert len(in_details) == 3, f"Expected 3 inputs, got {len(in_details)}"

img_idx = next(d['index'] for d in in_details if 'image'     in d['name'].lower())
lm_idx  = next(d['index'] for d in in_details if 'landmark'  in d['name'].lower())
eng_idx = next(d['index'] for d in in_details if 'engineered' in d['name'].lower())
out_idx = out_detail['index']

tfl_pred = []
for i in range(min(100, N)):
    interp.set_tensor(img_idx, X_test[i:i+1])
    interp.set_tensor(lm_idx,  L_test[i:i+1])
    interp.set_tensor(eng_idx, E_test[i:i+1])
    interp.invoke()
    tfl_pred.append(int(np.argmax(interp.get_tensor(out_idx)[0])))
tfl_pred = np.array(tfl_pred)
tfl_acc = (tfl_pred == y4[:len(tfl_pred)]).mean()
match   = (tfl_pred == keras_pred[:len(tfl_pred)]).mean()

print(f"\nTFLite acc (first {len(tfl_pred)}): {tfl_acc * 100:.2f}%")
print(f"Match with Keras:                 {match * 100:.2f}%")

if match >= 0.98 and tfl_acc >= 0.80:
    print("\n✅ conversion correct — 3 inputs, shapes clean, acc matches Keras")
else:
    print("\n⚠️  drift detected — paste this output")
"""
reexport_emotion_v12.py
Rebuild v12 architecture from scratch, transfer weights, export TFLite.
Bypasses Keras 3 Lambda serialization issues.
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
H5_PATH     = f"{BASE}/models/face_fer2025_hybrid_v12/emotion_hybrid_v12_seed42.h5"
CKPT_PATH   = f"{BASE}/models/face_fer2025_hybrid_v12/best_v12_seed42.h5"
TFLITE_PATH = f"{BASE}/convert/output/emotion_model_v12.tflite"
DATA_PATH   = f"{BASE}/landmarks/fer2025_v12_balanced"

IMG_SIZE      = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2
NUM_CLASSES   = 4
USE_ENGINEERED = True

# ── Copy these functions from train_hybrid_v12.py ──
# build_preprocessing_layers()
# se_block()
# residual_conv_block()
# residual_mlp_block()

# (paste the exact function bodies from train_hybrid_v12.py)

def build_v12_model():
    # Exact copy of the model-building code from train_hybrid_v12.py
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    xi = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
    xi = residual_conv_block(xi, 32, drop=0.10)
    xi = residual_conv_block(xi, 64, stride=2, drop=0.15)
    xi = residual_conv_block(xi, 128, stride=2, drop=0.20)
    xi = residual_conv_block(xi, 256, stride=2, drop=0.25)
    xi = layers.GlobalAveragePooling2D()(xi)
    xi = layers.Dense(128, use_bias=False)(xi)
    xi = layers.BatchNormalization()(xi); xi = layers.Activation('relu')(xi)
    xi = layers.Dropout(0.4)(xi)

    lm_input = layers.Input(shape=(RAW_LANDMARK_DIM,), name="landmarks")
    lm_preprocessed = build_preprocessing_layers()(lm_input)

    xl = layers.Dense(512, use_bias=False)(lm_preprocessed)
    xl = layers.BatchNormalization()(xl)
    xl = layers.Activation('relu')(xl)
    xl = layers.Dropout(0.3)(xl)
    xl = residual_mlp_block(xl, 512, drop=0.3)
    xl = residual_mlp_block(xl, 256, drop=0.3)
    xl = residual_mlp_block(xl, 128, drop=0.3)
    xl = layers.Dense(128, use_bias=False)(xl)
    xl = layers.BatchNormalization()(xl); xl = layers.Activation('relu')(xl)
    xl = layers.Dropout(0.25)(xl)

    combined = layers.Concatenate()([xi, xl])
    combined = layers.Dense(128, use_bias=False)(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(combined)

    return keras.Model(inputs=[img_input, lm_input], outputs=outputs,
                       name="emotion_hybrid_v12_rebuilt")

# 1. Build clean model
model = build_v12_model()
print(f"params: {model.count_params()}")

# 2. Load weights
for candidate in (CKPT_PATH, H5_PATH):
    if not os.path.exists(candidate): continue
    try:
        model.load_weights(candidate)
        print(f"✅ Weights loaded from {os.path.basename(candidate)}")
        break
    except Exception as e:
        print(f"  load_weights failed: {e}")

# 3. Sanity check
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")
GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)
y4 = GROUP_MAP[y_test_orig]

N = 500
keras_pred = model.predict([X_test[:N], L_test[:N]], verbose=0).argmax(axis=1)
keras_acc = (keras_pred == y4[:N]).mean()
print(f"Keras acc: {keras_acc*100:.2f}%")

# 4. Freeze + convert
@tf.function
def serve(img, lm):
    return model([img, lm], training=False)

concrete = serve.get_concrete_function(
    tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 1], tf.float32, name='image'),
    tf.TensorSpec([1, RAW_LANDMARK_DIM], tf.float32, name='landmarks'),
)
frozen = convert_variables_to_constants_v2(concrete)
converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

try:
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
    print(f"✅ TFLite saved: {TFLITE_PATH} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"❌ TFLite failed: {e}")
    print(f"   → Retrain with USE_ENGINEERED = False")
    raise

# 5. Verify
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
in_details = interp.get_input_details()
out_detail = interp.get_output_details()[0]

print(f"\nInputs:")
for d in in_details:
    print(f"  {d['name']}: {d['shape']}")
print(f"Output: {out_detail['shape']}")

img_idx = next(d['index'] for d in in_details if 'image' in d['name'].lower())
lm_idx = next(d['index'] for d in in_details if 'landmark' in d['name'].lower())
out_idx = out_detail['index']

tfl_pred = []
for i in range(min(100, N)):
    interp.set_tensor(img_idx, X_test[i:i+1])
    interp.set_tensor(lm_idx, L_test[i:i+1])
    interp.invoke()
    tfl_pred.append(int(np.argmax(interp.get_tensor(out_idx)[0])))
tfl_pred = np.array(tfl_pred)
match = (tfl_pred == keras_pred[:len(tfl_pred)]).mean()
print(f"\nTFLite matches Keras: {match*100:.2f}%")
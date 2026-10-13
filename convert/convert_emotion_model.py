"""
convert_emotion_model.py
Convert emotion model via frozen graph (fixes Keras 3 + BN Cast bug).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

BASE = os.path.expanduser("~/Signify/Signify_Model")
H5_PATH  = f"{BASE}/models/face_fer2025_hybrid_v11_8/emotion_hybrid_v11_8_seed42.h5"
TFLITE_PATH = f"{BASE}/convert/output/emotion_model.tflite"
DATA_PATH = f"{BASE}/landmarks/fer2025_v11"

IMG_SIZE = 48
NUM_LANDMARKS = 936

print("=" * 60)
print("😊 EMOTION MODEL CONVERSION (frozen graph)")
print("=" * 60)

# ---- load ----
print("\nLoading .h5...")
model = tf.keras.models.load_model(H5_PATH, compile=False)
print(f"✅ Inputs:  {[i.shape for i in model.inputs]}")
print(f"✅ Outputs: {[o.shape for o in model.outputs]}")

# ---- build concrete function with FIXED batch=1 ----
print("\nBuilding concrete function (batch=1)...")

@tf.function
def serve(img, lm):
    return model([img, lm], training=False)

concrete = serve.get_concrete_function(
    tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 1], tf.float32, name='image'),
    tf.TensorSpec([1, NUM_LANDMARKS], tf.float32, name='landmarks'),
)

# ---- freeze: variables → constants (this is the key fix) ----
print("Freezing variables to constants...")
frozen = convert_variables_to_constants_v2(concrete)

# ---- convert from frozen graph ----
print("Converting frozen graph → TFLite...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"\n✅ Saved: {TFLITE_PATH} ({size_mb:.2f} MB)")

# ---- verify ----
print("\nLoading test data...")
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test_orig = np.load(f"{DATA_PATH}/y_test.npy")

GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)
y_test4 = GROUP_MAP[y_test_orig]

N = 300
X_small = X_test[:N]
L_small = L_test[:N]
y_small = y_test4[:N]

print("Running Keras predictions...")
keras_probs = model.predict([X_small, L_small], verbose=0)
keras_labels = keras_probs.argmax(axis=1)
keras_acc = (keras_labels == y_small).mean()

print("Running TFLite predictions...")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
in_details = interp.get_input_details()
out_detail = interp.get_output_details()[0]

print(f"\nTFLite inputs:")
for d in in_details:
    print(f"  name={d['name']} shape={d['shape']} dtype={d['dtype']}")
print(f"TFLite output: shape={out_detail['shape']}")

img_idx = None
lm_idx = None
for d in in_details:
    nm = d['name'].lower()
    if 'image' in nm:
        img_idx = d['index']
    elif 'landmark' in nm:
        lm_idx = d['index']
if img_idx is None or lm_idx is None:
    img_idx, lm_idx = in_details[0]['index'], in_details[1]['index']

tflite_labels = []
tflite_probs_all = []
for i in range(N):
    interp.set_tensor(img_idx, X_small[i:i+1])
    interp.set_tensor(lm_idx,  L_small[i:i+1])
    interp.invoke()
    out = interp.get_tensor(out_detail['index'])[0]
    tflite_probs_all.append(out)
    tflite_labels.append(out.argmax())

tflite_labels = np.array(tflite_labels)
tflite_probs_all = np.array(tflite_probs_all)
tflite_acc = (tflite_labels == y_small).mean()
match_rate = (keras_labels == tflite_labels).mean()
max_prob_diff = np.abs(keras_probs - tflite_probs_all).max()

print("\n" + "=" * 60)
print("📊 VERIFICATION")
print("=" * 60)
print(f"Keras  accuracy (first {N}):  {keras_acc*100:.2f}%")
print(f"TFLite accuracy (first {N}):  {tflite_acc*100:.2f}%")
print(f"Prediction match rate:         {match_rate*100:.2f}%")
print(f"Max prob diff:                 {max_prob_diff:.5f}")

if match_rate >= 0.98 and max_prob_diff < 0.01:
    print("\n✅ Conversion is accurate")
else:
    print("\n⚠️  Some drift")
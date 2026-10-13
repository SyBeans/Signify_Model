"""
convert_sign_model.py
Convert sign_model_v4.h5 (BiLSTM, input (30, 252)) to TFLite.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

# ---- paths ----
BASE = os.path.expanduser("~/Signify/Signify_Model")
H5_PATH  = f"{BASE}/models/hand/sign_model_v4.h5"
TFLITE_PATH = f"{BASE}/convert/output/sign_model.tflite"

print("=" * 60)
print("🖐️  SIGN MODEL CONVERSION")
print("=" * 60)
print(f"Input:  {H5_PATH}")
print(f"Output: {TFLITE_PATH}")

if not os.path.exists(H5_PATH):
    raise FileNotFoundError(f"Model not found: {H5_PATH}")

# ---- load ----
print("\nLoading .h5...")
model = tf.keras.models.load_model(H5_PATH, compile=False)
print(f"✅ Loaded. Input:  {model.inputs[0].shape}")
print(f"✅ Loaded. Output: {model.outputs[0].shape}")

# ---- convert ----
print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No quantization — preserve fp32 accuracy
converter.optimizations = []
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"\n✅ Saved: {TFLITE_PATH} ({size_mb:.2f} MB)")

# ---- quick verify with test data ----
DATA_PATH = f"{BASE}/landmarks/hand"
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
y_test = np.load(f"{DATA_PATH}/y_test.npy")

N = 200  # verify on first 200 samples
X_small = X_test[:N]

# Keras predictions
keras_preds = model.predict(X_small, verbose=0)
keras_labels = keras_preds.argmax(axis=1)
keras_acc = (keras_labels == y_test[:N]).mean()

# TFLite predictions
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
in_detail = interp.get_input_details()
out_detail = interp.get_output_details()
print(f"\nTFLite input:  {in_detail[0]['shape']} {in_detail[0]['dtype']}")
print(f"TFLite output: {out_detail[0]['shape']}")

tflite_labels = []
for i in range(N):
    x = X_small[i:i+1]
    interp.set_tensor(in_detail[0]['index'], x)
    interp.invoke()
    out = interp.get_tensor(out_detail[0]['index'])[0]
    tflite_labels.append(out.argmax())

tflite_labels = np.array(tflite_labels)
tflite_acc = (tflite_labels == y_test[:N]).mean()
match_rate = (keras_labels == tflite_labels).mean()

print("\n" + "=" * 60)
print("📊 VERIFICATION")
print("=" * 60)
print(f"Keras  accuracy (first {N}):  {keras_acc*100:.2f}%")
print(f"TFLite accuracy (first {N}):  {tflite_acc*100:.2f}%")
print(f"Prediction match rate:         {match_rate*100:.2f}%")
print(f"Keras vs TFLite gap:           {(keras_acc - tflite_acc)*100:+.2f}%")

if match_rate >= 0.98:
    print("\n✅ Conversion is accurate — TFLite matches Keras")
else:
    print("\n⚠️  Some drift — investigate if gap > 2%")
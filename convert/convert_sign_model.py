"""
convert_sign_model.py
Convert sign_model_v4.h5 (BiLSTM) to TFLite via SavedModel.

Fix for Keras 3 + TF 2.16 LSTM conversion bug:
  1. Save as SavedModel first
  2. Convert from SavedModel (preserves graph structure)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

BASE = os.path.expanduser("~/Signify/Signify_Model")
H5_PATH  = f"{BASE}/models/hand/sign_model_v4.h5"
SAVEDMODEL_PATH = f"{BASE}/convert/output/sign_model_saved"
TFLITE_PATH = f"{BASE}/convert/output/sign_model.tflite"

print("=" * 60)
print("🖐️  SIGN MODEL CONVERSION (via SavedModel)")
print("=" * 60)

# ---- load ----
print("\nLoading .h5...")
model = tf.keras.models.load_model(H5_PATH, compile=False)
print(f"✅ Input:  {model.inputs[0].shape}")
print(f"✅ Output: {model.outputs[0].shape}")

# ---- step 1: save as SavedModel ----
print(f"\nSaving as SavedModel → {SAVEDMODEL_PATH}")
model.export(SAVEDMODEL_PATH)
print("✅ SavedModel ready")

# ---- step 2: convert from SavedModel ----
print("\nConverting SavedModel → TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_PATH)

# Use BOTH builtin and TF ops (needed for some LSTM ops)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"\n✅ Saved: {TFLITE_PATH} ({size_mb:.2f} MB)")

# ---- verify ----
DATA_PATH = f"{BASE}/landmarks/hand"
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
y_test = np.load(f"{DATA_PATH}/y_test.npy")

N = 200
X_small = X_test[:N]

keras_preds = model.predict(X_small, verbose=0)
keras_labels = keras_preds.argmax(axis=1)
keras_acc = (keras_labels == y_test[:N]).mean()

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

if match_rate >= 0.98:
    print("\n✅ Conversion is accurate")
else:
    print("\n⚠️  Some drift — investigate")
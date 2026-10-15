"""
convert_sign_model_unrolled.py
Rebuild the model with LSTM unroll=True (removes TensorList ops),
copy the trained weights, then convert with pure TFLITE_BUILTINS.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

BASE = os.path.expanduser("~/Signify/Signify_Model")
H5_PATH = f"{BASE}/models/hand/sign_model_v4.h5"
TFLITE_PATH = f"{BASE}/convert/output/sign_model_builtin.tflite"
DATA_PATH = f"{BASE}/landmarks/hand"

NUM_FRAMES = 30
NUM_FEATURES = 252
NUM_CLASSES = 105

print("=" * 60)
print("SIGN MODEL — UNROLLED CONVERSION (no Flex)")
print("=" * 60)

# ---- load original ----
print("\nLoading .h5...")
orig = keras.models.load_model(H5_PATH, compile=False)
print(f"✅ Loaded. Weights: {len(orig.get_weights())} tensors")

# ---- build parallel with unroll=True ----
print("\nBuilding unrolled model...")
inputs = layers.Input(shape=(NUM_FRAMES, NUM_FEATURES))
x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inputs)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, unroll=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, unroll=True))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

attn_scores = layers.Dense(1, activation='tanh')(x)
attn_weights = layers.Softmax(axis=1)(attn_scores)
weighted = layers.Multiply()([x, attn_weights])

x = layers.GlobalAveragePooling1D()(weighted)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

new_model = models.Model(inputs=inputs, outputs=outputs)

# ---- copy weights ----
print("Copying weights from original...")
try:
    new_model.set_weights(orig.get_weights())
    print("✅ Weights copied (by order)")
except Exception as e:
    print(f"❌ Order copy failed: {e}")
    print("Trying by layer name...")
    for layer in new_model.layers:
        try:
            orig_layer = orig.get_layer(layer.name)
            layer.set_weights(orig_layer.get_weights())
        except Exception as inner:
            print(f"   skip {layer.name}: {inner}")

# ---- verify weights work ----
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
y_test = np.load(f"{DATA_PATH}/y_test.npy")
N = 200

orig_preds = orig.predict(X_test[:N], verbose=0).argmax(axis=1)
new_preds = new_model.predict(X_test[:N], verbose=0).argmax(axis=1)
match = (orig_preds == new_preds).mean()
print(f"Original vs unrolled pred match: {match*100:.2f}%")

if match < 0.99:
    print("⚠️  Weight copy didn't fully work — predictions differ")
    print("Consider using by_name weight save/load instead")
else:
    print("✅ Weights preserved perfectly")

# ---- convert ----
print("\nBuilding concrete function...")

@tf.function
def serve(x):
    return new_model(x, training=False)

concrete = serve.get_concrete_function(
    tf.TensorSpec([1, NUM_FRAMES, NUM_FEATURES], tf.float32, name='input')
)

print("Freezing variables...")
frozen = convert_variables_to_constants_v2(concrete)

print("Converting...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_bytes = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_bytes)

size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"\n✅ Saved: {TFLITE_PATH} ({size_mb:.2f} MB)")

# ---- check ops ----
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
ops = [op['op_name'] for op in interp._get_ops_details()]
flex_ops = [o for o in ops if o.startswith('Flex') or o.startswith('flex')]

print("\n" + "=" * 60)
print("OPS CHECK")
print("=" * 60)
print(f"Total ops:   {len(ops)}")
print(f"Flex ops:    {len(flex_ops)}")
if flex_ops:
    print(f"⚠️  Still has Flex: {set(flex_ops)}")
else:
    print("✅ No Flex ops — pure TFLITE_BUILTINS!")

# ---- verify with interpreter ----
print("\nVerifying TFLite output...")
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']

tflite_preds = []
for i in range(N):
    interp.set_tensor(in_idx, X_test[i:i+1])
    interp.invoke()
    tflite_preds.append(interp.get_tensor(out_idx)[0].argmax())

tflite_preds = np.array(tflite_preds)
tflite_acc = (tflite_preds == y_test[:N]).mean()
keras_acc = (orig_preds == y_test[:N]).mean()

print(f"\nKeras accuracy:  {keras_acc*100:.2f}%")
print(f"TFLite accuracy: {tflite_acc*100:.2f}%")

if len(flex_ops) == 0 and tflite_acc >= keras_acc - 0.01:
    print("\n🎉 SUCCESS — pure builtins, no Flex, full accuracy")
else:
    print("\n⚠️  Check output above")
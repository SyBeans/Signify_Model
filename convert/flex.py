import tensorflow as tf
interp = tf.lite.Interpreter("convert/output/sign_model.tflite")
print([op['op_name'] for op in interp._get_ops_details()])
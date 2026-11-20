"""
test_webcam_v12_2.py
Webcam test for v12.2 (3-input: image + landmarks + 26 engineered features).

Matches v12.2 training:
  - Same 26 engineered features (identical to precompute_engineered_v13.py)
  - Same landmark normalization
  - Same crop extraction
  - Uniform post-softmax weights [1,1,1,1] (model trained balanced — no reweighting needed)

Load strategy:
  Rebuild architecture from code + load_weights() — because Keras 3 cannot
  deserialize the Lambda layers (lm_center, lm_maxd, lm_norm) in a fresh scope.

Usage:
  python test_webcam_v12_2.py
  python test_webcam_v12_2.py --model models/face_fer2025_hybrid_v12_2/emotion_v12_2_best.keras
  python test_webcam_v12_2.py --no-flip
"""
import argparse
import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers
from collections import deque, Counter

keras.config.enable_unsafe_deserialization()

# ============================================
# ARGUMENTS
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/face_fer2025_hybrid_v12_2/emotion_v12_2_best.keras",
                    help="Path to v12.2 .keras model")
parser.add_argument("--no-flip", action="store_true",
                    help="Don't mirror the webcam (matches training extraction more exactly)")
args = parser.parse_args()

# ============================================
# CONFIG
# ============================================
MODEL_PATH = args.model
IMG_SIZE = 48
NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2
ENGINEERED_DIM = 26
NUM_CLASSES = 4

SMOOTHING_LEN = 8
CONFIDENCE_THRESHOLD = 0.40

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]
CLASS_ICONS = {"Positive": "😊", "Negative": "😢", "Surprise": "😲", "Neutral": "😐"}
CLASS_COLORS = {
    "Positive": (0, 255, 0), "Negative": (0, 0, 255),
    "Surprise": (255, 255, 0), "Neutral": (200, 200, 200),
}
CLASS_PUNCTUATION = {"Positive": "!", "Negative": "!", "Surprise": "?", "Neutral": "."}

# ============================================
# REBUILD v12.2 ARCHITECTURE (must match training)
# ============================================
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
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
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

# ============================================
# ENGINEERED FEATURES (same as precompute_engineered_v13.py)
# ============================================
def engineered_features_numpy(L_936):
    """L_936: (1, 936) or (936,) raw landmarks → (1, 26) features.
    MUST match precompute_engineered_v13.py EXACTLY."""
    if L_936.ndim == 1:
        L_936 = L_936.reshape(1, -1)
    lm = L_936.reshape(-1, NUM_LANDMARKS, 2).astype(np.float32)

    center = lm.mean(axis=1, keepdims=True)
    c = lm - center
    max_d = np.linalg.norm(c, axis=-1).max(axis=1, keepdims=True)[..., None] + 1e-6
    n = c / max_d

    def dist(a, b):
        return np.linalg.norm(n[:, a] - n[:, b], axis=-1) + 1e-6
    def y(i):
        return n[:, i, 1]

    left_ear_num  = dist(159, 145);  left_ear_den = dist(33, 133)
    right_ear_num = dist(386, 374);  right_ear_den = dist(362, 263)
    left_ear  = left_ear_num  / left_ear_den
    right_ear = right_ear_num / right_ear_den
    mouth_h = dist(13, 14); mouth_w = dist(61, 291)
    mar = mouth_h / mouth_w

    mouth_center_y = (y(13) + y(14)) / 2.0
    corners_avg    = (y(61) + y(291)) / 2.0
    mouth_curve    = mouth_center_y - corners_avg

    lb_above = y(159) - y(105)
    rb_above = y(386) - y(334)
    face_aspect = dist(10, 152) / dist(234, 454)

    eye_avg_y    = (y(159) + y(386)) / 2.0
    nose_vs_eyes = y(1) - eye_avg_y

    feats = np.stack([
        left_ear, right_ear, mar, mouth_curve,
        dist(61, 133), dist(291, 362), dist(105, 159), dist(334, 386), dist(107, 336),
        lb_above, rb_above, face_aspect,
        dist(1, 13), dist(152, 14), dist(172, 397), dist(93, 323),
        nose_vs_eyes,
        np.abs(left_ear - right_ear),
        np.abs(y(61) - y(291)),
        np.abs(lb_above - rb_above),
        left_ear_den / right_ear_den,
        corners_avg - y(1),
        (dist(107, 1) + dist(336, 1)) / 2.0,
        mouth_w, mouth_h, left_ear_den,
    ], axis=-1).astype(np.float32)
    return feats   # (1, 26)

# ============================================
# LOAD MODEL (rebuild + load_weights — avoids Lambda deserialization issue)
# ============================================
print("=" * 60)
print(f"🔨 Building v12.2 architecture...")
model = build_v12_2()

print(f"📥 Loading weights from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n❌ Model not found: {MODEL_PATH}\n"
        f"   Train v12.2 first, or specify a different path with --model\n"
    )
model.load_weights(MODEL_PATH)
print(f"✅ Model loaded")
print(f"   Inputs: {[(i.name, i.shape) for i in model.inputs]}")
print(f"   Outputs: {[(o.name, o.shape) for o in model.outputs]}")
print(f"   Uniform weights [1,1,1,1] — model trained balanced")
if args.no_flip:
    print(f"   Frame flip: DISABLED (raw webcam)")
else:
    print(f"   Frame flip: ENABLED (mirror view)")
print("=" * 60)

# ============================================
# MEDIAPIPE
# ============================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ============================================
# CAMERA
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prediction_buffer = deque(maxlen=SMOOTHING_LEN)
current_class = "Waiting..."
confidence = 0.0
all_probs = np.zeros(4)

# ============================================
# EXTRACTION (matches training crop)
# ============================================
def extract_all(image_bgr, results):
    """Return (img_in, lm_in, eng_in, bbox) or (None, None, None, None)."""
    if not results.multi_face_landmarks:
        return None, None, None, None

    h, w = image_bgr.shape[:2]
    lm = results.multi_face_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    size = max(x_max - x_min, y_max - y_min) * 1.55

    y0 = max(0.0, cy - size * 0.65); y1 = min(1.0, cy + size * 0.35)
    x0 = max(0.0, cx - size * 0.50); x1 = min(1.0, cx + size * 0.50)
    px1, py1 = int(x0 * w), int(y0 * h)
    px2, py2 = int(x1 * w), int(y1 * h)
    if px2 <= px1 or py2 <= py1:
        return None, None, None, None

    crop = image_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return None, None, None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_in = (gray.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

    coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
    lm_vec = coords.flatten()
    lm_in = lm_vec.reshape(1, RAW_LANDMARK_DIM)

    eng_in = engineered_features_numpy(lm_vec)

    return img_in, lm_in, eng_in, (px1, py1, px2, py2)

def predict_single(img_in, lm_in, eng_in):
    """3-input single-pass — matches Flutter app after TFLite wrapper."""
    return model.predict([img_in, lm_in, eng_in], verbose=0)[0]

# ============================================
# LOOP
# ============================================
print("🎥 Webcam started. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not args.no_flip:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        fl = results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=fl,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    img_in, lm_in, eng_in, bbox = extract_all(frame, results)

    if img_in is not None:
        probs = predict_single(img_in, lm_in, eng_in)
        all_probs = probs

        top_prob = float(probs.max())
        if top_prob >= CONFIDENCE_THRESHOLD:
            pred = int(np.argmax(probs))
            confidence = top_prob * 100
            prediction_buffer.append(CLASS_NAMES[pred])
            current_class = Counter(prediction_buffer).most_common(1)[0][0]

        px1, py1, px2, py2 = bbox
        color = CLASS_COLORS.get(current_class, (255, 255, 255))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
    else:
        prediction_buffer.clear()
        current_class = "No face detected"
        confidence = 0.0
        all_probs = np.zeros(4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 190), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if current_class in CLASS_ICONS:
        icon = CLASS_ICONS[current_class]
        color = CLASS_COLORS[current_class]
        punct = CLASS_PUNCTUATION[current_class]
        cv2.putText(frame, f"{current_class.upper()} {icon} {punct}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for i, cname in enumerate(CLASS_NAMES):
            c_color = CLASS_COLORS[cname]
            bar_len = int(all_probs[i] * 200)
            cv2.rectangle(frame, (15, 110 + i * 18),
                          (15 + bar_len, 124 + i * 18), c_color, -1)
            cv2.putText(frame, f"{cname}: {all_probs[i]*100:.0f}%",
                        (230, 124 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    else:
        cv2.putText(frame, current_class, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Signify - v12.2 Emotion Test (3-input)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\n✅ Test complete!")
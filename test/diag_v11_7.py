"""
diag_v11_7.py
Feeds one webcam frame through the model AND one test sample.
Prints raw softmax, saves crop to disk for visual inspection.
"""
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

MODEL_PATH = "models/face_fer2025_hybrid_v11_7/emotion_hybrid_v11_7_seed42.h5"
DATA_PATH  = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v11")
IMG_SIZE = 48

CLASS_NAMES = ["Positive", "Negative", "Surprise", "Neutral"]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ---- Step 1: sanity check the model on a known test sample ----
X_test = np.load(f"{DATA_PATH}/X_test.npy").astype(np.float32)
L_test = np.load(f"{DATA_PATH}/L_test.npy").astype(np.float32)
y_test = np.load(f"{DATA_PATH}/y_test.npy")  # 7-class labels

GROUP_MAP = np.array([1, 1, 1, 0, 3, 1, 2], dtype=np.int32)
y_test4 = GROUP_MAP[y_test]

print("\n=== SANITY CHECK: 20 test samples ===")
for i in range(20):
    p = model.predict([X_test[i:i+1], L_test[i:i+1]], verbose=0)[0]
    pred = np.argmax(p)
    true = y_test4[i]
    mark = "✅" if pred == true else "❌"
    print(f"  {mark} true={CLASS_NAMES[true]:<10} pred={CLASS_NAMES[pred]:<10} "
          f"probs={np.round(p, 3)}")

# ---- Step 2: capture ONE webcam frame ----
print("\n=== WEBCAM CAPTURE ===")
print("Press SPACE to capture, Q to quit.")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    cv2.imshow("Press SPACE to capture, Q to quit", frame)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

    if k == ord(' '):
        print("\nCaptured frame. Processing...")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("❌ NO FACE DETECTED by MediaPipe. That's the issue.")
            continue

        lm = results.multi_face_landmarks[0]
        xs = [p.x for p in lm.landmark]; ys = [p.y for p in lm.landmark]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
        size = max(x_max-x_min, y_max-y_min) * 1.55
        print(f"  face bbox: w={x_max-x_min:.3f} h={y_max-y_min:.3f} "
              f"cx={cx:.3f} cy={cy:.3f} crop_size={size:.3f}")
        if size < 0.15 or size > 0.90:
            print("  ⚠️ face size outside training range [0.15, 0.90]")

        h, w = frame.shape[:2]
        y0 = max(0.0, cy - size * 0.65); y1 = min(1.0, cy + size * 0.35)
        x0 = max(0.0, cx - size * 0.50); x1 = min(1.0, cx + size * 0.50)
        px1, py1 = int(x0*w), int(y0*h); px2, py2 = int(x1*w), int(y1*h)
        crop = frame[py1:py2, px1:px2]
        print(f"  crop pixel size: {crop.shape}")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Save for visual inspection
        cv2.imwrite("diag_full_frame.jpg", frame)
        cv2.imwrite("diag_crop_raw.jpg", crop)
        cv2.imwrite("diag_crop_48x48.png", gray_small)
        print("  ✅ saved diag_full_frame.jpg / diag_crop_raw.jpg / diag_crop_48x48.png")

        img_in = (gray_small.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)
        coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
        lm_in = coords.flatten().reshape(1, 936)

        print(f"  img_in range: [{img_in.min():.3f}, {img_in.max():.3f}] mean={img_in.mean():.3f}")
        print(f"  lm_in range:  [{lm_in.min():.3f}, {lm_in.max():.3f}]")

        p1 = model.predict([img_in, lm_in], verbose=0)[0]
        img_flip = img_in[:, :, ::-1, :]
        p2 = model.predict([img_flip, lm_in], verbose=0)[0]
        probs = (p1 + p2) / 2.0

        print(f"\n  p1 (raw):       {np.round(p1, 4)}")
        print(f"  p2 (flip TTA):  {np.round(p2, 4)}")
        print(f"  averaged probs: {np.round(probs, 4)}")
        print(f"  ARGMAX: {CLASS_NAMES[int(np.argmax(probs))]}")

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("\nDone. Look at the 3 saved images.")
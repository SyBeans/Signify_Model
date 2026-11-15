"""
precompute_engineered.py
Run ONCE. Reads L_train/val/test.npy (raw 936 landmarks) and writes
E_train/val/test.npy (26 engineered features) next to them.

These 26 features are what the model consumes as its 3rd input in
train_hybrid_v12.py — computed here in NumPy so training stays fast
(no ~50 Lambda ops in the graph).
"""
import os
import numpy as np

DATA_PATH = os.path.expanduser("~/Signify/Signify_Model/landmarks/fer2025_v12_balanced")

NUM_LANDMARKS = 468
RAW_LANDMARK_DIM = NUM_LANDMARKS * 2   # 936


def engineered_features_numpy(L_936):
    """L_936: (N, 936) raw landmarks → (N, 26) features.
    Matches the op-order of the old v12 engineered() Lambdas exactly."""
    lm = L_936.reshape(-1, NUM_LANDMARKS, 2).astype(np.float32)

    # --- same normalization as v12 model ---
    center = lm.mean(axis=1, keepdims=True)
    c = lm - center
    max_d = np.linalg.norm(c, axis=-1).max(axis=1, keepdims=True)[..., None] + 1e-6
    n = c / max_d                                            # (N, 468, 2)

    def dist(a, b):
        return np.linalg.norm(n[:, a] - n[:, b], axis=-1) + 1e-6

    def y(i):
        return n[:, i, 1]

    # 1-2. Eye aspect ratios
    left_ear_num  = dist(159, 145);  left_ear_den = dist(33, 133)
    right_ear_num = dist(386, 374);  right_ear_den = dist(362, 263)
    left_ear  = left_ear_num  / left_ear_den
    right_ear = right_ear_num / right_ear_den

    # 3. Mouth aspect ratio
    mouth_h = dist(13, 14)
    mouth_w = dist(61, 291)
    mar     = mouth_h / mouth_w

    # 4. Mouth curve
    mouth_center_y = (y(13) + y(14)) / 2.0
    corners_avg    = (y(61) + y(291)) / 2.0
    mouth_curve    = mouth_center_y - corners_avg

    # 10-11. Brow heights
    lb_above = y(159) - y(105)
    rb_above = y(386) - y(334)

    # 12. Face aspect
    face_aspect = dist(10, 152) / dist(234, 454)

    # 17. Nose vs eyes
    eye_avg_y    = (y(159) + y(386)) / 2.0
    nose_vs_eyes = y(1) - eye_avg_y

    # Stack in the exact order the model's Concatenate expects
    feats = np.stack([
        left_ear,                               # 1
        right_ear,                              # 2
        mar,                                    # 3
        mouth_curve,                            # 4
        dist(61, 133),                          # 5
        dist(291, 362),                         # 6
        dist(105, 159),                         # 7
        dist(334, 386),                         # 8
        dist(107, 336),                         # 9
        lb_above,                               # 10
        rb_above,                               # 11
        face_aspect,                            # 12
        dist(1, 13),                            # 13
        dist(152, 14),                          # 14
        dist(172, 397),                         # 15
        dist(93, 323),                          # 16
        nose_vs_eyes,                           # 17
        np.abs(left_ear - right_ear),           # 18
        np.abs(y(61) - y(291)),                 # 19
        np.abs(lb_above - rb_above),            # 20
        left_ear_den / right_ear_den,           # 21
        corners_avg - y(1),                     # 22
        (dist(107, 1) + dist(336, 1)) / 2.0,    # 23
        mouth_w,                                # 24
        mouth_h,                                # 25
        left_ear_den,                           # 26
    ], axis=-1).astype(np.float32)

    return feats   # (N, 26)


if __name__ == "__main__":
    print(f"📂 Data path: {DATA_PATH}")
    for split in ("train", "val", "test"):
        L_file = f"{DATA_PATH}/L_{split}.npy"
        E_file = f"{DATA_PATH}/E_{split}.npy"
        print(f"\n▶ {split}")
        L = np.load(L_file).astype(np.float32)
        E = engineered_features_numpy(L)
        np.save(E_file, E)
        print(f"  L shape: {L.shape}  →  E shape: {E.shape}")
        print(f"  E mean (first 4 dims): {E.mean(0)[:4]}")
        print(f"  E std  (first 4 dims): {E.std(0)[:4]}")
        print(f"  ✅ wrote {E_file}")

    print("\n🎉 Done. Now run: python train_hybrid_v12.py")
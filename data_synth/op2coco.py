'''OpenPose-137 → COCO-WholeBody-133 lossless mapping.

OpenPose JSON has body_25 + face_70 + hand_left_21 + hand_right_21 = 137 keypoints.
COCO-WholeBody-133 has body_17 + foot_6 + face_68 + hand_left_21 + hand_right_21.

OpenPose face has 70 points (drop OP[68], OP[69] = pupils → 68 face points).
Hands and feet map directly. Body needs index reorder body_25 → COCO-17 + COCO-feet.
'''
import os
import json
import numpy as np

# COCO-17 body indices ← OpenPose body_25 indices
# COCO order: 0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear,
#             5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow,
#             9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip,
#             13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle
OP_BODY25_TO_COCO17 = [
    0,   # 0 nose         ← OP 0  nose
    16,  # 1 L-eye        ← OP 16 left eye
    15,  # 2 R-eye        ← OP 15 right eye
    18,  # 3 L-ear        ← OP 18 left ear
    17,  # 4 R-ear        ← OP 17 right ear
    5,   # 5 L-shoulder   ← OP 5  left shoulder
    2,   # 6 R-shoulder   ← OP 2  right shoulder
    6,   # 7 L-elbow      ← OP 6  left elbow
    3,   # 8 R-elbow      ← OP 3  right elbow
    7,   # 9 L-wrist      ← OP 7  left wrist
    4,   # 10 R-wrist     ← OP 4  right wrist
    12,  # 11 L-hip       ← OP 12 left hip
    9,   # 12 R-hip       ← OP 9  right hip
    13,  # 13 L-knee      ← OP 13 left knee
    10,  # 14 R-knee      ← OP 10 right knee
    14,  # 15 L-ankle     ← OP 14 left ankle
    11,  # 16 R-ankle     ← OP 11 right ankle
]

# COCO-feet indices 17..22 ← OP body_25 indices 19..24
# Order: L-bigtoe, L-smtoe, L-heel, R-bigtoe, R-smtoe, R-heel
OP_BODY25_TO_COCO_FOOT = [19, 20, 21, 22, 23, 24]


def op_json_to_coco133(json_path: str) -> np.ndarray:
    # Load one OpenPose frame JSON, return (133, 3) float32 in COCO-WholeBody layout.
    # Returns zeros if no person detected. Confidence preserved per joint.
    with open(json_path, "r") as f:
        data = json.load(f)
        
    out = np.zeros((133, 3), dtype=np.float32)
    if not data.get('people'): return out
    p = data['people'][0]
    op_body = np.array(p['pose_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
    op_face = np.array(p['face_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
    op_lh = np.array(p['hand_left_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
    op_rh = np.array(p['hand_right_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
    
    if op_body.shape[0] >= 25:
        out[0:17]  = op_body[OP_BODY25_TO_COCO17]
        out[17:23] = op_body[OP_BODY25_TO_COCO_FOOT]
    if op_face.shape[0] >= 68: out[23:91] = op_face[0:68]  # drop pupils 68, 69
    if op_lh.shape[0]   >= 21: out[91:112] = op_lh
    if op_rh.shape[0]   >= 21: out[112:133] = op_rh
    out[..., 2] = np.clip(out[..., 2], 0.0, 1.0)
    return out


def load_clip_pose(json_dir: str) -> np.ndarray:
    '''Load every *_keypoints.json in `json_dir`, sort by frame index in filename, return (T, 133, 3).

    Filename pattern: <prefix>_<framenum>_keypoints.json (zero-padded). Sort lexicographically (= numeric)
    since zero-padded indices guarantee correct ordering.
    '''
    if not os.path.isdir(json_dir): return np.zeros((0, 133, 3), dtype=np.float32)
    files = sorted(f for f in os.listdir(json_dir) if f.endswith('_keypoints.json'))
    if not files: return np.zeros((0, 133, 3), dtype=np.float32)
    arr = np.stack([op_json_to_coco133(os.path.join(json_dir, f)) for f in files], axis=0)
    return arr.astype(np.float32)
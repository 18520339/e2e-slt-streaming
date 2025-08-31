import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def normalize_keypoints(keypoints: np.ndarray, width: int=444, height: int=444) -> np.ndarray:
    '''
    CoSign keypoints normalize: Group-specific centralization (subtract root per group).
    
    Groups: Body (root: mid-shoulders), Left hand (wrist), Right hand (wrist), Face (nose).
    
    Args:
        keypoints (np.ndarray): (frames, 133, 3) x,y,confidence
        width (int): Image width for global scaling.
        height (int): Image height for global scaling.

    Returns:
        np.ndarray: Normalized keypoints.
    '''
    if keypoints.shape[1] != 133: 
        raise ValueError(f'Invalid pose shape: {keypoints.shape}, expected (frames, 133, 3)')
    
    # Split into groups
    body_kpts = keypoints[:, BODY_IDS]
    left_hand_kpts = keypoints[:, LEFT_HAND_IDS]
    right_hand_kpts = keypoints[:, RIGHT_HAND_IDS]
    mouth_kpts = keypoints[:, MOUTH_IDS]
    face_kpts = keypoints[:, FACE_IDS]

    # Root keypoints
    root_body = (body_kpts[:, 3] + body_kpts[:, 4]) / 2  # Mid-shoulder
    root_left = left_hand_kpts[:, 0]  # Wrist index 0 in hand
    root_right = right_hand_kpts[:, 0]  # Wrist index 0 in hand
    root_nose = face_kpts[:, -1]  # Nose as root

    # Group lengths
    shoulder_length = np.linalg.norm(body_kpts[:, 3] - body_kpts[:, 4], axis=-1)[:, np.newaxis, np.newaxis] + 1e-6  # Avoid div by 0
    left_hand_length = np.linalg.norm(left_hand_kpts[:, 0] - left_hand_kpts[:, 9], axis=-1)[:, np.newaxis, np.newaxis] + 1e-6
    right_hand_length = np.linalg.norm(right_hand_kpts[:, 0] - right_hand_kpts[:, 9], axis=-1)[:, np.newaxis, np.newaxis] + 1e-6
    mouth_length = np.linalg.norm(mouth_kpts[:, 0] - mouth_kpts[:, 4], axis=-1)[:, np.newaxis, np.newaxis] + 1e-6
    face_length = np.linalg.norm(face_kpts[:, 0] - face_kpts[:, 8], axis=-1)[:, np.newaxis, np.newaxis] + 1e-6

    # Normalize keypoints
    norm_kpts = np.zeros((keypoints.shape[0], 77, 3), dtype=np.float32)
    norm_kpts[:, 0:9] = (body_kpts - root_body[:, np.newaxis]) / shoulder_length * 3 # 0 to 8 (9 points)
    norm_kpts[:, 9:30] = (left_hand_kpts - root_left[:, np.newaxis]) / left_hand_length * 2 # 9 to 29 (21 points)
    norm_kpts[:, 30:51] = (right_hand_kpts - root_right[:, np.newaxis]) / right_hand_length * 2 # 30 to 50 (21 points)
    norm_kpts[:, 51:59] = (mouth_kpts - root_nose[:, np.newaxis]) / mouth_length # 51 to 58 (8 points)
    norm_kpts[:, 59:77] = (face_kpts - root_nose[:, np.newaxis]) / face_length * 2 # 59 to 76 (18 points)

    # Global scaling (assume video width=444, height=444; adjust if needed)
    if width is not None and height is not None:
        norm_kpts[:, :, 0] /= width  # x
        norm_kpts[:, :, 1] /= height  # y
    return norm_kpts


def threshold_confidence(keypoints: np.ndarray) -> np.ndarray:
    '''
    Set low-conf keypoints to 0 (x,y=0, conf=0 if < threshold)
    Input/Output: keypoints (frames, kpts, 3)
    '''
    low_conf_mask = keypoints[:, :, 2] < CONF_THRESHOLD
    keypoints[low_conf_mask] = 0  # Zero out x,y,conf
    print(f'Applied confidence threshold: {np.sum(low_conf_mask)} keypoints zeroed')
    return keypoints


def subsample_frames(keypoints: np.ndarray, target_fps: float) -> np.ndarray:
    '''
    Subsample to lower FPS if needed (e.g., from 12.5 to 5 fps)
    Input: keypoints (frames, kpts, 3), target_fps (float)
    Output: subsampled keypoints
    '''
    if target_fps >= FPS: return keypoints
    step = int(FPS / target_fps)
    subsampled = keypoints[::step, :, :]
    print(f'Subsampled from {keypoints.shape[0]} to {subsampled.shape[0]} frames (target FPS: {target_fps})')
    return subsampled


if __name__ == '__main__':
    dummy_keypoints = np.random.rand(100, 133, 3)  # frames, kpts, (x,y,conf)
    dummy_keypoints[:, :, 2] = np.random.uniform(0.4, 1.0, (100, 133))  # Random conf
    normalized = normalize_keypoints(dummy_keypoints)
    thresholded = threshold_confidence(normalized)
    subsampled = subsample_frames(thresholded, target_fps=6.0)
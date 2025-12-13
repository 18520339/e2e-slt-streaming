# Pose augmentation approach from best solution of Kaggle's GISLR competition
# https://www.kaggle.com/code/hoyso48/1st-place-solution-training?scriptVersionId=128283887&cellId=9
import sys
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def interp1d_(keypoints: np.ndarray, target_len: int, method: str = 'random', rng: np.random.Generator | None = None) -> np.ndarray:
    '''Resize along the time axis (axis 0) using spline interpolation for pose sequences.

    This mimics tf.image.resize used on a (T, K, C) tensor by treating T as the resized dimension and keeping K,C unchanged.
    Args:
        keypoints: (T, K, C)
        target_len: desired T'
        method: 'random' | 'bilinear' | 'bicubic' | 'nearest'
    '''
    keypoints = np.asarray(keypoints)
    if keypoints.ndim != 3: raise ValueError(f'Expected keypoints with shape (T, K, C), got {keypoints.shape}')

    target_len = int(max(1, target_len))
    src_len = int(keypoints.shape[0])
    if src_len <= 0: return np.zeros((target_len, *keypoints.shape[1:]), dtype=keypoints.dtype)
    if src_len <= 1: return np.repeat(keypoints[:1], target_len, axis=0)
    if src_len == target_len: return keypoints

    if rng is None: rng = np.random.default_rng()
    if method == 'random':
        p = float(rng.random())
        if p < 0.33: order = 1  # bilinear-like
        else: order = 3 if rng.random() < 0.5 else 0  # bicubic-like or nearest
    else:
        method = method.lower()
        if method in {'nearest', 'nearest_neighbor'}: order = 0
        elif method in {'bilinear', 'linear'}: order = 1
        elif method in {'bicubic', 'cubic'}: order = 3
        else: raise ValueError(f'Unknown method: {method}')

    zoom_factor = (target_len / src_len, 1.0, 1.0)
    x = zoom(x, zoom=zoom_factor, order=order, mode='nearest', prefilter=(order > 1))

    # ndimage.zoom can be off by 1 due to rounding; enforce exact length
    if keypoints.shape[0] > target_len: keypoints = keypoints[:target_len]
    elif keypoints.shape[0] < target_len: 
        pad_len = target_len - keypoints.shape[0]
        # If you want to pad with the last frame instead of zeros, uncomment below:
        # pad = np.repeat(keypoints[-1:, :, :], pad_len, axis=0) if keypoints.shape[0] > 0 \
        # else np.zeros((pad_len, keypoints.shape[1], keypoints.shape[2]), dtype=keypoints.dtype)
        # keypoints = np.concatenate([keypoints, pad], axis=0)
        keypoints = np.pad(keypoints, ((0, pad_len), (0, 0), (0, 0)))
    return keypoints


def resample(keypoints: np.ndarray, rate: tuple[float, float] = (0.8, 1.2), rng: np.random.Generator | None = None) -> np.ndarray:
	if rng is None: rng = np.random.default_rng()
	r = float(rng.uniform(rate[0], rate[1]))
	length = int(keypoints.shape[0])
	new_size = int(max(1, round(r * length)))
	return interp1d_(keypoints, new_size, method='random', rng=rng)


def flip_lr(keypoints: np.ndarray) -> np.ndarray:
    '''Flip horizontally and swap left/right for COCO-WholeBody 133 keypoints.

    Notes:
    - Mirrors x around the image centerline (x -> width - x).
    - Swaps the left/right hand groups to preserve semantic indices.
    - Swaps a small set of body left/right pairs (eyes/shoulders/elbows/wrists).
    '''
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3: raise ValueError(f'Expected (T, K, 3), got {keypoints.shape}')
    keypoints = np.asarray(keypoints).copy()
    conf = keypoints[..., 2]
    valid = conf > 0
    keypoints[..., 0] = np.where(valid, WIDTH - keypoints[..., 0], keypoints[..., 0]) # Horizontal flip in pixel coordinates

    # Swap hands as groups (left 91-111, right 112-132)
    left_hand = keypoints[:, LEFT_HAND_IDS, :].copy()
    right_hand = keypoints[:, RIGHT_HAND_IDS, :].copy()
    keypoints[:, LEFT_HAND_IDS, :] = right_hand
    keypoints[:, RIGHT_HAND_IDS, :] = left_hand

    # Swap common body left/right semantic pairs.
    body_lr_pairs = [
        (1, 2),  # left/right eye
        (5, 6),  # left/right shoulder
        (7, 8),  # left/right elbow
        (9, 10),  # left/right wrist
    ]
    for li, ri in body_lr_pairs:
        tmp = keypoints[:, li, :].copy()
        keypoints[:, li, :] = keypoints[:, ri, :]
        keypoints[:, ri, :] = tmp

    keypoints[:, FACE_IDS] = keypoints[:, FACE_IDS][:, ::-1].copy() # Face contour 23..39 (17 points)
    keypoints[:, MOUTH_IDS] = keypoints[:, MOUTH_IDS][:, ::-1].copy() # Inner mouth 83..90 (8 points)
    return keypoints


def spatial_random_affine(
	keypoints: np.ndarray,
	scale: tuple[float, float] | None = (0.8, 1.2),
	shear: tuple[float, float] | None = (-0.15, 0.15),
	shift: tuple[float, float] | None = (-0.1, 0.1),
	degree: tuple[float, float] | None = (-30, 30),
	rng: np.random.Generator | None = None,
) -> np.ndarray:
    '''Apply random affine to x/y (leaves conf intact).

    Designed for raw pixel coordinates (0..WIDTH, 0..HEIGHT).
    Only transforms keypoints with conf > 0 to avoid moving padded/invalid points.
    '''
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3: raise ValueError(f'Expected (T, K, 3), got {keypoints.shape}')
    if rng is None: rng = np.random.default_rng()
    keypoints = np.asarray(keypoints).copy()
    xy = keypoints[..., :2]
    conf = keypoints[..., 2]
    valid = conf > 0
    center = np.array([WIDTH / 2.0, HEIGHT / 2.0], dtype=np.float32)

    if scale is not None: # Scale around center
        s = float(rng.uniform(scale[0], scale[1]))
        # xy[valid] = xy[valid] * s # Incorrect: scales away from origin (0,0)
        xy[valid] = (xy[valid] - center) * s + center

    if shear is not None: # Shear around center
        shear_x = shear_y = float(rng.uniform(shear[0], shear[1]))
        if rng.random() < 0.5: shear_x = 0.0
        else: shear_y = 0.0
        shear_mat = np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float32)
        # xy[valid] = xy[valid] @ shear_mat # Incorrect: shears away from origin (0,0)
        xy[valid] = (xy[valid] - center) @ shear_mat + center
        center = center + np.array([shear_y, shear_x], dtype=np.float32)

    if degree is not None: # Rotate around center
        deg = float(rng.uniform(degree[0], degree[1]))
        rad = deg * np.pi / 180.0 
        cos_v, sin_v = float(np.cos(rad)), float(np.sin(rad))
        rot = np.array([[cos_v, sin_v], [-sin_v, cos_v]], dtype=np.float32)
        xy[valid] = (xy[valid] - center) @ rot + center

    if shift is not None: # Shift in pixels (fraction of WIDTH/HEIGHT)
        sh = float(rng.uniform(shift[0], shift[1]))
        xy[valid, 0] = xy[valid, 0] + sh * WIDTH
        xy[valid, 1] = xy[valid, 1] + sh * HEIGHT

    keypoints[..., :2] = xy
    return keypoints


def temporal_crop(keypoints: np.ndarray, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
	if rng is None: rng = np.random.default_rng()
	l = int(keypoints.shape[0])
    # offset = int(rng.uniform(0, np.clip(l - length, 1, length)))
 
	if l <= length: return keypoints
	max_offset = max(1, l - length)
	offset = int(rng.integers(0, max_offset))
	return keypoints[offset : offset + length]


def temporal_mask(
	keypoints: np.ndarray,
	size: tuple[float, float] = (0.2, 0.4),
	mask_value: float = float('nan'),
	rng: np.random.Generator | None = None,
) -> np.ndarray:
	if rng is None: rng = np.random.default_rng()
	keypoints = np.asarray(keypoints).copy()
	l = int(keypoints.shape[0])
	if l <= 1: return keypoints

	frac = float(rng.uniform(size[0], size[1]))
	mask_size = int(max(1, round(l * frac)))
    # offset = int(rng.uniform(0, np.clip(l - mask_size, 1, l)))
 
	max_offset = max(1, l - mask_size)
	offset = int(rng.integers(0, max_offset))
	keypoints[offset : offset + mask_size, :, :] = mask_value
	return keypoints


def spatial_mask(
	keypoints: np.ndarray,
	size: tuple[float, float] = (0.2, 0.4),
	mask_value: float = float('nan'),
	rng: np.random.Generator | None = None,
) -> np.ndarray:
	if rng is None: rng = np.random.default_rng()
	keypoints = np.asarray(keypoints).copy()

	offset_y = float(rng.uniform(0.0, HEIGHT))
	offset_x = float(rng.uniform(0.0, WIDTH))
	frac = float(rng.uniform(size[0], size[1]))
	box = frac * min(WIDTH, HEIGHT)

	mask_x = (offset_x < keypoints[..., 0]) & (keypoints[..., 0] < offset_x + box)
	mask_y = (offset_y < keypoints[..., 1]) & (keypoints[..., 1] < offset_y + box)
	mask = mask_x & mask_y
	keypoints[mask, :] = mask_value
	return keypoints


def augment_keypoints(
	keypoints: np.ndarray,
	always: bool = False,
	max_len: int | None = None,
	rng: np.random.Generator | None = None,
) -> np.ndarray:
	'''NumPy port of the provided TF augmentation pipeline.

	WARNING: If you use this for DVC event localization, `resample` and `temporal_crop`
	will change the temporal alignment between frames and subtitle timings.
	'''
	if rng is None: rng = np.random.default_rng()
	if (rng.random() < 0.8) or always: keypoints = resample(keypoints, (0.5, 1.5), rng=rng)
	if (rng.random() < 0.5) or always: keypoints = flip_lr(keypoints)
 
	if max_len is not None: keypoints = temporal_crop(keypoints, max_len, rng=rng)
	if (rng.random() < 0.75) or always: keypoints = spatial_random_affine(keypoints, rng=rng)
	if (rng.random() < 0.5) or always: keypoints = temporal_mask(keypoints, rng=rng)
	if (rng.random() < 0.5) or always: keypoints = spatial_mask(keypoints, rng=rng)
	return keypoints


def augment_dvc_window(keypoints: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
	'''Augmentations safe for window-level DVC (do not change sequence length).

	Applies: flip_lr, spatial_random_affine, temporal_mask, spatial_mask.
	'''
	if rng is None: rng = np.random.default_rng()
	if rng.random() < 0.5: keypoints = flip_lr(keypoints)
	if rng.random() < 0.75: keypoints = spatial_random_affine(keypoints, rng=rng)
	if rng.random() < 0.5: keypoints = temporal_mask(keypoints, rng=rng)
	if rng.random() < 0.5: keypoints = spatial_mask(keypoints, rng=rng)
	return keypoints
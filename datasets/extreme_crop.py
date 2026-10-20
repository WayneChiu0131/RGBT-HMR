"""Extreme-crop augmentation ported from TokenHMR.

Keypoint layout assumed: SPIN J49 (25 OpenPose + 24 GT-extra), which matches
TokenHMR's `25 + i` indexing exactly (see constants.JOINT_NAMES).

Input keypoints are in **full-image pixel coordinates** with (N, 3) layout
where column -1 holds visibility/confidence. The function returns a new
(center, scale_spin) pair, where `scale_spin = max(bbox_w, bbox_h) / 200`
follows SPIN's convention.
"""
from typing import Tuple

import numpy as np


_SPIN_SCALE_UNIT = 200.0
_MIN_BBOX_PX = 4


def _get_bbox(keypoints_2d: np.ndarray, rescale: float = 1.2):
    """Return (center_xy, bbox_wh) from visible keypoints."""
    valid = keypoints_2d[:, -1] > 0
    valid_kp = keypoints_2d[valid][:, :-1]
    center = 0.5 * (valid_kp.max(axis=0) + valid_kp.min(axis=0))
    bbox_size = (valid_kp.max(axis=0) - valid_kp.min(axis=0))
    return center, bbox_size


def _apply_mask(keypoints_2d: np.ndarray, drop_idx) -> np.ndarray:
    kp = keypoints_2d.copy()
    kp[drop_idx, :] = 0
    return kp


def crop_to_hips(cx, cy, w, h, keypoints_2d):
    drop = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24,
            25 + 0, 25 + 1, 25 + 4, 25 + 5]
    kp = _apply_mask(keypoints_2d, drop)
    if kp[:, -1].sum() > 1:
        c, s = _get_bbox(kp)
        return c[0], c[1], 1.1 * s[0], 1.1 * s[1]
    return cx, cy, w, h


def crop_to_shoulders(cx, cy, w, h, keypoints_2d):
    drop = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            19, 20, 21, 22, 23, 24] + \
           [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    kp = _apply_mask(keypoints_2d, drop)
    if kp[:, -1].sum() > 1:
        c, s = _get_bbox(kp)
        return c[0], c[1], 1.2 * s[0], 1.2 * s[1]
    return cx, cy, w, h


def crop_to_head(cx, cy, w, h, keypoints_2d):
    drop = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            19, 20, 21, 22, 23, 24] + \
           [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    kp = _apply_mask(keypoints_2d, drop)
    if kp[:, -1].sum() > 1:
        c, s = _get_bbox(kp)
        return c[0], c[1], 1.3 * s[0], 1.3 * s[1]
    return cx, cy, w, h


def full_body(keypoints_2d: np.ndarray) -> bool:
    body_kp_op = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_kp = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    visible = np.maximum(
        keypoints_2d[body_kp, -1], keypoints_2d[body_kp_op, -1]
    ) > 0
    return int(visible.sum()) == len(body_kp)


def upper_body(keypoints_2d: np.ndarray) -> bool:
    lower_kp_op = [10, 11, 13, 14]
    lower_kp = [25 + i for i in [1, 0, 4, 5]]
    upper_kp_op = [0, 1, 15, 16, 17, 18]
    upper_kp = [25 + i for i in [8, 9, 12, 13, 17, 18]]
    no_lower = (keypoints_2d[lower_kp + lower_kp_op, -1] > 0).sum() == 0
    enough_upper = (keypoints_2d[upper_kp + upper_kp_op, -1] > 0).sum() >= 2
    return bool(no_lower and enough_upper)


def _extreme_crop_level0(cx, cy, w, h, keypoints_2d):
    """3-way branch: hips / shoulders / head, only when full or upper body visible."""
    p = np.random.rand()
    if full_body(keypoints_2d):
        if p < 0.7:
            cx, cy, w, h = crop_to_hips(cx, cy, w, h, keypoints_2d)
        elif p < 0.9:
            cx, cy, w, h = crop_to_shoulders(cx, cy, w, h, keypoints_2d)
        else:
            cx, cy, w, h = crop_to_head(cx, cy, w, h, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            cx, cy, w, h = crop_to_shoulders(cx, cy, w, h, keypoints_2d)
        else:
            cx, cy, w, h = crop_to_head(cx, cy, w, h, keypoints_2d)
    new_side = max(w, h)
    return cx, cy, new_side, new_side


def apply_extreme_crop(
    center: np.ndarray,
    scale_spin: float,
    keypoints_fullimg: np.ndarray,
    level: int = 0,
) -> Tuple[np.ndarray, float]:
    """SPIN-facing wrapper.

    Args:
        center: (2,) np.ndarray, pixel coords.
        scale_spin: SPIN-convention scalar (= max(w,h) / 200).
        keypoints_fullimg: (N, 3) np.ndarray in raw pixel coords, N >= 49.
        level: 0 → level-0 branching (hips/shoulders/head only).

    Returns:
        (new_center, new_scale_spin). If keypoint signal is too weak or the
        new bbox shrinks below `_MIN_BBOX_PX`, the original (center, scale) is
        returned unchanged.
    """
    if level != 0:
        return center, scale_spin

    cx, cy = float(center[0]), float(center[1])
    side = float(scale_spin) * _SPIN_SCALE_UNIT
    w = h = side

    new_cx, new_cy, new_w, new_h = _extreme_crop_level0(cx, cy, w, h, keypoints_fullimg)

    new_side = max(new_w, new_h)
    if new_side < _MIN_BBOX_PX:
        return center, scale_spin

    new_center = np.array([new_cx, new_cy], dtype=center.dtype)
    new_scale = float(new_side / _SPIN_SCALE_UNIT)
    return new_center, new_scale

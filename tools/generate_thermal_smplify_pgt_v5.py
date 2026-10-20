#!/usr/bin/env python3
"""v5: Generate SPIN-style SMPLify pseudo-GT NPZs for thermal pose datasets.

v5 additions over v4-safe:
  - CameraHMR as initializer (--init_model camerahmr, default).
    CameraHMR replaces local HMR for pose/shape/camera initialization.
    ThermalCamSMPLify_v4 remains unchanged as the optimizer.
  - Focal-length semantics split:
      camerahmr_condition_focal  = sqrt(W^2+H^2) per image  (CameraHMR conditioning)
      smplify_crop_focal         = 5000 px (fixed)           (SMPLify crop projection)
    These two are never mixed.
  - LLVIP: paired RGB for CameraHMR init (--llvip_camerahmr_input paired_rgb).
    Thermal fallback when RGB missing; no silent local-HMR fallback.
  - OTP2: thermal image, single-channel duplicated to 3ch.
  - New NPZ fields: init_cam_wp, init_cam_t_crop, camerahmr_condition_focal,
    crop_focal_length, crop_camera_center, has_full_camera, camera_model_code.
    cam_t = final crop-perspective translation (camera_model_code=0).
    has_full_camera=0 for all samples (no metric full-image camera).
  - selected_init_type source codes:
      0 = local thermal HMR  1 = local LLVIP RGB HMR
      2 = CameraHMR thermal  3 = CameraHMR LLVIP paired RGB

v4 features retained: paired-RGB init for local HMR, OTP2 instance-level
curation, tiered reproj thresholds, freeze_betas_for_all, ThermalCamSMPLify_v4.

The saved NPZ does not contain key "S"; samples without an accepted SMPLify fit
remain usable as 2D-only supervision.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import yaml
from torchvision.transforms import Normalize
from tqdm import tqdm

import config
import constants
from utils.geometry import (
    bbox_info_from_center_scale,
    cam_crop2full,
    cam_crop2perspective_crop,
    cam_full2crop_perspective,
    perspective_projection,
    project_points_full_to_crop,
)
from utils.imutils import crop, transform

try:
    from config import apply_paths_config
except Exception:  # pragma: no cover - older forks may not expose it
    apply_paths_config = None

# ── CameraHMR image normalization (ImageNet) ─────────────────────────────────
CAMERAHMR_IMAGE_MEAN = [0.485, 0.456, 0.406]
CAMERAHMR_IMAGE_STD  = [0.229, 0.224, 0.225]
CAMERAHMR_CROP_RES   = 256   # CameraHMR always uses 256×256 square crop

# source_code values stored in selected_init_type NPZ field
SRC_LOCAL_THERMAL  = 0
SRC_LOCAL_RGB      = 1
SRC_CAMERAHMR_THERMAL = 2
SRC_CAMERAHMR_RGB     = 3

# LLVIP-Pose 17-keypoint order from the annotation JSON:
# 0 nose, 1 head_bottom, 2 head_top, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
#
# SPIN GT24 does not have a native head_bottom keypoint. We therefore use
# LLVIP head_bottom as a neck/base-of-head proxy at GT24 index 12, and map
# LLVIP head_top directly to GT24 head_top at index 13.
LLVIP17_TO_SPIN_GT24 = {
    16: 0,
    14: 1,
    12: 2,
    11: 3,
    13: 4,
    15: 5,
    10: 6,
    8: 7,
    6: 8,
    5: 9,
    7: 10,
    9: 11,
    1: 12,
    2: 13,
    0: 19,
    3: 22,
    4: 23,
}

OTP2_COCO17_TO_SPIN_GT24 = {
    16: 0,
    14: 1,
    12: 2,
    11: 3,
    13: 4,
    15: 5,
    10: 6,
    8: 7,
    6: 8,
    5: 9,
    7: 10,
    9: 11,
    0: 19,
    1: 20,
    2: 21,
    3: 22,
    4: 23,
}

TRUNC_NONE = 0
TRUNC_LOWER = 1
TRUNC_UPPER = 2
TRUNC_LEFT = 3
TRUNC_RIGHT = 4
TRUNC_SEVERE = 5

TRUNCATION_NAMES = {
    TRUNC_NONE: "none",
    TRUNC_LOWER: "lower",
    TRUNC_UPPER: "upper",
    TRUNC_LEFT: "left",
    TRUNC_RIGHT: "right",
    TRUNC_SEVERE: "severe",
}

PGT_QUALITY_REJECTED = 0
PGT_QUALITY_WEAK = 1
PGT_QUALITY_FULL = 2

CURATION_REMOVE  = 0
CURATION_WEAK    = 1
CURATION_FULL    = 2
CURATION_UNKNOWN = 3

CURATION_NAMES = {
    CURATION_REMOVE:  "remove",
    CURATION_WEAK:    "weak",
    CURATION_FULL:    "full",
    CURATION_UNKNOWN: "unknown",
}

CURATION_DIR_ALIASES = {
    "remove": CURATION_REMOVE,
    "weak":   CURATION_WEAK,
    "weal":   CURATION_WEAK,   # typo alias
    "full":   CURATION_FULL,
}

GT24_LOWER = (0, 1, 2, 3, 4, 5)
GT24_UPPER = (6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23)
GT24_LEFT_SIDE = (3, 4, 5, 9, 10, 11)
GT24_RIGHT_SIDE = (0, 1, 2, 6, 7, 8)

OUTPUT_NAMES = {
    ("llvip", "train"): "llvip_train_smplify_pgt.npz",
    ("llvip", "test"): "llvip_test_smplify_pgt.npz",
    ("otp2", "train"): "otp2_train_smplify_pgt.npz",
    ("otp2", "test"): "otp2_test_smplify_pgt.npz",
    ("otp2", "val"): "otp2_val_smplify_pgt.npz",
}

REQUIRED_NPZ_KEYS = (
    "imgname",
    "center",
    "scale",
    "part",
    "openpose",
    "pose",
    "shape",
    "has_smpl",
    "valid_fit",
    "reprojection_error",
    "cam_t",
    "bbox",
)

OPTIONAL_NPZ_KEYS = (
    "init_reprojection_error",
    "improvement_ratio",
    "fitting_loss_total",
    "fitting_loss_2d",
    "fitting_loss_pose_init",
    "fitting_loss_beta_init",
    "fitting_loss_verts_init",
    "fitting_loss_angle",
    "pose_change",
    "verts_init_dist",
    "truncation_type",
    "pgt_quality",
    "smpl_weight",
    "pose_weight",
    "shape_weight",
    "cam_weight",
    "used_init_fallback",
    "proj_height_ratio",
    "proj_width_ratio",
    "proj_area_ratio",
    "proj_center_offset",
    "global_orient_change",
    "body_pose_change",
    "upright_ok",
    "reject_low_joints",
    "reject_reproj",
    "reject_no_improvement",
    "reject_depth",
    "reject_beta",
    "reject_pose_change",
    "reject_verts_init",
    "reject_bbox_sanity",
    "reject_upright",
    "reject_global_orient_change",
    "reject_truncation",
    "reject_nan",
    "reject_read",
    "curation_type",
    "downgraded_full_to_weak",
    # v4 new scalar fields (shape (n,))
    "overlap_ratio",
    "bbox_height_ratio",
    "bbox_width_ratio",
    "instance_id",
    "instance_curation_reason",
    "selected_init_type",
    # v4 multi-dim fields NOT in scalar check: cam_t_full (n,3), focal_length_npz (n,), img_center_npz (n,2)
    # v5 scalar fields
    "camerahmr_condition_focal",
    "crop_focal_length",
    "has_full_camera",
    "camera_model_code",
)

# v4: multi-dim optional arrays – validated separately (shape != (n,))
V4_MULTIDIM_OPTIONAL_KEYS = ("cam_t_full", "focal_length_npz", "img_center_npz")

# v5: additional multi-dim optional arrays
V5_MULTIDIM_OPTIONAL_KEYS = (
    "init_cam_wp",        # (n, 3)
    "init_cam_t_crop",    # (n, 3)
    "crop_camera_center", # (n, 2)
)


@dataclass
class Sample:
    dataset: str
    split: str
    imgname: str
    img_path: str
    center: np.ndarray
    scale: float
    part: np.ndarray
    openpose: np.ndarray
    bbox: np.ndarray
    img_w: int
    img_h: int
    valid_joint_count: int
    fit_eligible: bool
    truncation_type: int
    pgt_quality: int
    smpl_weight: float
    pose_weight: float
    shape_weight: float
    cam_weight: float
    curation_type: int = CURATION_FULL
    # v4 fields
    overlap_ratio: float = 0.0
    bbox_height_ratio: float = 0.0
    bbox_width_ratio: float = 0.0
    instance_id: int = 0
    instance_curation_reason: int = 0  # 0=full_clear,1=small_bbox,2=crowded,3=lower_missing,4=torso_missing,5=trunc_downgrade


@dataclass
class HMRInitOutput:
    """Unified output from any HMR initializer (local or CameraHMR)."""
    pose_aa:          torch.Tensor   # [B, 72]  axis-angle
    betas:            torch.Tensor   # [B, 10]
    cam_wp:           torch.Tensor   # [B, 3]   weak perspective [s, tx, ty]
    cam_t_crop:       torch.Tensor   # [B, 3]   crop perspective translation
    condition_focal:  torch.Tensor   # [B]      focal used to condition model
    valid_mask:       torch.Tensor   # [B] bool  False = image load fail / NaN
    source_code:      torch.Tensor   # [B] uint8 SRC_* constants above


# v4 instance curation reason codes
ICURE_FULL_CLEAR      = 0
ICURE_SMALL_BBOX      = 1
ICURE_CROWDED_OVERLAP = 2
ICURE_LOWER_MISSING   = 3
ICURE_TORSO_MISSING   = 4
ICURE_TRUNC_DOWNGRADE = 5


def normalize_relpath(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def rel_to_root(path: str, root: str) -> str:
    try:
        return normalize_relpath(os.path.relpath(path, root))
    except ValueError:
        return normalize_relpath(path)


def resolve_llvip_rgb_image(thermal_path: str, rgb_root: Optional[str] = None) -> Optional[str]:
    """Return the paired LLVIP RGB image path for a given thermal image path.

    Default strategy: replace '/infrared/' with '/visible/' in the path.
    If rgb_root is given, look for the same filename under that root.
    """
    if rgb_root is not None:
        basename = os.path.basename(thermal_path)
        # LLVIP has train/test sub-dirs inside each root
        for sub in ("train", "test", ""):
            candidate = os.path.join(rgb_root, sub, basename) if sub else os.path.join(rgb_root, basename)
            if os.path.isfile(candidate):
                return candidate
        return None
    # Auto-derive: swap infrared ↔ visible
    rgb_path = thermal_path.replace("/infrared/", "/visible/")
    if rgb_path != thermal_path and os.path.isfile(rgb_path):
        return rgb_path
    return None


def compute_instance_overlaps(bboxes: List[np.ndarray]) -> List[float]:
    """Compute per-instance max overlap ratio (intersection / own_area) against siblings."""
    n = len(bboxes)
    if n <= 1:
        return [0.0] * n
    overlaps: List[float] = []
    for i, bi in enumerate(bboxes):
        x1i, y1i = float(bi[0]), float(bi[1])
        x2i, y2i = float(bi[0] + bi[2]), float(bi[1] + bi[3])
        area_i = max(float(bi[2] * bi[3]), 1e-6)
        max_ov = 0.0
        for j, bj in enumerate(bboxes):
            if i == j:
                continue
            ix1 = max(x1i, float(bj[0]))
            iy1 = max(y1i, float(bj[1]))
            ix2 = min(x2i, float(bj[0] + bj[2]))
            iy2 = min(y2i, float(bj[1] + bj[3]))
            if ix2 > ix1 and iy2 > iy1:
                max_ov = max(max_ov, (ix2 - ix1) * (iy2 - iy1) / area_i)
        overlaps.append(max_ov)
    return overlaps


def resolve_llvip_annotation(root: str, split: str) -> str:
    candidates = [
        os.path.join(root, "annotations", f"{split}_llvip.json"),
        os.path.join(root, f"{split}_llvip.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"LLVIP annotation not found for split={split}: {candidates}")


def resolve_llvip_image(root: str, split: str, file_name: str) -> Optional[str]:
    file_name = normalize_relpath(file_name)
    candidates = [
        os.path.join(root, file_name),
        os.path.join(root, "infrared", split, file_name),
        os.path.join(root, split, file_name),
        os.path.join(root, "images", split, file_name),
        os.path.join(root, "images", file_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def image_size(path: str) -> Optional[Tuple[int, int]]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(w), int(h)


def load_rgb(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = img[:, :, ::-1]
    return img.astype(np.float32)


def bbox_to_center_scale(bbox: Sequence[float], bbox_scale_factor: float) -> Tuple[np.ndarray, float]:
    x, y, w, h = [float(v) for v in bbox]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    scale = float(max(w, h) / 200.0 * bbox_scale_factor)
    return center, scale


def convert_17_to_spin_part(
    keypoints17: np.ndarray,
    mapping: Dict[int, int],
    is_llvip: bool,
) -> np.ndarray:
    """Map dataset-specific 17-keypoint annotations into SPIN GT24 space.

    For LLVIP, keypoints 1 and 2 are ``head_bottom`` and ``head_top`` rather
    than COCO ``left_eye`` and ``right_eye``. The existing LLVIP mapping uses
    ``head_bottom -> GT24[12]`` as a neck/base-of-head proxy and
    ``head_top -> GT24[13]``.
    """
    part = np.zeros((24, 3), dtype=np.float32)
    for src_idx, dst_idx in mapping.items():
        x, y, conf = keypoints17[src_idx]
        if conf > 0:
            part[dst_idx] = np.array([x, y, conf], dtype=np.float32)

    if is_llvip and part[12, 2] > 0:
        # LLVIP GT24[12] comes from head_bottom, a proxy for neck/base-of-head.
        part[12, 2] *= 0.7

    derive_midpoint(part, 14, 3, 2)   # pelvis = midpoint(left_hip, right_hip)
    derive_midpoint(part, 15, 9, 8)   # thorax = midpoint(left_shoulder, right_shoulder)
    return part


def derive_midpoint(part: np.ndarray, dst: int, a: int, b: int) -> None:
    if part[a, 2] > 0 and part[b, 2] > 0:
        part[dst, :2] = 0.5 * (part[a, :2] + part[b, :2])
        part[dst, 2] = min(float(part[a, 2]), float(part[b, 2]))


def apply_thermal_joint_reliability(part: np.ndarray, dataset_name: str, args: argparse.Namespace) -> np.ndarray:
    """Down-weight sparse thermal joints before SMPLify uses confidence as loss weight."""
    if getattr(args, "disable_thermal_joint_reweight", False):
        return part.astype(np.float32, copy=False)

    out = part.astype(np.float32, copy=True)
    conf = out[:, 2]
    multipliers = np.ones((24,), dtype=np.float32)

    # Extremities are frequently blurred or partially truncated in thermal data.
    multipliers[[0, 5, 6, 11]] *= 0.75  # ankles, wrists
    # Hip and neck annotations are comparatively ambiguous in both thermal sets.
    multipliers[[2, 3]] *= 0.70
    multipliers[12] *= 0.60
    # Midpoints are synthetic joints, not directly annotated.
    multipliers[[14, 15]] *= 0.65
    # Facial landmarks on thermal imagery can be weak, especially ears.
    multipliers[[20, 21, 22, 23]] *= 0.85
    if dataset_name == "llvip":
        # LLVIP head annotations are nose/head_bottom/head_top/ears, not COCO eyes.
        # The mapped head_top, nose, and ear proxies should not dominate fitting.
        multipliers[[13, 19, 22, 23]] *= 0.90

    visible = conf > 0
    conf[visible] *= multipliers[visible]
    conf[conf < float(args.thermal_conf_thresh)] = 0.0
    out[:, 2] = conf
    return out


def count_effective_keypoints(kps49: np.ndarray, args: argparse.Namespace) -> int:
    return int((kps49[:, 2] > float(args.thermal_conf_thresh)).sum())


def _visible_count(part: np.ndarray, joint_ids: Sequence[int], conf_thresh: float) -> int:
    return int(sum(1 for joint_id in joint_ids if part[joint_id, 2] > conf_thresh))


def detect_truncation(
    bbox: Sequence[float],
    img_w: int,
    img_h: int,
    part: np.ndarray,
    args: argparse.Namespace,
) -> int:
    x, y, w, h = [float(v) for v in bbox]
    margin = float(args.trunc_margin_px)
    left_touch = x <= margin
    right_touch = x + w >= float(img_w) - margin
    top_touch = y <= margin
    bottom_touch = y + h >= float(img_h) - margin

    conf_thresh = float(args.thermal_conf_thresh)
    total_visible = int((part[:, 2] > conf_thresh).sum())
    if total_visible < int(args.severe_trunc_min_joints):
        return TRUNC_SEVERE

    lower_count = _visible_count(part, GT24_LOWER, conf_thresh)
    upper_count = _visible_count(part, GT24_UPPER, conf_thresh)
    left_count = _visible_count(part, GT24_LEFT_SIDE, conf_thresh)
    right_count = _visible_count(part, GT24_RIGHT_SIDE, conf_thresh)

    directions: List[int] = []
    if bottom_touch and lower_count < int(args.min_fullbody_lower_joints):
        directions.append(TRUNC_LOWER)
    if top_touch and upper_count < int(args.min_side_joints):
        directions.append(TRUNC_UPPER)
    if left_touch and left_count < int(args.min_side_joints):
        directions.append(TRUNC_LEFT)
    if right_touch and right_count < int(args.min_side_joints):
        directions.append(TRUNC_RIGHT)

    if len(directions) >= 2:
        return TRUNC_SEVERE
    if len(directions) == 1:
        return directions[0]
    return TRUNC_NONE


def pgt_strategy_for_truncation(truncation_type: int, args: argparse.Namespace) -> Tuple[int, float, float, float, float]:
    if truncation_type == TRUNC_SEVERE:
        return PGT_QUALITY_REJECTED, 0.0, 0.0, 0.0, 0.0
    if truncation_type == TRUNC_NONE:
        return (
            PGT_QUALITY_REJECTED,
            float(args.fullbody_smpl_weight),
            float(args.fullbody_pose_weight),
            float(args.fullbody_shape_weight),
            float(args.thermal_cam_weight),
        )
    return (
        PGT_QUALITY_REJECTED,
        float(args.truncated_smpl_weight),
        float(args.truncated_pose_weight),
        float(args.truncated_shape_weight),
        float(args.thermal_cam_weight),
    )


def build_keypoints49(openpose: np.ndarray, part: np.ndarray) -> np.ndarray:
    return np.concatenate([openpose, part], axis=0).astype(np.float32)


def keypoints_full_to_crop(kps49: np.ndarray, center: np.ndarray, scale: float, img_res: int) -> np.ndarray:
    out = kps49.copy().astype(np.float32)
    for idx in range(out.shape[0]):
        if out[idx, 2] > 0:
            out[idx, :2] = transform(out[idx, :2] + 1.0, center, scale, [img_res, img_res], rot=0)
    return out


def parse_llvip_split(args: argparse.Namespace, split: str) -> List[Sample]:
    ann_path = resolve_llvip_annotation(args.llvip_root, split)
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    image_cache: Dict[int, Optional[Tuple[str, int, int]]] = {}
    samples: List[Sample] = []
    skipped_no_kp = 0
    skipped_missing_image = 0
    skipped_bad_bbox = 0

    for ann in tqdm(data.get("annotations", []), desc=f"llvip/{split} parse"):
        if int(ann.get("num_keypoints", 0)) == 0:
            skipped_no_kp += 1
            continue
        keypoints = ann.get("keypoints", [])
        if len(keypoints) != 17 * 3:
            continue
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            skipped_bad_bbox += 1
            continue

        image_id = ann.get("image_id")
        image_info = images.get(image_id)
        if image_info is None:
            skipped_missing_image += 1
            continue

        if image_id not in image_cache:
            img_path = resolve_llvip_image(args.llvip_root, split, str(image_info.get("file_name", "")))
            if img_path is None:
                image_cache[image_id] = None
            else:
                size = image_size(img_path)
                image_cache[image_id] = (img_path, size[0], size[1]) if size is not None else None
        cached = image_cache[image_id]
        if cached is None:
            skipped_missing_image += 1
            continue
        img_path, img_w, img_h = cached

        kp17 = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
        kp17[:, 2] = np.clip(kp17[:, 2] / 2.0, 0.0, 1.0)
        part = convert_17_to_spin_part(kp17, LLVIP17_TO_SPIN_GT24, is_llvip=True)
        part = apply_thermal_joint_reliability(part, "llvip", args)
        openpose = np.zeros((25, 3), dtype=np.float32)
        center, scale = bbox_to_center_scale(bbox, args.bbox_scale_factor)
        valid_joint_count = count_effective_keypoints(build_keypoints49(openpose, part), args)
        truncation_type = detect_truncation(bbox, img_w, img_h, part, args)
        pgt_quality, smpl_weight, pose_weight, shape_weight, cam_weight = pgt_strategy_for_truncation(
            truncation_type, args
        )
        fit_eligible = valid_joint_count >= args.min_valid_joints
        if args.reject_severe_truncation and truncation_type == TRUNC_SEVERE:
            fit_eligible = False

        samples.append(
            Sample(
                dataset="llvip",
                split=split,
                imgname=rel_to_root(img_path, args.llvip_root),
                img_path=img_path,
                center=center,
                scale=scale,
                part=part,
                openpose=openpose,
                bbox=np.asarray(bbox, dtype=np.float32),
                img_w=img_w,
                img_h=img_h,
                valid_joint_count=valid_joint_count,
                fit_eligible=fit_eligible,
                truncation_type=truncation_type,
                pgt_quality=pgt_quality,
                smpl_weight=smpl_weight,
                pose_weight=pose_weight,
                shape_weight=shape_weight,
                cam_weight=cam_weight,
            )
        )

    print(
        f"[llvip/{split}] samples={len(samples)} skipped_no_keypoints={skipped_no_kp} "
        f"skipped_missing_image={skipped_missing_image} skipped_bad_bbox={skipped_bad_bbox}"
    )
    return samples


def build_stem_to_image(images_dir: str) -> Dict[str, str]:
    stem_to_image: Dict[str, str] = {}
    if not os.path.isdir(images_dir):
        return stem_to_image
    for name in os.listdir(images_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            stem_to_image.setdefault(os.path.splitext(name)[0], name)
    return stem_to_image


def parse_otp2_line(line: str) -> Optional[np.ndarray]:
    if not line.strip():
        return None
    tokens = line.strip().split()
    if len(tokens) != 1 + 4 + 17 * 3:
        return None
    try:
        return np.asarray([float(v) for v in tokens], dtype=np.float32)
    except ValueError:
        return None


COCO_FACE_IDS  = (0, 1, 2, 3, 4)
COCO_TORSO_IDS = (5, 6, 11, 12)
COCO_BODY_IDS  = tuple(range(5, 17))
COCO_LOWER_IDS = (11, 12, 13, 14, 15, 16)

GT24_TORSO_FOR_CURATION = (2, 3, 8, 9, 12, 14, 15)


def coco_visible_count(kp17: np.ndarray, ids: tuple, conf_thresh: float) -> int:
    return int(sum(kp17[i, 2] > conf_thresh for i in ids))


def is_face_only_person(kp17: np.ndarray, args: argparse.Namespace) -> bool:
    th = float(args.thermal_conf_thresh)
    face_count  = coco_visible_count(kp17, COCO_FACE_IDS,  th)
    torso_count = coco_visible_count(kp17, COCO_TORSO_IDS, th)
    body_count  = coco_visible_count(kp17, COCO_BODY_IDS,  th)
    lower_count = coco_visible_count(kp17, COCO_LOWER_IDS, th)
    return (face_count >= 2 and torso_count <= 1 and lower_count <= 1 and body_count <= 3)


def build_otp2_curation_map(args: argparse.Namespace, split: str) -> dict:
    if not getattr(args, "otp2_use_curation_folders", False):
        return {}
    split_root = Path(args.otp2_root) / split
    curation_map: dict = {}
    conflicts = []
    for dirname, ctype in CURATION_DIR_ALIASES.items():
        folder = split_root / dirname
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".txt"}:
                continue
            stem = path.stem
            if stem in curation_map and curation_map[stem] != ctype:
                conflicts.append((stem, CURATION_NAMES[curation_map[stem]], CURATION_NAMES[ctype]))
            curation_map[stem] = ctype
    if conflicts:
        raise ValueError(f"Curation folder conflicts (stem appears in multiple folders): {conflicts[:10]}")
    counts = {name: sum(1 for v in curation_map.values() if v == ct) for ct, name in CURATION_NAMES.items()}
    print(
        f"[otp2/{split}] curation_map: full={counts['full']}, weak={counts['weak']}, "
        f"remove={counts['remove']}, unknown={counts['unknown']}, total={len(curation_map)}"
    )
    return curation_map


def parse_otp2_split(args: argparse.Namespace, split: str) -> List[Sample]:
    split_root = os.path.join(args.otp2_root, split)
    images_dir = os.path.join(split_root, "images")
    labels_dir = os.path.join(split_root, "labels")
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"OTP2 split dirs not found: images={images_dir}, labels={labels_dir}")

    exclude_stems: set = set()
    exclude_list_path = getattr(args, "otp2_exclude_list", None)
    if exclude_list_path and os.path.isfile(exclude_list_path):
        with open(exclude_list_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                stem = _line.strip()
                if stem:
                    exclude_stems.add(stem)
        print(f"[otp2/{split}] loaded {len(exclude_stems)} excluded stems from {exclude_list_path}")

    stem_to_image = build_stem_to_image(images_dir)
    curation_map = build_otp2_curation_map(args, split)
    use_curation = bool(curation_map)
    # v4 instance-level curation flag
    v4_instance = getattr(args, "v4_instance_level_curation", True)
    samples: List[Sample] = []
    skipped_missing_image    = 0
    skipped_bad_line         = 0
    skipped_bad_bbox         = 0
    skipped_excluded         = 0
    skipped_curated_remove   = 0
    skipped_unknown_curation = 0
    skipped_weak_2d_only     = 0
    skipped_face_only_person = 0
    v4_downgraded_full_to_weak = 0

    label_paths = sorted(Path(labels_dir).glob("*.txt"))
    for label_path in tqdm(label_paths, desc=f"otp2/{split} parse"):
        if label_path.stem in exclude_stems:
            skipped_excluded += 1
            continue
        image_name = stem_to_image.get(label_path.stem)
        if image_name is None:
            skipped_missing_image += 1
            continue
        img_path = os.path.join(images_dir, image_name)
        size = image_size(img_path)
        if size is None:
            skipped_missing_image += 1
            continue
        img_w, img_h = size

        # --- per-image curation gate ---
        if use_curation:
            manual_curation = curation_map.get(label_path.stem, CURATION_UNKNOWN)
            if manual_curation == CURATION_UNKNOWN:
                policy = getattr(args, "otp2_unknown_curation_policy", "skip")
                if policy == "skip":
                    skipped_unknown_curation += 1
                    continue
                elif policy == "weak":
                    manual_curation = CURATION_WEAK
                elif policy == "full":
                    manual_curation = CURATION_FULL
            if manual_curation == CURATION_REMOVE:
                skipped_curated_remove += 1
                continue
        else:
            manual_curation = CURATION_FULL

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # --- v4: parse all valid instances first to compute per-image overlap ---
        img_instances: List[Tuple] = []  # (row, bbox, kp17)
        for line in lines:
            row = parse_otp2_line(line)
            if row is None:
                skipped_bad_line += 1
                continue
            cx = float(row[1] * img_w)
            cy = float(row[2] * img_h)
            bw = float(row[3] * img_w)
            bh = float(row[4] * img_h)
            if bw <= 0 or bh <= 0:
                skipped_bad_bbox += 1
                continue
            bbox = np.array([cx - bw * 0.5, cy - bh * 0.5, bw, bh], dtype=np.float32)
            kp17 = row[5:].reshape(17, 3).copy()
            kp17[:, 0] *= img_w
            kp17[:, 1] *= img_h
            kp17[:, 2] = np.clip(kp17[:, 2] / 2.0, 0.0, 1.0)
            img_instances.append((row, bbox, kp17))

        # Compute per-instance max overlap ratio
        instance_bboxes = [inst[1] for inst in img_instances]
        overlap_ratios = compute_instance_overlaps(instance_bboxes)

        for inst_id, ((row, bbox, kp17), overlap_ratio) in enumerate(zip(img_instances, overlap_ratios)):
            # face-only filter for weak curations
            if (use_curation and manual_curation == CURATION_WEAK
                    and getattr(args, "drop_face_only_persons_in_weak", True)):
                if is_face_only_person(kp17, args):
                    skipped_face_only_person += 1
                    continue

            bw, bh = float(bbox[2]), float(bbox[3])
            bbox_h_ratio = bh / max(img_h, 1)
            bbox_w_ratio = bw / max(img_w, 1)
            center, scale = bbox_to_center_scale(bbox, args.bbox_scale_factor)
            part = convert_17_to_spin_part(kp17, OTP2_COCO17_TO_SPIN_GT24, is_llvip=False)
            part = apply_thermal_joint_reliability(part, "otp2", args)
            openpose = np.zeros((25, 3), dtype=np.float32)
            valid_joint_count = count_effective_keypoints(build_keypoints49(openpose, part), args)
            truncation_type = detect_truncation(bbox, img_w, img_h, part, args)

            # --- v4 instance-level curation for curated_full ---
            effective_curation = manual_curation
            icure_reason = ICURE_FULL_CLEAR
            if v4_instance and use_curation and manual_curation == CURATION_FULL:
                conf_thresh = float(args.thermal_conf_thresh)
                lower_count = _visible_count(part, GT24_LOWER, conf_thresh)
                torso_count_v4 = _visible_count(part, GT24_TORSO_FOR_CURATION, conf_thresh)
                min_h_ratio = float(getattr(args, "instance_min_bbox_height_ratio", 0.08))
                overlap_thresh = float(getattr(args, "instance_overlap_thresh", 0.20))
                min_lower = int(getattr(args, "instance_min_lower_joints", 2))
                min_torso = int(getattr(args, "instance_min_torso_joints", 2))

                if bbox_h_ratio < min_h_ratio:
                    effective_curation = CURATION_WEAK
                    icure_reason = ICURE_SMALL_BBOX
                elif overlap_ratio > overlap_thresh:
                    effective_curation = CURATION_WEAK
                    icure_reason = ICURE_CROWDED_OVERLAP
                elif lower_count < min_lower:
                    effective_curation = CURATION_WEAK
                    icure_reason = ICURE_LOWER_MISSING
                elif torso_count_v4 < min_torso:
                    effective_curation = CURATION_WEAK
                    icure_reason = ICURE_TORSO_MISSING
                elif truncation_type == TRUNC_SEVERE:
                    effective_curation = CURATION_WEAK
                    icure_reason = ICURE_TRUNC_DOWNGRADE

                if effective_curation == CURATION_WEAK:
                    v4_downgraded_full_to_weak += 1

            # --- eligibility and weight assignment based on effective curation type ---
            if use_curation:
                torso_count_gt24 = _visible_count(part, GT24_TORSO_FOR_CURATION, float(args.thermal_conf_thresh))

                if effective_curation == CURATION_FULL:
                    fit_eligible = (
                        valid_joint_count >= args.min_valid_joints_full
                        and torso_count_gt24 >= args.min_torso_joints_full
                    )
                    init_smpl_weight  = float(args.fullbody_smpl_weight)
                    init_pose_weight  = float(args.fullbody_pose_weight)
                    init_shape_weight = float(args.fullbody_shape_weight)
                    init_cam_weight   = 0.0
                elif effective_curation == CURATION_WEAK:
                    if getattr(args, "otp2_curation_mode", "weighted") == "full_only":
                        fit_eligible      = False
                        init_smpl_weight  = 0.0
                        init_pose_weight  = 0.0
                        init_shape_weight = 0.0
                        init_cam_weight   = 0.0
                        skipped_weak_2d_only += 1
                    else:
                        fit_eligible = (
                            valid_joint_count >= args.min_valid_joints_weak
                            and torso_count_gt24 >= args.min_torso_joints_weak
                        )
                        init_smpl_weight  = float(args.weak_smpl_weight)
                        init_pose_weight  = float(args.weak_pose_weight)
                        init_shape_weight = 0.0
                        init_cam_weight   = 0.0
                else:  # UNKNOWN passed through
                    fit_eligible = (
                        valid_joint_count >= args.min_valid_joints_weak
                        and torso_count_gt24 >= args.min_torso_joints_weak
                    )
                    init_smpl_weight  = float(args.weak_smpl_weight)
                    init_pose_weight  = float(args.weak_pose_weight)
                    init_shape_weight = 0.0
                    init_cam_weight   = 0.0

                if args.reject_severe_truncation and truncation_type == TRUNC_SEVERE:
                    fit_eligible = False
                pgt_quality = PGT_QUALITY_REJECTED
                smpl_weight  = init_smpl_weight
                pose_weight  = init_pose_weight
                shape_weight = init_shape_weight
                cam_weight   = init_cam_weight
            else:
                # original path: no curation
                pgt_quality, smpl_weight, pose_weight, shape_weight, cam_weight = pgt_strategy_for_truncation(
                    truncation_type, args
                )
                fit_eligible = valid_joint_count >= args.min_valid_joints
                if args.reject_severe_truncation and truncation_type == TRUNC_SEVERE:
                    fit_eligible = False

            samples.append(
                Sample(
                    dataset="otp2",
                    split=split,
                    imgname=rel_to_root(img_path, args.otp2_root),
                    img_path=img_path,
                    center=center,
                    scale=scale,
                    part=part,
                    openpose=openpose,
                    bbox=bbox,
                    img_w=img_w,
                    img_h=img_h,
                    valid_joint_count=valid_joint_count,
                    fit_eligible=fit_eligible,
                    truncation_type=truncation_type,
                    pgt_quality=pgt_quality,
                    smpl_weight=smpl_weight,
                    pose_weight=pose_weight,
                    shape_weight=shape_weight,
                    cam_weight=cam_weight,
                    curation_type=effective_curation,
                    overlap_ratio=float(overlap_ratio),
                    bbox_height_ratio=float(bbox_h_ratio),
                    bbox_width_ratio=float(bbox_w_ratio),
                    instance_id=inst_id,
                    instance_curation_reason=icure_reason,
                )
            )

    print(
        f"[otp2/{split}] samples={len(samples)} skipped_excluded={skipped_excluded} "
        f"skipped_missing_image={skipped_missing_image} "
        f"skipped_bad_line={skipped_bad_line} skipped_bad_bbox={skipped_bad_bbox} "
        f"skipped_curated_remove={skipped_curated_remove} "
        f"skipped_unknown_curation={skipped_unknown_curation} "
        f"skipped_weak_2d_only={skipped_weak_2d_only} "
        f"skipped_face_only_person={skipped_face_only_person} "
        f"v4_instance_downgraded_to_weak={v4_downgraded_full_to_weak}"
    )
    return samples


def split_values(split: str, dataset: str = "llvip") -> List[str]:
    if split != "all":
        return [split]
    if dataset == "otp2":
        return ["train", "test", "val"]
    return ["train", "test"]


def dataset_values(dataset: str) -> List[str]:
    return ["llvip", "otp2"] if dataset == "both" else [dataset]


def output_path(out_dir: str, dataset: str, split: str) -> str:
    return os.path.join(out_dir, OUTPUT_NAMES[(dataset, split)])


def rotmat_to_axis_angle_cv2(rotmat: torch.Tensor) -> torch.Tensor:
    rot_np = rotmat.detach().cpu().reshape(-1, 3, 3).numpy().astype(np.float64)
    aa = np.zeros((rot_np.shape[0], 3), dtype=np.float32)
    for idx, mat in enumerate(rot_np):
        rvec, _ = cv2.Rodrigues(mat)
        aa[idx] = rvec.reshape(3)
    return torch.from_numpy(aa).to(rotmat.device, dtype=rotmat.dtype)


def load_hmr_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from models.hmr import hmr

    infer_hmr_model_args(args)

    hrnet_pretrained = args.hrnet_pretrained
    if hrnet_pretrained is None:
        hrnet_pretrained = getattr(config, "HRNET_PRETRAINED", None)

    model = hmr(
        config.SMPL_MEAN_PARAMS,
        pretrained=False,
        backbone=args.backbone,
        hrnet_pretrained=hrnet_pretrained if args.backbone == "hrnet_w48" else None,
        img_res=args.img_res,
        strict_hrnet=args.strict_hrnet,
        hrnet_backbone_input_height=args.hrnet_backbone_input_height,
        hrnet_backbone_input_width=args.hrnet_backbone_input_width,
        hrnet_center_crop_input=args.hrnet_center_crop_input,
        modality_embed_dim=args.modality_embed_dim,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{args.checkpoint}' into backbone={args.backbone}, "
            f"img_res={args.img_res}, modality_embed_dim={args.modality_embed_dim}. "
            f"Use --backbone/--modality_embed_dim matching the checkpoint, or let the script auto-infer them. "
            f"Original error: {exc}"
        ) from exc
    print(f"[model] checkpoint={args.checkpoint}")
    print(
        f"[model] backbone={args.backbone} img_res={args.img_res} "
        f"modality_embed_dim={args.modality_embed_dim}"
    )
    print(f"[model] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing[:5]:
        print(f"[model] missing first 5: {missing[:5]}")
    if unexpected[:5]:
        print(f"[model] unexpected first 5: {unexpected[:5]}")
    return model.to(device).eval()


def load_camerahmr_model(args: argparse.Namespace, device: torch.device):
    """Load CameraHMR body regressor only (no detector, no HumanFoV)."""
    camerahmr_root = args.camerahmr_root
    ckpt_path = args.camerahmr_checkpoint

    if not os.path.isdir(camerahmr_root):
        raise FileNotFoundError(f"CameraHMR repo not found: {camerahmr_root}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"CameraHMR checkpoint not found: {ckpt_path}")

    # Insert CameraHMR root into sys.path so `core.*` imports resolve.
    # SPIN_infrared has no `core/` package so there is no collision.
    if camerahmr_root not in sys.path:
        sys.path.insert(0, camerahmr_root)

    from core.camerahmr_model import CameraHMR

    model = CameraHMR(model_type="smpl")

    # Load checkpoint — CameraHMR uses pytorch-lightning format
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[camerahmr] checkpoint={ckpt_path}")
    print(f"[camerahmr] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing[:5]:
        print(f"[camerahmr] missing first 5: {missing[:5]}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def prepare_camerahmr_item(
    image_rgb: np.ndarray,
    center: np.ndarray,
    scale: float,
    cam_int: np.ndarray,
    source_code: int,
) -> Optional[dict]:
    """Prepare a single CameraHMR input item from a raw RGB image.

    Uses CameraHMR official preprocessing (generate_image_patch_cv2 + ImageNet norm).
    Returns None if the image cannot be processed.
    """
    try:
        from core.datasets.utils import generate_image_patch_cv2
    except ImportError:
        return None

    if image_rgb is None or image_rgb.ndim < 2:
        return None

    img_h, img_w = image_rgb.shape[:2]
    # Ensure 3-channel uint8 for cv2 warpAffine
    if image_rgb.ndim == 2:
        img3 = np.stack([image_rgb] * 3, axis=2)
    elif image_rgb.shape[2] == 1:
        img3 = np.concatenate([image_rgb] * 3, axis=2)
    elif image_rgb.shape[2] == 4:
        img3 = image_rgb[:, :, :3]
    else:
        img3 = image_rgb[:, :, :3]

    img3_u8 = np.clip(img3, 0, 255).astype(np.uint8)

    cx, cy = float(center[0]), float(center[1])
    box_size = float(scale * 200.0)

    # CameraHMR uses square crop with box_size as both width and height
    patch, _ = generate_image_patch_cv2(
        img3_u8, cx, cy,
        box_size, box_size,
        CAMERAHMR_CROP_RES, CAMERAHMR_CROP_RES,
        do_flip=False, scale=1.0, rot=0.0,
    )

    # Normalize with ImageNet mean/std
    patch_f = patch.astype(np.float32) / 255.0
    mean = np.array(CAMERAHMR_IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(CAMERAHMR_IMAGE_STD,  dtype=np.float32).reshape(1, 1, 3)
    patch_f = (patch_f - mean) / std

    img_tensor = torch.from_numpy(patch_f.transpose(2, 0, 1)).float()  # [3, 256, 256]

    return {
        "img":        img_tensor,
        "box_center": torch.tensor([cx, cy], dtype=torch.float32),
        "box_size":   torch.tensor(box_size, dtype=torch.float32),
        "img_size":   torch.tensor([img_h, img_w], dtype=torch.float32),
        "cam_int":    torch.from_numpy(cam_int.astype(np.float32)),
        "source_code": source_code,
    }


def _build_condition_K(img_w: int, img_h: int) -> np.ndarray:
    """Build camera intrinsic matrix using image-diagonal as focal length."""
    f = float(np.sqrt(img_w ** 2 + img_h ** 2))
    return np.array([
        [f,   0.0, img_w / 2.0],
        [0.0, f,   img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def predict_camerahmr_initialization(
    model,
    batch_items: List[dict],
    smplify_crop_focal: float,
    device: torch.device,
) -> HMRInitOutput:
    """Run CameraHMR forward on a batch of items and return HMRInitOutput."""
    B = len(batch_items)

    def _stack(key):
        return torch.stack([it[key] for it in batch_items]).to(device)

    batch = {
        "img":        _stack("img"),
        "box_center": _stack("box_center"),
        "box_size":   _stack("box_size"),
        "img_size":   _stack("img_size"),
        "cam_int":    _stack("cam_int"),
    }

    with torch.no_grad():
        pred_smpl_params, pred_cam_wp, fl_h = model(batch)

    # global_orient [B,1,3,3] + body_pose [B,23,3,3] → [B,24,3,3]
    rotmat = torch.cat([
        pred_smpl_params["global_orient"],
        pred_smpl_params["body_pose"],
    ], dim=1)
    pose_aa = rotmat_to_axis_angle_cv2(rotmat).view(B, 72)
    pose_aa[torch.isnan(pose_aa)] = 0.0

    betas = pred_smpl_params["betas"][:, :10]

    # Weak camera → crop perspective (tz = 2*f / (crop_res * s))
    init_cam_t_crop = cam_crop2perspective_crop(pred_cam_wp, smplify_crop_focal, CAMERAHMR_CROP_RES)

    # Validity: s>0, tz>0, no NaN/Inf
    s = pred_cam_wp[:, 0]
    valid = (
        s.gt(0)
        & torch.isfinite(init_cam_t_crop).all(dim=1)
        & torch.isfinite(pose_aa).all(dim=1)
        & torch.isfinite(betas).all(dim=1)
        & init_cam_t_crop[:, 2].gt(0)
    )

    condition_focal = torch.stack([it["cam_int"][0, 0] for it in batch_items]).to(device)
    source_code_t = torch.tensor(
        [it["source_code"] for it in batch_items], dtype=torch.uint8, device=device
    )

    return HMRInitOutput(
        pose_aa=pose_aa,
        betas=betas,
        cam_wp=pred_cam_wp,
        cam_t_crop=init_cam_t_crop,
        condition_focal=condition_focal,
        valid_mask=valid,
        source_code=source_code_t,
    )


class CameraHMRInitializer:
    """Initializer that uses CameraHMR to produce SMPL pose/shape/camera."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.model = load_camerahmr_model(args, device)
        self.device = device
        self.normalize_local = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def predict_batch(
        self,
        batch_samples: List[Sample],
        device: torch.device,
        dataset: str,
        args: argparse.Namespace,
    ) -> HMRInitOutput:
        B = len(batch_samples)
        batch_items: List[Optional[dict]] = []
        valid_flags: List[bool] = []

        use_paired_rgb = (
            dataset == "llvip"
            and getattr(args, "llvip_camerahmr_input", "paired_rgb") == "paired_rgb"
        )
        llvip_rgb_root = getattr(args, "llvip_rgb_root", None)

        for sample in batch_samples:
            cam_int = _build_condition_K(sample.img_w, sample.img_h)
            item = None
            src = SRC_CAMERAHMR_THERMAL

            if use_paired_rgb:
                rgb_path = resolve_llvip_rgb_image(sample.img_path, llvip_rgb_root)
                if rgb_path is not None:
                    img = load_rgb(rgb_path)
                    if img is not None:
                        item = prepare_camerahmr_item(img, sample.center, sample.scale, cam_int, SRC_CAMERAHMR_RGB)
                        src = SRC_CAMERAHMR_RGB

            if item is None:
                img = load_rgb(sample.img_path)
                if img is not None:
                    item = prepare_camerahmr_item(img, sample.center, sample.scale, cam_int, SRC_CAMERAHMR_THERMAL)

            batch_items.append(item)
            valid_flags.append(item is not None)

        # Build dummy outputs for failed items; mark invalid
        valid_tensor = torch.tensor(valid_flags, dtype=torch.bool, device=device)

        valid_items = [it for it in batch_items if it is not None]
        if not valid_items:
            dummy = HMRInitOutput(
                pose_aa=torch.zeros(B, 72, device=device),
                betas=torch.zeros(B, 10, device=device),
                cam_wp=torch.zeros(B, 3, device=device),
                cam_t_crop=torch.zeros(B, 3, device=device),
                condition_focal=torch.zeros(B, device=device),
                valid_mask=valid_tensor,
                source_code=torch.zeros(B, dtype=torch.uint8, device=device),
            )
            return dummy

        valid_idx = [i for i, v in enumerate(valid_flags) if v]
        sub_out = predict_camerahmr_initialization(
            self.model, valid_items, float(getattr(args, "smplify_crop_focal", args.focal_length)), device
        )

        # Scatter sub_out back into full-batch tensors
        pose_aa       = torch.zeros(B, 72, device=device)
        betas         = torch.zeros(B, 10, device=device)
        cam_wp        = torch.zeros(B, 3, device=device)
        cam_t_crop    = torch.zeros(B, 3, device=device)
        cond_focal    = torch.zeros(B, device=device)
        src_code      = torch.zeros(B, dtype=torch.uint8, device=device)

        for out_i, global_i in enumerate(valid_idx):
            if sub_out.valid_mask[out_i]:
                pose_aa[global_i]    = sub_out.pose_aa[out_i]
                betas[global_i]      = sub_out.betas[out_i]
                cam_wp[global_i]     = sub_out.cam_wp[out_i]
                cam_t_crop[global_i] = sub_out.cam_t_crop[out_i]
                cond_focal[global_i] = sub_out.condition_focal[out_i]
                src_code[global_i]   = sub_out.source_code[out_i]
            else:
                valid_tensor[global_i] = False

        return HMRInitOutput(
            pose_aa=pose_aa,
            betas=betas,
            cam_wp=cam_wp,
            cam_t_crop=cam_t_crop,
            condition_focal=cond_focal,
            valid_mask=valid_tensor,
            source_code=src_code,
        )


class LocalHMRInitializer:
    """Initializer that uses the existing local HMR model (v4 behavior)."""

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.normalize = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def predict_batch(
        self,
        batch_samples: List[Sample],
        device: torch.device,
        dataset: str,
        args: argparse.Namespace,
    ) -> HMRInitOutput:
        B = len(batch_samples)
        use_rgb_init = getattr(args, "llvip_use_rgb_init", False) and dataset == "llvip"
        llvip_rgb_root = getattr(args, "llvip_rgb_root", None)

        batch_images: List[Optional[torch.Tensor]] = []
        valid_flags: List[bool] = []
        source_codes: List[int] = []

        for sample in batch_samples:
            tensor = None
            src = SRC_LOCAL_THERMAL

            if use_rgb_init:
                rgb_path = resolve_llvip_rgb_image(sample.img_path, llvip_rgb_root)
                if rgb_path is not None:
                    tensor = prepare_image_tensor(rgb_path, sample.center, sample.scale, args.img_res, self.normalize)
                    if tensor is not None:
                        src = SRC_LOCAL_RGB

            if tensor is None:
                tensor = prepare_image_tensor(sample.img_path, sample.center, sample.scale, args.img_res, self.normalize)

            batch_images.append(tensor)
            valid_flags.append(tensor is not None)
            source_codes.append(src)

        valid_tensor = torch.tensor(valid_flags, dtype=torch.bool, device=device)
        valid_idx = [i for i, v in enumerate(valid_flags) if v]

        if not valid_idx:
            return HMRInitOutput(
                pose_aa=torch.zeros(B, 72, device=device),
                betas=torch.zeros(B, 10, device=device),
                cam_wp=torch.zeros(B, 3, device=device),
                cam_t_crop=torch.zeros(B, 3, device=device),
                condition_focal=torch.full((B,), float(args.focal_length), device=device),
                valid_mask=valid_tensor,
                source_code=torch.zeros(B, dtype=torch.uint8, device=device),
            )

        valid_samples = [batch_samples[i] for i in valid_idx]
        images_t = torch.stack([batch_images[i] for i in valid_idx]).to(device)
        center_t = torch.from_numpy(np.asarray([s.center for s in valid_samples], dtype=np.float32)).to(device)
        scale_t  = torch.from_numpy(np.asarray([s.scale for s in valid_samples],  dtype=np.float32)).to(device)
        img_focal_t = torch.full((len(valid_samples),), float(args.focal_length), dtype=torch.float32, device=device)
        img_center_t = torch.from_numpy(
            np.asarray([[s.img_w / 2.0, s.img_h / 2.0] for s in valid_samples], dtype=np.float32)
        ).to(device)

        with torch.no_grad():
            bbox_info = bbox_info_from_center_scale(center_t, scale_t, img_focal_t, img_center_t)
            pred_rotmat, pred_betas_sub, pred_cam_wp_sub = self.model(images_t, bbox_info=bbox_info)
            pose_aa_sub = rotmat_to_axis_angle_cv2(pred_rotmat).contiguous().view(len(valid_samples), -1)
            pose_aa_sub[torch.isnan(pose_aa_sub)] = 0.0
            cam_t_crop_sub = cam_crop2perspective_crop(pred_cam_wp_sub, args.focal_length, args.img_res)

        # Scatter back to full-batch tensors
        pose_aa    = torch.zeros(B, 72, device=device)
        betas      = torch.zeros(B, 10, device=device)
        cam_wp     = torch.zeros(B, 3, device=device)
        cam_t_crop = torch.zeros(B, 3, device=device)
        cond_focal = torch.full((B,), float(args.focal_length), device=device)
        src_code_t = torch.tensor(source_codes, dtype=torch.uint8, device=device)

        for out_i, global_i in enumerate(valid_idx):
            pose_aa[global_i]    = pose_aa_sub[out_i]
            betas[global_i]      = pred_betas_sub[out_i]
            cam_wp[global_i]     = pred_cam_wp_sub[out_i]
            cam_t_crop[global_i] = cam_t_crop_sub[out_i]

        return HMRInitOutput(
            pose_aa=pose_aa,
            betas=betas,
            cam_wp=cam_wp,
            cam_t_crop=cam_t_crop,
            condition_focal=cond_focal,
            valid_mask=valid_tensor,
            source_code=src_code_t,
        )


def infer_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def read_checkpoint_train_config(checkpoint_path: str) -> Dict[str, object]:
    ckpt_path = Path(checkpoint_path).resolve()
    candidates = [
        ckpt_path.parent.parent / "train_config.yaml",
        ckpt_path.parent.parent / "config.yaml",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    return {}


def infer_backbone_from_state(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    fc1 = state_dict.get("fc1.weight")
    if fc1 is None or fc1.ndim != 2:
        return None
    in_features = int(fc1.shape[1])
    embed_dim = int(state_dict.get("modality_embed.weight", torch.zeros(2, 0)).shape[1]) if "modality_embed.weight" in state_dict else 0
    base_dim = in_features - 160 - embed_dim
    if base_dim == 384:
        return "hrnet_w48"
    if base_dim == 2048:
        return "resnet50"
    return None


def infer_hmr_model_args(args: argparse.Namespace) -> None:
    config_data = read_checkpoint_train_config(args.checkpoint)
    model_cfg = config_data.get("MODEL", {}) if isinstance(config_data.get("MODEL", {}), dict) else {}
    state_dict = infer_checkpoint_state_dict(args.checkpoint)

    inferred_backbone = None
    inferred_source = None
    if isinstance(model_cfg.get("BACKBONE"), str) and model_cfg.get("BACKBONE"):
        inferred_backbone = str(model_cfg["BACKBONE"])
        inferred_source = "train_config.yaml"
    if inferred_backbone is None:
        inferred_backbone = infer_backbone_from_state(state_dict)
        inferred_source = "checkpoint_state" if inferred_backbone is not None else None

    if args.backbone is None and inferred_backbone is not None:
        args.backbone = inferred_backbone
        print(f"[model] inferred backbone={args.backbone} from {inferred_source}")
    elif args.backbone is None:
        args.backbone = "resnet50"
        print("[model] could not infer backbone from checkpoint; falling back to resnet50")

    if getattr(args, "modality_embed_dim", None) in [None]:
        embed_weight = state_dict.get("modality_embed.weight")
        if embed_weight is not None and embed_weight.ndim == 2:
            args.modality_embed_dim = int(embed_weight.shape[1])
            print(f"[model] inferred modality_embed_dim={args.modality_embed_dim} from checkpoint_state")
        elif isinstance(model_cfg.get("MODALITY_EMBED_DIM"), int):
            args.modality_embed_dim = int(model_cfg["MODALITY_EMBED_DIM"])
            print(f"[model] inferred modality_embed_dim={args.modality_embed_dim} from train_config.yaml")
        else:
            args.modality_embed_dim = 0

    if getattr(args, "img_res", None) in [None]:
        if isinstance(model_cfg.get("IMG_RES"), int):
            args.img_res = int(model_cfg["IMG_RES"])
            print(f"[model] inferred img_res={args.img_res} from train_config.yaml")
        else:
            args.img_res = 224

    if getattr(args, "hrnet_backbone_input_height", None) in [None]:
        if isinstance(model_cfg.get("HRNET_BACKBONE_INPUT_HEIGHT"), int):
            args.hrnet_backbone_input_height = int(model_cfg["HRNET_BACKBONE_INPUT_HEIGHT"])
        else:
            args.hrnet_backbone_input_height = 256

    if getattr(args, "hrnet_backbone_input_width", None) in [None]:
        if isinstance(model_cfg.get("HRNET_BACKBONE_INPUT_WIDTH"), int):
            args.hrnet_backbone_input_width = int(model_cfg["HRNET_BACKBONE_INPUT_WIDTH"])
        else:
            args.hrnet_backbone_input_width = 192


def build_smplify(args: argparse.Namespace, device: torch.device):
    if args.smplify_mode == "thermal_camsmplify":
        # v4: use ThermalCamSMPLify_v4 (supports lambda_bbox)
        from smplify import ThermalCamSMPLify_v4 as ThermalCamSMPLify

        print(
            "[smplify] mode=thermal_camsmplify "
            f"phases=({args.thermal_phase1_iters},{args.thermal_phase2_iters},{args.thermal_phase3_iters}) "
            f"lr={args.thermal_smplify_lr}"
        )
        return ThermalCamSMPLify(
            batch_size=args.batch_size,
            focal_length=args.focal_length,
            device=device,
            img_res=args.img_res,
            phase1_iters=args.thermal_phase1_iters,
            phase2_iters=args.thermal_phase2_iters,
            phase3_iters=args.thermal_phase3_iters,
            lr=args.thermal_smplify_lr,
            lambda_2d=args.lambda_2d,
            lambda_pose_init=args.lambda_pose_init,
            lambda_beta_init_phase2=args.lambda_beta_init_phase2,
            lambda_beta_init_phase3=args.lambda_beta_init_phase3,
            lambda_verts_init_phase3=args.lambda_verts_init_phase3,
            lambda_angle=args.lambda_angle_thermal,
            gmof_sigma=args.thermal_gmof_sigma,
            lambda_bbox=float(getattr(args, "lambda_bbox", 0.0)),
            truncated_anchor_multiplier=args.truncated_anchor_multiplier,
            fullcam=False,
        )
    if args.smplify_mode != "spin":
        raise ValueError(f"Unknown smplify_mode={args.smplify_mode}")

    from smplify import SMPLify

    sig = inspect.signature(SMPLify.__init__)
    kwargs = {
        "step_size": args.smplify_step_size,
        "batch_size": args.batch_size,
        "num_iters": args.num_smplify_iters,
        "focal_length": args.focal_length,
        "device": device,
    }
    optional = {
        "camera_num_iters": args.num_smplify_iters,
        "body_num_iters": args.num_smplify_iters,
        "pose_prior_weight": args.pose_prior_weight,
        "shape_prior_weight": args.shape_prior_weight,
        "angle_prior_weight": args.angle_prior_weight,
        "sigma": args.smplify_sigma,
        "crop_res": args.img_res,
    }
    for key, value in optional.items():
        if key in sig.parameters:
            kwargs[key] = value
    print(f"[smplify] mode=spin num_iters={args.num_smplify_iters} step_size={args.smplify_step_size}")
    return SMPLify(**kwargs)


def smplify_uses_fullcam(smplify) -> bool:
    params = inspect.signature(smplify.__call__).parameters
    return "img_focal" in params or "init_cam_full" in params


def prepare_image_tensor(path: str, center: np.ndarray, scale: float, img_res: int, normalize: Normalize) -> Optional[torch.Tensor]:
    img_rgb = load_rgb(path)
    if img_rgb is None:
        return None
    cropped = crop(img_rgb, center, scale, [img_res, img_res], rot=0)
    tensor = torch.from_numpy(np.transpose(cropped / 255.0, (2, 0, 1))).float()
    return normalize(tensor)


def project_with_crop_camera(
    joints: torch.Tensor,
    cam_t: torch.Tensor,
    focal_length: float,
    img_res: int,
) -> torch.Tensor:
    batch_size = joints.shape[0]
    camera_center = torch.full((batch_size, 2), img_res / 2.0, device=joints.device, dtype=joints.dtype)
    rotation = torch.eye(3, device=joints.device, dtype=joints.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    return perspective_projection(
        joints,
        rotation=rotation,
        translation=cam_t,
        focal_length=focal_length,
        camera_center=camera_center,
    )


def compute_reprojection_error(
    smplify,
    opt_pose: torch.Tensor,
    opt_betas: torch.Tensor,
    cam_for_smplify: torch.Tensor,
    cam_t_crop: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    img_focal: torch.Tensor,
    img_center: torch.Tensor,
    keypoints_crop: torch.Tensor,
    args: argparse.Namespace,
    fullcam: bool,
) -> np.ndarray:
    with torch.no_grad():
        smpl_out = smplify.smpl(
            global_orient=opt_pose[:, :3],
            body_pose=opt_pose[:, 3:],
            betas=opt_betas,
        )
        joints = smpl_out.joints
        if fullcam:
            _, proj_crop, _ = project_points_full_to_crop(
                joints,
                cam_for_smplify,
                center=center,
                scale=scale,
                img_focal=img_focal,
                img_center=img_center,
                crop_res=args.img_res,
            )
        else:
            proj_crop = project_with_crop_camera(joints, cam_t_crop, args.focal_length, args.img_res)

        conf = keypoints_crop[:, :, 2]
        err = torch.norm(proj_crop - keypoints_crop[:, :, :2], dim=-1)
        denom = conf.sum(dim=-1).clamp(min=1.0)
        mean_err = (err * conf).sum(dim=-1) / denom
        return mean_err.detach().cpu().numpy().astype(np.float32)


def project_smpl_joints_to_crop(
    smpl,
    pose: torch.Tensor,
    betas: torch.Tensor,
    cam_t_crop: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    with torch.no_grad():
        smpl_out = smpl(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=betas)
        proj = project_with_crop_camera(smpl_out.joints, cam_t_crop, args.focal_length, args.img_res)
    return proj.detach().cpu().numpy().astype(np.float32)


def bbox_from_visible_points(points: np.ndarray, conf: np.ndarray, conf_thresh: float) -> Optional[np.ndarray]:
    visible = (conf > conf_thresh) & np.isfinite(points).all(axis=1)
    if int(visible.sum()) < 2:
        return None
    xy = points[visible]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    return np.array([x0, y0, max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)], dtype=np.float32)


def compute_projected_bbox_sanity(
    projected_joints: np.ndarray,
    keypoints_crop: np.ndarray,
    truncation_types: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    batch_size = projected_joints.shape[0]
    height_ratio = np.full((batch_size,), np.inf, dtype=np.float32)
    width_ratio = np.full((batch_size,), np.inf, dtype=np.float32)
    area_ratio = np.full((batch_size,), np.inf, dtype=np.float32)
    center_offset = np.full((batch_size,), np.inf, dtype=np.float32)
    ok = np.zeros((batch_size,), dtype=bool)

    conf_thresh = float(args.thermal_conf_thresh)
    eps = 1e-6
    for i in range(batch_size):
        conf = keypoints_crop[i, :, 2]
        anno_bbox = bbox_from_visible_points(keypoints_crop[i, :, :2], conf, conf_thresh)
        proj_bbox = bbox_from_visible_points(projected_joints[i], conf, conf_thresh)
        if anno_bbox is None or proj_bbox is None:
            continue

        aw, ah = float(anno_bbox[2]), float(anno_bbox[3])
        pw, ph = float(proj_bbox[2]), float(proj_bbox[3])
        height_ratio[i] = ph / max(ah, eps)
        width_ratio[i] = pw / max(aw, eps)
        area_ratio[i] = (pw * ph) / max(aw * ah, eps)
        anno_center = anno_bbox[:2] + 0.5 * anno_bbox[2:]
        proj_center = proj_bbox[:2] + 0.5 * proj_bbox[2:]
        center_offset[i] = float(np.linalg.norm(proj_center - anno_center) / max(max(aw, ah), eps))

        min_h = 0.05 if truncation_types[i] == TRUNC_LOWER else float(args.min_proj_height_ratio)
        min_w = float(args.min_proj_width_ratio)
        max_h = float(args.max_proj_height_ratio)
        max_w = float(args.max_proj_width_ratio)
        ok[i] = (
            height_ratio[i] <= max_h
            and height_ratio[i] >= min_h
            and width_ratio[i] <= max_w
            and width_ratio[i] >= min_w
            and center_offset[i] <= float(args.max_proj_center_offset)
        )
    return {
        "ok": ok,
        "height_ratio": height_ratio,
        "width_ratio": width_ratio,
        "area_ratio": area_ratio,
        "center_offset": center_offset,
    }


def _joint_visible(kps49: np.ndarray, idx: int, conf_thresh: float) -> bool:
    return bool(kps49[idx, 2] > conf_thresh and np.isfinite(kps49[idx, :2]).all())


def compute_upright_sanity(
    projected_joints: np.ndarray,
    keypoints_crop: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    batch_size = projected_joints.shape[0]
    ok = np.ones((batch_size,), dtype=bool)
    if not bool(args.enable_upright_check):
        return ok

    conf_thresh = float(args.thermal_conf_thresh)
    tol = 5.0
    for i in range(batch_size):
        kps = keypoints_crop[i]
        proj = projected_joints[i]
        checks = 0
        violations = 0

        def add_check(a: int, b: int, expect_a_above_b: bool = True) -> None:
            nonlocal checks, violations
            if not (_joint_visible(kps, a, conf_thresh) and _joint_visible(kps, b, conf_thresh)):
                return
            checks += 1
            if expect_a_above_b:
                violations += int(proj[a, 1] > proj[b, 1] + tol)
            else:
                violations += int(proj[a, 1] + tol < proj[b, 1])

        # 49-joint indices: openpose 25 + GT24 index.
        pelvis = 25 + 14
        thorax = 25 + 15
        neck = 25 + 12
        head_top = 25 + 13
        right_shoulder = 25 + 8
        left_shoulder = 25 + 9
        right_hip = 25 + 2
        left_hip = 25 + 3
        right_knee = 25 + 1
        left_knee = 25 + 4
        right_ankle = 25 + 0
        left_ankle = 25 + 5

        add_check(head_top, pelvis)
        add_check(neck, pelvis)
        add_check(thorax, pelvis)
        add_check(right_shoulder, right_hip)
        add_check(left_shoulder, left_hip)
        add_check(right_hip, right_knee)
        add_check(right_knee, right_ankle)
        add_check(left_hip, left_knee)
        add_check(left_knee, left_ankle)

        if checks > 0:
            ok[i] = not (violations >= 2 or (checks <= 2 and violations >= 1))
    return ok


def compute_vertices_init_distance(
    smpl,
    init_pose: torch.Tensor,
    init_betas: torch.Tensor,
    opt_pose: torch.Tensor,
    opt_betas: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        init_out = smpl(global_orient=init_pose[:, :3], body_pose=init_pose[:, 3:], betas=init_betas)
        opt_out = smpl(global_orient=opt_pose[:, :3], body_pose=opt_pose[:, 3:], betas=opt_betas)
        dist = torch.norm(opt_out.vertices - init_out.vertices, dim=-1).mean(dim=-1)
    return dist.detach().cpu().numpy().astype(np.float32)


def generate_split(
    args: argparse.Namespace,
    dataset: str,
    split: str,
    samples: List[Sample],
    initializer,   # CameraHMRInitializer or LocalHMRInitializer
    smplify,
    device: torch.device,
) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = output_path(args.out_dir, dataset, split)
    n = len(samples)
    print(f"[{dataset}/{split}] generating {n} samples -> {out_file}")
    print(f"[{dataset}/{split}] smplify_mode={args.smplify_mode}")

    pose = np.zeros((n, 72), dtype=np.float32)
    shape = np.zeros((n, 10), dtype=np.float32)
    has_smpl = np.zeros((n,), dtype=np.uint8)
    valid_fit = np.zeros((n,), dtype=np.uint8)
    reprojection_error = np.full((n,), np.inf, dtype=np.float32)
    cam_t = np.zeros((n, 3), dtype=np.float32)
    init_reprojection_error = np.full((n,), np.inf, dtype=np.float32)
    improvement_ratio = np.full((n,), np.inf, dtype=np.float32)
    fitting_loss_total = np.full((n,), np.inf, dtype=np.float32)
    fitting_loss_2d = np.full((n,), np.inf, dtype=np.float32)
    fitting_loss_pose_init = np.full((n,), np.inf, dtype=np.float32)
    fitting_loss_beta_init = np.full((n,), np.inf, dtype=np.float32)
    fitting_loss_verts_init = np.full((n,), np.inf, dtype=np.float32)
    fitting_loss_angle = np.full((n,), np.inf, dtype=np.float32)
    pose_change = np.full((n,), np.inf, dtype=np.float32)
    verts_init_dist = np.full((n,), np.inf, dtype=np.float32)
    truncation_type = np.asarray([s.truncation_type for s in samples], dtype=np.uint8) if n else np.zeros((0,), dtype=np.uint8)
    pgt_quality = np.asarray([s.pgt_quality for s in samples], dtype=np.uint8) if n else np.zeros((0,), dtype=np.uint8)
    smpl_weight = np.asarray([s.smpl_weight for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    pose_weight = np.asarray([s.pose_weight for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    shape_weight = np.asarray([s.shape_weight for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    cam_weight = np.asarray([s.cam_weight for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    used_init_fallback = np.zeros((n,), dtype=np.uint8)
    proj_height_ratio = np.full((n,), np.inf, dtype=np.float32)
    proj_width_ratio = np.full((n,), np.inf, dtype=np.float32)
    proj_area_ratio = np.full((n,), np.inf, dtype=np.float32)
    proj_center_offset = np.full((n,), np.inf, dtype=np.float32)
    global_orient_change = np.full((n,), np.inf, dtype=np.float32)
    body_pose_change = np.full((n,), np.inf, dtype=np.float32)
    upright_ok = np.zeros((n,), dtype=np.uint8)
    reject_low_joints = np.zeros((n,), dtype=np.uint8)
    reject_reproj = np.zeros((n,), dtype=np.uint8)
    reject_no_improvement = np.zeros((n,), dtype=np.uint8)
    reject_depth = np.zeros((n,), dtype=np.uint8)
    reject_beta = np.zeros((n,), dtype=np.uint8)
    reject_pose_change = np.zeros((n,), dtype=np.uint8)
    reject_verts_init = np.zeros((n,), dtype=np.uint8)
    reject_bbox_sanity = np.zeros((n,), dtype=np.uint8)
    reject_upright = np.zeros((n,), dtype=np.uint8)
    reject_global_orient_change = np.zeros((n,), dtype=np.uint8)
    reject_truncation = np.zeros((n,), dtype=np.uint8)
    reject_nan = np.zeros((n,), dtype=np.uint8)
    reject_read = np.zeros((n,), dtype=np.uint8)
    curation_type_arr = np.asarray([s.curation_type for s in samples], dtype=np.uint8) if n else np.zeros((0,), dtype=np.uint8)
    downgraded_full_to_weak = np.zeros((n,), dtype=np.uint8)
    # v4 new arrays
    overlap_ratio_arr = np.asarray([s.overlap_ratio for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    bbox_height_ratio_arr = np.asarray([s.bbox_height_ratio for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    bbox_width_ratio_arr = np.asarray([s.bbox_width_ratio for s in samples], dtype=np.float32) if n else np.zeros((0,), dtype=np.float32)
    instance_id_arr = np.asarray([s.instance_id for s in samples], dtype=np.int32) if n else np.zeros((0,), dtype=np.int32)
    instance_curation_reason_arr = np.asarray([s.instance_curation_reason for s in samples], dtype=np.uint8) if n else np.zeros((0,), dtype=np.uint8)
    selected_init_type_arr = np.zeros((n,), dtype=np.uint8)   # SRC_* constants
    cam_t_full_arr = np.zeros((n, 3), dtype=np.float32)
    focal_length_arr = np.zeros((n,), dtype=np.float32)
    img_center_arr = np.zeros((n, 2), dtype=np.float32)
    # v5 CameraHMR / crop-camera fields
    _smplify_crop_focal = float(getattr(args, "smplify_crop_focal", args.focal_length))
    init_cam_wp_arr           = np.zeros((n, 3), dtype=np.float32)
    init_cam_t_crop_arr       = np.zeros((n, 3), dtype=np.float32)
    camerahmr_condition_focal_arr = np.zeros((n,), dtype=np.float32)
    crop_focal_length_arr     = np.full((n,), _smplify_crop_focal, dtype=np.float32)
    crop_camera_center_arr    = np.full((n, 2), float(args.img_res) / 2.0, dtype=np.float32)
    has_full_camera_arr       = np.zeros((n,), dtype=np.uint8)   # always 0
    camera_model_code_arr     = np.zeros((n,), dtype=np.uint8)   # 0 = crop_perspective

    if n == 0:
        imgname = np.asarray([], dtype=str)
        center = np.zeros((0, 2), dtype=np.float32)
        scale = np.zeros((0,), dtype=np.float32)
        part = np.zeros((0, 24, 3), dtype=np.float32)
        openpose = np.zeros((0, 25, 3), dtype=np.float32)
        bbox = np.zeros((0, 4), dtype=np.float32)
        save_npz(
            out_file,
            imgname,
            center,
            scale,
            part,
            openpose,
            pose,
            shape,
            has_smpl,
            valid_fit,
            reprojection_error,
            cam_t,
            bbox,
            init_reprojection_error=init_reprojection_error,
            improvement_ratio=improvement_ratio,
            fitting_loss_total=fitting_loss_total,
            fitting_loss_2d=fitting_loss_2d,
            fitting_loss_pose_init=fitting_loss_pose_init,
            fitting_loss_beta_init=fitting_loss_beta_init,
            fitting_loss_verts_init=fitting_loss_verts_init,
            fitting_loss_angle=fitting_loss_angle,
            pose_change=pose_change,
            verts_init_dist=verts_init_dist,
            truncation_type=truncation_type,
            pgt_quality=pgt_quality,
            smpl_weight=smpl_weight,
            pose_weight=pose_weight,
            shape_weight=shape_weight,
            cam_weight=cam_weight,
            used_init_fallback=used_init_fallback,
            proj_height_ratio=proj_height_ratio,
            proj_width_ratio=proj_width_ratio,
            proj_area_ratio=proj_area_ratio,
            proj_center_offset=proj_center_offset,
            global_orient_change=global_orient_change,
            body_pose_change=body_pose_change,
            upright_ok=upright_ok,
            reject_low_joints=reject_low_joints,
            reject_reproj=reject_reproj,
            reject_no_improvement=reject_no_improvement,
            reject_depth=reject_depth,
            reject_beta=reject_beta,
            reject_pose_change=reject_pose_change,
            reject_verts_init=reject_verts_init,
            reject_bbox_sanity=reject_bbox_sanity,
            reject_upright=reject_upright,
            reject_global_orient_change=reject_global_orient_change,
            reject_truncation=reject_truncation,
            reject_nan=reject_nan,
            reject_read=reject_read,
            curation_type=curation_type_arr,
            downgraded_full_to_weak=downgraded_full_to_weak,
        )
        return out_file

    imgname = np.asarray([s.imgname for s in samples])
    center = np.asarray([s.center for s in samples], dtype=np.float32)
    scale = np.asarray([s.scale for s in samples], dtype=np.float32)
    part = np.asarray([s.part for s in samples], dtype=np.float32)
    openpose = np.asarray([s.openpose for s in samples], dtype=np.float32)
    bbox = np.asarray([s.bbox for s in samples], dtype=np.float32)

    fullcam = smplify_uses_fullcam(smplify) if args.smplify_mode == "spin" else False
    eligible_indices = [idx for idx, sample in enumerate(samples) if sample.fit_eligible]
    skipped_low_joints = n - len(eligible_indices)
    if skipped_low_joints:
        reject_low_joints[[idx for idx, sample in enumerate(samples) if not sample.fit_eligible]] = 1
    severe_indices = [idx for idx, sample in enumerate(samples) if sample.truncation_type == TRUNC_SEVERE]
    if severe_indices:
        reject_truncation[severe_indices] = 1
        smpl_weight[severe_indices] = 0.0
        pose_weight[severe_indices] = 0.0
        shape_weight[severe_indices] = 0.0
        cam_weight[severe_indices] = 0.0
    skipped_read = 0

    # In v5, image loading is delegated to the initializer.
    # smplify_crop_focal is used for all crop-space projections (default 5000).
    _smplify_crop_focal = float(getattr(args, "smplify_crop_focal", args.focal_length))
    _batch_size = int(getattr(args, "camerahmr_batch_size", None) or args.batch_size)

    for start in tqdm(range(0, len(eligible_indices), _batch_size), desc=f"{dataset}/{split} smplify"):
        batch_indices = eligible_indices[start : start + _batch_size]
        batch_candidate_indices: List[int] = list(batch_indices)
        batch_candidate_samples: List[Sample] = [samples[i] for i in batch_candidate_indices]

        if not batch_candidate_samples:
            continue

        # Delegate image loading + HMR inference to initializer
        init_out = initializer.predict_batch(batch_candidate_samples, device, dataset, args)

        valid_local = init_out.valid_mask.cpu().numpy().astype(bool)
        for local_i, idx in enumerate(batch_candidate_indices):
            if not valid_local[local_i]:
                skipped_read += 1
                reject_read[idx] = 1
                reject_nan[idx] = 1

        valid_local_idx = np.where(valid_local)[0]
        if len(valid_local_idx) == 0:
            continue

        batch_kept_indices = [batch_candidate_indices[i] for i in valid_local_idx]
        batch_samples = [batch_candidate_samples[i] for i in valid_local_idx]
        batch_npz_indices = np.asarray(batch_kept_indices, dtype=np.int64)

        # Extract initializer outputs for valid subset
        pred_pose    = init_out.pose_aa[valid_local_idx].to(device)
        pred_betas   = init_out.betas[valid_local_idx].to(device)
        pred_cam_crop = init_out.cam_wp[valid_local_idx].to(device)   # weak-perspective [s,tx,ty]
        init_cam_crop = init_out.cam_t_crop[valid_local_idx].to(device)
        condition_focal_batch = init_out.condition_focal[valid_local_idx].to(device)
        source_codes_np = init_out.source_code[valid_local_idx].cpu().numpy()

        # Update v5 diagnostic arrays
        selected_init_type_arr[batch_npz_indices]         = source_codes_np
        init_cam_wp_arr[batch_npz_indices]                = pred_cam_crop.detach().cpu().numpy()
        init_cam_t_crop_arr[batch_npz_indices]            = init_cam_crop.detach().cpu().numpy()
        camerahmr_condition_focal_arr[batch_npz_indices]  = condition_focal_batch.detach().cpu().numpy()

        B_valid = len(batch_samples)
        center_t = torch.from_numpy(np.asarray([s.center for s in batch_samples], dtype=np.float32)).to(device)
        scale_t  = torch.from_numpy(np.asarray([s.scale  for s in batch_samples], dtype=np.float32)).to(device)
        # img_focal_t uses smplify_crop_focal for crop-space projection (not camerahmr_condition_focal)
        img_focal_t = torch.full((B_valid,), _smplify_crop_focal, dtype=torch.float32, device=device)
        img_center_t = torch.from_numpy(
            np.asarray([[s.img_w / 2.0, s.img_h / 2.0] for s in batch_samples], dtype=np.float32)
        ).to(device)
        keypoints_crop_np = np.stack([
            keypoints_full_to_crop(build_keypoints49(s.openpose, s.part), s.center, s.scale, args.img_res)
            for s in batch_samples
        ]).astype(np.float32)
        keypoints_crop_t = torch.from_numpy(keypoints_crop_np).to(device)

        # init_cam_full is only needed for fullcam SMPLify (spin mode)
        init_cam_full = cam_crop2full(pred_cam_crop, center_t, scale_t, img_focal_t, img_center_t)

        if fullcam:
            init_reproj = compute_reprojection_error(
                smplify,
                pred_pose,
                pred_betas,
                init_cam_full,
                init_cam_crop,
                center_t,
                scale_t,
                img_focal_t,
                img_center_t,
                keypoints_crop_t,
                args,
                fullcam=True,
            )
        else:
            init_reproj = compute_reprojection_error(
                smplify,
                pred_pose,
                pred_betas,
                init_cam_crop,
                init_cam_crop,
                center_t,
                scale_t,
                img_focal_t,
                img_center_t,
                keypoints_crop_t,
                args,
                fullcam=False,
            )
        init_reprojection_error[batch_npz_indices] = init_reproj

        try:
            if args.smplify_mode == "spin" and fullcam:
                _, _, opt_pose, opt_betas, opt_cam_full, _ = smplify(
                    pred_pose.detach(),
                    pred_betas.detach(),
                    init_cam_full.detach(),
                    center_t,
                    scale_t,
                    img_focal_t,
                    img_center_t,
                    keypoints_crop_t,
                )
                opt_cam_crop = cam_full2crop_perspective(
                    opt_cam_full,
                    center_t,
                    scale_t,
                    img_focal_t,
                    img_center_t,
                    args.focal_length,
                    args.img_res,
                )
                fit_reproj = compute_reprojection_error(
                    smplify,
                    opt_pose,
                    opt_betas,
                    opt_cam_full,
                    opt_cam_crop,
                    center_t,
                    scale_t,
                    img_focal_t,
                    img_center_t,
                    keypoints_crop_t,
                    args,
                    fullcam=True,
                )
                fitting_valid_np = np.ones((len(batch_samples),), dtype=bool)
                loss_dict = None
            elif args.smplify_mode == "spin":
                camera_center = torch.full(
                    (len(batch_samples), 2),
                    args.img_res / 2.0,
                    dtype=torch.float32,
                    device=device,
                )
                _, _, opt_pose, opt_betas, opt_cam_crop, _ = smplify(
                    pred_pose.detach(),
                    pred_betas.detach(),
                    init_cam_crop.detach(),
                    camera_center,
                    keypoints_crop_t,
                )
                fit_reproj = compute_reprojection_error(
                    smplify,
                    opt_pose,
                    opt_betas,
                    opt_cam_crop,
                    opt_cam_crop,
                    center_t,
                    scale_t,
                    img_focal_t,
                    img_center_t,
                    keypoints_crop_t,
                    args,
                    fullcam=False,
                )
                fitting_valid_np = np.ones((len(batch_samples),), dtype=bool)
                loss_dict = None
            elif args.smplify_mode == "thermal_camsmplify":
                batch_trunc_np    = truncation_type[batch_npz_indices].astype(np.uint8)
                batch_curation_np = curation_type_arr[batch_npz_indices].astype(np.uint8)
                weak_or_trunc_np  = (batch_trunc_np != TRUNC_NONE) | (batch_curation_np == CURATION_WEAK)
                # v4: --freeze_betas_for_all overrides per-sample logic
                if getattr(args, "freeze_betas_for_all", False):
                    _freeze_np = np.ones(len(batch_npz_indices), dtype=bool)
                else:
                    _freeze_np = weak_or_trunc_np & bool(args.freeze_betas_for_truncation)
                freeze_betas_t = torch.from_numpy(_freeze_np).to(device)
                strong_anchor_t = torch.from_numpy(weak_or_trunc_np).to(device)
                result = smplify(
                    pred_pose.detach(),
                    pred_betas.detach(),
                    init_cam_crop.detach(),
                    keypoints_crop_t,
                    center_t,
                    scale_t,
                    img_focal_t,
                    img_center_t,
                    img_res=args.img_res,
                    fullcam=False,
                    freeze_betas=freeze_betas_t,
                    strong_anchor=strong_anchor_t,
                )
                opt_pose = result["opt_pose"]
                opt_betas = result["opt_betas"]
                opt_cam_crop = result["opt_cam"]
                fitting_valid_np = result["valid_mask"].detach().cpu().numpy().astype(bool)
                loss_dict = result.get("loss_dict", {})
                fit_reproj = compute_reprojection_error(
                    smplify,
                    opt_pose,
                    opt_betas,
                    opt_cam_crop,
                    opt_cam_crop,
                    center_t,
                    scale_t,
                    img_focal_t,
                    img_center_t,
                    keypoints_crop_t,
                    args,
                    fullcam=False,
                )
                fit_reproj[~fitting_valid_np] = np.inf
            else:
                raise ValueError(f"Unknown smplify_mode={args.smplify_mode}")
        except Exception as exc:
            print(f"[{dataset}/{split}] SMPLify failed for batch start={start}: {exc}")
            reject_nan[batch_npz_indices] = 1
            continue

        opt_pose_np = opt_pose.detach().cpu().numpy().astype(np.float32)
        opt_betas_np = opt_betas.detach().cpu().numpy().astype(np.float32)
        opt_cam_crop_np = opt_cam_crop.detach().cpu().numpy().astype(np.float32)
        pred_pose_np = pred_pose.detach().cpu().numpy().astype(np.float32)
        pred_betas_np = pred_betas.detach().cpu().numpy().astype(np.float32)

        fit_reproj = fit_reproj.astype(np.float32)

        if loss_dict:
            for key, target in (
                ("loss_total", fitting_loss_total),
                ("loss_2d", fitting_loss_2d),
                ("loss_pose_init", fitting_loss_pose_init),
                ("loss_beta_init", fitting_loss_beta_init),
                ("loss_verts_init", fitting_loss_verts_init),
                ("loss_angle", fitting_loss_angle),
                ("verts_init_dist", verts_init_dist),
            ):
                if key in loss_dict:
                    target[batch_npz_indices] = loss_dict[key].detach().cpu().numpy().astype(np.float32)

        need_verts_dist = (
            (args.max_verts_init_dist > 0 or getattr(args, "max_verts_init_dist_weak", 0.0) > 0)
            and not np.isfinite(verts_init_dist[batch_npz_indices]).all()
        )
        if need_verts_dist:
            verts_init_dist[batch_npz_indices] = compute_vertices_init_distance(
                smplify.smpl,
                pred_pose.detach(),
                pred_betas.detach(),
                opt_pose.detach(),
                opt_betas.detach(),
            )

        fallback_mask = (
            bool(args.allow_init_fallback)
            & np.isfinite(init_reproj)
            & (init_reproj < float(args.init_good_reproj_thresh))
            & ((~np.isfinite(fit_reproj)) | (fit_reproj > init_reproj))
        )
        final_pose_np = opt_pose_np.copy()
        final_betas_np = opt_betas_np.copy()
        final_cam_crop_np = opt_cam_crop_np.copy()
        final_reproj = fit_reproj.copy()
        final_pose_np[fallback_mask] = pred_pose_np[fallback_mask]
        final_betas_np[fallback_mask] = pred_betas_np[fallback_mask]
        final_cam_crop_np[fallback_mask] = init_cam_crop.detach().cpu().numpy().astype(np.float32)[fallback_mask]
        final_reproj[fallback_mask] = init_reproj[fallback_mask]
        if fallback_mask.any():
            verts_init_dist[batch_npz_indices[fallback_mask]] = 0.0

        final_pose_t = torch.from_numpy(final_pose_np).to(device=device, dtype=pred_pose.dtype)
        final_betas_t = torch.from_numpy(final_betas_np).to(device=device, dtype=pred_betas.dtype)
        final_cam_crop_t = torch.from_numpy(final_cam_crop_np).to(device=device, dtype=init_cam_crop.dtype)
        projected_joints_np = project_smpl_joints_to_crop(
            smplify.smpl,
            final_pose_t,
            final_betas_t,
            final_cam_crop_t,
            args,
        )
        bbox_sanity = compute_projected_bbox_sanity(
            projected_joints_np,
            keypoints_crop_np,
            truncation_type[batch_npz_indices],
            args,
        )
        upright_ok_np = compute_upright_sanity(projected_joints_np, keypoints_crop_np, args)

        ratio = final_reproj / np.maximum(init_reproj, 1e-6)
        ratio[~np.isfinite(ratio)] = np.inf
        pose_change_np = np.mean(np.abs(final_pose_np - pred_pose_np), axis=1).astype(np.float32)
        global_change_np = np.mean(np.abs(final_pose_np[:, :3] - pred_pose_np[:, :3]), axis=1).astype(np.float32)
        body_change_np = np.mean(np.abs(final_pose_np[:, 3:] - pred_pose_np[:, 3:]), axis=1).astype(np.float32)

        valid_joint_counts = np.asarray([s.valid_joint_count for s in batch_samples], dtype=np.int32)
        beta_ok = np.max(np.abs(final_betas_np), axis=1) < args.max_abs_beta
        depth_ok = final_cam_crop_np[:, 2] > 0
        # v4: tiered reproj thresholds — full pGT uses full_reproj_thresh (tight),
        # weak pGT uses reproj_thresh (loose). reject_reproj flag uses weak threshold.
        _full_reproj_thresh = float(getattr(args, "full_reproj_thresh", args.reproj_thresh))
        reproj_ok_full = final_reproj < _full_reproj_thresh
        reproj_ok_weak = final_reproj < args.reproj_thresh
        reproj_ok = reproj_ok_weak  # used for reject_reproj flag
        count_ok = valid_joint_counts >= args.min_valid_joints
        improvement_ok = (
            np.ones_like(reproj_ok, dtype=bool)
            if args.disable_improvement_check
            else ((ratio < args.min_improvement_ratio) | (final_reproj < args.very_good_reproj_thresh))
        )
        improvement_or_fallback_ok = improvement_ok | fallback_mask
        global_ok = global_change_np < args.max_global_orient_change
        body_ok = body_change_np < args.max_body_pose_change
        pose_ok = global_ok & body_ok
        bbox_sanity_ok = bbox_sanity["ok"]
        truncation_ok = ~(
            (truncation_type[batch_npz_indices] == TRUNC_SEVERE)
            & bool(args.reject_severe_truncation)
        )
        finite_ok = (
            (fitting_valid_np | fallback_mask)
            & np.isfinite(final_reproj)
            & np.isfinite(final_pose_np).all(axis=1)
            & np.isfinite(final_betas_np).all(axis=1)
            & np.isfinite(final_cam_crop_np).all(axis=1)
        )

        # --- curation-aware fit_ok split ---
        batch_curation_np2 = curation_type_arr[batch_npz_indices].astype(np.uint8)
        curated_full = batch_curation_np2 == CURATION_FULL
        curated_weak = batch_curation_np2 == CURATION_WEAK
        full_body_ok = truncation_type[batch_npz_indices] == TRUNC_NONE
        non_severe   = truncation_type[batch_npz_indices] != TRUNC_SEVERE

        _base_ok = (
            count_ok & finite_ok & depth_ok & beta_ok
            & improvement_or_fallback_ok & pose_ok & bbox_sanity_ok
            & upright_ok_np & truncation_ok
        )
        # full pGT: tight reproj threshold; weak pGT: loose threshold
        common_ok_full = _base_ok & reproj_ok_full
        common_ok_weak = _base_ok & reproj_ok_weak
        common_ok = common_ok_weak  # backward-compat alias

        if args.max_verts_init_dist > 0:
            verts_ok_full = verts_init_dist[batch_npz_indices] < args.max_verts_init_dist
        else:
            verts_ok_full = np.ones(len(batch_samples), dtype=bool)

        max_verts_weak = getattr(args, "max_verts_init_dist_weak", args.max_verts_init_dist)
        good_reproj_override_thresh = float(getattr(args, "good_reproj_override_thresh", 6.0))
        good_reproj_override = (
            (final_reproj < good_reproj_override_thresh) & bbox_sanity_ok
            & upright_ok_np & beta_ok & depth_ok & np.isfinite(final_reproj)
        )
        if max_verts_weak > 0:
            verts_ok_weak = verts_ok_full | (
                (verts_init_dist[batch_npz_indices] < max_verts_weak) & good_reproj_override
            )
        else:
            verts_ok_weak = np.ones(len(batch_samples), dtype=bool)

        # v4: fit_ok_full uses tight reproj (common_ok_full); fit_ok_weak uses loose (common_ok_weak)
        fit_ok_full = common_ok_full & curated_full & full_body_ok & verts_ok_full

        curation_mode = getattr(args, "otp2_curation_mode", "weighted")
        if curation_mode == "weighted":
            fit_ok_weak = (
                common_ok_weak & (~fit_ok_full) & non_severe & verts_ok_weak
                & (curated_weak | curated_full)
            )
        else:
            # full_only: only curated_full can degrade to weak pGT
            fit_ok_weak = (
                common_ok_weak & (~fit_ok_full) & non_severe & verts_ok_weak & curated_full
            )

        fit_ok = fit_ok_full | fit_ok_weak
        verts_ok = verts_ok_full  # used by reject_verts_init flag

        reprojection_error[batch_npz_indices] = final_reproj
        cam_save = final_cam_crop_np.copy()
        cam_save[~np.isfinite(cam_save).all(axis=1)] = 0.0
        cam_save[~fit_ok] = 0.0
        cam_t[batch_npz_indices] = cam_save
        improvement_ratio[batch_npz_indices] = ratio
        pose_change[batch_npz_indices] = pose_change_np
        global_orient_change[batch_npz_indices] = global_change_np
        body_pose_change[batch_npz_indices] = body_change_np
        used_init_fallback[batch_npz_indices] = fallback_mask.astype(np.uint8)
        proj_height_ratio[batch_npz_indices] = bbox_sanity["height_ratio"]
        proj_width_ratio[batch_npz_indices] = bbox_sanity["width_ratio"]
        proj_area_ratio[batch_npz_indices] = bbox_sanity["area_ratio"]
        proj_center_offset[batch_npz_indices] = bbox_sanity["center_offset"]
        upright_ok[batch_npz_indices] = upright_ok_np.astype(np.uint8)
        valid_fit[batch_npz_indices] = fit_ok.astype(np.uint8)
        has_smpl[batch_npz_indices]  = fit_ok.astype(np.uint8)
        pose[batch_npz_indices[fit_ok]]  = final_pose_np[fit_ok]
        shape[batch_npz_indices[fit_ok]] = final_betas_np[fit_ok]

        # pgt_quality
        pgt_quality_batch = np.zeros((len(batch_samples),), dtype=np.uint8)
        pgt_quality_batch[fit_ok_full] = PGT_QUALITY_FULL
        pgt_quality_batch[fit_ok_weak] = PGT_QUALITY_WEAK
        pgt_quality[batch_npz_indices] = pgt_quality_batch

        # weights: zero all first, then fill by tier
        smpl_weight[batch_npz_indices]  = 0.0
        pose_weight[batch_npz_indices]  = 0.0
        shape_weight[batch_npz_indices] = 0.0
        cam_weight[batch_npz_indices]   = 0.0

        full_idx = batch_npz_indices[fit_ok_full]
        smpl_weight[full_idx]  = float(args.fullbody_smpl_weight)
        pose_weight[full_idx]  = float(args.fullbody_pose_weight)
        shape_weight[full_idx] = float(args.fullbody_shape_weight)
        cam_weight[full_idx]   = 0.0

        weak_idx = batch_npz_indices[fit_ok_weak]
        smpl_weight[weak_idx]  = float(getattr(args, "weak_smpl_weight", 0.3))
        pose_weight[weak_idx]  = float(getattr(args, "weak_pose_weight", 0.3))
        shape_weight[weak_idx] = 0.0
        cam_weight[weak_idx]   = 0.0

        # diagnostic: curated_full samples that could only reach weak pGT
        downgraded_full_to_weak[batch_npz_indices[curated_full & fit_ok_weak]] = 1

        reject_low_joints[batch_npz_indices[~count_ok]] = 1
        reject_reproj[batch_npz_indices[~reproj_ok]] = 1
        reject_no_improvement[batch_npz_indices[~improvement_or_fallback_ok]] = 1
        reject_depth[batch_npz_indices[~depth_ok]] = 1
        reject_beta[batch_npz_indices[~beta_ok]] = 1
        reject_pose_change[batch_npz_indices[~pose_ok]] = 1
        reject_global_orient_change[batch_npz_indices[~global_ok]] = 1
        reject_verts_init[batch_npz_indices[~verts_ok]] = 1
        reject_bbox_sanity[batch_npz_indices[~bbox_sanity_ok]] = 1
        reject_upright[batch_npz_indices[~upright_ok_np]] = 1
        reject_truncation[batch_npz_indices[~truncation_ok]] = 1
        reject_nan[batch_npz_indices[~finite_ok]] = 1

    # v5: print init-source statistics
    src_counts = {
        "local_thermal":    int((selected_init_type_arr == SRC_LOCAL_THERMAL).sum()),
        "local_rgb":        int((selected_init_type_arr == SRC_LOCAL_RGB).sum()),
        "camerahmr_thermal":int((selected_init_type_arr == SRC_CAMERAHMR_THERMAL).sum()),
        "camerahmr_rgb":    int((selected_init_type_arr == SRC_CAMERAHMR_RGB).sum()),
    }
    print(f"[{dataset}/{split}] init_source_counts={src_counts}")

    invalid = valid_fit == 0
    pgt_quality[invalid] = PGT_QUALITY_REJECTED
    smpl_weight[invalid] = 0.0
    pose_weight[invalid] = 0.0
    shape_weight[invalid] = 0.0
    cam_weight[invalid] = 0.0

    finite_init = init_reprojection_error[np.isfinite(init_reprojection_error)]
    finite_opt = reprojection_error[np.isfinite(reprojection_error)]
    finite_ratio = improvement_ratio[np.isfinite(improvement_ratio)]
    print(
        f"[{dataset}/{split}] valid_fit={int(valid_fit.sum())}/{n} "
        f"({100.0 * float(valid_fit.sum()) / max(n, 1):.2f}%) "
        f"skipped_low_joints={skipped_low_joints} skipped_read={skipped_read}"
    )
    print(
        f"[{dataset}/{split}] init_reproj_median={float(np.median(finite_init)) if finite_init.size else float('nan'):.4f} "
        f"opt_reproj_median={float(np.median(finite_opt)) if finite_opt.size else float('nan'):.4f} "
        f"improvement_ratio_median={float(np.median(finite_ratio)) if finite_ratio.size else float('nan'):.4f}"
    )
    trunc_counts = {
        TRUNCATION_NAMES.get(int(t), str(int(t))): int((truncation_type == int(t)).sum())
        for t in np.unique(truncation_type)
    }
    accepted = valid_fit.astype(bool)
    print(
        f"[{dataset}/{split}] truncation_counts={trunc_counts} "
        f"used_init_fallback={int(used_init_fallback.sum())} "
        f"accepted_smpl_weight_mean={float(np.mean(smpl_weight[accepted])) if accepted.any() else 0.0:.4f}"
    )
    reject_stats = {
        "low_joints": int(reject_low_joints.sum()),
        "reproj": int(reject_reproj.sum()),
        "no_improvement": int(reject_no_improvement.sum()),
        "depth": int(reject_depth.sum()),
        "beta": int(reject_beta.sum()),
        "pose_change": int(reject_pose_change.sum()),
        "global_orient_change": int(reject_global_orient_change.sum()),
        "verts_init": int(reject_verts_init.sum()),
        "bbox_sanity": int(reject_bbox_sanity.sum()),
        "upright": int(reject_upright.sum()),
        "truncation": int(reject_truncation.sum()),
        "nan": int(reject_nan.sum()),
        "read": int(reject_read.sum()),
    }
    print(f"[{dataset}/{split}] rejection_stats={reject_stats}")
    save_npz(
        out_file,
        imgname,
        center,
        scale,
        part,
        openpose,
        pose,
        shape,
        has_smpl,
        valid_fit,
        reprojection_error,
        cam_t,
        bbox,
        init_reprojection_error=init_reprojection_error,
        improvement_ratio=improvement_ratio,
        fitting_loss_total=fitting_loss_total,
        fitting_loss_2d=fitting_loss_2d,
        fitting_loss_pose_init=fitting_loss_pose_init,
        fitting_loss_beta_init=fitting_loss_beta_init,
        fitting_loss_verts_init=fitting_loss_verts_init,
        fitting_loss_angle=fitting_loss_angle,
        pose_change=pose_change,
        verts_init_dist=verts_init_dist,
        truncation_type=truncation_type,
        pgt_quality=pgt_quality,
        smpl_weight=smpl_weight,
        pose_weight=pose_weight,
        shape_weight=shape_weight,
        cam_weight=cam_weight,
        used_init_fallback=used_init_fallback,
        proj_height_ratio=proj_height_ratio,
        proj_width_ratio=proj_width_ratio,
        proj_area_ratio=proj_area_ratio,
        proj_center_offset=proj_center_offset,
        global_orient_change=global_orient_change,
        body_pose_change=body_pose_change,
        upright_ok=upright_ok,
        reject_low_joints=reject_low_joints,
        reject_reproj=reject_reproj,
        reject_no_improvement=reject_no_improvement,
        reject_depth=reject_depth,
        reject_beta=reject_beta,
        reject_pose_change=reject_pose_change,
        reject_verts_init=reject_verts_init,
        reject_bbox_sanity=reject_bbox_sanity,
        reject_upright=reject_upright,
        reject_global_orient_change=reject_global_orient_change,
        reject_truncation=reject_truncation,
        reject_nan=reject_nan,
        reject_read=reject_read,
        curation_type=curation_type_arr,
        downgraded_full_to_weak=downgraded_full_to_weak,
        # v4 instance-level curation fields
        overlap_ratio=overlap_ratio_arr,
        bbox_height_ratio=bbox_height_ratio_arr,
        bbox_width_ratio=bbox_width_ratio_arr,
        instance_id=instance_id_arr,
        instance_curation_reason=instance_curation_reason_arr,
        selected_init_type=selected_init_type_arr,
        # v4 full-image camera fields (multi-dim; zeros until explicitly computed)
        cam_t_full=cam_t_full_arr,
        focal_length_npz=focal_length_arr,
        img_center_npz=img_center_arr,
        # v5 crop-camera semantics fields
        init_cam_wp=init_cam_wp_arr,
        init_cam_t_crop=init_cam_t_crop_arr,
        camerahmr_condition_focal=camerahmr_condition_focal_arr,
        crop_focal_length=crop_focal_length_arr,
        crop_camera_center=crop_camera_center_arr,
        has_full_camera=has_full_camera_arr,
        camera_model_code=camera_model_code_arr,
    )
    return out_file


def save_npz(
    out_file: str,
    imgname: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    part: np.ndarray,
    openpose: np.ndarray,
    pose: np.ndarray,
    shape: np.ndarray,
    has_smpl: np.ndarray,
    valid_fit: np.ndarray,
    reprojection_error: np.ndarray,
    cam_t: np.ndarray,
    bbox: np.ndarray,
    **extra_arrays: np.ndarray,
) -> None:
    payload = dict(
        imgname=imgname,
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        part=part.astype(np.float32),
        openpose=openpose.astype(np.float32),
        pose=pose.astype(np.float32),
        shape=shape.astype(np.float32),
        has_smpl=has_smpl.astype(np.uint8),
        valid_fit=valid_fit.astype(np.uint8),
        reprojection_error=reprojection_error.astype(np.float32),
        cam_t=cam_t.astype(np.float32),
        bbox=bbox.astype(np.float32),
    )
    for key, value in extra_arrays.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"f", "c"}:
            arr = arr.astype(np.float32)
        elif arr.dtype.kind in {"i", "u", "b"}:
            arr = arr.astype(np.uint8) if key.startswith("reject_") or key in {
                "truncation_type",
                "pgt_quality",
                "used_init_fallback",
                "upright_ok",
                "curation_type",
                "downgraded_full_to_weak",
                "instance_curation_reason",
                "selected_init_type",
                # v5 uint8 fields
                "has_full_camera",
                "camera_model_code",
            } else arr
        payload[key] = arr
    np.savez(out_file, **payload)
    print(f"[save] {out_file}")


def parse_samples(args: argparse.Namespace, dataset: str, split: str) -> List[Sample]:
    if dataset == "llvip":
        samples = parse_llvip_split(args, split)
    elif dataset == "otp2":
        samples = parse_otp2_split(args, split)
    else:
        raise ValueError(dataset)
    max_n = getattr(args, "max_samples", None)
    if max_n and max_n > 0:
        samples = samples[:max_n]
        print(f"[{dataset}/{split}] --max_samples={max_n}: truncated to {len(samples)} samples")
    return samples


def resolve_npz_image_path(dataset: str, split: str, imgname: str, args: argparse.Namespace) -> Optional[str]:
    root = args.llvip_root if dataset == "llvip" else args.otp2_root
    candidates = [
        os.path.join(root, imgname),
        os.path.join(root, split, imgname),
        os.path.join(root, "infrared", split, os.path.basename(imgname)),
        os.path.join(root, split, "images", os.path.basename(imgname)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def validate_npz_shapes(npz: np.lib.npyio.NpzFile, path: str) -> None:
    keys = set(npz.files)
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in keys]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    if "S" in keys:
        raise ValueError(f"{path} must not contain key 'S'")

    n = len(npz["imgname"])
    expected = {
        "center": (n, 2),
        "scale": (n,),
        "part": (n, 24, 3),
        "openpose": (n, 25, 3),
        "pose": (n, 72),
        "shape": (n, 10),
        "has_smpl": (n,),
        "valid_fit": (n,),
        "reprojection_error": (n,),
        "cam_t": (n, 3),
        "bbox": (n, 4),
    }
    for key, shape in expected.items():
        if npz[key].shape != shape:
            raise ValueError(f"{path} key={key} shape={npz[key].shape}, expected={shape}")
    for key in OPTIONAL_NPZ_KEYS:
        if key in keys and npz[key].shape != (n,):
            raise ValueError(f"{path} optional key={key} shape={npz[key].shape}, expected={(n,)}")
    for key in V4_MULTIDIM_OPTIONAL_KEYS:
        if key in keys and npz[key].shape[0] != n:
            raise ValueError(f"{path} v4 key={key} shape={npz[key].shape}, expected first dim={n}")
    for key in V5_MULTIDIM_OPTIONAL_KEYS:
        if key in keys and npz[key].shape[0] != n:
            raise ValueError(f"{path} v5 key={key} shape={npz[key].shape}, expected first dim={n}")


def draw_points(img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], radius: int = 3) -> np.ndarray:
    out = img.copy()
    for x, y, conf in points:
        if conf > 0 and np.isfinite(x) and np.isfinite(y):
            cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_bbox_xywh(img: np.ndarray, bbox: Optional[np.ndarray], color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    if bbox is None or not np.isfinite(bbox).all():
        return out
    x, y, w, h = [float(v) for v in bbox]
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(out, p1, p2, color, 2, lineType=cv2.LINE_AA)
    return out


def project_npz_pose_to_crop(
    smpl: SMPL,
    pose: np.ndarray,
    betas: np.ndarray,
    cam_t: np.ndarray,
    img_res: int,
    focal_length: float,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        pose_t = torch.from_numpy(pose[None].astype(np.float32)).to(device)
        betas_t = torch.from_numpy(betas[None].astype(np.float32)).to(device)
        cam_t_t = torch.from_numpy(cam_t[None].astype(np.float32)).to(device)
        out = smpl(global_orient=pose_t[:, :3], body_pose=pose_t[:, 3:], betas=betas_t)
        proj = project_with_crop_camera(out.joints, cam_t_t, focal_length, img_res)
    return proj[0].detach().cpu().numpy().astype(np.float32)


def visualize_npz_samples(
    npz: np.lib.npyio.NpzFile,
    npz_path: str,
    dataset: str,
    split: str,
    args: argparse.Namespace,
    indices: Sequence[int],
) -> None:
    if args.num_vis <= 0:
        return
    from models.smpl import SMPL

    vis_dir = args.vis_dir or os.path.join(args.out_dir, "sanity_vis", f"{dataset}_{split}")
    os.makedirs(vis_dir, exist_ok=True)

    device = select_device(args.device, allow_cpu_fallback=True)
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)

    for idx in indices:
        imgname = str(npz["imgname"][idx])
        img_path = resolve_npz_image_path(dataset, split, imgname, args)
        if img_path is None:
            continue
        img_rgb = load_rgb(img_path)
        if img_rgb is None:
            continue

        center = npz["center"][idx].astype(np.float32)
        scale = float(npz["scale"][idx])
        part = npz["part"][idx].astype(np.float32)
        openpose = npz["openpose"][idx].astype(np.float32)
        kps49 = build_keypoints49(openpose, part)
        crop_img = crop(img_rgb, center, scale, [args.img_res, args.img_res], rot=0).astype(np.uint8)
        crop_kps = keypoints_full_to_crop(kps49, center, scale, args.img_res)

        orig_bgr = draw_points(img_rgb.astype(np.uint8)[:, :, ::-1], part, (0, 255, 0), radius=3)
        crop_bgr = draw_points(crop_img[:, :, ::-1], crop_kps[25:], (0, 255, 0), radius=3)
        anno_bbox_crop = bbox_from_visible_points(crop_kps[:, :2], crop_kps[:, 2], float(args.thermal_conf_thresh))
        crop_bgr = draw_bbox_xywh(crop_bgr, anno_bbox_crop, (0, 255, 0))

        if int(npz["valid_fit"][idx]) > 0 and int(npz["has_smpl"][idx]) > 0:
            proj = project_npz_pose_to_crop(
                smpl,
                npz["pose"][idx],
                npz["shape"][idx],
                npz["cam_t"][idx],
                args.img_res,
                args.focal_length,
                device,
            )
            proj_kps = np.concatenate([proj, np.ones((proj.shape[0], 1), dtype=np.float32)], axis=1)
            crop_bgr = draw_points(crop_bgr, proj_kps, (0, 0, 255), radius=2)
            proj_bbox_crop = bbox_from_visible_points(proj, crop_kps[:, 2], float(args.thermal_conf_thresh))
            crop_bgr = draw_bbox_xywh(crop_bgr, proj_bbox_crop, (0, 0, 255))

        init_reproj = float(npz["init_reprojection_error"][idx]) if "init_reprojection_error" in npz.files else float("nan")
        opt_reproj = float(npz["reprojection_error"][idx]) if "reprojection_error" in npz.files else float("nan")
        ratio = float(npz["improvement_ratio"][idx]) if "improvement_ratio" in npz.files else float("nan")
        trunc = int(npz["truncation_type"][idx]) if "truncation_type" in npz.files else TRUNC_NONE
        quality = int(npz["pgt_quality"][idx]) if "pgt_quality" in npz.files else -1
        weight = float(npz["smpl_weight"][idx]) if "smpl_weight" in npz.files else float("nan")
        fallback = int(npz["used_init_fallback"][idx]) if "used_init_fallback" in npz.files else 0
        h_ratio = float(npz["proj_height_ratio"][idx]) if "proj_height_ratio" in npz.files else float("nan")
        w_ratio = float(npz["proj_width_ratio"][idx]) if "proj_width_ratio" in npz.files else float("nan")
        upright = int(npz["upright_ok"][idx]) if "upright_ok" in npz.files else -1
        text_lines = [
            f"valid={int(npz['valid_fit'][idx])} trunc={TRUNCATION_NAMES.get(trunc, trunc)} q={quality}",
            f"w={weight:.2f} fallback={fallback} upright={upright}",
            f"opt={opt_reproj:.2f} init={init_reproj:.2f}",
            f"ratio={ratio:.2f} bbox_h/w={h_ratio:.2f}/{w_ratio:.2f}",
        ]
        for line_no, text in enumerate(text_lines):
            cv2.putText(
                crop_bgr,
                text,
                (8, 18 + 18 * line_no),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                crop_bgr,
                text,
                (8, 18 + 18 * line_no),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        orig_scale = min(1.0, 640.0 / max(orig_bgr.shape[1], 1))
        orig_small = cv2.resize(orig_bgr, (int(orig_bgr.shape[1] * orig_scale), int(orig_bgr.shape[0] * orig_scale)))
        canvas = np.zeros((max(orig_small.shape[0], crop_bgr.shape[0]), orig_small.shape[1] + crop_bgr.shape[1], 3), dtype=np.uint8)
        canvas[: orig_small.shape[0], : orig_small.shape[1]] = orig_small
        canvas[: crop_bgr.shape[0], orig_small.shape[1] :] = crop_bgr
        cv2.imwrite(os.path.join(vis_dir, f"{idx:06d}.jpg"), canvas)

    print(f"[check] visualization saved under {vis_dir}")


def sanity_check_npz(args: argparse.Namespace, dataset: str, split: str, npz_path: str) -> None:
    if not os.path.isfile(npz_path):
        print(f"[check] missing {npz_path}")
        return
    npz = np.load(npz_path, allow_pickle=True)
    validate_npz_shapes(npz, npz_path)

    n = len(npz["imgname"])
    valid = npz["valid_fit"].astype(bool)
    finite_reproj = npz["reprojection_error"][np.isfinite(npz["reprojection_error"])]
    if finite_reproj.size:
        mean_reproj = float(np.mean(finite_reproj))
        median_reproj = float(np.median(finite_reproj))
    else:
        mean_reproj = float("nan")
        median_reproj = float("nan")

    print(
        f"[check] {dataset}/{split}: samples={n} valid_fit={int(valid.sum())} "
        f"ratio={float(valid.mean()) if n else 0.0:.4f} "
        f"reproj_mean={mean_reproj:.4f} reproj_median={median_reproj:.4f}"
    )
    if "init_reprojection_error" in npz.files:
        init_vals = npz["init_reprojection_error"][np.isfinite(npz["init_reprojection_error"])]
        if init_vals.size:
            print(
                f"[check] init_reproj_mean={float(np.mean(init_vals)):.4f} "
                f"init_reproj_median={float(np.median(init_vals)):.4f}"
            )
    if "improvement_ratio" in npz.files:
        ratio_vals = npz["improvement_ratio"][np.isfinite(npz["improvement_ratio"])]
        valid_ratio_vals = npz["improvement_ratio"][valid & np.isfinite(npz["improvement_ratio"])]
        if ratio_vals.size:
            print(
                f"[check] improvement_ratio_mean={float(np.mean(ratio_vals)):.4f} "
                f"improvement_ratio_median={float(np.median(ratio_vals)):.4f}"
            )
        if valid_ratio_vals.size:
            print(
                f"[check] valid_fit_improvement_ratio_mean={float(np.mean(valid_ratio_vals)):.4f} "
                f"valid_fit_improvement_ratio_median={float(np.median(valid_ratio_vals)):.4f}"
            )
    reject_keys = [key for key in OPTIONAL_NPZ_KEYS if key.startswith("reject_") and key in npz.files]
    if reject_keys:
        reject_stats = {key.replace("reject_", ""): int(npz[key].astype(bool).sum()) for key in reject_keys}
        print(f"[check] rejection_stats={reject_stats}")
    if "truncation_type" in npz.files:
        trunc_stats = {
            TRUNCATION_NAMES.get(int(t), str(int(t))): int((npz["truncation_type"] == int(t)).sum())
            for t in np.unique(npz["truncation_type"])
        }
        print(f"[check] truncation_stats={trunc_stats}")
    if "used_init_fallback" in npz.files:
        print(f"[check] used_init_fallback={int(npz['used_init_fallback'].astype(bool).sum())}")
    if "proj_height_ratio" in npz.files and "proj_width_ratio" in npz.files:
        finite_h = npz["proj_height_ratio"][np.isfinite(npz["proj_height_ratio"])]
        finite_w = npz["proj_width_ratio"][np.isfinite(npz["proj_width_ratio"])]
        if finite_h.size and finite_w.size:
            print(
                f"[check] proj_height_ratio_median={float(np.median(finite_h)):.4f} "
                f"proj_width_ratio_median={float(np.median(finite_w)):.4f}"
            )
    if "upright_ok" in npz.files:
        print(f"[check] upright_ok_ratio={float(npz['upright_ok'].astype(bool).mean()) if n else 0.0:.4f}")
    if "curation_type" in npz.files:
        ct = npz["curation_type"]
        pq = npz["pgt_quality"] if "pgt_quality" in npz.files else None
        curation_stats = {name: int((ct == cval).sum()) for cval, name in CURATION_NAMES.items()}
        print(f"[check] curation_stats={curation_stats}")
        if pq is not None:
            is_cfull = ct == CURATION_FULL
            is_cweak = ct == CURATION_WEAK
            print(
                f"[check] curation=full -> full_pgt={int(((pq == PGT_QUALITY_FULL) & is_cfull).sum())} "
                f"weak_pgt={int(((pq == PGT_QUALITY_WEAK) & is_cfull).sum())} "
                f"rejected={int(((pq == PGT_QUALITY_REJECTED) & is_cfull).sum())}"
            )
            print(
                f"[check] curation=weak -> weak_pgt={int(((pq == PGT_QUALITY_WEAK) & is_cweak).sum())} "
                f"rejected={int(((pq == PGT_QUALITY_REJECTED) & is_cweak).sum())}"
            )
    if "pgt_quality" in npz.files:
        pq = npz["pgt_quality"]
        pq_stats = {
            "rejected": int((pq == PGT_QUALITY_REJECTED).sum()),
            "weak":     int((pq == PGT_QUALITY_WEAK).sum()),
            "full":     int((pq == PGT_QUALITY_FULL).sum()),
        }
        print(f"[check] pgt_quality_stats={pq_stats}")
    for wkey in ("smpl_weight", "pose_weight", "shape_weight", "cam_weight"):
        if wkey in npz.files:
            valid_w = npz[wkey][valid]
            print(
                f"[check] {wkey}_mean={float(np.mean(npz[wkey])):.4f} "
                f"valid_{wkey}_mean={float(np.mean(valid_w)) if valid.any() else 0.0:.4f}"
            )
    if "downgraded_full_to_weak" in npz.files:
        print(f"[check] downgraded_full_to_weak={int(npz['downgraded_full_to_weak'].astype(bool).sum())}")

    out_of_bounds = 0
    missing_images = 0
    for idx in range(n):
        imgname = str(npz["imgname"][idx])
        img_path = resolve_npz_image_path(dataset, split, imgname, args)
        if img_path is None:
            missing_images += 1
            continue
        size = image_size(img_path)
        if size is None:
            missing_images += 1
            continue
        img_w, img_h = size
        part = npz["part"][idx]
        visible = part[:, 2] > 0
        if visible.any():
            xy = part[visible, :2]
            bad = (xy[:, 0] < 0) | (xy[:, 0] >= img_w) | (xy[:, 1] < 0) | (xy[:, 1] >= img_h)
            out_of_bounds += int(bad.sum())

    print(f"[check] image_missing_or_unreadable={missing_images} visible_keypoints_out_of_bounds={out_of_bounds}")

    rng = random.Random(args.seed)
    vis_n = min(args.num_vis, n)
    if vis_n > 0:
        indices = rng.sample(range(n), vis_n) if n > vis_n else list(range(n))
        visualize_npz_samples(npz, npz_path, dataset, split, args, indices)


def select_device(device_arg: str, allow_cpu_fallback: bool = False) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        if allow_cpu_fallback:
            print("[device] CUDA unavailable; using CPU for this step.")
            return torch.device("cpu")
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(device_arg)


def apply_optional_paths_config(args: argparse.Namespace) -> None:
    if args.paths_cfg and apply_paths_config is not None and os.path.isfile(args.paths_cfg):
        apply_paths_config(args.paths_cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate thermal SPIN-style SMPLify pseudo-GT NPZs.")
    parser.add_argument("--dataset", default="both", choices=["llvip", "otp2", "both"])
    parser.add_argument("--split", default="all", choices=["train", "test", "val", "all"])
    parser.add_argument("--llvip_root", default="/home/q36131134/HMR_models/dataset/infrared/llvip-pose")
    parser.add_argument("--otp2_root", default="/home/q36131134/HMR_models/dataset/infrared/otp2_dataset")
    parser.add_argument(
        "--otp2_exclude_list",
        default=None,
        help="Path to a txt file listing image stems (one per line) to exclude from OTP2 NPZ generation.",
    )
    parser.add_argument("--checkpoint", default=None, help="HMR checkpoint. Required unless --check_only is set.")
    parser.add_argument("--out_dir", default="data/dataset_extras/thermal_pgt_camsmplify_v3_relaxed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_smplify_iters", type=int, default=50)
    parser.add_argument("--bbox_scale_factor", type=float, default=1.2)
    parser.add_argument("--img_res", type=int, default=None)
    parser.add_argument("--focal_length", type=float, default=5000.0)
    parser.add_argument("--reproj_thresh", type=float, default=25.0)
    parser.add_argument("--min_valid_joints", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--num_vis", type=int, default=50)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--paths_cfg", default="configs/paths/local.yaml")

    parser.add_argument("--backbone", default=None, choices=["resnet50", "hrnet_w48"])
    parser.add_argument("--hrnet_pretrained", default=None)
    parser.add_argument("--strict_hrnet", action="store_true")
    parser.add_argument("--hrnet_backbone_input_height", type=int, default=None)
    parser.add_argument("--hrnet_backbone_input_width", type=int, default=None)
    hrnet_crop_group = parser.add_mutually_exclusive_group()
    hrnet_crop_group.add_argument("--hrnet_center_crop_input", dest="hrnet_center_crop_input", action="store_true")
    hrnet_crop_group.add_argument("--no-hrnet_center_crop_input", dest="hrnet_center_crop_input", action="store_false")
    parser.set_defaults(hrnet_center_crop_input=True)
    parser.add_argument("--modality_embed_dim", type=int, default=None)

    parser.add_argument("--smplify_mode", default="thermal_camsmplify", choices=["spin", "thermal_camsmplify"])

    parser.add_argument("--smplify_step_size", type=float, default=1e-2)
    parser.add_argument("--pose_prior_weight", type=float, default=4.78)
    parser.add_argument("--shape_prior_weight", type=float, default=5.0)
    parser.add_argument("--angle_prior_weight", type=float, default=15.2)
    parser.add_argument("--smplify_sigma", type=float, default=100.0)
    parser.add_argument("--max_abs_beta", type=float, default=3.0)

    parser.add_argument("--thermal_phase1_iters", type=int, default=50)
    parser.add_argument("--thermal_phase2_iters", type=int, default=50)
    parser.add_argument("--thermal_phase3_iters", type=int, default=100)
    parser.add_argument("--thermal_smplify_lr", type=float, default=1e-3)
    parser.add_argument("--thermal_gmof_sigma", type=float, default=100.0)

    parser.add_argument("--lambda_2d", type=float, default=1.0)
    parser.add_argument("--lambda_pose_init", type=float, default=50.0)
    parser.add_argument("--lambda_beta_init_phase2", type=float, default=0.01)
    parser.add_argument("--lambda_beta_init_phase3", type=float, default=30.0)
    parser.add_argument("--lambda_verts_init_phase3", type=float, default=10.0)
    parser.add_argument("--lambda_angle_thermal", type=float, default=2.0)

    parser.add_argument("--thermal_conf_thresh", type=float, default=0.25)
    parser.add_argument("--disable_thermal_joint_reweight", action="store_true")

    parser.add_argument("--min_improvement_ratio", type=float, default=0.85)
    parser.add_argument("--very_good_reproj_thresh", type=float, default=12.0)
    parser.add_argument("--max_pose_change", type=float, default=4.0)
    parser.add_argument("--max_verts_init_dist", type=float, default=0.20)
    parser.add_argument("--disable_improvement_check", action="store_true")
    parser.add_argument("--trunc_margin_px", type=float, default=5.0)
    parser.add_argument("--min_fullbody_lower_joints", type=int, default=3)
    parser.add_argument("--min_side_joints", type=int, default=3)
    parser.add_argument("--severe_trunc_min_joints", type=int, default=6)
    parser.add_argument("--reject_severe_truncation", dest="reject_severe_truncation", action="store_true", default=True)
    parser.add_argument("--no-reject_severe_truncation", dest="reject_severe_truncation", action="store_false")
    parser.add_argument("--freeze_betas_for_truncation", dest="freeze_betas_for_truncation", action="store_true", default=True)
    parser.add_argument("--no-freeze_betas_for_truncation", dest="freeze_betas_for_truncation", action="store_false")
    parser.add_argument("--truncated_smpl_weight", type=float, default=0.3)
    parser.add_argument("--truncated_pose_weight", type=float, default=0.3)
    parser.add_argument("--truncated_shape_weight", type=float, default=0.0)
    parser.add_argument("--fullbody_smpl_weight", type=float, default=1.0)
    parser.add_argument("--fullbody_pose_weight", type=float, default=1.0)
    parser.add_argument("--fullbody_shape_weight", type=float, default=0.3)
    parser.add_argument("--thermal_cam_weight", type=float, default=0.0)
    parser.add_argument("--truncated_anchor_multiplier", type=float, default=2.0)
    parser.add_argument("--max_proj_height_ratio", type=float, default=2.2)
    parser.add_argument("--min_proj_height_ratio", type=float, default=0.4)
    parser.add_argument("--max_proj_width_ratio", type=float, default=2.5)
    parser.add_argument("--min_proj_width_ratio", type=float, default=0.35)
    parser.add_argument("--max_proj_center_offset", type=float, default=0.35)
    parser.add_argument("--enable_upright_check", dest="enable_upright_check", action="store_true", default=True)
    parser.add_argument("--no-enable_upright_check", dest="enable_upright_check", action="store_false")
    parser.add_argument("--max_global_orient_change", type=float, default=1.2)
    parser.add_argument("--max_body_pose_change", type=float, default=0.35)
    parser.add_argument("--allow_init_fallback", dest="allow_init_fallback", action="store_true", default=True)
    parser.add_argument("--no-allow_init_fallback", dest="allow_init_fallback", action="store_false")
    parser.add_argument("--init_good_reproj_thresh", type=float, default=15.0)

    # OTP2 curation folders
    parser.add_argument("--otp2_use_curation_folders", action="store_true", default=False)
    parser.add_argument("--otp2_curation_mode", choices=["full_only", "weighted"], default="weighted")
    parser.add_argument("--otp2_unknown_curation_policy", choices=["auto", "skip", "weak", "full"], default="skip")

    # per-curation eligibility thresholds
    parser.add_argument("--min_valid_joints_full", type=int, default=10)
    parser.add_argument("--min_torso_joints_full", type=int, default=3)
    parser.add_argument("--min_valid_joints_weak", type=int, default=6)
    parser.add_argument("--min_torso_joints_weak", type=int, default=2)

    # weak pGT weights
    parser.add_argument("--weak_smpl_weight",  type=float, default=0.3)
    parser.add_argument("--weak_pose_weight",  type=float, default=0.3)
    parser.add_argument("--weak_shape_weight", type=float, default=0.0)

    # weak-mode verts / reproj overrides
    parser.add_argument("--max_verts_init_dist_weak",    type=float, default=0.20)
    parser.add_argument("--good_reproj_override_thresh", type=float, default=6.0)

    # face-only filter for weak
    parser.add_argument("--drop_face_only_persons_in_weak", dest="drop_face_only_persons_in_weak", action="store_true", default=True)
    parser.add_argument("--no-drop_face_only_persons_in_weak", dest="drop_face_only_persons_in_weak", action="store_false")

    # ── v4 new arguments ──────────────────────────────────────────────────────

    # default out_dir for v4
    # (only applied if user didn't pass --out_dir; controlled by changing default above)
    # Change the default out_dir to v4 directory
    parser.set_defaults(out_dir="data/dataset_extras/thermal_pgt_camsmplify_v4")

    # tiered reproj thresholds
    parser.add_argument(
        "--full_reproj_thresh", type=float, default=10.0,
        help="Reproj threshold (px) for pgt_full tier. Default 10px (stricter than weak).",
    )
    # reproj_thresh (already defined above, ~25px) is used as weak_reproj_thresh

    # LLVIP RGB initialization
    parser.add_argument(
        "--llvip_use_rgb_init", action="store_true", default=False,
        help="For LLVIP, load the paired RGB image for HMR initialization instead of thermal.",
    )
    parser.add_argument(
        "--llvip_rgb_root", default=None,
        help="Root dir containing LLVIP RGB images. If None, auto-derived by replacing "
             "'/infrared/' → '/visible/' in llvip_root.",
    )

    # freeze betas for ALL samples
    parser.add_argument(
        "--freeze_betas_for_all", action="store_true", default=False,
        help="Freeze beta for every sample during SMPLify (overrides freeze_betas_for_truncation).",
    )

    # v4 instance-level curation (OTP2)
    parser.add_argument(
        "--v4_instance_level_curation", action="store_true", default=False,
        help="Enable per-instance checks for curated_full OTP2 images (bbox ratio, overlap, joints).",
    )
    parser.add_argument(
        "--instance_overlap_thresh", type=float, default=0.3,
        help="Max overlap_ratio (intersection/own_area) before downgrading curated_full → weak.",
    )
    parser.add_argument(
        "--instance_min_bbox_height_ratio", type=float, default=0.15,
        help="Min bbox_height/image_height ratio for curated_full instances.",
    )
    parser.add_argument(
        "--instance_min_lower_joints", type=int, default=2,
        help="Min lower-body keypoints visible for curated_full instances.",
    )
    parser.add_argument(
        "--instance_min_torso_joints", type=int, default=2,
        help="Min torso keypoints visible for curated_full instances.",
    )

    # bbox center alignment loss in ThermalCamSMPLify_v4
    parser.add_argument(
        "--lambda_bbox", type=float, default=0.0,
        help="Weight for bbox center alignment loss in ThermalCamSMPLify_v4. 0=disabled.",
    )

    # ── v5 new arguments ──────────────────────────────────────────────────────

    parser.add_argument(
        "--init_model",
        choices=["local_hmr", "camerahmr"],
        default="camerahmr",
        help="Which initializer to use: local HMR (v4 behavior) or CameraHMR (v5 default).",
    )
    parser.add_argument(
        "--camerahmr_root",
        default="/home/q36131134/HMR_models/CameraHMR",
        help="Path to CameraHMR repository root (must contain core/).",
    )
    parser.add_argument(
        "--camerahmr_checkpoint",
        default="/home/q36131134/HMR_models/CameraHMR/scripts/data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt",
        help="Path to CameraHMR pretrained checkpoint.",
    )
    parser.add_argument(
        "--llvip_camerahmr_input",
        choices=["paired_rgb", "thermal"],
        default="paired_rgb",
        help="Image source for CameraHMR on LLVIP: paired visible-RGB (default) or thermal.",
    )
    parser.add_argument(
        "--camerahmr_focal_policy",
        choices=["image_diagonal", "fixed"],
        default="image_diagonal",
        help="How to compute conditioning focal length for CameraHMR: "
             "image_diagonal=sqrt(W²+H²) (default), fixed=--camerahmr_fixed_focal.",
    )
    parser.add_argument(
        "--camerahmr_fixed_focal",
        type=float,
        default=1000.0,
        help="Conditioning focal length used when --camerahmr_focal_policy=fixed.",
    )
    parser.add_argument(
        "--smplify_crop_focal",
        type=float,
        default=5000.0,
        help="Focal length (px) for crop-space SMPLify projection. "
             "Replaces focal_length in crop-space ops when init_model=camerahmr.",
    )
    parser.add_argument(
        "--camerahmr_batch_size",
        type=int,
        default=None,
        help="Batch size for CameraHMR inference. Defaults to --batch_size if not set.",
    )
    parser.add_argument(
        "--lambda_bbox_center",
        type=float,
        default=0.1,
        help="Weight for bbox center alignment loss (v5). Reserved for future use.",
    )
    parser.add_argument(
        "--lambda_bbox_size",
        type=float,
        default=0.05,
        help="Weight for bbox size alignment loss (v5). Reserved for future use.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Truncate dataset to at most N samples. For smoke testing only.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_optional_paths_config(args)
    os.makedirs(args.out_dir, exist_ok=True)

    selected = [(dataset, split) for dataset in dataset_values(args.dataset) for split in split_values(args.split, dataset)]

    if args.check_only:
        for dataset, split in selected:
            sanity_check_npz(args, dataset, split, output_path(args.out_dir, dataset, split))
        return

    device = select_device(args.device)

    if args.init_model == "camerahmr":
        # CameraHMR always uses 256×256 crops; set img_res if not explicitly provided.
        if args.img_res is None:
            args.img_res = CAMERAHMR_CROP_RES
            print(f"[v5] img_res not set — defaulting to CAMERAHMR_CROP_RES={CAMERAHMR_CROP_RES}")
        # When using CameraHMR, smplify_crop_focal is the definitive crop-space focal.
        # Override focal_length so internal helpers (compute_reprojection_error, etc.)
        # automatically use the correct value without extra plumbing.
        args.focal_length = args.smplify_crop_focal
        print(f"[v5] init_model=camerahmr — focal_length overridden to smplify_crop_focal={args.focal_length}")
        initializer = CameraHMRInitializer(args, device)
    else:
        # local_hmr: requires --checkpoint; preserves v4 behavior
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --init_model local_hmr")
        model = load_hmr_model(args, device)
        initializer = LocalHMRInitializer(model, args, device)
        print(f"[v5] init_model=local_hmr — using checkpoint={args.checkpoint}")

    smplify = build_smplify(args, device)

    for dataset, split in selected:
        samples = parse_samples(args, dataset, split)
        npz_path = generate_split(args, dataset, split, samples, initializer, smplify, device)
        sanity_check_npz(args, dataset, split, npz_path)


if __name__ == "__main__":
    main()

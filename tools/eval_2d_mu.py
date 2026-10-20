#!/usr/bin/env python3
"""eval_2d_mu.py — evaluate an MMPedUnifiedHMR2 checkpoint on LLVIP-Pose / OTP2.

Metrics: PCK@0.05 / PCK@0.10 (bbox-diagonal normalised), 2D-MPJPE (px), OKS AP
(COCO). With --eval_3d: 3D-MPJPE / PA-MPJPE against the SMPLify pseudo-GT NPZ.

Eval modes (--eval_mode):
  rgb_ir       — paired RGB+IR forward (default for LLVIP)
  ir_only      — IR-only forward via modality_mask=[0,1] (default for OTP2)
  rgb_only     — RGB-only path model(x=rgb); bit-exact original HMR2.0
  baseline_ir  — original HMR2.0 on IR: model(x=ir), bypasses the unified
                 encoder and all adapters (frozen core only)
  baseline_rgb — same as rgb_only, named symmetrically with baseline_ir
  rgb_only_fused, ir_core_go, rgb_as_ir — diagnostics (see run_batch)

Camera: pred_cam is decoded with --decode_focal (default 5000, HMR2's native
convention) for both cam_crop2full and the 2D projection.

pGT matching (match_persons_to_pgt) drops invalid (valid_fit=False) NPZ entries
BEFORE the nearest-centre search, so a valid but slightly farther entry can still
match a person.

Usage:
    python tools/eval_2d_mu.py \\
        --checkpoint weights/main/model_final.pt \\
        --train_cfg  configs/train/main/s5_dcg.yaml \\
        --dataset llvip --eval_mode rgb_ir --eval_3d
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize

_HERE = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.dirname(_HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
import constants
from utils.geometry import cam_crop2full, bbox_info_from_center_scale
from utils.imutils import crop as spin_crop

# ── Dataset roots (overridden by paths_cfg if provided) ───────────────────────
LLVIP_ROOT = config.LLVIP_ROOT
OTP2_ROOT  = config.OTP2_DATASET_ROOT


# LLVIP visible (RGB) partner root. IR frames live under
# <LLVIP_ROOT>/infrared/test/<name>.jpg; their pixel-aligned visible partner is
# <LLVIP_VISIBLE_ROOT>/test/<name>.jpg — NOT under LLVIP_ROOT. This mirrors
# datasets/llvip_paired_dataset.py (config.LLVIP_VISIBLE_ROOT + strip 'infrared/').
LLVIP_VISIBLE_ROOT = config.LLVIP_VISIBLE_ROOT

BBOX_SCALE_FACTOR = 1.2

# H36M-24 → COCO-17 (joint re-index for projection comparison)
H36M_JOINT_START = 25
SPIN24_TO_COCO17 = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

# Root joint for 3D-MPJPE recentering, within the 24-joint GT-superset block
# sliced at H36M_JOINT_START (constants.py JOINT_NAMES[25:49]). Set from
# --root_joint in main(); default 'hip_mid' matches this repo's established
# convention (train/loss_mixer.py:keypoint_3d_loss's gt_pelvis/pred_pelvis).
#   'ankle'       : block index 0  = 'Right Ankle'. An extremity, far
#                   (~900-1550mm) from most other joints -> non-PA 3D-MPJPE is
#                   heavily inflated by any global-orientation mismatch between
#                   pred and pGT (lever-arm effect), on top of genuine
#                   articulated-pose error.
#   'hip_mid'     : mean(block index 2 'Right Hip', index 3 'Left Hip') --
#                   matches train/loss_mixer.py:keypoint_3d_loss's convention.
#   'pelvis_mpii' : block index 14 = 'Pelvis (MPII)', SMPL's own root/pelvis joint.
ROOT_JOINT_MODE = "hip_mid"


def _recenter_joints(joints_h36m):
    """Root-relative recentering of a (B, 24, 3) H36M-block joint tensor,
    per the module-level ROOT_JOINT_MODE."""
    if ROOT_JOINT_MODE == "pelvis_mpii":
        root = joints_h36m[:, 14:15, :]
    elif ROOT_JOINT_MODE == "ankle":
        root = joints_h36m[:, 0:1, :]
    else:  # 'hip_mid' (default, backward-compatible with the prior _pelvis_center)
        root = (joints_h36m[:, 2:3, :] + joints_h36m[:, 3:4, :]) / 2.0
    return joints_h36m - root

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ── GT loading ─────────────────────────────────────────────────────────────────

def _bbox_xywh_to_center_scale(x, y, w, h):
    cx = float(x) + float(w) / 2.0
    cy = float(y) + float(h) / 2.0
    box_size = max(float(w), float(h)) * BBOX_SCALE_FACTOR
    return np.array([cx, cy], dtype=np.float32), float(box_size / 200.0)


def load_llvip_persons(llvip_root: str) -> List[Dict]:
    ann_path = osp.join(llvip_root, "annotations", "test_llvip.json")
    with open(ann_path) as f:
        data = json.load(f)
    img_info_map = {img["id"]: img for img in data["images"]}
    persons = []
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_info = img_info_map[ann["image_id"]]
        img_w, img_h = int(img_info["width"]), int(img_info["height"])
        fname = img_info["file_name"]
        ir_path  = osp.join(llvip_root, "infrared", "test", fname)
        # Visible partner lives under LLVIP_VISIBLE_ROOT/test/<name> (strip the
        # 'infrared/' prefix), matching LLVIPPairedDataset used at training time.
        rgb_path = osp.join(LLVIP_VISIBLE_ROOT, "test", fname)
        gt_kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        gt_kps[1, 2] = 0.0  # LLVIP joints 1,2 are head_bottom/top, not eyes
        gt_kps[2, 2] = 0.0
        if (gt_kps[:, 2] > 0).sum() == 0:
            continue
        x, y, w, h = ann["bbox"]
        center, scale = _bbox_xywh_to_center_scale(x, y, w, h)
        persons.append({
            "ir_path":   ir_path,
            "rgb_path":  rgb_path,
            "center":    center,
            "scale":     scale,
            "gt_kps":    gt_kps,
            "img_w":     img_w,
            "img_h":     img_h,
            "bbox_w":    float(w),
            "bbox_h":    float(h),
            "bbox_area": float(ann.get("area", w * h)),
        })
    return persons


def load_otp2_persons(otp2_root: str) -> List[Dict]:
    label_dir = osp.join(otp2_root, "test", "labels")
    img_dir   = osp.join(otp2_root, "test", "full")
    persons = []
    for lbl_file in sorted(os.listdir(label_dir)):
        if not lbl_file.endswith(".txt"):
            continue
        stem = osp.splitext(lbl_file)[0]
        img_path = osp.join(img_dir, stem + ".png")
        if not osp.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        with open(osp.join(label_dir, lbl_file)) as f:
            lines = f.read().strip().splitlines()
        for line in lines:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 56:
                continue
            _, x_n, y_n, w_n, h_n = vals[:5]
            w_px = w_n * img_w
            h_px = h_n * img_h
            x_px = x_n * img_w
            y_px = y_n * img_h
            center, scale = _bbox_xywh_to_center_scale(
                x_px - w_px / 2.0, y_px - h_px / 2.0, w_px, h_px
            )
            kps_raw = np.array(vals[5:5 + 51], dtype=np.float32).reshape(17, 3)
            gt_kps = np.empty_like(kps_raw)
            gt_kps[:, 0] = kps_raw[:, 0] * img_w
            gt_kps[:, 1] = kps_raw[:, 1] * img_h
            gt_kps[:, 2] = kps_raw[:, 2]
            persons.append({
                "ir_path":   img_path,
                "rgb_path":  None,
                "center":    center,
                "scale":     scale,
                "gt_kps":    gt_kps,
                "img_w":     img_w,
                "img_h":     img_h,
                "bbox_w":    float(w_px),
                "bbox_h":    float(h_px),
                "bbox_area": float(w_px * h_px),
            })
    return persons


# ── Dataset ────────────────────────────────────────────────────────────────────

def _load_crop(img_path, center, scale, img_res, grayscale=False):
    """Load an image and return a normalized (3, H, W) tensor. Returns zeros on failure."""
    normalize = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    if img_path is None or not osp.isfile(img_path):
        return torch.zeros(3, img_res, img_res), False
    bgr = cv2.imread(img_path)
    if bgr is None:
        return torch.zeros(3, img_res, img_res), False
    rgb = bgr[:, :, ::-1].copy()
    crop = spin_crop(rgb, center, scale, (img_res, img_res)).astype(np.float32) / 255.0
    if grayscale:
        gray = (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2])
        crop = np.stack([gray, gray, gray], axis=-1)
    t = normalize(torch.from_numpy(crop).permute(2, 0, 1))
    return t, True


class PairedPersonDataset(Dataset):
    def __init__(self, persons: List[Dict], img_res: int, eval_mode: str):
        self.persons   = persons
        self.img_res   = img_res
        self.eval_mode = eval_mode  # 'rgb_ir' | 'ir_only' | 'rgb_only'

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        p     = self.persons[idx]
        c, s  = p["center"], p["scale"]
        img_h, img_w = float(p["img_h"]), float(p["img_w"])

        ir_t,  ir_ok  = _load_crop(p["ir_path"],  c, s, self.img_res, grayscale=True)
        rgb_t, rgb_ok = _load_crop(p["rgb_path"], c, s, self.img_res, grayscale=False)

        # True diagonal focal estimate sqrt(W^2+H^2) (matches this checkpoint
        # family's training-time CAMERA.FOCAL_ESTIMATE_MODE=diagonal, see
        # utils/geometry.py:estimate_focal_length), not eval_hmr_2d.py's fixed
        # HMR2_FOCAL=5000. Overridden by --decode_focal in run_batch (default 5000).
        # BUGFIX 2026-07-05: this previously used max(img_w, img_h) ("max" mode),
        # which silently diverged from the "diagonal" convention the comment
        # claimed to match — a ~28% focal error on LLVIP (1280 vs 1639px) that
        # explained part of a spurious internal-vs-external MPJPE gap.
        img_focal = float((img_w ** 2 + img_h ** 2) ** 0.5)
        valid = float(ir_ok or rgb_ok)

        return {
            "img_ir":    ir_t,
            "img_rgb":   rgb_t,
            "ir_valid":  float(ir_ok),
            "rgb_valid": float(rgb_ok),
            "center":    torch.tensor(c,        dtype=torch.float32),
            "scale":     torch.tensor(s,        dtype=torch.float32),
            "img_h":     torch.tensor(img_h,    dtype=torch.float32),
            "img_w":     torch.tensor(img_w,    dtype=torch.float32),
            "img_focal": torch.tensor(img_focal, dtype=torch.float32),
            "valid":     torch.tensor(valid,     dtype=torch.float32),
            "idx":       torch.tensor(idx,       dtype=torch.long),
        }


# ── Model building ─────────────────────────────────────────────────────────────

def _strip_module(sd):
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _load_mmped_model_cfg(train_cfg_path: Optional[str]):
    """Pull MODEL.UNIFIED_ENCODER / IR_TOKEN_ADAPTER / BODY_ADAPTER dicts from a
    train YAML. Unlike LT-RHMR, this wrapper is NOT safely auto-detectable:
    MMPedStyleHMR2Unifier unconditionally builds ALL of its sub-heads
    (per_token_conf_mlp, direct_conf_mlp, conf_mlp) regardless of which fusion
    mode is active, so a missing key can never reveal that e.g.
    USE_PER_TOKEN_CONFIDENCE was True at train time — guessing wrong silently
    changes which fusion codepath runs. nn.MultiheadAttention's HEADS count is
    likewise invisible in its weight shapes. --train_cfg is therefore required.
    """
    if not train_cfg_path:
        raise ValueError(
            "MMPedUnifiedHMR2 checkpoints require --train_cfg: hyperparameters "
            "like UNIFIED_ENCODER.{HEADS,USE_PER_TOKEN_CONFIDENCE,"
            "USE_DIRECT_CONFIDENCE,MAA_RESIDUAL_MAX_SCALE} and BODY_ADAPTER.* "
            "are plain Python attributes (not nn.Parameters) that cannot be "
            "recovered from the checkpoint state_dict, and guessing wrong "
            "silently runs the wrong fusion codepath instead of failing loudly."
        )
    cfg = config.load_any_config(train_cfg_path)
    model_section = cfg.get("MODEL", {}) if isinstance(cfg, dict) else {}
    ue_cfg = dict(model_section.get("UNIFIED_ENCODER", {}) or {})
    ir_cfg = dict(model_section.get("IR_TOKEN_ADAPTER", {}) or {})
    ba_cfg = dict(model_section.get("BODY_ADAPTER", {}) or {})
    return ue_cfg, ir_cfg, ba_cfg, model_section


# Released weights store only the trainable modules (ARG / MA / DCG). The frozen
# HMR2.0 backbone + head are bit-identical to MODEL.HMR2_CHECKPOINT, so they are
# re-read from there (see tools/export_release_weights.py).
TRAINABLE_ONLY_FORMAT = "mu_trainable_only_v1"


def _merge_frozen_hmr2(state_dict, train_cfg_path):
    from models.hmr2_core import build_hmr2_core
    if not train_cfg_path:
        raise ValueError("Trainable-only checkpoints require --train_cfg (MODEL.HMR2_CHECKPOINT).")
    hmr2_path = config.load_any_config(train_cfg_path)["MODEL"]["HMR2_CHECKPOINT"]
    if not osp.isabs(hmr2_path):
        hmr2_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), hmr2_path)
    core = build_hmr2_core(config.SMPL_MEAN_PARAMS, ckpt_path=hmr2_path, freeze=True)
    frozen = core.state_dict()
    overlap = set(frozen) & set(state_dict)
    if overlap:
        raise RuntimeError(f"trainable-only checkpoint overlaps HMR2.0 keys: {sorted(overlap)[:5]}")
    merged = dict(frozen)
    merged.update(state_dict)
    print(f"[build] trainable-only checkpoint: {len(state_dict)} keys + {len(frozen)} frozen HMR2.0 keys")
    return merged


def build_mu_hmr(checkpoint_path: str, device: torch.device, img_res: int = 256,
                 train_cfg_path: Optional[str] = None):
    """Build MMPedUnifiedHMR2 (HMR2 core + unified RGB-T encoder) from a checkpoint.

    Architecture hyper-parameters that are not recoverable from tensor shapes
    come from --train_cfg (MODEL.UNIFIED_ENCODER / IR_TOKEN_ADAPTER / BODY_ADAPTER
    / GO_RESIDUAL_*). Loading is strict: any unexpected key is an error, so an
    optional module present in the checkpoint can never be silently dropped.
    """
    from models.hmr2_core import build_hmr2_core
    from models.multimodal import (
        build_mmped_hmr2_unified_encoder, build_bounded_ir_token_adapter,
        build_body_adapters, MMPedUnifiedHMR2,
    )
    from models.multimodal.mmped_unified_hmr2 import BoundedGlobalOrientResidual

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _strip_module(
        ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    )
    if isinstance(ckpt, dict) and ckpt.get("format") == TRAINABLE_ONLY_FORMAT:
        state_dict = _merge_frozen_hmr2(state_dict, train_cfg_path)
    if not any(k.startswith("unified_encoder.") for k in state_dict):
        raise ValueError("Not an MMPedUnifiedHMR2 checkpoint (no 'unified_encoder.*' keys).")

    head_kwargs = {}
    if "head.transformer.pos_embedding" in state_dict:
        head_kwargs["dim"] = int(state_dict["head.transformer.pos_embedding"].shape[-1])
    import re as _re
    depths = set()
    for k in state_dict:
        m = _re.search(r"head\.transformer\.transformer\.layers\.(\d+)\.", k)
        if m:
            depths.add(int(m.group(1)))
    if depths:
        head_kwargs["depth"] = max(depths) + 1
    if "head.transformer.transformer.layers.0.1.fn.to_kv.weight" in state_dict:
        head_kwargs["context_dim"] = int(
            state_dict["head.transformer.transformer.layers.0.1.fn.to_kv.weight"].shape[1]
        )
    print(f"[build] core=hmr2 | head_kwargs={head_kwargs}")
    hmr_core = build_hmr2_core(
        config.SMPL_MEAN_PARAMS,
        ckpt_path=None,
        freeze=False,
        head_kwargs=head_kwargs or None,
    )

    ue_cfg, ir_cfg, ba_cfg, model_section = _load_mmped_model_cfg(train_cfg_path)
    # Checkpoints without CMT weights (no unified_encoder.blocks.*) -> do not build the CMT.
    if not any(k.startswith("unified_encoder.blocks.") for k in state_dict):
        ue_cfg["USE_CMT"] = False
        print("[build] checkpoint has no CMT weights -> USE_CMT=False")
    unified_encoder = build_mmped_hmr2_unified_encoder(ue_cfg)

    has_ir_adapter = any(k.startswith("ir_token_adapter.") for k in state_dict)
    ir_adapter = build_bounded_ir_token_adapter(ir_cfg) if has_ir_adapter else None

    has_body_adapters = any(k.startswith("body_adapters.") for k in state_dict)
    if has_body_adapters:
        body_adapters, body_adapter_block_indices = build_body_adapters(ba_cfg)
    else:
        body_adapters, body_adapter_block_indices = None, None

    ir_patch_stem = None
    if any(k.startswith("ir_patch_stem.") for k in state_dict):
        from models.multimodal.mmped_unified_hmr2 import IRPatchStem
        ips_cfg = dict(model_section.get("IR_PATCH_STEM", {}) or {})
        ir_patch_stem = IRPatchStem(hmr_core.backbone.build_ir_patch_stem(),
                                    max_scale=float(ips_cfg.get("MAX_SCALE", 1.0)),
                                    init_raw_scale=float(ips_cfg.get("INIT_RAW_SCALE", 0.0)))

    go_residual = None
    if any(k.startswith("go_residual.") for k in state_dict):
        go_residual = BoundedGlobalOrientResidual(
            dim=int(model_section.get("GO_RESIDUAL_DIM", 1280)),
            bottleneck_dim=int(model_section.get("GO_RESIDUAL_BOTTLENECK_DIM", 128)),
            max_delta=float(model_section.get("GO_RESIDUAL_MAX_DELTA", 0.2)),
            init_raw_scale=float(model_section.get("GO_RESIDUAL_INIT_RAW_SCALE", 0.0)))

    print(f"[build] unified_encoder: dim={ue_cfg.get('DIM', 1280)} "
          f"depth={ue_cfg.get('DEPTH', 1)} heads={ue_cfg.get('HEADS', 8)} "
          f"per_token_conf={ue_cfg.get('USE_PER_TOKEN_CONFIDENCE', False)}")
    print(f"[build] ir_token_adapter={'on' if has_ir_adapter else 'off'} | "
          f"body_adapters={('on(' + str(len(body_adapters)) + ')') if has_body_adapters else 'off'} | "
          f"ir_patch_stem={'on' if ir_patch_stem is not None else 'off'} | "
          f"go_residual={'on' if go_residual is not None else 'off'}")

    # Deep Cross-modal Gate (DCG): built only when the checkpoint has xmodal_deep_adapters.*.
    xm_adapters, xm_blocks = None, None
    if any(k.startswith("xmodal_deep_adapters.") for k in state_dict):
        from models.multimodal.mmped_unified_hmr2 import build_xmodal_deep_modules
        xm_blocks = [int(b) for b in model_section["XMODAL_DEEP_BLOCKS"]]
        xm_adapters = build_xmodal_deep_modules(
            xm_blocks, str(model_section.get("XMODAL_DEEP_TYPE", "adapter")),
            int(model_section.get("XMODAL_DEEP_BOTTLENECK_DIM", 128)), 0.0,
            float(model_section.get("XMODAL_DEEP_MAX_REL_SCALE", 0.0)),
            float(model_section.get("XMODAL_DEEP_INIT_LOGIT", -4.0)))
        print(f"[build] xmodal_deep_adapters=on({len(xm_blocks)}) blocks={xm_blocks}")

    model = MMPedUnifiedHMR2(
        hmr_core, unified_encoder, ir_token_adapter=ir_adapter,
        body_adapters=body_adapters, body_adapter_block_indices=body_adapter_block_indices,
        ir_patch_stem=ir_patch_stem,
        freeze_hmr2_body=True, freeze_head=True,
        go_residual=go_residual,
        xmodal_deep_adapters=xm_adapters, xmodal_deep_block_indices=xm_blocks,
    )
    model.xmodal_ref_input = str(model_section.get("XMODAL_DEEP_REF_INPUT", "raw"))

    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys:
        print(f"[build] missing_keys={incompat.missing_keys[:10]}")
    if incompat.unexpected_keys:
        raise RuntimeError(f"[build] unexpected checkpoint keys (module not built): "
                           f"{incompat.unexpected_keys[:10]}")

    model = model.to(device).eval()
    print(f"[build] mmped_unified_hmr2 loaded from {checkpoint_path}")
    return model

# ── Per-batch inference ────────────────────────────────────────────────────────

def _project_joints(joints_h36m, pred_cam_full, img_focal_t, img_w_t, img_h_t):
    joints_cam = joints_h36m + pred_cam_full.unsqueeze(1)
    f  = img_focal_t.unsqueeze(1)
    cx = (img_w_t * 0.5).unsqueeze(1)
    cy = (img_h_t * 0.5).unsqueeze(1)
    Z  = joints_cam[:, :, 2].clamp(min=1e-6)
    x_2d = f * joints_cam[:, :, 0] / Z + cx
    y_2d = f * joints_cam[:, :, 1] / Z + cy
    pred_h36m = torch.stack([x_2d, y_2d, torch.ones_like(x_2d)], dim=-1)
    pred_coco = pred_h36m[:, SPIN24_TO_COCO17, :].cpu().numpy()
    return pred_coco, x_2d.cpu().numpy(), y_2d.cpu().numpy()


@torch.no_grad()
def run_batch(batch, model, smpl, device, eval_mode: str, decode_focal="diagonal"):
    from models import SMPL

    img_ir  = batch["img_ir"].to(device).float()
    img_rgb = batch["img_rgb"].to(device).float()
    center    = batch["center"].to(device).float()
    scale     = batch["scale"].to(device).float()
    img_focal = batch["img_focal"].to(device).float()
    img_h     = batch["img_h"].to(device).float()
    img_w     = batch["img_w"].to(device).float()
    B = img_ir.shape[0]

    # P0 (2026-07-05): camera-convention fix. HMR2's head was trained to decode
    # pred_cam under FOCAL_LENGTH=5000, so reporting against the eval_hmr_2d.py
    # HMR2 baseline (which uses fixed 5000) requires the SAME focal here. Because
    # `img_focal` feeds BOTH cam_crop2full (weak->full decode, tz=2f/(s*b)) and
    # _project_joints (x=f*X/Z+cx), overriding it in one place keeps the whole
    # camera chain self-consistent. decode_focal="diagonal" keeps the legacy
    # per-sample sqrt(W^2+H^2) estimate for backward comparison.
    if str(decode_focal).lower() != "diagonal":
        img_focal = torch.full_like(img_focal, float(decode_focal))

    if eval_mode in ("rgb_only", "baseline_rgb"):
        # baseline_rgb: original HMR2.0 on RGB (frozen-core legacy path,
        # bypasses unified_encoder/adapters, see mmped_unified_hmr2.py:89-97).
        # Same call as rgb_only — kept as a distinct eval_mode name only so
        # result tables can label it symmetrically with baseline_ir.
        pred_rotmat, pred_betas, pred_camera = model(x=img_rgb)
    elif eval_mode == "baseline_ir":
        # ORIGINAL HMR2.0 on IR: legacy single-image path bypasses the unified
        # encoder + all adapters (both img_rgb/img_ir None -> frozen core only).
        # patch_embed expects 3ch, so expand grayscale IR like the fusion path does.
        ir3 = img_ir.repeat(1, 3, 1, 1) if img_ir.shape[1] == 1 else img_ir
        pred_rotmat, pred_betas, pred_camera = model(x=ir3)
    elif eval_mode == "ir_core_go":
        # EXPERIMENTAL: articulated pose (joints 1-23) from the IR adapter path,
        # global orient (joint 0) from the FROZEN CORE (baseline_ir legacy path).
        # Tests the global-orient decomposition: the adapters improve articulated
        # pose (PA-MPJPE) but degrade global orient (raw 3D-MPJPE); the frozen
        # core has the better global orient. Camera kept from the adapter path
        # (3D-MPJPE is root-relative so camera is irrelevant to it).
        mask = torch.tensor([[0.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_ir=img_ir, modality_mask=mask)
        ir3c = img_ir.repeat(1, 3, 1, 1) if img_ir.shape[1] == 1 else img_ir
        core_rotmat, _, _ = model(x=ir3c)
        pred_rotmat = torch.cat([core_rotmat[:, :1], pred_rotmat[:, 1:]], dim=1)
    elif eval_mode == "ir_only":
        mask = torch.tensor([[0.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_ir=img_ir, modality_mask=mask)
    elif eval_mode == "rgb_only_fused":
        # RGB 走「融合 wrapper」而非 legacy 單影像路徑：modality_mask=[1,0]。
        # 與 eval_mode='rgb_only' 的差別：後者呼叫 model(x=rgb)，直接繞過
        # unified_encoder 與所有 adapter，因此與 baseline_rgb 逐位元相同（同一行程式碼）。
        # 本模式會真正經過 unified_encoder 的 rgb_only hard-routing 與 body adapter 的
        # adapter_mask 判斷 —— 也就是訓練時 rgb_only 列實際走的路徑。
        # 配合 --force_body_adapter_all_rows 即可量化「解除 adapter 封印」的效果。
        mask = torch.tensor([[1.0, 0.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(
            img_rgb=img_rgb, modality_mask=mask)
    elif eval_mode == "rgb_as_ir":
        # DIAGNOSTIC: real RGB crop fed into the img_ir slot, mask=[0,1] (ir_only
        # hard-routing). img_rgb left None so the wrapper auto-zero-fills it
        # (unused anyway — ir_only routing never reads tokens_rgb). Same GT/images
        # as rgb_only, isolates the effect of running RGB content through the
        # IR-token adapter + body-adapter (both trained on real thermal stats).
        mask = torch.tensor([[0.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_ir=img_rgb, modality_mask=mask)
    else:  # rgb_ir
        mask = torch.tensor([[1.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(
            img_rgb=img_rgb, img_ir=img_ir, modality_mask=mask)

    pred_output = smpl(
        betas=pred_betas,
        body_pose=pred_rotmat[:, 1:],
        global_orient=pred_rotmat[:, 0].unsqueeze(1),
        pose2rot=False,
    )
    joints_h36m  = pred_output.joints[:, H36M_JOINT_START:H36M_JOINT_START + 24, :]
    img_center   = torch.stack([img_w * 0.5, img_h * 0.5], dim=-1)
    pred_cam_full = cam_crop2full(pred_camera, center, scale, img_focal, img_center)

    pred_coco, x2d, y2d = _project_joints(joints_h36m, pred_cam_full, img_focal, img_w, img_h)
    cam_z     = pred_cam_full[:, 2].cpu().numpy()
    beta_norm = pred_betas.norm(dim=-1).cpu().numpy()
    joints_rr = _recenter_joints(joints_h36m).cpu().numpy()
    return pred_coco, cam_z, beta_norm, x2d, y2d, img_w.cpu().numpy(), img_h.cpu().numpy(), joints_rr


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_pck(gt_kps_all, pred_kps_all, persons, threshold=0.05):
    per_correct = np.zeros(17, dtype=np.float64)
    per_total   = np.zeros(17, dtype=np.float64)
    for gt, pred, p in zip(gt_kps_all, pred_kps_all, persons):
        thresh_px = threshold * np.sqrt(p["bbox_w"] ** 2 + p["bbox_h"] ** 2)
        for j in range(17):
            if gt[j, 2] <= 0:
                continue
            per_total[j] += 1
            if np.linalg.norm(gt[j, :2] - pred[j, :2]) <= thresh_px:
                per_correct[j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        pck = np.where(per_total > 0, per_correct / per_total, np.nan)
    return pck, float(np.nanmean(pck))


def compute_mpjpe_px(gt_kps_all, pred_kps_all):
    per_dist  = np.zeros(17, dtype=np.float64)
    per_total = np.zeros(17, dtype=np.float64)
    for gt, pred in zip(gt_kps_all, pred_kps_all):
        for j in range(17):
            if gt[j, 2] <= 0:
                continue
            per_dist[j]  += np.linalg.norm(gt[j, :2] - pred[j, :2])
            per_total[j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        per_j = np.where(per_total > 0, per_dist / per_total, np.nan)
    return per_j, float(np.nanmean(per_j))


def compute_oks_ap(gt_kps_all, pred_kps_all, centers, scales, persons):
    try:
        from pycocotools.coco     import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        try:
            from xtcocotools.coco     import COCO
            from xtcocotools.cocoeval import COCOeval
        except ImportError:
            print("[WARNING] pycocotools / xtcocotools not installed — OKS AP skipped.")
            return None

    def _cxywh(c, s):
        sz = float(s) * 200.0
        return [float(c[0]) - sz / 2, float(c[1]) - sz / 2, sz, sz]

    images, annotations = [], []
    for i, kps in enumerate(gt_kps_all):
        images.append({"id": i, "width": 4096, "height": 4096})
        nkp = int((kps[:, 2] > 0).sum())
        kps_flat = [float(v) for kp in kps for v in [kp[0], kp[1], int(kp[2] > 0)]]
        annotations.append({
            "id": i + 1, "image_id": i, "category_id": 1,
            "keypoints": kps_flat, "num_keypoints": nkp,
            "bbox": _cxywh(centers[i], scales[i]),
            "area": persons[i]["bbox_area"], "iscrowd": 0,
        })
    gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person",
                        "keypoints": COCO_KP_NAMES, "skeleton": [],
                        "supercategory": "person"}],
    }
    dt_list = [{
        "image_id": i, "category_id": 1,
        "keypoints": pred_kps_all[i].flatten().tolist(),
        "score": float(pred_kps_all[i, :, 2].mean()) if pred_kps_all[i, :, 2].max() > 0 else 0.01,
    } for i in range(len(pred_kps_all))]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); gt_p = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dt_list, f); dt_p = f.name
    try:
        import io, contextlib
        coco_gt = COCO(gt_p)
        coco_dt = coco_gt.loadRes(dt_p)
        ev = COCOeval(coco_gt, coco_dt, "keypoints")
        ev.params.imgIds = list(range(len(gt_kps_all)))
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(); ev.accumulate(); ev.summarize()
        return {
            "AP":   float(ev.stats[0]),
            "AP50": float(ev.stats[1]),
            "AP75": float(ev.stats[2]),
            "AP_M": float(ev.stats[3]),
            "AP_L": float(ev.stats[4]),
            "AR":   float(ev.stats[5]),
        }
    finally:
        os.unlink(gt_p); os.unlink(dt_p)


# ── Optional 3D metrics (pGT-matched) ─────────────────────────────────────────

_PGT_V5_DIR = osp.join(REPO_ROOT, "data", "dataset_extras", "thermal_pgt_camsmplify_v5")
DEFAULT_PGT_NPZ = {
    "llvip": osp.join(_PGT_V5_DIR, "llvip_test_smplify_pgt.npz"),
    "otp2":  osp.join(_PGT_V5_DIR, "otp2_test_smplify_pgt.npz"),
}


def match_persons_to_pgt(persons, npz_path, center_tol=5.0):
    from collections import defaultdict
    data      = np.load(npz_path, allow_pickle=False)
    npz_names = data["imgname"]
    npz_cen   = data["center"]
    npz_poses = data["pose"]
    npz_shapes= data["shape"]
    npz_camt  = data["cam_t"]
    npz_valid = data["valid_fit"].astype(bool)
    n = len(persons)
    out_pose  = np.zeros((n, 72),  dtype=np.float32)
    out_shape = np.zeros((n, 10),  dtype=np.float32)
    out_camt  = np.zeros((n, 3),   dtype=np.float32)
    out_valid = np.zeros(n,        dtype=bool)
    by_name   = defaultdict(list)
    for eidx, nm in enumerate(npz_names):
        by_name[osp.basename(str(nm))].append(eidx)
    for i, p in enumerate(persons):
        pc = p["center"]
        bname = osp.basename(p["ir_path"])
        best_dist, best_idx = float("inf"), None
        # Excludes invalid candidates BEFORE the nearest-center search (unlike
        # eval_hmr_2d.py, which finds the globally nearest entry first and only
        # checks valid_fit afterward) — see module docstring, deviation
        # confirmed 2026-07-04: 501/3431 LLVIP basenames have mixed
        # valid/invalid entries, so this recovers real matches eval_hmr_2d.py
        # would silently drop.
        for eidx in by_name.get(bname, []):
            if not npz_valid[eidx]:
                continue
            d = float(np.linalg.norm(pc - npz_cen[eidx]))
            if d < best_dist:
                best_dist, best_idx = d, eidx
        if best_idx is not None and best_dist < center_tol and npz_valid[best_idx]:
            out_pose[i]  = npz_poses[best_idx]
            out_shape[i] = npz_shapes[best_idx]
            out_camt[i]  = npz_camt[best_idx]
            out_valid[i] = True
    print(f"[pGT] {out_valid.sum()}/{n} matched (center ≤{center_tol}px)")
    return out_pose, out_shape, out_camt, out_valid


def procrustes_align_np(S1, S2):
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2   = S1 - mu1, S2 - mu2
    ss1 = (X1 ** 2).sum()
    if ss1 < 1e-12:
        return S1.copy()
    K  = X2.T @ X1
    U, sigma, Vt = np.linalg.svd(K)
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U * np.array([1.0, 1.0, d])) @ Vt
    c = (sigma * np.array([1.0, 1.0, d])).sum() / ss1
    return c * (R @ X1.T).T + mu2


def compute_3d_metrics(gt_j3d, pred_j3d, gt_valid, pred_valid):
    mask = gt_valid & pred_valid
    n_ok = int(mask.sum())
    if n_ok == 0:
        return float("nan"), float("nan")
    gt_s   = gt_j3d[mask][:, SPIN24_TO_COCO17, :]
    pr_s   = pred_j3d[mask][:, SPIN24_TO_COCO17, :]
    mpjpe  = float(np.linalg.norm(gt_s - pr_s, axis=-1).mean() * 1000.0)
    pa_err = np.empty(n_ok, dtype=np.float64)
    for k, (g, p) in enumerate(zip(gt_s, pr_s)):
        try:
            pa_err[k] = np.linalg.norm(g - procrustes_align_np(p, g), axis=-1).mean()
        except np.linalg.LinAlgError:
            pa_err[k] = np.linalg.norm(g - p, axis=-1).mean()
    pampjpe = float(pa_err.mean() * 1000.0)
    print(f"[3D] n_valid={n_ok}  MPJPE={mpjpe:.2f}mm  PA-MPJPE={pampjpe:.2f}mm")
    return mpjpe, pampjpe


def print_joint_table(pck, pck_mean, mpjpe, mpjpe_mean, threshold):
    C0, C1, C2 = 20, 14, 12
    print(f"\n┌{'─'*(C0+2)}┬{'─'*(C1+2)}┬{'─'*(C2+2)}┐")
    print(f"│ {'Joint':<{C0}} │ {'PCK@'+str(threshold)+'(%)':^{C1}} │ {'MPJPE(px)':^{C2}} │")
    print(f"├{'─'*(C0+2)}┼{'─'*(C1+2)}┼{'─'*(C2+2)}┤")
    for j, name in enumerate(COCO_KP_NAMES):
        pc = f"{pck[j]*100:.1f}" if not np.isnan(pck[j]) else "—"
        mp = f"{mpjpe[j]:.1f}"   if not np.isnan(mpjpe[j]) else "—"
        print(f"│ {name:<{C0}} │ {pc:^{C1}} │ {mp:^{C2}} │")
    print(f"├{'─'*(C0+2)}┼{'─'*(C1+2)}┼{'─'*(C2+2)}┤")
    print(f"│ {'MEAN':<{C0}} │ {pck_mean*100:^{C1}.1f} │ {mpjpe_mean:^{C2}.1f} │")
    print(f"└{'─'*(C0+2)}┴{'─'*(C1+2)}┴{'─'*(C2+2)}┘")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="2D eval for ModalityUnifiedHMR on LLVIP / OTP2")
    p.add_argument("--checkpoint",       required=True,
                   help="MMPedUnifiedHMR2 checkpoint (.pt).")
    p.add_argument("--model_type",       default="mu_hmr", choices=["mu_hmr"],
                   help="Kept for command-line compatibility; only 'mu_hmr' is supported.")
    p.add_argument("--dataset",          required=True, choices=["llvip", "otp2"])
    p.add_argument("--eval_mode",        default=None,
                   choices=["rgb_ir", "ir_only", "rgb_only", "rgb_only_fused", "baseline_ir", "baseline_rgb", "ir_core_go", "rgb_as_ir"],
                   help="Fusion mode. Defaults to rgb_ir for llvip, ir_only for otp2. "
                        "'baseline_ir' feeds IR through the frozen-core legacy path "
                        "(model(x=ir)) — bit-exact ORIGINAL HMR2.0 on IR, for a fair "
                        "same-pipeline baseline vs the fusion modes. 'baseline_rgb' is "
                        "the RGB counterpart (model(x=rgb)) — bit-exact ORIGINAL HMR2.0 "
                        "on RGB; numerically identical to 'rgb_only', provided as a "
                        "separate name for symmetric baseline_ir/baseline_rgb reporting. "
                        "'rgb_as_ir' feeds the REAL RGB crop into the img_ir argument "
                        "with modality_mask=[0,1] (ir_only routing) — same GT/images as "
                        "rgb_only, but the RGB tokens are pushed through the IR-token "
                        "adapter + body-adapter path that was trained on real thermal "
                        "statistics. Diagnostic mode: quantifies how much the IR-specific "
                        "trainable modules perturb output when the modality flag is wrong.")
    p.add_argument("--train_cfg",        default=None,
                   help="Path to the train YAML that produced --checkpoint (required: "
                        "fusion hyper-parameters are not stored in the state_dict).")
    p.add_argument("--paths_cfg",        default=None)
    p.add_argument("--img_res",          type=int, default=256)
    p.add_argument("--batch_size",       type=int, default=32)
    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--pck_threshold",    type=float, default=0.05)
    p.add_argument("--decode_focal",     default="5000",
                   help="Camera decode/projection focal. '5000' = HMR2 native "
                        "convention (fair vs eval_hmr_2d.py baseline, DEFAULT); "
                        "'diagonal' = legacy per-sample sqrt(W^2+H^2). P0 fix "
                        "2026-07-05.")
    p.add_argument("--eval_3d",          action="store_true",
                   help="Also compute 3D MPJPE / PA-MPJPE against pGT NPZ.")
    p.add_argument("--pgt_npz",          default=None,
                   help="Path to pGT NPZ. Defaults to thermal_pgt_camsmplify_v5/.")
    p.add_argument("--root_joint",       default="hip_mid",
                   choices=["ankle", "hip_mid", "pelvis_mpii"],
                   help="Root joint for non-PA 3D-MPJPE recentering (PA-MPJPE is "
                        "translation-invariant and unaffected by this choice). "
                        "'hip_mid' (default) = mean(Right Hip, Left Hip), matches "
                        "train/loss_mixer.py:keypoint_3d_loss's convention. "
                        "'ankle' = block idx 0 'Right Ankle' (an extremity -> "
                        "non-PA 3D-MPJPE inflated by global-orientation mismatch, "
                        "lever-arm effect). 'pelvis_mpii' = block idx 14, SMPL's "
                        "own pelvis joint.")
    p.add_argument("--force_body_adapter_all_rows", action="store_true",
                   help="診斷用（推理期）：強制 body adapter 的 adapter_mask=1，讓 "
                        "rgb_only 的列也吃到 adapter 殘差。預設情況下 rgb_only 的 "
                        "adapter_mask=0（見 mmped_unified_hmr2.py），使 RGB 路徑與凍結 "
                        "HMR2.0 逐位元相同；本旗標用來量化該設計選擇的代價/收益。"
                        "僅影響推理，不改變任何已訓練權重。")
    p.add_argument("--out",              default=None, help="Save result JSON.")
    return p.parse_args()


def main():
    args = parse_args()

    global ROOT_JOINT_MODE
    ROOT_JOINT_MODE = args.root_joint
    if args.eval_3d:
        print(f"[3D] root_joint={ROOT_JOINT_MODE} "
              f"(non-PA 3D-MPJPE recentering anchor; PA-MPJPE unaffected)")

    if args.eval_mode is None:
        args.eval_mode = "ir_only" if args.dataset == "otp2" else "rgb_ir"

    print(f"Eval | model={args.model_type}  dataset={args.dataset}  "
          f"mode={args.eval_mode}  "
          f"device={args.device}  batch={args.batch_size}  img_res={args.img_res}")
    _df = "diagonal sqrt(W^2+H^2)" if str(args.decode_focal).lower() == "diagonal" \
          else f"native f={args.decode_focal} (HMR2 convention)"
    print(f"  decode_focal={_df}")
    print(f"  ckpt={args.checkpoint}")

    # Apply paths config
    paths_cfg_path = args.paths_cfg or osp.join(REPO_ROOT, "configs", "paths", "local.yaml")
    if osp.isfile(paths_cfg_path):
        path_cfg = config.load_paths_config(paths_cfg_path)
        config.apply_paths_config(path_cfg, verbose=False)
    # Dataset roots come from configs/paths/local.yaml (via config).
    global LLVIP_ROOT, OTP2_ROOT, LLVIP_VISIBLE_ROOT
    LLVIP_ROOT = config.LLVIP_ROOT
    OTP2_ROOT = config.OTP2_DATASET_ROOT
    LLVIP_VISIBLE_ROOT = config.LLVIP_VISIBLE_ROOT

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load GT annotations
    if args.dataset == "llvip":
        persons = load_llvip_persons(LLVIP_ROOT)
    else:
        persons = load_otp2_persons(OTP2_ROOT)
        if args.eval_mode == "rgb_ir":
            print("[WARNING] OTP2 has no paired RGB — falling back to ir_only.")
            args.eval_mode = "ir_only"
    n = len(persons)
    print(f"Samples | total={n}")

    gt_kps_all  = np.array([p["gt_kps"]  for p in persons], dtype=np.float32)
    centers_all = np.array([p["center"]  for p in persons], dtype=np.float32)
    scales_all  = np.array([p["scale"]   for p in persons], dtype=np.float32)

    # Build model
    model = build_mu_hmr(args.checkpoint, device, img_res=args.img_res,
                         train_cfg_path=args.train_cfg)
    if getattr(args, 'force_body_adapter_all_rows', False):
        n_ad = len(getattr(model, 'body_adapters', []) or [])
        if n_ad == 0:
            raise ValueError('--force_body_adapter_all_rows 需要 checkpoint 含 '
                             'body_adapters，但這個模型沒有。')
        model.force_body_adapter_all_rows = True
        print(f'[build] ★ force_body_adapter_all_rows=True '
              f'({n_ad} 個 body adapter 對所有列生效，含 rgb_only)')

    from models import SMPL
    smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)

    pred_kps_all   = np.zeros((n, 17, 3), dtype=np.float32)
    pred_j3d_all   = np.zeros((n, 24, 3), dtype=np.float32)
    pred_valid_arr = np.zeros(n, dtype=bool)
    cam_z_arr      = np.full(n, np.inf,  dtype=np.float32)
    beta_norm_arr  = np.full(n, np.inf,  dtype=np.float32)

    print("Running inference ...")
    # Build dataset + loader (per-crop path)
    dataset = PairedPersonDataset(persons, img_res=args.img_res, eval_mode=args.eval_mode)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    report_at = max(1, n // 10)
    processed = 0
    with torch.no_grad():
        for batch in loader:
            valid   = batch["valid"].numpy()
            indices = batch["idx"].numpy()

            pred_coco, cam_z, beta_norm, x2d, y2d, iw, ih, joints_rr = \
                run_batch(batch, model, smpl, device, args.eval_mode,
                          decode_focal=args.decode_focal)

            for b, idx in enumerate(indices):
                if valid[b] < 0.5:
                    continue
                pred_kps_all[idx]   = pred_coco[b]
                pred_j3d_all[idx]   = joints_rr[b]
                pred_valid_arr[idx] = True
                cam_z_arr[idx]      = float(cam_z[b])
                beta_norm_arr[idx]  = float(beta_norm[b])

            processed += len(indices)
            if processed % report_at < len(indices) or processed >= n:
                print(f"  [{processed}/{n}]  {processed*100//n}%")

    # 2D metrics
    pck_per_j, pck_mean = compute_pck(gt_kps_all, pred_kps_all, persons, args.pck_threshold)
    pck10_per_j, pck10_mean = compute_pck(gt_kps_all, pred_kps_all, persons, threshold=0.10)
    mpjpe_per_j, mpjpe_mean = compute_mpjpe_px(gt_kps_all, pred_kps_all)
    ap_metrics = compute_oks_ap(gt_kps_all, pred_kps_all, centers_all, scales_all, persons)

    print_joint_table(pck_per_j, pck_mean, mpjpe_per_j, mpjpe_mean, args.pck_threshold)
    if ap_metrics:
        print(f"\nOKS mAP | " + " | ".join(f"{k}={v:.4f}" for k, v in ap_metrics.items()))

    print(f"\n*** Final Results on {args.dataset} (mode={args.eval_mode}, "
          f"decode_focal={args.decode_focal}) ***")
    print(f"PCK@{args.pck_threshold}:  {pck_mean*100:.2f}%")
    print(f"PCK@0.10:  {pck10_mean*100:.2f}%")
    print(f"MPJPE:    {mpjpe_mean:.2f} px")

    results = {
        "dataset":      args.dataset,
        "model_type":   args.model_type,
        "eval_mode":    args.eval_mode,
        "checkpoint":   args.checkpoint,
        "decode_focal": args.decode_focal,
        "pck_mean":     pck_mean,
        "pck10_mean":   pck10_mean,
        "mpjpe_mean":   mpjpe_mean,
        **(ap_metrics or {}),
    }

    # Optional 3D metrics
    if args.eval_3d:
        npz_path = args.pgt_npz or DEFAULT_PGT_NPZ.get(args.dataset)
        if npz_path and osp.isfile(npz_path):
            from utils.geometry import batch_rodrigues
            gt_pose, gt_shape, gt_camt, gt_valid = match_persons_to_pgt(persons, npz_path)
            gt_j3d  = np.zeros((n, 24, 3), dtype=np.float32)
            idx_ok  = np.where(gt_valid)[0]
            for start in range(0, len(idx_ok), 128):
                bi = idx_ok[start: start + 128]
                pt = torch.from_numpy(gt_pose[bi]).float().to(device)
                st = torch.from_numpy(gt_shape[bi]).float().to(device)
                rm = batch_rodrigues(pt.reshape(-1, 3)).reshape(-1, 24, 3, 3)
                with torch.no_grad():
                    out = smpl(betas=st, body_pose=rm[:, 1:],
                               global_orient=rm[:, 0].unsqueeze(1), pose2rot=False)
                j  = out.joints[:, H36M_JOINT_START:H36M_JOINT_START + 24, :]
                gt_j3d[bi] = _recenter_joints(j).cpu().numpy()
            mpjpe_3d, pa_mpjpe = compute_3d_metrics(gt_j3d, pred_j3d_all, gt_valid, pred_valid_arr)
            results["mpjpe_3d_mm"]   = mpjpe_3d
            results["pa_mpjpe_3d_mm"] = pa_mpjpe
            print(f"3D MPJPE:    {mpjpe_3d:.2f} mm")
            print(f"3D PA-MPJPE: {pa_mpjpe:.2f} mm")
        else:
            print(f"[WARNING] pGT NPZ not found: {npz_path} — 3D eval skipped.")

    if args.out:
        import json
        os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in results.items()}, f, indent=2)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()

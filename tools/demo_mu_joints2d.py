#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把我的模型（MMPedUnifiedHMR2）的 SMPL 輸出重投影成 2D 關節點，畫成骨架圖。
========================================================================
跟 demo_mu_advanced.py 走同一條推論路徑（build_mu_hmr / _load_crop / GT bbox
/ cam_crop2full），差別只在最後不渲染 mesh 表面、也不疊在原圖上，而是把
SMPL 的 49 個關節（25 OpenPose + 24 GT，見 constants.JOINT_NAMES）投影到
全圖像素座標後，每個人各自重新置中縮放，畫成白底的 2D 骨架圖（只畫後 24
個 GT 關節；OpenPose 那 25 個目前沒有輸入監督，數值沒有意義）。

用法
------------------------------------------------------------------------
  python tools/demo_mu_joints2d.py \\
      --checkpoint weights/main/model_final.pt \\
      --train_cfg configs/train/main/s5_dcg.yaml \\
      --dataset LLVIP --mode rgb_ir \\
      --image <LLVIP_ROOT>/infrared/test/190538.jpg \\
      --bbox_file <bbox json> \\
      --output_folder demo_output/190538/portraits
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
import constants
from utils.geometry import cam_crop2full, perspective_projection
import tools.eval_2d_mu as E
import tools.demo_mu as D

HMR2_FOCAL = 5000.0

# GT24（constants.JOINT_NAMES 的第 25~48 個）局部索引下的骨架連線。
# 0 R-Ankle 1 R-Knee 2 R-Hip 3 L-Hip 4 L-Knee 5 L-Ankle 6 R-Wrist 7 R-Elbow
# 8 R-Shoulder 9 L-Shoulder 10 L-Elbow 11 L-Wrist 12 Neck 13 HeadTop
# 14 Pelvis 15 Thorax 16 Spine 17 Jaw 18 Head 19 Nose 20 L-Eye 21 R-Eye
# 22 L-Ear 23 R-Ear
GT24_BONES = [
    (0, 1), (1, 2), (2, 14), (3, 14), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 15), (9, 15), (9, 10), (10, 11),
    (14, 15), (15, 12), (12, 13), (12, 18), (18, 19),
    (19, 20), (19, 21), (20, 22), (21, 23),
]
GT24_LEFT = {3, 4, 5, 9, 10, 11, 20, 22}
GT24_RIGHT = {0, 1, 2, 6, 7, 8, 21, 23}
# 配色比照 results/190538/portraits/190538_person00_skeleton.png：
# 左側藍、右側橘（matplotlib tab:blue / tab:orange），中軸灰，白底、
# 圓點帶深色描邊。
COLOR_LEFT = (180, 119, 31)    # BGR，tab:blue
COLOR_RIGHT = (14, 127, 255)   # BGR，tab:orange
COLOR_MID = (110, 110, 110)    # 中軸灰
COLOR_OUTLINE = (60, 60, 60)


def _bone_color(a: int, b: int):
    if a in GT24_LEFT or b in GT24_LEFT:
        return COLOR_LEFT
    if a in GT24_RIGHT or b in GT24_RIGHT:
        return COLOR_RIGHT
    return COLOR_MID


def draw_skeleton(canvas_bgr: np.ndarray, joints_gt24_xy: np.ndarray) -> np.ndarray:
    out = canvas_bgr
    for a, b in GT24_BONES:
        pa, pb = joints_gt24_xy[a], joints_gt24_xy[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(out, tuple(pa.astype(int)), tuple(pb.astype(int)), _bone_color(a, b), 3, cv2.LINE_AA)
    for idx in range(24):
        p = joints_gt24_xy[idx]
        if not np.all(np.isfinite(p)):
            continue
        color = COLOR_LEFT if idx in GT24_LEFT else COLOR_RIGHT if idx in GT24_RIGHT else COLOR_MID
        cv2.circle(out, tuple(p.astype(int)), 6, color, -1, cv2.LINE_AA)
        cv2.circle(out, tuple(p.astype(int)), 6, COLOR_OUTLINE, 1, cv2.LINE_AA)
    return out


def render_person_skeleton_portrait(joints_gt24_xy: np.ndarray, canvas_size: int = 640,
                                     margin_ratio: float = 0.12) -> np.ndarray:
    """把單一人的 GT24 2D 關節（任意原圖像素尺度）重新置中、縮放到填滿一張
    白底正方形畫布，不依賴原圖大小/該人在畫面裡的實際位置。"""
    valid = np.isfinite(joints_gt24_xy).all(axis=1)
    pts = joints_gt24_xy[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, 1e-3)
    margin = canvas_size * margin_ratio
    scale = (canvas_size - 2 * margin) / span
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    shifted = (joints_gt24_xy - np.array([cx, cy])) * scale + canvas_size / 2.0

    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    return draw_skeleton(canvas, shifted)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="SMPL 關節點重投影 demo")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--dataset", required=True, choices=["3DPW", "H36M", "LLVIP", "OTP2"])
    ap.add_argument("--mode", default="auto", choices=["auto", "rgb_only", "ir_only", "rgb_ir"])
    ap.add_argument("--image", required=True, help="單張圖片路徑")
    ap.add_argument("--bbox_file", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--decode_focal", type=float, default=HMR2_FOCAL)
    ap.add_argument("--img_res", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    mode = D.DATASET_DEFAULT_MODE[args.dataset] if args.mode == "auto" else args.mode
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    paths_cfg = os.path.join(REPO_ROOT, "configs", "paths", "local.yaml")
    if os.path.isfile(paths_cfg):
        try:
            config.apply_paths_config(config.load_paths_config(paths_cfg), verbose=False)
        except Exception as e:
            print(f"[WARN] paths config 套用失敗：{e}")

    from models import SMPL
    print(f"[joints2d] 載入模型 {args.checkpoint}")
    model = E.build_mu_hmr(args.checkpoint, device, img_res=args.img_res, train_cfg_path=args.train_cfg)
    model.eval()
    smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)

    bbox_map = json.load(open(args.bbox_file))
    fname = os.path.basename(args.image)
    boxes = bbox_map.get(fname) or bbox_map.get(Path(args.image).stem)
    if not boxes:
        raise RuntimeError(f"{fname} 在 {args.bbox_file} 裡沒有 bbox")

    ir_path, rgb_path, bg_bgr = D.resolve_inputs(args.dataset, mode, args.image)
    if bg_bgr is None:
        raise RuntimeError("找不到背景圖")
    img_h, img_w = bg_bgr.shape[:2]

    ir_list, rgb_list, cs_list = [], [], []
    for xyxy in boxes:
        center, scale = D.xyxy_to_center_scale(xyxy)
        need_ir = mode in ("ir_only", "rgb_ir")
        need_rgb = mode in ("rgb_only", "rgb_ir")
        ir_t = (E._load_crop(ir_path, center, scale, args.img_res, grayscale=True)[0]
                if need_ir else torch.zeros(3, args.img_res, args.img_res))
        rgb_t = (E._load_crop(rgb_path, center, scale, args.img_res, grayscale=False)[0]
                 if need_rgb else torch.zeros(3, args.img_res, args.img_res))
        ir_list.append(ir_t)
        rgb_list.append(rgb_t)
        cs_list.append((center, scale))

    img_ir = torch.stack(ir_list).to(device).float()
    img_rgb = torch.stack(rgb_list).to(device).float()
    centers = torch.tensor(np.stack([c for c, _ in cs_list]), dtype=torch.float32, device=device)
    scales = torch.tensor(np.array([s for _, s in cs_list]), dtype=torch.float32, device=device)
    B = img_ir.shape[0]
    focal_t = torch.full((B,), float(args.decode_focal), device=device)

    if mode == "rgb_only":
        pred_rotmat, pred_betas, pred_camera = model(x=img_rgb)
    elif mode == "ir_only":
        mask = torch.tensor([[0.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_ir=img_ir, modality_mask=mask)
    else:
        mask = torch.tensor([[1.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_rgb=img_rgb, img_ir=img_ir, modality_mask=mask)

    out = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
               global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
    joints49 = out.joints  # [B, 49, 3]：0-24 OpenPose（未監督，忽略）、25-48 GT24

    # 跟 demo_mu.py 的 mesh 疊圖走同一條相機還原路徑：cam_crop2full 直接吃
    # 模型輸出的 crop 弱透視相機 [s,tx,ty]，換算成全圖相機平移，再用標準
    # 透視投影把任意 3D 點（這裡是關節而非 mesh 頂點）投影回全圖像素座標。
    img_center = torch.tensor([[img_w * 0.5, img_h * 0.5]], device=device).expand(B, 2)
    pred_cam_full = cam_crop2full(pred_camera, centers, scales, focal_t, img_center)
    rot_identity = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
    proj49 = perspective_projection(
        joints49, rotation=rot_identity, translation=pred_cam_full,
        focal_length=args.decode_focal, camera_center=img_center,
    ).detach().cpu().numpy()

    os.makedirs(args.output_folder, exist_ok=True)
    stem = Path(args.image).stem
    for i in range(B):
        proj_gt24 = proj49[i, 25:49, :]
        portrait = render_person_skeleton_portrait(proj_gt24)
        out_path = os.path.join(args.output_folder, f"{stem}_person{i:02d}_joints2d.png")
        cv2.imwrite(out_path, portrait)
    print(f"[joints2d] {B} 人 → {args.output_folder}/{stem}_person##_joints2d.png")


if __name__ == "__main__":
    main()

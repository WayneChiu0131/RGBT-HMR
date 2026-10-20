#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
我的 RGB-T 模型（MMPedUnifiedHMR2 / scheme3c）demo 渲染腳本
========================================================================
在完整解析度原圖上疊加預測的 SMPL mesh，支援 4 個資料集與 LLVIP 的三種輸入
模式（RGB-only / IR-only / RGB-IR）。

與評估流程的一致性
------------------------------------------------------------------------
本腳本 **直接 import tools.eval_2d_mu**（記為 E），重用其 build_mu_hmr()、
_load_crop()、_bbox_xywh_to_center_scale() 與相機鏈，確保 demo 所渲染的 mesh
與 eval_2d_mu.py 產生指標時走的是**完全相同的推論路徑**：
  - IR 影像同樣做 grayscale（0.299R+0.587G+0.114B）再疊三通道
  - 同樣以 f=5000（HMR2 原生慣例，見 docs/rootcause_..._0705.md 的 P0）解碼相機
  - 模式派發同 run_batch()：
      rgb_only → model(x=img_rgb)                      （凍結核心單張）
      ir_only  → model(img_ir=, modality_mask=[[0,1]])
      rgb_ir   → model(img_rgb=, img_ir=, modality_mask=[[1,1]])

各資料集的預設模式 / 輸入
------------------------------------------------------------------------
  3DPW / H36M : rgb_only（可見光單張，examples 影像即輸入）
  OTP2        : ir_only （純熱像，examples 影像即輸入）
  LLVIP       : rgb_ir（預設）/ ir_only / rgb_only 三選一
                IR = examples 影像；RGB = 對應 visible 配對圖

渲染背景
------------------------------------------------------------------------
  ir_only / rgb_ir → 疊在 IR（熱像）上，展示熱域結果
  rgb_only         → 疊在 RGB（可見光）上

用法
------------------------------------------------------------------------
  python tools/demo_mu.py \
      --checkpoint weights/main/model_final.pt \
      --train_cfg configs/train/main/s5_dcg.yaml \
      --dataset LLVIP --mode rgb_ir \
      --image_folder <LLVIP_ROOT>/infrared/test \
      --bbox_file <bbox json> \
      --output_folder demo_output/MyModel/LLVIP/rgb_ir
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
import constants
from utils.geometry import cam_crop2full
import tools.eval_2d_mu as E  # 重用 build_mu_hmr / _load_crop / bbox 轉換

try:
    import pyrender
    import trimesh
except Exception as e:  # pragma: no cover
    pyrender = None
    trimesh = None
    _IMPORT_ERR = e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
HMR2_FOCAL = 5000.0
# mesh 頂點座標系是 Y-down（v=f*Y/Z+cy，跟 image row 同向）、Z 為離相機的深度。
# 對著相機的人套用 R_y(+deg)（cv2.Rodrigues([0,rad,0])）會讓人物的臉轉向畫面
# 右側、即觀者視角的「順時針」——已用實際渲染結果核對過方向。
ROTATE_Y_SIGN = 1.0

# LLVIP visible（RGB）配對根目錄，與 eval_2d_mu 一致
LLVIP_VISIBLE_ROOT = config.LLVIP_VISIBLE_ROOT

DATASET_DEFAULT_MODE = {
    "3DPW": "rgb_only",
    "H36M": "rgb_only",
    "OTP2": "ir_only",
    "LLVIP": "rgb_ir",
}


# ─── 渲染器（沿用 SPIN_infrared/demo.py 的 FullImageRenderer）──────────────

class FullImageRenderer:
    # 與 demo_baselines.py / SPIN_infrared/demo.py 統一：同色相加深加飽和的藍
    # （原始 HMR/4D-Humans LIGHT_BLUE 在灰階熱影像背景上太淡看不出輪廓）。
    def __init__(self, faces, mesh_color=(0.24, 0.34, 0.56, 1.0)):
        if pyrender is None or trimesh is None:
            raise ImportError(f"需要 pyrender/trimesh/PyOpenGL：{_IMPORT_ERR}")
        self.faces = faces
        self.mesh_color = mesh_color

    def _make_mesh(self, vertices):
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=self.mesh_color)
        return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    @staticmethod
    def _add_lights(scene):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        for tx, ty, tz in [(0, -1, 1), (0, 1, 1), (1, 1, 2)]:
            pose = np.eye(4)
            pose[:3, 3] = np.array([tx, ty, tz])
            scene.add(light, pose=pose)

    def render_front_view(self, vertices_list, bg_img_rgb, focal_length, supersample=3):
        # supersample 抗鋸齒：N 倍解析度渲染 mesh-only 後用 INTER_AREA 降採樣，
        # 邊緣才不會是硬遮罩造成的鋸齒（會被誤以為「解析度變差」）。背景圖本身
        # 解析度不變。
        bg = bg_img_rgb.astype(np.float32) / 255.0 if bg_img_rgb.dtype == np.uint8 \
            else np.clip(bg_img_rgb.astype(np.float32), 0.0, 1.0)
        img_h, img_w = bg.shape[:2]
        ss = max(1, int(supersample))
        W, H = img_w * ss, img_h * ss
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.25, 0.25, 0.25])
        camera = pyrender.IntrinsicsCamera(
            fx=float(focal_length) * ss, fy=float(focal_length) * ss,
            cx=W / 2.0, cy=H / 2.0)
        scene.add(camera, pose=np.eye(4))
        self._add_lights(scene)
        for v in vertices_list:
            scene.add(self._make_mesh(v))
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        color = color.astype(np.float32) / 255.0
        alpha = (depth > 0).astype(np.float32)
        if ss > 1:
            color = cv2.resize(color[:, :, :3], (img_w, img_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (img_w, img_h), interpolation=cv2.INTER_AREA)
        else:
            color = color[:, :, :3]
        out = bg * (1.0 - alpha[:, :, None]) + color * alpha[:, :, None]
        return np.clip(out, 0.0, 1.0)

    def render_mesh_transparent(self, vertices_list, img_h, img_w, focal_length, supersample=3,
                                margin=18):
        """去背 mesh：只渲染 mesh、透明背景，回傳 (H', W', 4) uint8 BGRA。

        自動外擴畫布（auto-fit）：先用與 mesh 疊圖相同的透視投影
        （u=focal*X/Z+w/2, v=focal*Y/Z+h/2）算出所有 mesh 頂點的投影範圍，
        再把畫布擴到剛好同時涵蓋『原圖框 + 全部 mesh 投影 + margin』，並平移
        principal point。這樣不論人物的腳（或被 bbox 截半的人所預測出的完整
        下半身）延伸到原圖邊界外多遠，都完整入鏡不被截斷。"""
        all_v = np.concatenate([np.asarray(v, dtype=np.float32) for v in vertices_list], axis=0)
        z = np.clip(all_v[:, 2], 1e-6, None)
        u = focal_length * all_v[:, 0] / z + img_w / 2.0
        v = focal_length * all_v[:, 1] / z + img_h / 2.0
        left = int(np.floor(min(0.0, u.min() - margin)))
        right = int(np.ceil(max(float(img_w), u.max() + margin)))
        top = int(np.floor(min(0.0, v.min() - margin)))
        bottom = int(np.ceil(max(float(img_h), v.max() + margin)))
        out_w, out_h = right - left, bottom - top
        ss = max(1, int(supersample))
        W, H = out_w * ss, out_h * ss
        cx = (img_w / 2.0 - left) * ss
        cy = (img_h / 2.0 - top) * ss
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.25, 0.25, 0.25])
        camera = pyrender.IntrinsicsCamera(
            fx=float(focal_length) * ss, fy=float(focal_length) * ss, cx=cx, cy=cy)
        scene.add(camera, pose=np.eye(4))
        self._add_lights(scene)
        for vv in vertices_list:
            scene.add(self._make_mesh(vv))
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        color = color[:, :, :3].astype(np.float32) / 255.0
        alpha = (depth > 0).astype(np.float32)
        if ss > 1:
            color = cv2.resize(color, (out_w, out_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_AREA)
        bgr = np.clip(color, 0.0, 1.0)[:, :, ::-1]
        bgra = np.concatenate([bgr, np.clip(alpha, 0.0, 1.0)[:, :, None]], axis=-1)
        return (bgra * 255).astype(np.uint8)


# ─── 同一人重複 mesh 去重 ────────────────────────────────────────────────

def dedup_person_meshes(vertices_cam_list, bbox_xyxy_list, focal_length,
                        img_w, img_h, iou_thresh=0.5):
    """去重：單人 HMR 對「含多人的方形裁切圖」可能對兩個相鄰 bbox 都輸出同一個
    人的姿態，造成同一人身上疊兩個 mesh。把每個 mesh 投影回全圖算 2D footprint，
    兩個 footprint 的 IoU 高於 iou_thresh 判定為同一人，只留一個。

    保留優先序：mesh 投影中心與其來源 bbox 中心偏移越小越優先（偏移小＝這個框
    確實框中這個人；偏移大＝該框本應框鄰居卻latch到別人，就是要丟的重複）。
    回傳保留的索引（依原順序）。
    """
    n = len(vertices_cam_list)
    if n <= 1:
        return list(range(n))
    cx, cy = img_w / 2.0, img_h / 2.0

    def proj(verts):
        z = np.clip(verts[:, 2], 1e-6, None)
        x = focal_length * verts[:, 0] / z + cx
        y = focal_length * verts[:, 1] / z + cy
        return (np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32),
                np.array([x.mean(), y.mean()], dtype=np.float32))

    pboxes, offsets = [], []
    for verts, bbox in zip(vertices_cam_list, bbox_xyxy_list):
        pb, pc = proj(np.asarray(verts, dtype=np.float32))
        bb = np.asarray(bbox, dtype=np.float32)
        bcenter = np.array([(bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0], dtype=np.float32)
        pboxes.append(pb)
        offsets.append(float(np.linalg.norm(pc - bcenter)))

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    kept = []
    for i in sorted(range(n), key=lambda k: offsets[k]):
        if all(iou(pboxes[i], pboxes[k]) <= iou_thresh for k in kept):
            kept.append(i)
    return sorted(kept)


# ─── 影像/背景/輸入解析 ──────────────────────────────────────────────────

def list_images(folder: str) -> List[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if os.path.splitext(f)[1] in IMG_EXTS)


def visible_partner_path(ir_example_path: str) -> Optional[str]:
    """LLVIP：由 IR examples 影像找對應 visible 配對圖（同 basename）。"""
    fname = os.path.basename(ir_example_path)
    cand = os.path.join(LLVIP_VISIBLE_ROOT, "test", fname)
    return cand if os.path.isfile(cand) else None


def resolve_inputs(dataset: str, mode: str, example_path: str):
    """回傳 (ir_path, rgb_path, background_bgr)。找不到背景回傳 None。"""
    ir_path = rgb_path = None
    if dataset == "LLVIP":
        ir_path = example_path
        rgb_path = visible_partner_path(example_path)
        if mode == "rgb_only":
            bg_src = rgb_path
        else:
            bg_src = ir_path
    elif dataset == "OTP2":
        ir_path = example_path
        bg_src = ir_path
    else:  # 3DPW / H36M：可見光單張
        rgb_path = example_path
        bg_src = rgb_path

    if bg_src is None or not os.path.isfile(bg_src):
        return ir_path, rgb_path, None
    bg_bgr = cv2.imread(bg_src)
    return ir_path, rgb_path, bg_bgr


def xyxy_to_center_scale(xyxy):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # 與 eval_2d_mu._bbox_xywh_to_center_scale 完全一致（含 BBOX_SCALE_FACTOR=1.2）
    return E._bbox_xywh_to_center_scale(x1, y1, w, h)


# ─── 單張推論 + 渲染 ─────────────────────────────────────────────────────

def rotate_vertices_y(vertices_list, deg):
    """繞每個人自己的重心、繞垂直 Y 軸旋轉 deg 度（正值＝順時針，見模組頂部
    ROTATE_Y_SIGN 的說明），用於生成側面/背面視角的 mesh 圖。"""
    if not deg:
        return vertices_list
    r = cv2.Rodrigues(np.array([0, np.radians(deg) * ROTATE_Y_SIGN, 0], dtype=np.float32))[0]
    out = []
    for v in vertices_list:
        center = v.mean(axis=0, keepdims=True)
        out.append((v - center) @ r.T + center)
    return out


@torch.no_grad()
def infer_and_render(example_path, boxes_xyxy, dataset, mode, model, smpl, renderer,
                     device, img_res, decode_focal, out_dir, transparent=False, rotate_y_deg=0.0):
    ir_path, rgb_path, bg_bgr = resolve_inputs(dataset, mode, example_path)
    if bg_bgr is None:
        return "no_background"
    img_h, img_w = bg_bgr.shape[:2]
    bg_rgb = bg_bgr[:, :, ::-1].copy()

    # 對每個人裁切
    ir_list, rgb_list, cs_list = [], [], []
    for xyxy in boxes_xyxy:
        center, scale = xyxy_to_center_scale(xyxy)
        need_ir = mode in ("ir_only", "rgb_ir")
        need_rgb = mode in ("rgb_only", "rgb_ir")
        ir_t = (E._load_crop(ir_path, center, scale, img_res, grayscale=True)[0]
                if need_ir else torch.zeros(3, img_res, img_res))
        rgb_t = (E._load_crop(rgb_path, center, scale, img_res, grayscale=False)[0]
                 if need_rgb else torch.zeros(3, img_res, img_res))
        ir_list.append(ir_t)
        rgb_list.append(rgb_t)
        cs_list.append((center, scale))
    if not cs_list:
        return "no_person"

    img_ir = torch.stack(ir_list).to(device).float()
    img_rgb = torch.stack(rgb_list).to(device).float()
    centers = torch.tensor(np.stack([c for c, _ in cs_list]), dtype=torch.float32, device=device)
    scales = torch.tensor(np.array([s for _, s in cs_list]), dtype=torch.float32, device=device)
    B = img_ir.shape[0]
    focal_t = torch.full((B,), float(decode_focal), device=device)

    # 模式派發（與 eval_2d_mu.run_batch 一致）
    if mode == "rgb_only":
        pred_rotmat, pred_betas, pred_camera = model(x=img_rgb)
    elif mode == "ir_only":
        mask = torch.tensor([[0.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_ir=img_ir, modality_mask=mask)
    else:  # rgb_ir
        mask = torch.tensor([[1.0, 1.0]], device=device).expand(B, 2)
        pred_rotmat, pred_betas, pred_camera = model(img_rgb=img_rgb, img_ir=img_ir,
                                                     modality_mask=mask)

    out = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
               global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
    img_center = torch.tensor([[img_w * 0.5, img_h * 0.5]], device=device).expand(B, 2)
    pred_cam_full = cam_crop2full(pred_camera, centers, scales, focal_t, img_center)
    verts_cam = (out.vertices + pred_cam_full.unsqueeze(1)).cpu().numpy()

    # 去重：避免同一個人被輸出兩個 mesh（見 dedup_person_meshes）
    keep_idx = dedup_person_meshes(list(verts_cam), boxes_xyxy, decode_focal, img_w, img_h)
    render_verts = [verts_cam[i] for i in keep_idx]
    render_verts = rotate_vertices_y(render_verts, rotate_y_deg)

    image_id = Path(example_path).stem
    tag = f"{dataset}_{mode}"
    rot_suffix = f"_rot{int(round(rotate_y_deg))}" if rotate_y_deg else ""
    if transparent:
        # 去背：只輸出 mesh、透明背景的 BGRA PNG
        bgra = renderer.render_mesh_transparent(render_verts, img_h, img_w, decode_focal)
        out_path = os.path.join(out_dir, f"{image_id}_{tag}_mesh{rot_suffix}.png")
        cv2.imwrite(out_path, bgra)
    else:
        front = renderer.render_front_view(render_verts, bg_rgb, decode_focal)
        out_path = os.path.join(out_dir, f"{image_id}_{tag}{rot_suffix}.png")
        cv2.imwrite(out_path, (np.clip(front * 255, 0, 255).astype(np.uint8))[:, :, ::-1])
    return "ok"


def main():
    ap = argparse.ArgumentParser(description="我的 RGB-T 模型 demo 渲染")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--dataset", required=True, choices=["3DPW", "H36M", "LLVIP", "OTP2"])
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "rgb_only", "ir_only", "rgb_ir"])
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--bbox_file", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--decode_focal", type=float, default=HMR2_FOCAL)
    ap.add_argument("--img_res", type=int, default=256)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--transparent", action="store_true",
                    help="去背模式：只輸出 mesh、透明背景的 BGRA PNG（不疊在原圖上）")
    ap.add_argument("--rotate_y_deg", type=float, default=0.0,
                    help="繞每個人重心、繞垂直 Y 軸旋轉幾度（正值＝順時針），0＝不旋轉")
    args = ap.parse_args()

    mode = DATASET_DEFAULT_MODE[args.dataset] if args.mode == "auto" else args.mode
    if mode != "ir_only" and args.dataset == "OTP2":
        print(f"[WARN] OTP2 純熱像無 RGB，強制 mode=ir_only（原要求 {mode}）")
        mode = "ir_only"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 套用 paths config（讓 config.* 路徑就緒）
    paths_cfg = os.path.join(REPO_ROOT, "configs", "paths", "local.yaml")
    if os.path.isfile(paths_cfg):
        try:
            config.apply_paths_config(config.load_paths_config(paths_cfg), verbose=False)
        except Exception as e:
            print(f"[WARN] paths config 套用失敗：{e}")
    global LLVIP_VISIBLE_ROOT
    LLVIP_VISIBLE_ROOT = config.LLVIP_VISIBLE_ROOT

    from models import SMPL
    print(f"[demo_mu] 載入模型 {args.checkpoint}")
    model = E.build_mu_hmr(args.checkpoint, device, img_res=args.img_res,
                           train_cfg_path=args.train_cfg)
    model.eval()
    smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    renderer = FullImageRenderer(smpl.faces)

    bbox_map = json.load(open(args.bbox_file))
    images = list_images(args.image_folder)
    if args.max_images:
        images = images[:args.max_images]
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"[demo_mu] dataset={args.dataset} mode={mode} | 圖片={len(images)} "
          f"| bbox_map={len(bbox_map)} | decode_focal={args.decode_focal}")

    stats = {"ok": 0, "no_background": 0, "no_person": 0, "no_bbox": 0}
    samples = []
    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        boxes = bbox_map.get(fname) or bbox_map.get(Path(img_path).stem)
        if not boxes:
            stats["no_bbox"] += 1
            continue
        res = infer_and_render(img_path, boxes, args.dataset, mode, model, smpl,
                               renderer, device, args.img_res, args.decode_focal,
                               args.output_folder, transparent=args.transparent,
                               rotate_y_deg=args.rotate_y_deg)
        stats[res] = stats.get(res, 0) + 1
        if res == "ok" and len(samples) < 3:
            samples.append(f"{fname}({len(boxes)}人)")

    print(f"[demo_mu] 完成：{stats} | 樣本：{samples}")
    print(f"[demo_mu] 輸出 → {args.output_folder}")


if __name__ == "__main__":
    main()

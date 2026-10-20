#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
我的 RGB-T 模型（MMPedUnifiedHMR2 / scheme3c）進階 demo
========================================================================
在 tools/demo_mu.py 的基礎上擴充輸出內容，針對單張或少量圖片產生完整的
展示素材組，而不是像 demo_mu.py 一樣只輸出一張疊圖。

與 demo_mu.py / eval_2d_mu.py 的一致性
------------------------------------------------------------------------
沿用 `tools.eval_2d_mu` 的 build_mu_hmr() / _load_crop() /
_bbox_xywh_to_center_scale()，以及 demo_mu.py 的 resolve_inputs() /
xyxy_to_center_scale()，確保推論路徑（裁切、灰階化、相機解碼）與既有
demo/eval 完全一致，只是把渲染與輸出這一段做得更完整。

每張輸入圖片會在 `<output_folder>/<image_stem>/` 底下產生：
  <stem>_original_ir.png       原始 IR（熱像）全幀，未經任何處理
  <stem>_original_rgb.png      原始 RGB 全幀（僅當有配對 RGB 時，如 LLVIP）
  <stem>_mesh_white.png        白底 SMPL mesh（去背、supersample 抗鋸齒）
  <stem>_mesh_overlay_ir.png   SMPL mesh 疊在原始 IR 圖上（無 IR 則疊 RGB）
  portraits/<stem>_person{idx:02d}_portrait.png
                                每個人物獨立的「產品照」：專屬相機置中放大、
                                單一主光源 + 地板陰影，解析度與框景一致

高解析度做法
------------------------------------------------------------------------
pyrender 用 `--supersample` 倍的視窗解析度渲染 mesh-only RGBA（透明背景），
再用 cv2.INTER_AREA 降採樣回原圖解析度，取得平滑抗鋸齒的邊緣後才用 alpha
與背景（白底或原圖）合成 —— 邊緣不是原始 demo_mu.py 那種硬遮罩（depth>0
布林遮罩），而是漸層 alpha，鋸齒感明顯較少。

單人 portrait 的光影/陰影做法
------------------------------------------------------------------------
群體疊圖用的是無陰影的三盞環繞光（`_add_lights`），單人 portrait 改用
「專屬相機 + 地板平面 + 單一主光源 SHADOWS_DIRECTIONAL 陰影」：
  - 每個人物用「原始 SMPL 頂點」（未加全圖相機平移）重新置中，依身高
    自動反推相機距離，讓每張輸出都固定佔畫面高度的 --portrait_fill_ratio。
  - 地板平面尺寸只取人物寬度的數倍（而非整個場景），這是刻意的：
    pyrender 0.1.45 的 shadow map 固定 2048px、且無 bias 參數可調，若地板
    （進而 scene bounding box）太大，陰影解析度被稀釋，會出現明顯階梯狀
    摩爾紋瑕疵（已用簡單場景實測驗證）；地板越貼近人物，陰影越乾淨。
  - 只用『一盞』會投影的主光源：pyrender 的 SHADOWS_DIRECTIONAL 是場景級
    flag，場景裡所有 DirectionalLight 都會一起投影，多盞光源會疊出多重
    陰影、觀感雜亂，因此只保留一盞主光源，其餘立體感交給 ambient_light。

用法
------------------------------------------------------------------------
  python tools/demo_mu_advanced.py \\
      --checkpoint weights/main/model_final.pt \\
      --train_cfg configs/train/main/s5_dcg.yaml \\
      --dataset LLVIP --mode rgb_ir \\
      --image <LLVIP_ROOT>/infrared/test/190538.jpg \\
      --bbox_file <bbox json> \\
      --output_folder demo_output/demo \\
      --supersample 3
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from utils.geometry import cam_crop2full
import tools.eval_2d_mu as E          # build_mu_hmr / _load_crop / bbox 轉換
import tools.demo_mu as D             # resolve_inputs / xyxy_to_center_scale / list_images

try:
    import pyrender
    import trimesh
except Exception as e:  # pragma: no cover
    pyrender = None
    trimesh = None
    _IMPORT_ERR = e

HMR2_FOCAL = 5000.0  # 與 demo_mu.py 預設一致（HMR2 原生慣例）


# ─── 進階渲染器：supersample + alpha 合成 ───────────────────────────────

class AdvancedRenderer(D.FullImageRenderer):
    """在 demo_mu.FullImageRenderer 基礎上加入 supersample 抗鋸齒與 alpha 合成。"""

    def render_mesh_rgba(self, vertices_list, img_w, img_h, focal_length, supersample=3):
        """在透明背景上渲染 mesh-only，回傳降採樣後的 (color_rgb[0,1], alpha[0,1])。"""
        ss = max(1, int(supersample))
        W, H = img_w * ss, img_h * ss
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
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

        color_f = color.astype(np.float32) / 255.0
        alpha_f = (depth > 0).astype(np.float32)
        if ss > 1:
            color_small = cv2.resize(color_f[:, :, :3], (img_w, img_h), interpolation=cv2.INTER_AREA)
            alpha_small = cv2.resize(alpha_f, (img_w, img_h), interpolation=cv2.INTER_AREA)
        else:
            color_small = color_f[:, :, :3]
            alpha_small = alpha_f
        return color_small, alpha_small

    @staticmethod
    def composite(mesh_rgb, alpha, background_rgb01):
        a = alpha[..., None]
        return np.clip(mesh_rgb * a + background_rgb01 * (1.0 - a), 0.0, 1.0)

    # ── 單人 portrait：專屬相機 + 地板陰影 ──────────────────────────────

    def render_person_portrait(self, raw_vertices, size=640, supersample=3,
                                fill_ratio=0.78, yfov_deg=32.0,
                                shadow_color=(0.10, 0.10, 0.10),
                                max_shadow_alpha=0.55):
        """單人置中放大渲染，只留人物（不透明）+ 地板陰影（半透明），其餘全透
        明，不合成任何底色/背景。raw_vertices 為 SMPL 原始輸出（未加全圖相機
        平移），(6890, 3)。回傳 (H, W, 4) uint8 BGRA，可直接 imwrite 成 PNG。

        做法：地板本身不出現在畫面上（不是一片實心色塊），而是渲染「有陰影」
        與「無陰影」兩個版本，用地板像素的亮度差反推陰影強度，當作陰影的
        alpha──陰影越深，alpha 越高；沒有陰影的地板region 直接是全透明。
        """
        ss = max(1, int(supersample))
        W = H = int(size) * ss

        # `_make_mesh` 內部會對輸入頂點做 180°(繞 X 軸) 旋轉，等同 (x,y,z)->(x,-y,-z)。
        # 為了在「旋轉後」的相機空間置中 + 貼地，先在旋轉後空間算出目標平移量，
        # 再映射回旋轉前（也就是這裡收到的 raw_vertices）空間，作為平移後再交給
        # `_make_mesh`。180° 旋轉是自身的反變換，平移量映射只需把 y、z 分量取負。
        flipped = raw_vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)
        v_min, v_max = flipped.min(axis=0), flipped.max(axis=0)
        person_height = float(v_max[1] - v_min[1])
        center_x = float((v_min[0] + v_max[0]) / 2.0)
        mid_y = float((v_min[1] + v_max[1]) / 2.0)
        center_z = float((v_min[2] + v_max[2]) / 2.0)
        footprint = float(max(v_max[0] - v_min[0], v_max[2] - v_min[2], 0.4))

        yfov = np.radians(yfov_deg)
        cam_dist = person_height / (2.0 * fill_ratio * np.tan(yfov / 2.0))
        target_z = -cam_dist  # 相機沿 -Z 看，物體須在負 Z 才可見

        # 相機無仰俯、正視 -Z，可視範圍在 Y 方向以 0 為中心對稱，因此要讓「身體
        # 垂直中點」對齊 Y=0，而不是腳底——腳底對齊 0 會讓全身都偏在光軸上方，
        # 頭會被裁到視野外（這是先前版本的 bug：誤把「貼地」和「對準鏡頭」用了
        # 同一個 Y=0 參考點）。貼地面另外算，見下面 floor_y_final。
        delta_flipped = np.array([
            0.0 - center_x,
            -mid_y,
            target_z - center_z,
        ], dtype=np.float32)
        # 映射回旋轉前空間：x 分量不變，y、z 分量取負
        delta_raw = np.array([delta_flipped[0], -delta_flipped[1], -delta_flipped[2]],
                              dtype=np.float32)
        centered_raw = raw_vertices + delta_raw

        gap = max(0.015, person_height * 0.01)
        new_feet_y = v_min[1] - mid_y  # 置中後腳底的新 Y 座標（= -person_height/2）
        floor_y_final = new_feet_y - gap
        floor_extent = footprint * 5.0

        # 地板要「躺平」：水平延伸在 X/Z，厚度薄在 Y。extents 順序是 (X,Y,Z)；
        # 先前寫成 (floor_extent, floor_extent, 0.01) 等於做出一片薄在 Z 方向、
        # 面向鏡頭的直立薄板（看起來像牆），已修正為薄在 Y 方向的水平板。
        plane = trimesh.creation.box(extents=[floor_extent, 0.01, floor_extent])
        plane.apply_translation([0.0, floor_y_final, target_z])
        floor_mesh = pyrender.Mesh.from_trimesh(
            plane,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.9, 0.9, 0.9, 1.0), metallicFactor=0.0, roughnessFactor=1.0),
        )

        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.2)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-42), [1, 0, 0])
        light_pose = light_pose @ trimesh.transformations.rotation_matrix(np.radians(28), [0, 1, 0])
        mesh_node = self._make_mesh(centered_raw)

        def _render(include_floor, cast_shadow):
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
            if include_floor:
                scene.add(floor_mesh)
            scene.add(mesh_node)
            scene.add(camera, pose=np.eye(4))
            scene.add(key_light, pose=light_pose)
            flags = pyrender.RenderFlags.RGBA
            if cast_shadow:
                flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            off = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
            c, d = off.render(scene, flags=flags)
            off.delete()
            return c.astype(np.float32) / 255.0, d

        color_shadow, depth_shadow = _render(include_floor=True, cast_shadow=True)
        color_flat, _ = _render(include_floor=True, cast_shadow=False)
        color_mesh, depth_mesh = _render(include_floor=False, cast_shadow=False)

        mesh_alpha = (depth_mesh > 0).astype(np.float32)
        mesh_rgb = color_mesh[:, :, :3]

        floor_only = (depth_shadow > 0).astype(np.float32) * (1.0 - mesh_alpha)
        lum_shadow = color_shadow[:, :, :3].mean(axis=-1)
        lum_flat = np.clip(color_flat[:, :, :3].mean(axis=-1), 1e-4, None)
        shadow_strength = np.clip(1.0 - lum_shadow / lum_flat, 0.0, 1.0) * floor_only
        shadow_alpha = shadow_strength * float(max_shadow_alpha)
        shadow_rgb = np.array(shadow_color, dtype=np.float32).reshape(1, 1, 3) * np.ones_like(mesh_rgb)

        # 人物在上（不透明）、陰影在下（半透明）、其餘全透明；用 premultiplied
        # alpha 做合成與降採樣，邊緣才不會被「透明=黑」污染出深色鑲邊。
        out_alpha = mesh_alpha + shadow_alpha * (1.0 - mesh_alpha)
        out_rgb_premult = (mesh_rgb * mesh_alpha[:, :, None]
                            + shadow_rgb * shadow_alpha[:, :, None] * (1.0 - mesh_alpha[:, :, None]))

        out_size = int(size)
        if ss > 1:
            premult_small = cv2.resize(out_rgb_premult, (out_size, out_size), interpolation=cv2.INTER_AREA)
            alpha_small = cv2.resize(out_alpha, (out_size, out_size), interpolation=cv2.INTER_AREA)
        else:
            premult_small = out_rgb_premult
            alpha_small = out_alpha

        safe_alpha = np.clip(alpha_small, 1e-6, None)
        rgb_small = np.where(alpha_small[:, :, None] > 1e-6, premult_small / safe_alpha[:, :, None], 0.0)

        bgr = np.clip(rgb_small, 0.0, 1.0)[:, :, ::-1]
        bgra = np.concatenate([bgr, np.clip(alpha_small, 0.0, 1.0)[:, :, None]], axis=-1)
        return (bgra * 255).astype(np.uint8)


# ─── 單張推論 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_vertices(example_path, boxes_xyxy, dataset, mode, model, smpl,
                    device, img_res, decode_focal):
    """回傳 (verts_cam_list, verts_raw_list, img_w, img_h, ir_path, rgb_path, bg_bgr)。

    verts_cam_list：加了全圖相機平移，用於群體疊圖（大家在同一個相機下的正確位置）。
    verts_raw_list：SMPL 原始輸出（未加平移），用於單人 portrait 重新置中取景。
    """
    ir_path, rgb_path, bg_bgr = D.resolve_inputs(dataset, mode, example_path)
    if bg_bgr is None:
        return None
    img_h, img_w = bg_bgr.shape[:2]

    ir_list, rgb_list, cs_list = [], [], []
    for xyxy in boxes_xyxy:
        center, scale = D.xyxy_to_center_scale(xyxy)
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
        return None

    img_ir = torch.stack(ir_list).to(device).float()
    img_rgb = torch.stack(rgb_list).to(device).float()
    centers = torch.tensor(np.stack([c for c, _ in cs_list]), dtype=torch.float32, device=device)
    scales = torch.tensor(np.array([s for _, s in cs_list]), dtype=torch.float32, device=device)
    B = img_ir.shape[0]
    focal_t = torch.full((B,), float(decode_focal), device=device)

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
    verts_raw = out.vertices.cpu().numpy()
    verts_cam = (out.vertices + pred_cam_full.unsqueeze(1)).cpu().numpy()

    return list(verts_cam), list(verts_raw), img_w, img_h, ir_path, rgb_path, bg_bgr


def process_image(example_path, boxes_xyxy, dataset, mode, model, smpl, renderer,
                   device, img_res, decode_focal, supersample, out_dir,
                   portrait_size, portrait_fill_ratio):
    result = infer_vertices(example_path, boxes_xyxy, dataset, mode, model, smpl,
                             device, img_res, decode_focal)
    if result is None:
        return "no_background"
    verts_list, verts_raw_list, img_w, img_h, ir_path, rgb_path, _bg_bgr = result

    stem = Path(example_path).stem
    sample_dir = os.path.join(out_dir, stem)
    os.makedirs(sample_dir, exist_ok=True)

    # 1) 原圖：IR / RGB 全幀原封不動複製（保留原副檔名，不轉檔避免失真）
    if ir_path and os.path.isfile(ir_path):
        ir_ext = os.path.splitext(ir_path)[1]
        shutil.copyfile(ir_path, os.path.join(sample_dir, f"{stem}_original_ir{ir_ext}"))
    if rgb_path and os.path.isfile(rgb_path):
        rgb_ext = os.path.splitext(rgb_path)[1]
        shutil.copyfile(rgb_path, os.path.join(sample_dir, f"{stem}_original_rgb{rgb_ext}"))

    # 2) mesh 渲染：白底 + 疊在 IR（無 IR 則疊 RGB）上
    mesh_rgb, alpha = renderer.render_mesh_rgba(verts_list, img_w, img_h, decode_focal,
                                                 supersample=supersample)

    white_bg = np.ones((img_h, img_w, 3), dtype=np.float32)
    mesh_on_white = renderer.composite(mesh_rgb, alpha, white_bg)
    cv2.imwrite(
        os.path.join(sample_dir, f"{stem}_mesh_white.png"),
        (mesh_on_white * 255).astype(np.uint8)[:, :, ::-1],
    )

    overlay_src_path = ir_path if (ir_path and os.path.isfile(ir_path)) else rgb_path
    overlay_tag = "ir" if overlay_src_path == ir_path else "rgb"
    overlay_bgr = cv2.imread(overlay_src_path)
    overlay_rgb01 = overlay_bgr[:, :, ::-1].astype(np.float32) / 255.0
    mesh_on_overlay = renderer.composite(mesh_rgb, alpha, overlay_rgb01)
    cv2.imwrite(
        os.path.join(sample_dir, f"{stem}_mesh_overlay_{overlay_tag}.png"),
        (mesh_on_overlay * 255).astype(np.uint8)[:, :, ::-1],
    )

    # 3) 每個人物獨立的放大 portrait（專屬相機置中 + 地板陰影）
    portrait_dir = os.path.join(sample_dir, "portraits")
    os.makedirs(portrait_dir, exist_ok=True)
    for idx, raw_v in enumerate(verts_raw_list):
        portrait_bgr = renderer.render_person_portrait(
            raw_v, size=portrait_size, supersample=supersample,
            fill_ratio=portrait_fill_ratio,
        )
        cv2.imwrite(
            os.path.join(portrait_dir, f"{stem}_person{idx:02d}_portrait.png"),
            portrait_bgr,
        )

    return "ok"


def main():
    ap = argparse.ArgumentParser(description="我的 RGB-T 模型進階 demo（白底 mesh + IR 疊圖 + 原圖輸出）")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--dataset", required=True, choices=["3DPW", "H36M", "LLVIP", "OTP2"])
    ap.add_argument("--mode", default="auto", choices=["auto", "rgb_only", "ir_only", "rgb_ir"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="單張圖片路徑")
    src.add_argument("--image_folder", help="批次資料夾（同 demo_mu.py）")
    ap.add_argument("--bbox_file", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--decode_focal", type=float, default=HMR2_FOCAL)
    ap.add_argument("--img_res", type=int, default=256)
    ap.add_argument("--supersample", type=int, default=3, help="mesh 抗鋸齒渲染倍率")
    ap.add_argument("--portrait_size", type=int, default=640, help="單人 portrait 輸出邊長（正方形）")
    ap.add_argument("--portrait_fill_ratio", type=float, default=0.78,
                     help="人物佔 portrait 畫面高度的比例")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    mode = D.DATASET_DEFAULT_MODE[args.dataset] if args.mode == "auto" else args.mode
    if mode != "ir_only" and args.dataset == "OTP2":
        print(f"[WARN] OTP2 純熱像無 RGB，強制 mode=ir_only（原要求 {mode}）")
        mode = "ir_only"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    paths_cfg = os.path.join(REPO_ROOT, "configs", "paths", "local.yaml")
    if os.path.isfile(paths_cfg):
        try:
            config.apply_paths_config(config.load_paths_config(paths_cfg), verbose=False)
        except Exception as e:
            print(f"[WARN] paths config 套用失敗：{e}")

    from models import SMPL
    print(f"[demo_mu_advanced] 載入模型 {args.checkpoint}")
    model = E.build_mu_hmr(args.checkpoint, device, img_res=args.img_res,
                            train_cfg_path=args.train_cfg)
    model.eval()
    smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    renderer = AdvancedRenderer(smpl.faces)

    bbox_map = json.load(open(args.bbox_file))
    if args.image:
        images = [args.image]
    else:
        images = D.list_images(args.image_folder)
        if args.max_images:
            images = images[:args.max_images]
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"[demo_mu_advanced] dataset={args.dataset} mode={mode} | 圖片={len(images)} "
          f"| decode_focal={args.decode_focal} | supersample={args.supersample}")

    stats = {"ok": 0, "no_background": 0, "no_bbox": 0}
    for img_path in images:
        fname = os.path.basename(img_path)
        boxes = bbox_map.get(fname) or bbox_map.get(Path(img_path).stem)
        if not boxes:
            stats["no_bbox"] += 1
            print(f"[skip] 找不到 bbox：{fname}")
            continue
        res = process_image(img_path, boxes, args.dataset, mode, model, smpl, renderer,
                             device, args.img_res, args.decode_focal, args.supersample,
                             args.output_folder, args.portrait_size, args.portrait_fill_ratio)
        stats[res] = stats.get(res, 0) + 1
        print(f"[{res}] {fname}（{len(boxes)} 人）")

    print(f"[demo_mu_advanced] 完成：{stats}")
    print(f"[demo_mu_advanced] 輸出 → {args.output_folder}")


if __name__ == "__main__":
    main()

import math
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import traceback
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from torchgeometry import rotation_matrix_to_angle_axis as tgm_rotation_matrix_to_angle_axis
except Exception:
    tgm_rotation_matrix_to_angle_axis = None

# torchgeometry 在目前 PyTorch 版本下會因 bool tensor 運算炸掉，
# 訓練時一律走 OpenCV fallback，避免 run_smplify=True 時出錯
tgm_rotation_matrix_to_angle_axis = None

from datasets import MixedDataset, ThermalValidationDataset
from models import SMPL
from smplify import SMPLify
from utils.geometry import (
    batch_rodrigues,
    perspective_projection,
    estimate_translation,
    bbox_info_from_center_scale,
    cam_crop2full,
    cam_crop2perspective_crop,
    cam_perspective_crop2cropcam,
    project_points_crop_full,
    estimate_translation_fullimg,
    project_points_full_to_crop,
    cam_full2crop_perspective,
)
from utils import BaseTrainer

import config
import constants
from .fits_dict import FitsDict
from .loss_mixer import ReFitStyleLossMixer, build_loss_weights, resolve_supervision_mode


def _rotation_matrix_to_angle_axis_cv2(rotmat_3x3):
    rot_np = rotmat_3x3.detach().cpu().numpy().astype(np.float32)
    aa_np = np.zeros((rot_np.shape[0], 3), dtype=np.float32)
    for i in range(rot_np.shape[0]):
        aa, _ = cv2.Rodrigues(rot_np[i])
        aa_np[i, :] = aa.reshape(3)
    return torch.from_numpy(aa_np).to(rotmat_3x3.device, dtype=rotmat_3x3.dtype)


def _resolve_pred_cam_full(model, pred_cam_raw, center, scale, img_focal, img_center):
    """Dispatch the model's raw camera output to full-image (tx, ty, tz).

    * ``camera_rep='weak_perspective'`` (legacy): raw output is (s, tx, ty) in
      crop space. Convert via CLIFF-style ``cam_crop2full``.
    * ``camera_rep='full_perspective'`` (Step 3): raw output is already
      (tx, ty, tz) in metres in the camera frame. Skip ``cam_crop2full`` so the
      depth is not silently re-derived from focal/bbox (the original source of
      camera instability documented in two_modality_plan §1.5).

    Downstream consumers (``project_points_full_to_crop``, ``cam_depth_loss``,
    ``cam_full_pair_loss``, SMPLify) only need (tx, ty, tz), so the rest of the
    pipeline is unchanged.
    """
    camera_rep = getattr(model, 'camera_rep', 'weak_perspective')
    if camera_rep == 'full_perspective':
        return pred_cam_raw
    return cam_crop2full(pred_cam_raw, center, scale, img_focal, img_center)


SPIN24_TO_COCO17 = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]


def _recenter_joints_h36m(joints_h36m, mode='pelvis'):
    """Root-relative recentering of a (B, 24, 3) H36M-block joint array/tensor.

    Works for both torch tensors and numpy arrays (only slicing + arithmetic).

    mode='pelvis' (default, REPORTING STANDARD)
        Root = mean('Right Hip' idx 2, 'Left Hip' idx 3). Matches
        tools/eval_2d_mu.py's default ROOT_JOINT_MODE='hip_mid' and
        train/loss_mixer.py:keypoint_3d_loss's gt_pelvis/pred_pelvis convention.
    mode='ankle' (LEGACY, comparison only)
        Root = 'Right Ankle' idx 0. This is what every run up to and including
        stage2c_smplsup_s1 logged as '3D-MPJPE'. The ankle is an extremity
        ~926mm (vs pelvis ~447mm) from the other joints on average, so any
        global-orientation mismatch between pred and pGT gets amplified by that
        lever arm — measured ~1.5x inflation for a pure 5-degree orient error.
        Kept so new runs stay numerically comparable with the historical logs.

    PA-MPJPE is root-independent (Procrustes removes translation), so it needs
    only one value regardless of mode.
    """
    if mode == 'ankle':
        root = joints_h36m[:, 0:1, :]
    elif mode == 'pelvis':
        root = (joints_h36m[:, 2:3, :] + joints_h36m[:, 3:4, :]) / 2.0
    else:
        raise ValueError(f"Unsupported root mode: {mode!r} (expected 'pelvis' or 'ankle')")
    return joints_h36m - root


def _mpjpe_3d_both_roots(gt_j3d_raw, pred_j3d_raw, mask):
    """3D-MPJPE under both root conventions + root-independent PA-MPJPE.

    gt_j3d_raw / pred_j3d_raw: (N, 24, 3) numpy, NOT yet recentered.
    Returns (mpjpe_pelvis_mm, mpjpe_ankle_mm, pampjpe_mm) — NaN if mask is empty.
    """
    n_ok = int(mask.sum())
    if n_ok <= 0:
        return float('nan'), float('nan'), float('nan')

    gt_raw   = gt_j3d_raw[mask]
    pred_raw = pred_j3d_raw[mask]

    out = {}
    for mode in ('pelvis', 'ankle'):
        g = _recenter_joints_h36m(gt_raw, mode)[:, SPIN24_TO_COCO17, :]
        p = _recenter_joints_h36m(pred_raw, mode)[:, SPIN24_TO_COCO17, :]
        out[mode] = float(np.linalg.norm(g - p, axis=-1).mean() * 1000.0)

    # PA-MPJPE: root-independent, so compute once off the pelvis-rooted pair.
    gt_sub   = _recenter_joints_h36m(gt_raw, 'pelvis')[:, SPIN24_TO_COCO17, :]
    pred_sub = _recenter_joints_h36m(pred_raw, 'pelvis')[:, SPIN24_TO_COCO17, :]
    pa_err = np.empty(n_ok, dtype=np.float64)
    for k in range(n_ok):
        try:
            pr_aligned = _procrustes_align_np(pred_sub[k], gt_sub[k])
        except np.linalg.LinAlgError:
            pr_aligned = pred_sub[k]
        pa_err[k] = np.linalg.norm(gt_sub[k] - pr_aligned, axis=-1).mean()

    return out['pelvis'], out['ankle'], float(pa_err.mean() * 1000.0)


def _procrustes_align_np(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Similarity transform (Umeyama 1991): align S1 to S2. Both (J,3). Returns aligned S1.
    Identical to procrustes_align_np in eval_hmr_2d.py.
    """
    mu1 = S1.mean(axis=0)
    mu2 = S2.mean(axis=0)
    X1  = S1 - mu1
    X2  = S2 - mu2
    ss1 = float((X1 ** 2).sum())   # Frobenius norm^2 — no /N
    if ss1 < 1e-12:
        return S1.copy()
    K = X2.T @ X1                   # (3,3): target^T @ source
    U, sigma, Vt = np.linalg.svd(K)
    d = float(np.sign(np.linalg.det(U @ Vt)))
    D = np.array([1.0, 1.0, d])
    R = (U * D) @ Vt                # rotation aligning X1 → X2
    c = float((sigma * D).sum()) / ss1
    return c * (R @ X1.T).T + mu2


class Trainer(BaseTrainer):
    def _resolve_train_dataset_list(self):
        dataset_list = getattr(self.options, 'dataset_list', None)
        if dataset_list is None and hasattr(self.options, 'DATASET'):
            dataset_list = getattr(self.options.DATASET, 'LIST', None)
        if dataset_list is None:
            dataset_list = config.DATASET_DEFAULT_LIST
        return list(dataset_list)

    def _resolve_train_dataset_partition(self):
        dataset_partition = getattr(self.options, 'dataset_partition', None)
        if dataset_partition is None and hasattr(self.options, 'DATASET'):
            dataset_partition = getattr(self.options.DATASET, 'PARTITION', None)
        if dataset_partition is None:
            dataset_partition = config.DATASET_DEFAULT_PARTITION
        return list(dataset_partition)

    def _resolve_validation_enabled(self):
        enabled = getattr(self.options, 'validation_enabled', None)
        if enabled is None and hasattr(self.options, 'VALIDATION'):
            enabled = getattr(self.options.VALIDATION, 'ENABLED', None)
        if enabled is None:
            enabled = True
        return bool(enabled)

    def _resolve_validation_dataset(self):
        dataset = getattr(self.options, 'validation_dataset', None)
        if dataset in [None, ''] and hasattr(self.options, 'VALIDATION'):
            dataset = getattr(self.options.VALIDATION, 'DATASET', None)
        if dataset in [None, '']:
            dataset = '3dpw'
        return str(dataset).lower()

    def _resolve_validation_dataset_names(self):
        """Return a non-empty list of validation dataset names (lowercased).

        Plan 06.28 §9.2.2: supports a `validation_datasets` list so a single
        run can validate on 3DPW + LLVIP + OTP2 simultaneously and pick the
        best checkpoint per dataset/metric.
        """
        names = getattr(self.options, 'validation_datasets', None)
        if (names in [None, '']) and hasattr(self.options, 'VALIDATION'):
            names = getattr(self.options.VALIDATION, 'DATASETS', None)
        if names in [None, '']:
            names = None
        if isinstance(names, (list, tuple)) and len(names) > 0:
            return [str(n).lower() for n in names]
        return [self._resolve_validation_dataset()]

    def _resolve_validation_primary_metric(self):
        """Return the flat key (e.g. '3dpw/MPJPE') used as overall `performance`.

        Falls back to ``<first_dataset>/<validation_metric>``.
        """
        prim = getattr(self.options, 'validation_primary_metric', None)
        if prim in [None, ''] and hasattr(self.options, 'VALIDATION'):
            prim = getattr(self.options.VALIDATION, 'PRIMARY_METRIC', None)
        if prim in [None, '']:
            return None
        return str(prim)

    def _resolve_validation_metric(self):
        metric = getattr(self.options, 'validation_metric', None)
        if metric in [None, ''] and hasattr(self.options, 'VALIDATION'):
            metric = getattr(self.options.VALIDATION, 'METRIC', None)
        if metric in [None, '']:
            metric = 'MPJPE'
        return str(metric).upper()

    def _run_preflight_checks(self):
        backbone = getattr(self.options, 'backbone', 'resnet50')
        core_type = str(getattr(self.options, 'core_type', 'vitpose') or 'vitpose')
        strict_hrnet = bool(getattr(self.options, 'strict_hrnet', True))
        render_summaries_requested = bool(getattr(self.options, 'render_summaries', True))

        dataset_list = self._resolve_train_dataset_list()
        config.validate_runtime_train_paths(
            dataset_names=dataset_list,
            require_renderer_files=render_summaries_requested,
            require_hrnet=(backbone == 'hrnet_w48' and strict_hrnet),
            require_smpl_gender=True,
        )
        config.validate_dataset_roots(dataset_list)
        config.validate_dataset_npzs(dataset_list, is_train=True)
        config.validate_dataset_npz_contents(dataset_list, is_train=True)
        if not getattr(self.options, 'skip_dataset_image_check', False):
            config.validate_dataset_image_paths(
                dataset_list,
                is_train=True,
                num_check=20,
                random_check=100,
                seed=123
            )
        if 'h36m' in dataset_list:
            h36m_npz = config.DATASET_FILES[1].get('h36m', '')
            if 'local_aligned' not in os.path.basename(h36m_npz):
                config.validate_h36m_refit_mapping(
                    num_check=20,
                    random_check=100,
                    seed=123,
                    allow_basename_fallback=False
                )

        if not os.path.isdir(config.SMPL_MODEL_DIR):
            raise FileNotFoundError(f"SMPL model dir not found: {config.SMPL_MODEL_DIR}")
        config.validate_smpl_files(require_gender=True)
        config.validate_core_data_files()

        gmm_prior = os.path.join(config.SMPLIFY_PRIOR_DIR, 'gmm_08.pkl')
        if not os.path.isfile(gmm_prior):
            raise FileNotFoundError(f"SMPLify prior file not found: {gmm_prior}")

        missing_renderer_assets = []
        for p in [config.VERTEX_TEXTURE_FILE, config.CUBE_PARTS_FILE]:
            if not os.path.isfile(p):
                missing_renderer_assets.append(p)
        self.render_summaries_preflight = render_summaries_requested
        if missing_renderer_assets and render_summaries_requested:
            print(
                f"[Preflight] Warning: renderer-related files missing, rendering summaries will be disabled: "
                f"{missing_renderer_assets}"
            )
            self.render_summaries_preflight = False

        use_augmentation = getattr(self.options, 'use_augmentation', True)
        occ_prob = float(getattr(self.options, 'occ_prob', 0.5))
        if use_augmentation and occ_prob > 0.0 and not os.path.isfile(config.PASCAL_OCCLUDERS):
            raise FileNotFoundError(
                f"Pascal occluders file not found but occ_prob={occ_prob}: {config.PASCAL_OCCLUDERS}"
            )

        self.hrnet_pretrained_resolved = None
        self.vitpose_checkpoint_resolved = None

        if core_type != 'hmr2':
            raise ValueError(
                f"MODEL.CORE_TYPE must be 'hmr2' (got {core_type!r}); "
                'the ViTPose / HRNet cores are not part of this release.')

    def init_fn(self):
        self._run_preflight_checks()
        self.train_ds = MixedDataset(
            self.options,
            ignore_3d=self.options.ignore_3d,
            is_train=True,
            use_augmentation=getattr(self.options, 'use_augmentation', True),
        )

        # Model core: ported 4D-Humans HMR2.0 (ViT-H + transformer SMPL head),
        # wrapped below by MMPedUnifiedHMR2 (RGB/IR patch tokens -> unified
        # encoder -> frozen HMR2 body -> frozen head).
        core_type = str(getattr(self.options, 'core_type', 'hmr2') or 'hmr2')
        _model_type0 = str(getattr(self.options, 'model_type', None) or '')
        self.is_pbf_hmr = False
        self.is_late_token_rgbt_hmr = False
        self.is_mu_hmr = False
        self.is_mmped_unified_hmr2 = _model_type0 in (
            'mmped_unified_hmr2', 'mmped_hmr2_unified_encoder', 'stage2c_mmped_hmr2')
        if core_type != 'hmr2' or not self.is_mmped_unified_hmr2:
            raise ValueError(
                f'Only MODEL.CORE_TYPE=hmr2 + MODEL.TYPE=mmped_unified_hmr2 is supported '
                f'(got CORE_TYPE={core_type!r}, TYPE={_model_type0!r}).')
        from models.hmr2_core import build_hmr2_core
        _hmr2_ckpt = getattr(self.options, 'hmr2_checkpoint', None)
        _model_cfg0 = getattr(self.options, 'MODEL', None)
        _freeze_core0 = bool(getattr(_model_cfg0, 'FREEZE_CORE', True)) if _model_cfg0 is not None else True
        self.model = build_hmr2_core(
            config.SMPL_MEAN_PARAMS,
            ckpt_path=_hmr2_ckpt,
            freeze=_freeze_core0,
            grad_through_frozen=False,
        ).to(self.device)
        print(f"[Init] core_type=hmr2 | hmr2_checkpoint={_hmr2_ckpt} | "
              f"freeze_core={_freeze_core0} | grad_through_frozen=False")

        # Stage2C MMPedUnifiedHMR2: wrap the HMR2 core so RGB/IR patch tokens go
        # through an MMPed-style unified encoder BEFORE the frozen HMR2 body.
        if getattr(self, 'is_mmped_unified_hmr2', False):
            if core_type != 'hmr2':
                raise ValueError('MMPedUnifiedHMR2 requires MODEL.CORE_TYPE=hmr2.')
            from models.multimodal import (
                build_mmped_hmr2_unified_encoder,
                build_bounded_ir_token_adapter, MMPedUnifiedHMR2)
            _model_cfg = getattr(self.options, 'MODEL', None)
            _ue_cfg = getattr(_model_cfg, 'UNIFIED_ENCODER', None) if _model_cfg is not None else None
            _freeze_body = bool(getattr(_model_cfg, 'FREEZE_HMR2_BODY', True)) if _model_cfg is not None else True
            _freeze_head = bool(getattr(_model_cfg, 'FREEZE_HEAD', True)) if _model_cfg is not None else True
            _unified_encoder = build_mmped_hmr2_unified_encoder(_ue_cfg)
            # Scheme-1 gate anchor target (nested cfg; defaults to CONF_INIT_IR).
            _conf_init_ir = float(getattr(_ue_cfg, 'CONF_INIT_IR', 0.70)) if _ue_cfg is not None else 0.70
            self._gate_anchor_ir = float(getattr(_ue_cfg, 'GATE_ANCHOR_IR', _conf_init_ir)) \
                if _ue_cfg is not None else _conf_init_ir
            # Stage B: build the bounded IR token adapter if enabled.
            _ir_cfg = getattr(_model_cfg, 'IR_TOKEN_ADAPTER', None) if _model_cfg is not None else None
            _ir_enable = bool(getattr(_ir_cfg, 'ENABLE', False)) if _ir_cfg is not None else False
            _ir_adapter = build_bounded_ir_token_adapter(_ir_cfg) if _ir_enable else None
            # Scheme 2: body block adapters (last-K frozen ViT blocks).
            _ba_cfg = getattr(_model_cfg, 'BODY_ADAPTER', None) if _model_cfg is not None else None
            _ba_enable = bool(getattr(_ba_cfg, 'ENABLE', False)) if _ba_cfg is not None else False
            _body_adapters = None
            _body_adapter_block_indices = None
            if _ba_enable:
                from models.multimodal import build_body_adapters
                _body_adapters, _body_adapter_block_indices = build_body_adapters(_ba_cfg)
                print(f'[Init][MMPedUnifiedHMR2] BODY_ADAPTER enabled: '
                      f'{len(_body_adapters)} adapters at blocks {_body_adapter_block_indices}')
            # P2b: trainable IR patch stem (first-layer RGB/IR domain-mismatch fix).
            _ips_cfg = getattr(_model_cfg, 'IR_PATCH_STEM', None) if _model_cfg is not None else None
            _ips_enable = bool(getattr(_ips_cfg, 'ENABLE', False)) if _ips_cfg is not None else False
            _ir_patch_stem = None
            if _ips_enable:
                from models.multimodal.mmped_unified_hmr2 import IRPatchStem
                _stem_module = self.model.backbone.build_ir_patch_stem()
                _ips_max_scale = float(getattr(_ips_cfg, 'MAX_SCALE', 1.0))
                _ips_init_raw = float(getattr(_ips_cfg, 'INIT_RAW_SCALE', 0.0))
                _ir_patch_stem = IRPatchStem(_stem_module, max_scale=_ips_max_scale,
                                             init_raw_scale=_ips_init_raw)
                print(f'[Init][MMPedUnifiedHMR2] IR_PATCH_STEM enabled: max_scale={_ips_max_scale} '
                      f'init_raw_scale={_ips_init_raw} '
                      f'({sum(p.numel() for p in _ir_patch_stem.parameters())/1e6:.2f}M params, '
                      f'step-0 residual=0 -> identity)')
            # Tier-1 global-orient bounded residual（2026-09-22）。預設停用 ->
            # 既有 run 的行為與權重結構完全不變。見 BoundedGlobalOrientResidual
            # 的 docstring：凍結 head 是 global orient 的限制，但直接解凍會發散
            # （unfreezehead_l2sp: 13.30 -> 18.77 於 1000 步內）。
            # 防呆（2026-09-22）：GO_RESIDUAL_* 必須在 MODEL 層級。放進
            # MODEL.UNIFIED_ENCODER 會被 getattr(_model_cfg, ...) 靜默忽略 ->
            # 整條鏈白跑 5 小時才發現模組根本沒建立。與 THERMAL_*_GLOBAL_WEIGHT
            # 的 two-valve 陷阱同一類失效模式：設定看起來有效，實際被丟棄。
            _ue_cfg_chk = getattr(_model_cfg, 'UNIFIED_ENCODER', None) if _model_cfg is not None else None
            if _ue_cfg_chk is not None:
                _stray = [k for k in ('GO_RESIDUAL_ENABLE', 'GO_RESIDUAL_MAX_DELTA',
                                      'GO_RESIDUAL_BOTTLENECK_DIM', 'GO_RESIDUAL_INIT_RAW_SCALE',
                                      'GO_RESIDUAL_DIM', 'TRAIN_GO_RESIDUAL')
                          if getattr(_ue_cfg_chk, k, None) is not None]
                if _stray:
                    raise ValueError(
                        f'[Config] {_stray} 出現在 MODEL.UNIFIED_ENCODER 之下，但 trainer '
                        f'只讀 MODEL 層級 -> 這些設定會被靜默忽略。請移到 MODEL: 底下（2 格縮排）。')
            _go_enable = bool(getattr(_model_cfg, 'GO_RESIDUAL_ENABLE', False)) if _model_cfg is not None else False
            _go_residual = None
            if _go_enable:
                from models.multimodal.mmped_unified_hmr2 import BoundedGlobalOrientResidual
                _go_dim = int(getattr(_model_cfg, 'GO_RESIDUAL_DIM', 1280))
                _go_bn = int(getattr(_model_cfg, 'GO_RESIDUAL_BOTTLENECK_DIM', 128))
                _go_max = float(getattr(_model_cfg, 'GO_RESIDUAL_MAX_DELTA', 0.2))
                _go_raw = float(getattr(_model_cfg, 'GO_RESIDUAL_INIT_RAW_SCALE', 0.0))
                _go_residual = BoundedGlobalOrientResidual(
                    dim=_go_dim, bottleneck_dim=_go_bn,
                    max_delta=_go_max, init_raw_scale=_go_raw)
                print(f'[Init][MMPedUnifiedHMR2] GO_RESIDUAL enabled: max_delta={_go_max} '
                      f'(6D L2 -> 約 {_go_max*40:.1f} 度旋轉上限) bottleneck={_go_bn} '
                      f'init_raw_scale={_go_raw} '
                      f'({sum(p.numel() for p in _go_residual.parameters())/1e3:.1f}K params, '
                      f'step-0 delta=0 -> 恆等)')
            # 2026-10-03 深層跨模態 adapter（CrossModalDeepAdapter）。MODEL.XMODAL_DEEP_*。
            _xm_enable = bool(getattr(_model_cfg, 'XMODAL_DEEP_ENABLE', False)) if _model_cfg is not None else False
            _xm_adapters, _xm_blocks = None, None
            if _xm_enable:
                from models.multimodal.mmped_unified_hmr2 import build_xmodal_deep_modules
                _xm_blocks = [int(b) for b in getattr(_model_cfg, 'XMODAL_DEEP_BLOCKS')]
                _xm_bn = int(getattr(_model_cfg, 'XMODAL_DEEP_BOTTLENECK_DIM', 128))
                _xm_adapters = build_xmodal_deep_modules(
                    _xm_blocks, str(getattr(_model_cfg, 'XMODAL_DEEP_TYPE', 'adapter')), _xm_bn,
                    float(getattr(_model_cfg, 'XMODAL_DEEP_DROPOUT', 0.0)),
                    float(getattr(_model_cfg, 'XMODAL_DEEP_MAX_REL_SCALE', 0.0)),
                    float(getattr(_model_cfg, 'XMODAL_DEEP_INIT_LOGIT', -4.0)))
                print(f'[Init][MMPedUnifiedHMR2] XMODAL_DEEP enabled: {len(_xm_blocks)} adapters at '
                      f'blocks {_xm_blocks} bottleneck={_xm_bn} '
                      f'({sum(p.numel() for p in _xm_adapters.parameters())/1e6:.2f}M params, step-0 恆等)')
            self.model = MMPedUnifiedHMR2(
                self.model, _unified_encoder, ir_token_adapter=_ir_adapter,
                body_adapters=_body_adapters,
                body_adapter_block_indices=_body_adapter_block_indices,
                ir_patch_stem=_ir_patch_stem,
                freeze_hmr2_body=_freeze_body, freeze_head=_freeze_head,
                go_residual=_go_residual,
                xmodal_deep_adapters=_xm_adapters, xmodal_deep_block_indices=_xm_blocks,
            ).to(self.device)
            self.model.xmodal_ref_input = str(getattr(_model_cfg, 'XMODAL_DEEP_REF_INPUT', 'raw')) if _model_cfg is not None else 'raw'
            assert self.model.xmodal_ref_input in ('raw', 'adapt'), self.model.xmodal_ref_input
            print(f'[Init][MMPedUnifiedHMR2] XMODAL_DEEP_REF_INPUT={self.model.xmodal_ref_input}')
            # optional per-module freeze (ablation / staging). Freezing (not just
            # LR=0) is required so the P0-6 requires_grad==lr>0 assert holds.
            _train_ir = bool(getattr(_model_cfg, 'TRAIN_IR_TOKEN_ADAPTER', True)) if _model_cfg is not None else True
            if _ir_adapter is not None and not _train_ir:
                for p in self.model.ir_token_adapter.parameters():
                    p.requires_grad = False
                print('[Init][MMPedUnifiedHMR2] TRAIN_IR_TOKEN_ADAPTER=false: IR adapter frozen')
            _train_ue = bool(getattr(_model_cfg, 'TRAIN_UNIFIED_ENCODER', True)) if _model_cfg is not None else True
            if not _train_ue:
                for p in self.model.unified_encoder.parameters():
                    p.requires_grad = False
                print('[Init][MMPedUnifiedHMR2] TRAIN_UNIFIED_ENCODER=false: unified encoder frozen')
            _train_ba = bool(getattr(_model_cfg, 'TRAIN_BODY_ADAPTER', True)) if _model_cfg is not None else True
            if _ba_enable and not _train_ba:
                for p in self.model.body_adapters.parameters():
                    p.requires_grad = False
                print('[Init][MMPedUnifiedHMR2] TRAIN_BODY_ADAPTER=false: body adapters frozen')
            _train_go = bool(getattr(_model_cfg, 'TRAIN_GO_RESIDUAL', True)) if _model_cfg is not None else True
            if _go_residual is not None and not _train_go:
                for p in self.model.go_residual.parameters():
                    p.requires_grad = False
                print('[Init][MMPedUnifiedHMR2] TRAIN_GO_RESIDUAL=false: global-orient residual frozen')
            _train_ips = bool(getattr(_model_cfg, 'TRAIN_IR_PATCH_STEM', True)) if _model_cfg is not None else True
            if _ips_enable and not _train_ips:
                for p in self.model.ir_patch_stem.parameters():
                    p.requires_grad = False
                print('[Init][MMPedUnifiedHMR2] TRAIN_IR_PATCH_STEM=false: IR patch stem frozen')
            # Scheme-3b: freeze ONLY the per-token gate (per_token_conf_mlp) while
            # the rest of the unified encoder + adapters keep training. Purpose:
            # cut the "IR improves -> rgbir pushes alpha_ir up" feedback loop that
            # made Scheme-3 regress RGBIR. The gate stays at the loaded (K=16)
            # balance; OTP2 then trains only body/IR adapters. Frozen (not LR=0)
            # so the P0-6 requires_grad==lr>0 assert holds.
            _train_gate = bool(getattr(_model_cfg, 'TRAIN_GATE', True)) if _model_cfg is not None else True
            if not _train_gate and getattr(self.model.unified_encoder, 'use_per_token_confidence', False):
                _n_gate = 0
                for p in self.model.unified_encoder.per_token_conf_mlp.parameters():
                    p.requires_grad = False
                    _n_gate += p.numel()
                print(f'[Init][MMPedUnifiedHMR2] TRAIN_GATE=false: per-token gate '
                      f'(per_token_conf_mlp, {_n_gate/1e3:.1f}K) frozen — gate fixed at '
                      f'loaded balance, OTP2 trains adapters only')
                # 2026-10-02: GATE_CMT_CONTEXT 時 CMT 也參與決定 alpha，gate 要「固定」
                # 就必須連整個 CMT（blocks / MAA / MAF / modality embed / 兩個調節 head）
                # 一起凍結，否則 S3b/S3c 的 alpha 仍會經由 CMT 漂移。
                _ue = self.model.unified_encoder
                if getattr(_ue, 'gate_cmt_context', False):
                    _n_cmt = 0
                    for _mod in (_ue.blocks, _ue.cmt_pair_mlp, _ue.cmt_token_mlp):
                        for p in _mod.parameters():
                            p.requires_grad = False
                            _n_cmt += p.numel()
                    for p in (_ue.maa, _ue.maf, _ue.rgb_mod_embed, _ue.ir_mod_embed):
                        p.requires_grad = False
                        _n_cmt += p.numel()
                    print(f'[Init][MMPedUnifiedHMR2] TRAIN_GATE=false + GATE_CMT_CONTEXT: '
                          f'CMT ({_n_cmt/1e6:.2f}M) frozen together with the gate')
            # scheme4: partial backbone unfreeze. With FREEZE_HMR2_BODY=false the
            # whole ViT-H (630M) is trainable -> AdamW state (~5.5G) + grads (~2.8G)
            # + the frozen teacher OOM a 24G GPU. UNFREEZE_LAST_N_BLOCKS>0 re-freezes
            # all but the last N transformer blocks (+ last_norm), matching v7's
            # proven recipe (last-6). Frozen early blocks store neither optimizer
            # state, gradients, nor (their own) activations -> fits memory AND
            # reduces catastrophic forgetting. Only meaningful when the body is not
            # already fully frozen.
            _unfreeze_n = int(getattr(_model_cfg, 'UNFREEZE_LAST_N_BLOCKS', 0) or 0) if _model_cfg is not None else 0
            if _unfreeze_n > 0 and not _freeze_body:
                _vit = self.model.backbone.backbone  # HMR2Encoder.backbone = ViT
                _blocks = getattr(_vit, 'blocks', None)
                if _blocks is not None:
                    _total = len(_blocks)
                    _keep = set(range(max(0, _total - _unfreeze_n), _total))
                    _frozen = 0
                    # freeze patch_embed + all blocks except the last N
                    for _pe in [getattr(_vit, 'patch_embed', None)]:
                        if _pe is not None:
                            for p in _pe.parameters():
                                p.requires_grad = False; _frozen += p.numel()
                    for _i, _blk in enumerate(_blocks):
                        if _i not in _keep:
                            for p in _blk.parameters():
                                p.requires_grad = False; _frozen += p.numel()
                    _bb_tr = sum(p.numel() for n, p in self.model.named_parameters()
                                 if n.startswith('backbone.backbone.') and p.requires_grad)
                    print(f'[Init][MMPedUnifiedHMR2] UNFREEZE_LAST_N_BLOCKS={_unfreeze_n}: '
                          f'froze {_frozen/1e6:.0f}M early backbone params | '
                          f'backbone trainable now {_bb_tr/1e6:.0f}M (last {_unfreeze_n} blocks + last_norm)')

            _ue_tr = sum(p.numel() for p in self.model.unified_encoder.parameters() if p.requires_grad)
            _ir_tr = sum(p.numel() for p in self.model.ir_token_adapter.parameters() if p.requires_grad) if _ir_adapter is not None else 0
            _ba_tr = sum(p.numel() for p in self.model.body_adapters.parameters() if p.requires_grad) if _ba_enable else 0
            print(
                f"[Init][MMPedUnifiedHMR2] wrapped HMR2 core (patch-token unified fusion) | "
                f"freeze_body={_freeze_body} | freeze_head={_freeze_head} | "
                f"unified_encoder={type(_unified_encoder).__name__}({_ue_tr/1e3:.1f}K) | "
                f"ir_token_adapter={_ir_tr/1e3:.1f}K | body_adapters={_ba_tr/1e3:.1f}K"
            )
            # Scheme-3c: thermal-teacher distillation. Load the thermal-finetuned
            # HMR2 (IR expert, IR-only 12.01) as a FROZEN forward-only teacher.
            # Its RGB/RGBIR are broken (22.45/12.09) but we only ever query it with
            # IR images on ir_only rows, so those outputs never enter any loss.
            self._teacher = None
            self._distill_pose_w = 0.0
            self._distill_beta_w = 0.0
            _dcfg = getattr(_model_cfg, 'DISTILL', None) if _model_cfg is not None else None
            _distill_enable = bool(getattr(_dcfg, 'ENABLE', False)) if _dcfg is not None else False
            if _distill_enable:
                _tck = getattr(_dcfg, 'TEACHER_CHECKPOINT', None)
                if not _tck or not os.path.isfile(_tck):
                    raise FileNotFoundError(
                        f'[Distill] DISTILL.ENABLE=true but TEACHER_CHECKPOINT missing/invalid: {_tck}')
                from models.hmr2_core.hmr2_core import build_hmr2_core as _bhc
                _teacher = _bhc(config.SMPL_MEAN_PARAMS, ckpt_path=_tck, freeze=True)
                _teacher.eval()
                for p in _teacher.parameters():
                    p.requires_grad = False
                self._teacher = _teacher.to(self.device)
                self._distill_pose_w = float(getattr(_dcfg, 'POSE_WEIGHT', 0.0) or 0.0)
                self._distill_beta_w = float(getattr(_dcfg, 'BETA_WEIGHT', 0.0) or 0.0)
                # When true, distill onto ALL IR-bearing rows (ir_only + rgbir),
                # not just ir_only. The teacher is a strong IR expert (far better
                # than the frozen-core+narrow-adapter student on both LLVIP and
                # OTP2), so guiding the fusion output too transfers more of it.
                # rgbir is 60% of LLVIP batches, so this ~3x's distill coverage.
                self._distill_apply_rgbir = bool(getattr(_dcfg, 'APPLY_TO_RGBIR', False))
                _rows_desc = ('all IR-bearing rows (ir_only + rgbir)'
                              if self._distill_apply_rgbir else 'ir_only rows only')
                print(f'[Init][Distill] thermal teacher loaded (frozen, forward-only) | '
                      f'ckpt={_tck} | pose_w={self._distill_pose_w} beta_w={self._distill_beta_w} | '
                      f'acts on {_rows_desc} (SMPL-param output distillation)')

        # Unified "modality (paired RGB-T) model" flag: late-token MU-HMR,
        # pre-backbone PBF-HMR2 and LT-RHMR all use the paired loader + train_step
        # modality forward + multi-mode validation. Non-modality runs keep False.
        self.is_modality_model = bool(
            getattr(self, 'is_mu_hmr', False)
            or getattr(self, 'is_pbf_hmr', False)
            or getattr(self, 'is_late_token_rgbt_hmr', False)
            or getattr(self, 'is_mmped_unified_hmr2', False)
        )

        # MU-HMR modality dropout config (train-time; applied in train_step on
        # fully-paired rows only). Read once here from MODALITY_DROPOUT.
        _md = getattr(self.options, 'MODALITY_DROPOUT', None)
        self._mu_dropout_enabled = bool(getattr(_md, 'ENABLED', False)) if _md is not None else False
        self._mu_dropout_probs = (
            (float(getattr(_md, 'LLVIP_RGB_ONLY_PROB', 0.30)),
             float(getattr(_md, 'LLVIP_IR_ONLY_PROB', 0.30)),
             float(getattr(_md, 'LLVIP_RGBIR_PROB', 0.40)))
            if _md is not None else (0.30, 0.30, 0.40)
        )
        if getattr(self, 'is_modality_model', False):
            _tag = ('MMPedUnifiedHMR2' if getattr(self, 'is_mmped_unified_hmr2', False)
                    else 'LT-RHMR' if getattr(self, 'is_late_token_rgbt_hmr', False)
                    else 'PBF-HMR2' if getattr(self, 'is_pbf_hmr', False) else 'MU-HMR')
            print(f"[Init][{_tag}] modality_dropout enabled={self._mu_dropout_enabled} "
                  f"probs(rgb/ir/both)={self._mu_dropout_probs}")

        # Supervision-coverage warning: LLVIP-pGT has no real 3D-joint / camera
        # GT, so keypoint_3d / cam_full / cam_depth losses are structurally
        # inactive (mask empty). Non-zero weights in config are misleading —
        # surface it so nobody reads 3D-MPJPE as a supervised metric.
        _ds_list = [str(d).lower() for d in getattr(self.options, 'dataset_list', []) or []]
        if any('llvip' in d for d in _ds_list):
            _w3d = float(getattr(self.options, 'keypoint_3d_loss_weight', 0.0) or 0.0)
            _wcf = float(getattr(self.options, 'cam_full_loss_weight', 0.0) or 0.0)
            _wcd = float(getattr(self.options, 'cam_depth_loss_weight', 0.0) or 0.0)
            if _w3d > 0 or _wcf > 0 or _wcd > 0:
                print("[WARN] Dataset llvip-pgt has no active 3D/camera supervision "
                      "(no 3D-joint / camera GT). "
                      f"KEYPOINT_3D({_w3d})/CAM_FULL({_wcf})/CAM_DEPTH({_wcd}) "
                      "losses will be INACTIVE despite non-zero weights; "
                      "validation 3D-MPJPE is an UNSUPERVISED reference only.")

        print(
            f"[Init] core=hmr2 | img_res={self.options.img_res} | "
            f"run_smplify={self.options.run_smplify}"
        )
        if core_type == 'hmr2':
            # HMR2 core: analogous param summary + freeze check on the ViT.
            _hmr2_freeze = bool(getattr(self.model.backbone, 'freeze', False))
            total_params = sum(p.numel() for p in self.model.parameters())
            total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            bb_params = sum(p.numel() for p in self.model.backbone.backbone.parameters())
            bb_trainable = sum(p.numel() for p in self.model.backbone.backbone.parameters() if p.requires_grad)
            print(f"[Init] core_type=hmr2 | feat_dim={self.model.feat_dim} | "
                  f"vit_freeze={_hmr2_freeze}")
            print(f"[Init] param_summary: total={total_params/1e6:.1f}M | "
                  f"trainable={total_trainable/1e6:.1f}M | "
                  f"hmr2_vit_trainable={bb_trainable/1e6:.1f}M")
            if _hmr2_freeze:
                bad = [n for n, p in self.model.backbone.backbone.named_parameters() if p.requires_grad]
                if bad:
                    raise RuntimeError(
                        f"[Init] HMR2 ViT freeze check FAILED: {len(bad)} params still trainable. "
                        f"First 3: {bad[:3]}"
                    )
                print("[Init] HMR2 ViT freeze check PASSED: all backbone params frozen")
        _head_type = str(getattr(self.options, 'head_type', 'iterative_mlp'))
        _camera_rep = str(getattr(self.options, 'camera_rep', 'weak_perspective'))
        print(
            f"[Init] hmr_head_type={_head_type} | "
            f"decoder_dim={getattr(self.options, 'decoder_dim', 1024)} | "
            f"decoder_depth={getattr(self.options, 'decoder_depth', 1)} | "
            f"decoder_heads={getattr(self.options, 'decoder_heads', 8)} | "
            f"decoder_mlp_ratio={getattr(self.options, 'decoder_mlp_ratio', 4.0)} | "
            f"decoder_dropout={getattr(self.options, 'decoder_dropout', 0.0)}"
        )
        print(
            f"[Init] camera_rep={_camera_rep} | "
            f"full_persp_init_depth={getattr(self.options, 'full_persp_init_depth', 5.0)} | "
            f"full_persp_min_depth={getattr(self.options, 'full_persp_min_depth', 1e-3)} | "
            f"full_persp_tz_param={getattr(self.options, 'full_persp_tz_param', 'clamp')}"
        )
        print(
            f"[Init] use_bbox_token={bool(getattr(self.options, 'use_bbox_token', False))} | "
            f"bbox_token_hidden={int(getattr(self.options, 'bbox_token_hidden', 128))}"
        )
        if _camera_rep == 'full_perspective':
            # Step 3B is done: trainer / eval / demo projection pipelines all
            # dispatch on model.camera_rep, so full_perspective is honoured
            # end-to-end. Print a short banner so the run config is visible.
            print(
                "[Init] camera_rep='full_perspective': model emits (tx,ty,tz) "
                "directly; trainer/eval/demo skip cam_crop2full and project "
                "with image focal/centre."
            )
        if getattr(self.options, 'paths_config_source', None):
            print(f"[Init] paths_cfg={self.options.paths_config_source}")
        print(
            f"[Init] batch_size={self.options.batch_size} | "
            f"num_workers={self.options.num_workers} | "
            f"summary_steps={self.options.summary_steps} | "
            f"save_step={getattr(self.options, 'save_step', None)} | "
            f"valid_step={getattr(self.options, 'valid_step', None)}"
        )

        opt_name = str(getattr(self.options, 'optimizer', 'adamw')).lower()
        if opt_name not in ('adam', 'adamw'):
            raise ValueError(
                f"Unsupported TRAIN.OPTIMIZER='{opt_name}'. Must be 'adam' or 'adamw'."
            )

        # Differential LR param groups: backbone / cam_head / head.
        # Each parameter belongs to at most one group (id() dedup); a group with
        # zero trainable params is skipped (so VITPOSE_FREEZE=true is safe).
        # When per-group LR is None, fall back to the global TRAIN.LR — this
        # preserves existing behavior for YAMLs that only set TRAIN.LR.
        base_lr = float(self.options.lr)
        _b_opt = getattr(self.options, 'backbone_lr', None)
        _h_opt = getattr(self.options, 'head_lr', None)
        _c_opt = getattr(self.options, 'cam_head_lr', None)
        # MU-HMR (plan §0.2 P0-6): adapter / unifier get their own LR groups so
        # they never silently land in the 'head' group at HEAD_LR=0.
        _u_opt = getattr(self.options, 'unifier_lr', None)
        _a_opt = getattr(self.options, 'thermal_adapter_lr', None)
        # PBF-HMR2: pre-backbone input adapter / spatial gate LR groups.
        _ia_opt = getattr(self.options, 'input_adapter_lr', None)
        _sg_opt = getattr(self.options, 'spatial_gate_lr', None)
        # LT-RHMR: late-token thermal token adapter / reliability unifier groups.
        _tta_opt = getattr(self.options, 'thermal_token_adapter_lr', None)
        _ltu_opt = getattr(self.options, 'late_token_unifier_lr', None)
        _ue_opt = getattr(self.options, 'unified_encoder_lr', None)
        _irta_opt = getattr(self.options, 'ir_token_adapter_lr', None)
        _ba_opt = getattr(self.options, 'body_adapter_lr', None)
        _ips_opt = getattr(self.options, 'ir_patch_stem_lr', None)
        _go_opt = getattr(self.options, 'go_residual_lr', None)
        _xm_opt = getattr(self.options, 'xmodal_deep_lr', None)
        xmodal_deep_lr_val = float(_xm_opt) if _xm_opt is not None else base_lr
        backbone_lr_val = float(_b_opt) if _b_opt is not None else base_lr
        head_lr_val = float(_h_opt) if _h_opt is not None else base_lr
        cam_head_lr_val = float(_c_opt) if _c_opt is not None else base_lr
        unifier_lr_val = float(_u_opt) if _u_opt is not None else base_lr
        adapter_lr_val = float(_a_opt) if _a_opt is not None else base_lr
        input_adapter_lr_val = float(_ia_opt) if _ia_opt is not None else base_lr
        spatial_gate_lr_val = float(_sg_opt) if _sg_opt is not None else base_lr
        thermal_token_adapter_lr_val = float(_tta_opt) if _tta_opt is not None else base_lr
        late_token_unifier_lr_val = float(_ltu_opt) if _ltu_opt is not None else base_lr
        unified_encoder_lr_val = float(_ue_opt) if _ue_opt is not None else base_lr
        ir_token_adapter_lr_val = float(_irta_opt) if _irta_opt is not None else base_lr
        body_adapter_lr_val = float(_ba_opt) if _ba_opt is not None else base_lr
        ir_patch_stem_lr_val = float(_ips_opt) if _ips_opt is not None else base_lr
        go_residual_lr_val = float(_go_opt) if _go_opt is not None else base_lr

        backbone_ids = set()
        backbone_params = []
        cam_head_ids = set()
        cam_head_params = []
        adapter_ids = set()
        adapter_params = []
        unifier_ids = set()
        unifier_params = []
        input_adapter_ids = set()
        input_adapter_params = []
        spatial_gate_ids = set()
        spatial_gate_params = []
        thermal_token_adapter_ids = set()
        thermal_token_adapter_params = []
        late_token_unifier_ids = set()
        late_token_unifier_params = []
        unified_encoder_ids = set()
        unified_encoder_params = []
        ir_token_adapter_ids = set()
        ir_token_adapter_params = []
        body_adapter_ids = set()
        body_adapter_params = []
        ir_patch_stem_ids = set()
        ir_patch_stem_params = []
        go_residual_params = []
        go_residual_ids = set()
        xmodal_deep_params = []
        xmodal_deep_ids = set()
        head_params = []
        # Match camera-head params across legacy MLP head ('deccam') and the
        # HMR2.0 transformer-decoder head ('dec_cam_t', 'cam_head', etc).
        CAM_HEAD_PATTERNS = ('deccam', 'dec_cam_t', 'cam_head', 'camera_head', 'head.camera')
        for n, p in self.model.named_parameters():
            if n.startswith('backbone.'):
                backbone_params.append(p)
                backbone_ids.add(id(p))
        for n, p in self.model.named_parameters():
            if id(p) in backbone_ids:
                continue
            if any(pat in n for pat in CAM_HEAD_PATTERNS):
                cam_head_params.append(p)
                cam_head_ids.add(id(p))
        # MU-HMR adapter / unifier groups MUST be collected before 'head'
        # (everything-else) so they are not swallowed at head_lr.
        for n, p in self.model.named_parameters():
            if id(p) in backbone_ids or id(p) in cam_head_ids:
                continue
            if n.startswith('thermal_adapter.'):
                adapter_params.append(p)
                adapter_ids.add(id(p))
            elif n.startswith('modality_unifier.'):
                unifier_params.append(p)
                unifier_ids.add(id(p))
            elif n.startswith('thermal_input_adapter.'):
                input_adapter_params.append(p)
                input_adapter_ids.add(id(p))
            elif n.startswith('spatial_fusion_gate.'):
                spatial_gate_params.append(p)
                spatial_gate_ids.add(id(p))
            elif n.startswith('thermal_token_adapter.'):
                thermal_token_adapter_params.append(p)
                thermal_token_adapter_ids.add(id(p))
            elif n.startswith('token_unifier.'):
                late_token_unifier_params.append(p)
                late_token_unifier_ids.add(id(p))
            elif n.startswith('unified_encoder.'):
                unified_encoder_params.append(p)
                unified_encoder_ids.add(id(p))
            elif n.startswith('ir_token_adapter.'):
                ir_token_adapter_params.append(p)
                ir_token_adapter_ids.add(id(p))
            elif n.startswith('body_adapters.'):
                body_adapter_params.append(p)
                body_adapter_ids.add(id(p))
            elif n.startswith('ir_patch_stem.'):
                ir_patch_stem_params.append(p)
                ir_patch_stem_ids.add(id(p))
            elif n.startswith('go_residual.'):
                go_residual_params.append(p)
                go_residual_ids.add(id(p))
            elif n.startswith('xmodal_deep_adapters.'):
                xmodal_deep_params.append(p)
                xmodal_deep_ids.add(id(p))
        for n, p in self.model.named_parameters():
            if (id(p) in backbone_ids or id(p) in cam_head_ids
                    or id(p) in adapter_ids or id(p) in unifier_ids
                    or id(p) in input_adapter_ids or id(p) in spatial_gate_ids
                    or id(p) in thermal_token_adapter_ids
                    or id(p) in late_token_unifier_ids
                    or id(p) in unified_encoder_ids
                    or id(p) in ir_token_adapter_ids
                    or id(p) in body_adapter_ids
                    or id(p) in ir_patch_stem_ids
                    or id(p) in go_residual_ids
                    or id(p) in xmodal_deep_ids):
                continue
            head_params.append(p)

        param_groups = []

        def _add_group(plist, lr_val, name):
            trainable = [p for p in plist if p.requires_grad]
            if len(trainable) == 0:
                print(f'[LR] group "{name}" has 0 trainable params (skipped)')
                return
            param_groups.append({'params': trainable, 'lr': float(lr_val), 'name': name})

        _add_group(backbone_params,       backbone_lr_val, 'backbone')
        _add_group(head_params,           head_lr_val,     'head')
        _add_group(cam_head_params,       cam_head_lr_val, 'cam_head')
        _add_group(adapter_params,        adapter_lr_val,  'thermal_adapter')
        _add_group(unifier_params,        unifier_lr_val,  'modality_unifier')
        _add_group(input_adapter_params,  input_adapter_lr_val, 'thermal_input_adapter')
        _add_group(spatial_gate_params,   spatial_gate_lr_val,  'spatial_fusion_gate')
        _add_group(thermal_token_adapter_params, thermal_token_adapter_lr_val, 'thermal_token_adapter')
        _add_group(late_token_unifier_params,    late_token_unifier_lr_val,    'late_token_unifier')
        _add_group(unified_encoder_params,       unified_encoder_lr_val,       'unified_encoder')
        _add_group(ir_token_adapter_params,      ir_token_adapter_lr_val,      'ir_token_adapter')
        _add_group(body_adapter_params,          body_adapter_lr_val,          'body_adapters')
        _add_group(go_residual_params,           go_residual_lr_val,           'go_residual')
        _add_group(xmodal_deep_params,           xmodal_deep_lr_val,           'xmodal_deep')
        _add_group(ir_patch_stem_params,         ir_patch_stem_lr_val,         'ir_patch_stem')

        if len(param_groups) == 0:
            raise RuntimeError('[LR] No trainable parameter group found; cannot build optimizer.')

        # Modality-model sanity (plan §0.2 P0-6): every requires_grad param must
        # live in an optimizer group with lr>0, else training silently no-ops.
        # Applies to both late-token MU-HMR and pre-backbone PBF-HMR2.
        if getattr(self, 'is_modality_model', False):
            n_req_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_lr_pos = sum(
                sum(p.numel() for p in g['params'])
                for g in param_groups if float(g['lr']) > 0.0
            )
            _mtag = ('MMPedUnifiedHMR2' if getattr(self, 'is_mmped_unified_hmr2', False)
                     else 'LT-RHMR' if getattr(self, 'is_late_token_rgbt_hmr', False)
                     else 'PBF-HMR2' if getattr(self, 'is_pbf_hmr', False) else 'MU-HMR')
            print(f'[LR][{_mtag}] requires_grad params={n_req_grad} | in lr>0 groups={n_lr_pos}')
            if n_req_grad != n_lr_pos:
                raise RuntimeError(
                    f'[LR][{_mtag}] trainable/optimizer mismatch: requires_grad='
                    f'{n_req_grad} but lr>0 groups hold {n_lr_pos}. Some trainable '
                    f'params are at lr=0 (silent no-op) or some lr>0 params are frozen.'
                )

        weight_decay = float(getattr(self.options, 'weight_decay', 0.0))
        opt_kwargs = dict(params=param_groups, weight_decay=weight_decay)
        if opt_name == 'adam':
            self.optimizer = torch.optim.Adam(**opt_kwargs)
        else:
            self.optimizer = torch.optim.AdamW(**opt_kwargs)

        for g in param_groups:
            n_params = sum(p.numel() for p in g['params'])
            print(f'[LR] group="{g["name"]}" trainable_params={n_params} lr={g["lr"]}')
        print(
            f"[Init] optimizer={opt_name} | base_lr={base_lr} | "
            f"backbone_lr={backbone_lr_val} | head_lr={head_lr_val} | cam_head_lr={cam_head_lr_val} | "
            f"weight_decay={weight_decay}"
        )
        def _make_lr_lambda(schedule, warmup_steps, max_step):
            def cosine_warmup(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, max_step - warmup_steps)
                return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
            return {'cosine_warmup': cosine_warmup}.get(schedule, lambda _: 1.0)

        lr_schedule = getattr(self.options, 'lr_schedule', 'constant')
        warmup_steps = getattr(self.options, 'warmup_steps', 0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=_make_lr_lambda(lr_schedule, warmup_steps, getattr(self.options, 'max_step', 1)),
        )
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        self.smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(self.device)
        self.smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(self.device)
        _kp2d_beta = float(getattr(self.options, 'keypoint_2d_loss_beta', 1.0) or 1.0)
        self.loss_mixer = ReFitStyleLossMixer(keypoint_2d_beta=_kp2d_beta).to(self.device)
        print(f'[Init] keypoint_2d_loss SmoothL1 beta={_kp2d_beta} '
              f'({"LINEAR/L1-like" if _kp2d_beta < 0.105 else "QUADRATIC/MSE-like"} '
              f'at the typical ~0.105 normalized error)')
        # Backward-compatible aliases for existing helper methods.
        self.criterion_shape = self.loss_mixer.criterion_shape
        self.criterion_keypoints = self.loss_mixer.criterion_keypoints
        self.criterion_regr = self.loss_mixer.criterion_regr
        self.loss_weights = build_loss_weights(self.options)
        self.supervision_mode = resolve_supervision_mode(self.options)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
        }
        self.focal_length = constants.FOCAL_LENGTH

        # Initialize SPIN-compatible SMPLify (with optional init regularization)
        # [TODO H7] crop_res is passed as self.options.img_res (256); verify SMPLify's
        # 2D reprojection geometry matches the external crop-space when the HRNet
        # backbone center-crops to 256x192 internally.
        self.smplify = SMPLify(
            step_size=getattr(self.options, 'smplify_step_size', 1e-2),
            batch_size=self.options.batch_size,
            num_iters=self.options.num_smplify_iters,
            camera_num_iters=getattr(self.options, 'num_smplify_cam_iters', 10),
            body_num_iters=getattr(self.options, 'num_smplify_body_iters', 20),
            focal_length=self.focal_length,
            device=self.device,
            pose_prior_weight=getattr(self.options, 'smplify_pose_prior_weight', 4.78),
            shape_prior_weight=getattr(self.options, 'smplify_shape_prior_weight', 5.0),
            angle_prior_weight=getattr(self.options, 'smplify_angle_prior_weight', 15.2),
            sigma=getattr(self.options, 'smplify_sigma', 100.0),
            init_pose_reg_weight=getattr(self.options, 'smplify_init_pose_reg_weight', 0.00),
            init_betas_reg_weight=getattr(self.options, 'smplify_init_betas_reg_weight', 0.00),
            init_global_orient_reg_weight=getattr(self.options, 'smplify_init_global_orient_reg_weight', 0.00),
            crop_res=self.options.img_res,
        )
        if self.options.pretrained_checkpoint not in [None, '']:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # L2-SP anchor for the (optionally unfrozen) SMPL head. Diagnosis
        # (2026-07-10): unfreezing head.* drifts the POSE (global-orient dir +
        # body pose) away from HMR2.0's calibrated prior -> validation breaks
        # while the crop-space keypoint loss stays low. Snapshot the pretrained
        # head weights NOW (post-load) so the L2-SP term can pull them back.
        self._head_l2sp_weight = float(getattr(self.options, 'head_l2sp_weight', 0.0) or 0.0)
        self._head_l2sp_anchor = None
        if self._head_l2sp_weight > 0.0:
            self._head_l2sp_anchor = {
                n: p.detach().clone()
                for n, p in self.model.named_parameters()
                if n.startswith('head.') and p.requires_grad
            }
            n_anchor = sum(v.numel() for v in self._head_l2sp_anchor.values())
            print(f'[Init] head L2-SP anchor: weight={self._head_l2sp_weight} '
                  f'| {len(self._head_l2sp_anchor)} tensors ({n_anchor:,} params)')

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)
        print("[Init] fits_dict=ready")
        print(
            f"[Init] supervision_mode={self.supervision_mode} | "
            f"loss_weights={self.loss_weights}"
        )

        # Create renderer (optional in headless/remote environments)
        self.render_summaries = bool(getattr(self, 'render_summaries_preflight', getattr(self.options, 'render_summaries', True)))
        self.renderer = None
        if self.render_summaries:
            # Mesh-rendering TensorBoard summaries are not part of this release.
            print("[Trainer] LOGGING.RENDER_SUMMARIES is not supported; disabled.")
            self.render_summaries = False

        # Validation setup (ReFit-style step-based validation).
        # Plan 06.28: support multi-dataset validation via `validation_datasets`.
        self.validation_loader = None
        self.validation_loaders = {}
        self.validation_metric_name = self._resolve_validation_metric()
        self.validation_dataset_names = self._resolve_validation_dataset_names()
        self.validation_dataset_name = self.validation_dataset_names[0]
        self.validation_primary_metric = self._resolve_validation_primary_metric()
        self.validation_enabled = self._resolve_validation_enabled()
        self.val_J_regressor = None
        self.val_smpl_neutral = None
        self.val_smpl_male = None
        self.val_smpl_female = None
        if self.validation_enabled:
            for _name in self.validation_dataset_names:
                self.validation_loader = None
                self.validation_dataset_name = _name
                self._build_validation_loader()
                if self.validation_loader is not None:
                    self.validation_loaders[_name] = self.validation_loader
            # restore primary dataset state for legacy single-loader code paths
            if self.validation_loaders:
                self.validation_dataset_name = self.validation_dataset_names[0]
                self.validation_loader = self.validation_loaders[self.validation_dataset_name]
        else:
            print("[Init] validation disabled by config.")

        # Per-dataset supervision coverage summary.
        try:
            _names = list(getattr(self.train_ds, 'dataset_list', []))
            _dsets = list(getattr(self.train_ds, 'datasets', []))
            _part = list(getattr(self.train_ds, 'partition', [0.0] * len(_names)))
            print("[SupervisionCoverage] per-dataset stats:")
            for _i, (_name, _ds) in enumerate(zip(_names, _dsets)):
                _n = int(len(_ds))
                _intr = float(np.mean(_ds.has_cam_intrinsics_arr)) if _n > 0 else 0.0
                _hs = float(np.mean(_ds.has_smpl)) if _n > 0 else 0.0
                # smpl_sup_rate: 實際能讓 SMPL pose/shape loss fire 的比例
                # (gt_only 用 has_smpl；fit_or_gt 的 valid_fit 因 `| has_smpl` 也必含這些)
                _smpl_sup = _hs
                # smpl_with_intrinsics_rate: 既有 SMPL pGT 又有相機內參的交集
                # (用來判斷哪些 sample 同時可餵 cam/depth loss)
                _smpl_with_intr = float(np.mean(
                    (_ds.has_cam_intrinsics_arr > 0) &
                    (_ds.has_smpl > 0)
                )) if _n > 0 else 0.0
                _p = float(_part[_i]) if _i < len(_part) else 0.0
                print(f"[SupervisionCoverage]   {_name}: n={_n} | partition={_p:.3f} | "
                      f"intrinsics_rate={_intr:.3f} | has_smpl_rate={_hs:.3f} | "
                      f"smpl_sup_rate={_smpl_sup:.3f} | "
                      f"smpl_with_intrinsics_rate={_smpl_with_intr:.3f}")
                if _smpl_sup == 0.0 and _p > 0.0:
                    print(f"[SupervisionCoverage][WARN] {_name} has smpl_sup_rate=0 "
                          f"but partition={_p:.3f} > 0; SMPL loss will not fire for this dataset.")
        except Exception as _e:
            print(f"[SupervisionCoverage][WARN] failed to summarize: {_e}")
            # 補上 traceback，避免 dataset wrapper 結構改動後靜默吞錯
            traceback.print_exc()

    def load_pretrained(self, checkpoint_file=None):
        if checkpoint_file in [None, '']:
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        loaded_any = False
        for model_key in self.models_dict:
            if model_key not in checkpoint:
                continue
            state = checkpoint[model_key]
            model = self.models_dict[model_key]
            model_sd = model.state_dict()
            if 'fc1.weight' in state and 'fc1.weight' in model_sd:
                old_w = state['fc1.weight']
                new_w = model_sd['fc1.weight']
                if (old_w.ndim == 2 and new_w.ndim == 2 and
                        old_w.shape[0] == new_w.shape[0] and
                        old_w.shape[1] < new_w.shape[1]):
                    pad = torch.zeros(
                        old_w.shape[0],
                        new_w.shape[1] - old_w.shape[1],
                        dtype=old_w.dtype,
                        device=old_w.device,
                    )
                    state['fc1.weight'] = torch.cat([old_w, pad], dim=1)
                    print(f'[Init] fc1.weight zero-padded {tuple(old_w.shape)} -> {tuple(state["fc1.weight"].shape)}')
            result = model.load_state_dict(state, strict=False)
            loaded_any = True
            # P0-8: report key-match so a re-parent / prefix mistake is visible.
            model_sd_keys = set(model_sd.keys())
            n_matched = sum(1 for k in state if k in model_sd_keys)
            print(f'[Init] load_pretrained[{model_key}]: ckpt_keys={len(state)} '
                  f'matched={n_matched} missing={len(result.missing_keys)} '
                  f'unexpected={len(result.unexpected_keys)}')
            if getattr(self, 'is_mu_hmr', False):
                # New modules SHOULD be the only missing keys; unexpected SHOULD be 0.
                ex_missing = [k for k in result.missing_keys
                              if not (k.startswith('thermal_adapter.')
                                      or k.startswith('modality_unifier.'))]
                if ex_missing:
                    print(f'[Init][MU-HMR][WARN] unexpected missing (non-adapter/unifier) '
                          f'keys: {len(ex_missing)} e.g. {ex_missing[:3]}')
                if result.unexpected_keys:
                    print(f'[Init][MU-HMR][WARN] unexpected ckpt keys not in model: '
                          f'{len(result.unexpected_keys)} e.g. {result.unexpected_keys[:3]}')
        if loaded_any:
            print(f'[Init] pretrained_checkpoint_loaded={checkpoint_file}')

    def finalize(self):
        pass

    # Keys printed by _maybe_print_losses (console only, not CSV).
    _PRINT_LOSS_KEYS = (
        'loss',
        'weighted_loss_keypoints_2d',
        'weighted_loss_keypoints_2d_pixel',
        'weighted_loss_cam_full',
        'gate_rgb_mean',
        'gate_ir_mean',
        'num_rgb_only_samples',
        'num_ir_only_samples',
        'num_rgbir_samples',
    )

    def _maybe_print_losses(self, epoch, step, losses):
        """Print selected losses every step (sanity) or every loss_record_steps.

        Override of the no-op in BaseTrainer. For sanity runs (max_step ≤ 10)
        we print every step so that 5-iter sanity is fully observable without
        waiting for the CSV-flush threshold (default 50 steps).
        """
        max_step = int(getattr(self.options, 'max_step', 0) or 0)
        is_sanity = (max_step > 0 and max_step <= 10)
        record_steps = max(1, self.loss_record_steps)
        if not is_sanity and self.step_count % record_steps != 0:
            return
        parts = [f'step={self.step_count}']
        for k in self._PRINT_LOSS_KEYS:
            if k not in losses:
                continue
            v = losses[k]
            if 'num_' in k:
                parts.append(f'{k}={int(v)}')
            else:
                parts.append(f'{k}={v:.4f}')
        print('[TrainStep] ' + ' | '.join(parts))

    def _thermal_sample_mask(self, dataset_name, batch_size):
        thermal_names = {'llvip', 'llvip-pgt', 'llvip-pgt-train90', 'otp2', 'otp2-pgt'}
        if isinstance(dataset_name, (list, tuple)):
            vals = [str(x) for x in dataset_name]
        else:
            vals = [str(dataset_name)] * batch_size
        return torch.tensor([v.lower() in thermal_names for v in vals], device=self.device, dtype=torch.bool)

    def _batch_float_or_default(self, input_batch, key, batch_size, default=1.0):
        value = input_batch.get(key, None)
        if value is None:
            return torch.full((batch_size,), float(default), device=self.device)
        return value.to(self.device).float().view(batch_size)

    def _build_validation_loader(self):
        name = self.validation_dataset_name
        val_batch_size = int(getattr(self.options, 'validation_batch_size', 0) or 0)
        if val_batch_size <= 0:
            val_batch_size = int(self.options.batch_size)
        val_num_workers = int(getattr(self.options, 'validation_num_workers', self.options.num_workers))

        if name in ('otp2', 'otp2-pgt'):
            self._build_thermal_validation_loader('otp2', val_batch_size, val_num_workers)
        elif name in ('llvip', 'llvip-pgt'):
            self._build_thermal_validation_loader('llvip', val_batch_size, val_num_workers)
        else:
            raise ValueError(
                f"Unsupported validation dataset: '{name}'. "
                "Supported: 'otp2', 'otp2-pgt', 'llvip', 'llvip-pgt'"
            )

    def _build_thermal_validation_loader(self, dataset_key, val_batch_size, val_num_workers):
        if dataset_key == 'otp2':
            npz_attr = 'validation_otp2_npz'
            val_attr = 'OTP2_NPZ'
            ds_name = 'otp2-pgt'
            default_npz = os.path.join(
                'data/dataset_extras/thermal_pgt_camsmplify_v3_relaxed/otp2_val_smplify_pgt.npz'
            )
        else:
            npz_attr = 'validation_llvip_npz'
            val_attr = 'LLVIP_NPZ'
            ds_name = 'llvip-pgt'
            # Use config.DATASET_FILES[0] (is_train=False) as the canonical source
            # so the default tracks any future NPZ version change in config.py.
            default_npz = config.DATASET_FILES[0].get('llvip-pgt', os.path.join(
                'data/dataset_extras/thermal_pgt_camsmplify_v5/llvip_test_smplify_pgt.npz'
            ))

        npz_path = getattr(self.options, npz_attr, None)
        if npz_path in [None, ''] and hasattr(self.options, 'VALIDATION'):
            npz_path = getattr(self.options.VALIDATION, val_attr, None)
        if npz_path in [None, '']:
            npz_path = default_npz

        # MU-HMR + LLVIP: use LLVIPPairedDataset for multi-mode validation.
        # The paired loader returns img_rgb / img_ir / modality_mask /
        # sample_index so _validate_llvip_multimode can do three forward passes.
        if dataset_key == 'llvip' and getattr(self, 'is_modality_model', False):
            from datasets.llvip_paired_dataset import LLVIPPairedDataset
            val_dataset = LLVIPPairedDataset(
                options=self.options,
                dataset=ds_name,
                is_train=False,
                use_augmentation=False,
                npz_path_override=npz_path,
                validate_samples=3,
            )
        else:
            val_dataset = ThermalValidationDataset(
                options=self.options,
                npz_path=npz_path,
                dataset_name=ds_name,
            )
        self.validation_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=bool(getattr(self.options, 'pin_memory', True)),
            drop_last=False,
        )
        self.val_smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(self.device)
        # Store npz_path so _validate_thermal_2d / _validate_llvip_multimode
        # can load GT arrays directly.
        if not hasattr(self, '_thermal_val_npz_paths'):
            self._thermal_val_npz_paths = {}
        self._thermal_val_npz_paths[dataset_key] = npz_path
        print(
            f"[Init] validation_dataset={dataset_key} | samples={len(val_dataset)} | "
            f"batch_size={val_batch_size} | npz={npz_path} | "
            f"mu_paired={'yes' if dataset_key == 'llvip' and getattr(self, 'is_modality_model', False) else 'no'}"
        )

    def _build_gendered_gt_vertices(self, gt_pose, gt_betas, gender):
        """Build GT vertices using gender-specific SMPL models (0=male, 1=female, else neutral)."""
        ref = self.smpl(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3],
        ).vertices
        gt_vertices = torch.zeros_like(ref)
        male_mask = gender == 0
        female_mask = gender == 1
        neutral_mask = ~(male_mask | female_mask)
        if male_mask.any():
            gt_vertices[male_mask] = self.smpl_male(
                betas=gt_betas[male_mask],
                body_pose=gt_pose[male_mask, 3:],
                global_orient=gt_pose[male_mask, :3],
            ).vertices
        if female_mask.any():
            gt_vertices[female_mask] = self.smpl_female(
                betas=gt_betas[female_mask],
                body_pose=gt_pose[female_mask, 3:],
                global_orient=gt_pose[female_mask, :3],
            ).vertices
        if neutral_mask.any():
            gt_vertices[neutral_mask] = ref[neutral_mask]
        return gt_vertices

    @torch.no_grad()
    def validate(self):
        if not self.validation_enabled or self.validation_loader is None:
            return None

        # Plan 06.28 §9.2: when multiple validation datasets are configured,
        # run each one and flatten metrics into a single dict whose keys are
        # `<dataset>/<metric>` (so the base trainer can save one best
        # checkpoint per metric automatically).
        if len(getattr(self, 'validation_loaders', {})) > 1:
            return self._validate_multi()

        name = self.validation_dataset_name
        if name in ('otp2', 'otp2-pgt', 'llvip', 'llvip-pgt'):
            # MU-HMR / PBF-HMR2 LLVIP: three-mode forward (rgb/ir/rgbir) via paired loader.
            if getattr(self, 'is_modality_model', False) and name in ('llvip', 'llvip-pgt'):
                return self._validate_llvip_multimode()
            return self._validate_thermal_2d()
        raise ValueError(f"Unsupported validation dataset: '{name}'")

    @torch.no_grad()
    def _validate_thermal_2d(self):
        """Thermal 2D/3D validation matching eval_hmr_2d.py protocol exactly.

        2D metrics: GT from NPZ 'part' field (H36M-24 full-image pixel coords) →
            SPIN24_TO_COCO17 → COCO-17. Prediction projected to full-image pixel
            via perspective projection. Per-joint accumulate (not per-sample mean).

        3D metrics: GT from NPZ 'pose'/'shape' (valid_fit mask). Pred root-relative
            H36M-24 joints. Both mapped to COCO-17 via SPIN24_TO_COCO17 before
            computing MPJPE/PA-MPJPE.

        PCK@0.05: threshold = 0.05 * sqrt(bbox_w^2 + bbox_h^2) per person.
        """
        name = self.validation_dataset_name
        dataset_key = 'otp2' if name in ('otp2', 'otp2-pgt') else 'llvip'
        print(f"[Validation] step={self.step_count} | {name} thermal eval ...")

        # ── Load GT arrays from NPZ directly ────────────────────────────────
        npz_path = getattr(self, '_thermal_val_npz_paths', {}).get(dataset_key)
        if npz_path is None or not os.path.isfile(npz_path):
            print(f"[Validation] {name}: npz_path not found ({npz_path}), skipping.")
            return None

        raw = np.load(npz_path, allow_pickle=False)
        n_total = int(len(raw['imgname']))

        # GT 2D: part (N,24,3) in full-image pixel, same convention as COCO GT
        gt_part       = raw['part'].astype(np.float32)         # (N,24,3)
        gt_coco17     = gt_part[:, SPIN24_TO_COCO17, :]        # (N,17,3)

        # PCK threshold per person
        gt_bbox       = raw['bbox'].astype(np.float32)         # (N,4) xywh
        gt_bbox_diag  = np.sqrt(gt_bbox[:, 2] ** 2 + gt_bbox[:, 3] ** 2)  # (N,)

        # GT 3D pGT
        gt_pose       = raw['pose'].astype(np.float32)         # (N,72)
        gt_shape      = raw['shape'].astype(np.float32)        # (N,10)
        gt_valid_fit  = raw['valid_fit'].astype(bool)          # (N,)

        # ── Compute GT 3D joints from pGT pose/shape ────────────────────────
        gt_j3d = np.zeros((n_total, 24, 3), dtype=np.float32)
        valid_idx = np.where(gt_valid_fit)[0]
        _batch3d = 128
        for _s in range(0, len(valid_idx), _batch3d):
            _bidx  = valid_idx[_s:_s + _batch3d]
            _pose  = torch.from_numpy(gt_pose[_bidx]).float().to(self.device)
            _shape = torch.from_numpy(gt_shape[_bidx]).float().to(self.device)
            _rot   = batch_rodrigues(_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
            _out   = self.val_smpl_neutral(
                betas=_shape,
                body_pose=_rot[:, 1:],
                global_orient=_rot[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            _j  = _out.joints[:, 25:49, :]         # H36M-24
            # Store RAW (un-recentered); both root conventions are applied later
            # in _mpjpe_3d_both_roots so we can report pelvis + legacy ankle.
            gt_j3d[_bidx] = _j.cpu().numpy()

        # ── Inference loop ───────────────────────────────────────────────────
        self.model.eval()
        pred_coco17_px = np.zeros((n_total, 17, 2), dtype=np.float32)
        pred_j3d       = np.zeros((n_total, 24, 3), dtype=np.float32)

        sample_idx = 0
        for batch in tqdm(self.validation_loader, desc=f'valid:{name}', leave=False):
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            B          = batch['img'].shape[0]
            images     = batch['img']
            center     = batch['center'].float()
            scale      = batch['scale'].float()
            img_center = batch['img_center'].float()
            img_focal  = batch['img_focal'].float().view(-1)
            img_h      = batch['img_h'].float().view(-1)
            img_w      = batch['img_w'].float().view(-1)

            bbox_info = bbox_info_from_center_scale(center, scale, img_focal, img_center)
            pred_rotmat, pred_betas, pred_cam_crop = self.model(
                images, bbox_info=bbox_info, modality=batch.get('modality', None))
            pred_cam_full = _resolve_pred_cam_full(
                self.model, pred_cam_crop, center, scale, img_focal, img_center)

            pred_output = self.val_smpl_neutral(
                betas=pred_betas,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            joints_h36m = pred_output.joints[:, 25:49, :]          # (B,24,3)

            # RAW 3D joints (recentering deferred to _mpjpe_3d_both_roots)
            pred_j3d[sample_idx:sample_idx + B] = joints_h36m.cpu().numpy()

            # Perspective-project to full-image 2D (matches _project_joints in eval_hmr_2d.py)
            joints_cam = joints_h36m + pred_cam_full.unsqueeze(1)   # (B,24,3)
            f   = img_focal.unsqueeze(1)                            # (B,1)
            cx  = (img_w * 0.5).unsqueeze(1)
            cy  = (img_h * 0.5).unsqueeze(1)
            Z   = joints_cam[:, :, 2].clamp(min=1e-6)
            x2d = f * joints_cam[:, :, 0] / Z + cx                 # (B,24)
            y2d = f * joints_cam[:, :, 1] / Z + cy
            pred_h36m = torch.stack([x2d, y2d], dim=-1)            # (B,24,2)
            pred_coco17_px[sample_idx:sample_idx + B] = (
                pred_h36m[:, SPIN24_TO_COCO17, :].cpu().numpy()
            )
            sample_idx += B

        self.model.train()

        # ── 2D-MPJPE & PCK@0.05: per-joint accumulate ───────────────────────
        per_joint_dist    = np.zeros(17, dtype=np.float64)
        per_joint_total   = np.zeros(17, dtype=np.float64)
        pck_joint_correct = np.zeros(17, dtype=np.float64)
        pck_joint_total   = np.zeros(17, dtype=np.float64)

        for i in range(n_total):
            thresh_px = 0.05 * gt_bbox_diag[i]
            for j in range(17):
                if gt_coco17[i, j, 2] <= 0:
                    continue
                dist = float(np.linalg.norm(gt_coco17[i, j, :2] - pred_coco17_px[i, j]))
                per_joint_dist[j]  += dist
                per_joint_total[j] += 1
                pck_joint_total[j] += 1
                if dist <= thresh_px:
                    pck_joint_correct[j] += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            mpjpe_per_joint = np.where(per_joint_total > 0,
                                        per_joint_dist / per_joint_total, np.nan)
            pck_per_joint   = np.where(pck_joint_total > 0,
                                        pck_joint_correct / pck_joint_total, np.nan)

        mpjpe_2d_px = float(np.nanmean(mpjpe_per_joint))
        pck_mean    = float(np.nanmean(pck_per_joint))

        # ── 3D-MPJPE (both roots) & 3D-PA-MPJPE ─────────────────────────────
        combined_mask = gt_valid_fit  # pred always valid
        n_ok = int(combined_mask.sum())
        mpjpe_3d_mm, mpjpe_3d_ankle_mm, pampjpe_3d_mm = _mpjpe_3d_both_roots(
            gt_j3d, pred_j3d, combined_mask
        )

        print(
            f"[Validation] {name} | 2D-MPJPE={mpjpe_2d_px:.2f}px | "
            f"PCK@0.05={pck_mean*100:.1f}% | "
            f"3D-MPJPE={mpjpe_3d_mm:.1f}mm (pelvis) | "
            f"3D-MPJPE-ankleroot={mpjpe_3d_ankle_mm:.1f}mm (legacy) | "
            f"3D-PA-MPJPE={pampjpe_3d_mm:.1f}mm "
            f"(3D: {n_ok}/{n_total} valid)"
        )
        return {
            '2D-MPJPE':     mpjpe_2d_px,
            'PCK@0.05':     pck_mean,
            # Pelvis-rooted = reporting standard (matches tools/eval_2d_mu.py).
            '3D-MPJPE':     mpjpe_3d_mm,
            # Ankle-rooted = legacy convention used by every run up to
            # stage2c_smplsup_s1; kept so new runs stay comparable to old logs.
            '3D-MPJPE-ankleroot': mpjpe_3d_ankle_mm,
            '3D-PA-MPJPE':  pampjpe_3d_mm,
            'performance':  mpjpe_2d_px,
            'performance_type': 'min',
        }

    @torch.no_grad()
    def _validate_llvip_multimode(self):
        """MU-HMR multi-mode validation on LLVIP-pGT.

        Runs three forward passes per batch with different modality_mask:
          rgb   : mask=[1,0]  -> measures pure-RGB accuracy
          ir    : mask=[0,1]  -> measures pure-IR accuracy
          rgbir : mask=[1,1]  -> measures fused accuracy + gate behavior

        Uses batch['sample_index'] for GT alignment (not sequential counter).
        No modality dropout applied. Gate sanity checks printed for rgb/ir modes.

        Returns a flat dict with keys:
          llvip-pgt/{mode}/2D-MPJPE, llvip-pgt/{mode}/PCK@0.05,
          llvip-pgt/{mode}/3D-MPJPE, llvip-pgt/{mode}/3D-PA-MPJPE,
          llvip-pgt/rgbir/gate_rgb_mean, llvip-pgt/rgbir/gate_ir_mean,
          llvip-pgt/rgbir/gate_rgb_std, llvip-pgt/rgbir/gate_ir_std,
          performance  (= rgbir/2D-MPJPE), performance_type = 'min'
        """
        name = self.validation_dataset_name
        dataset_key = 'llvip'
        print(f"[Validation] step={self.step_count} | {name} MU-HMR multi-mode eval ...")

        # ── Load GT arrays from NPZ ──────────────────────────────────────────
        npz_path = getattr(self, '_thermal_val_npz_paths', {}).get(dataset_key)
        if npz_path is None or not os.path.isfile(npz_path):
            print(f"[Validation] {name}: npz_path not found ({npz_path}), skipping.")
            return None

        raw = np.load(npz_path, allow_pickle=False)
        n_total = int(len(raw['imgname']))

        gt_part      = raw['part'].astype(np.float32)          # (N,24,3)
        gt_coco17    = gt_part[:, SPIN24_TO_COCO17, :]         # (N,17,3)
        gt_bbox      = raw['bbox'].astype(np.float32)          # (N,4) xywh
        gt_bbox_diag = np.sqrt(gt_bbox[:, 2]**2 + gt_bbox[:, 3]**2)
        gt_pose      = raw['pose'].astype(np.float32)          # (N,72)
        gt_shape     = raw['shape'].astype(np.float32)         # (N,10)
        gt_valid_fit = raw['valid_fit'].astype(bool)           # (N,)

        # Compute GT 3D joints from pGT pose/shape (same as _validate_thermal_2d)
        gt_j3d = np.zeros((n_total, 24, 3), dtype=np.float32)
        valid_idx = np.where(gt_valid_fit)[0]
        _batch3d = 128
        for _s in range(0, len(valid_idx), _batch3d):
            _bidx  = valid_idx[_s:_s + _batch3d]
            _pose  = torch.from_numpy(gt_pose[_bidx]).float().to(self.device)
            _shape = torch.from_numpy(gt_shape[_bidx]).float().to(self.device)
            _rot   = batch_rodrigues(_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
            _out   = self.val_smpl_neutral(
                betas=_shape,
                body_pose=_rot[:, 1:],
                global_orient=_rot[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            _j = _out.joints[:, 25:49, :]
            # RAW; both root conventions applied later in _mpjpe_3d_both_roots.
            gt_j3d[_bidx] = _j.cpu().numpy()

        # ── Three-mode inference ─────────────────────────────────────────────
        modes = [
            ('rgb',   torch.tensor([1.0, 0.0])),
            ('ir',    torch.tensor([0.0, 1.0])),
            ('rgbir', torch.tensor([1.0, 1.0])),
        ]
        mode_results = {}

        for mode_name, mask_tmpl in modes:
            pred_coco17_px = np.zeros((n_total, 17, 2), dtype=np.float32)
            pred_j3d_mode  = np.zeros((n_total, 24, 3), dtype=np.float32)
            gate_rgb_batch = []
            gate_ir_batch  = []

            self.model.eval()
            for batch in tqdm(
                self.validation_loader,
                desc=f'valid:{name}/{mode_name}',
                leave=False,
            ):
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                B = batch['img_ir'].shape[0]
                # GT positions: use sample_index for safe scatter (not cumulative counter)
                sample_indices = batch['sample_index'].cpu().numpy()

                center     = batch['center'].float()
                scale      = batch['scale'].float()
                img_center = batch['img_center'].float()
                img_focal  = batch['img_focal'].float().view(-1)
                img_h      = batch['img_h'].float().view(-1)
                img_w      = batch['img_w'].float().view(-1)

                modality_mask = mask_tmpl.unsqueeze(0).expand(B, -1).to(self.device)

                pred_rotmat, pred_betas, pred_cam_crop = self.model(
                    img_rgb=batch['img_rgb'],
                    img_ir=batch['img_ir'],
                    modality_mask=modality_mask,
                )
                pred_cam_full = _resolve_pred_cam_full(
                    self.model, pred_cam_crop, center, scale, img_focal, img_center
                )

                # Gate info (all modes, for sanity checks on rgb/ir)
                gi = self.model.last_gate_info
                if gi is not None:
                    gate_rgb_batch.append(gi['gate_rgb'].cpu())   # [B]
                    gate_ir_batch.append(gi['gate_ir'].cpu())     # [B]

                pred_output = self.val_smpl_neutral(
                    betas=pred_betas,
                    body_pose=pred_rotmat[:, 1:],
                    global_orient=pred_rotmat[:, 0].unsqueeze(1),
                    pose2rot=False,
                )
                joints_h36m = pred_output.joints[:, 25:49, :]    # (B,24,3)

                # RAW 3D joints at sample_index positions (recentering deferred)
                pred_j3d_mode[sample_indices] = joints_h36m.cpu().numpy()

                # Perspective-project to full-image 2D
                joints_cam = joints_h36m + pred_cam_full.unsqueeze(1)
                f   = img_focal.unsqueeze(1)
                cx  = (img_w * 0.5).unsqueeze(1)
                cy  = (img_h * 0.5).unsqueeze(1)
                Z   = joints_cam[:, :, 2].clamp(min=1e-6)
                x2d = f * joints_cam[:, :, 0] / Z + cx
                y2d = f * joints_cam[:, :, 1] / Z + cy
                pred_h36m = torch.stack([x2d, y2d], dim=-1)      # (B,24,2)
                pred_coco17_px[sample_indices] = (
                    pred_h36m[:, SPIN24_TO_COCO17, :].cpu().numpy()
                )

            # ── 2D-MPJPE & PCK@0.05 ─────────────────────────────────────────
            per_joint_dist    = np.zeros(17, dtype=np.float64)
            per_joint_total   = np.zeros(17, dtype=np.float64)
            pck_joint_correct = np.zeros(17, dtype=np.float64)
            pck_joint_total   = np.zeros(17, dtype=np.float64)
            for i in range(n_total):
                thresh_px = 0.05 * gt_bbox_diag[i]
                for j in range(17):
                    if gt_coco17[i, j, 2] <= 0:
                        continue
                    dist = float(np.linalg.norm(gt_coco17[i, j, :2] - pred_coco17_px[i, j]))
                    per_joint_dist[j]    += dist
                    per_joint_total[j]   += 1
                    pck_joint_total[j]   += 1
                    if dist <= thresh_px:
                        pck_joint_correct[j] += 1
            with np.errstate(invalid='ignore', divide='ignore'):
                mpjpe_per_j = np.where(per_joint_total > 0,
                                       per_joint_dist / per_joint_total, np.nan)
                pck_per_j   = np.where(pck_joint_total > 0,
                                       pck_joint_correct / pck_joint_total, np.nan)
            mpjpe_2d_px = float(np.nanmean(mpjpe_per_j))
            pck_mean    = float(np.nanmean(pck_per_j))

            # ── 3D-MPJPE (both roots) & 3D-PA-MPJPE ─────────────────────────
            n_ok = int(gt_valid_fit.sum())
            mpjpe_3d_mm, mpjpe_3d_ankle_mm, pampjpe_3d_mm = _mpjpe_3d_both_roots(
                gt_j3d, pred_j3d_mode, gt_valid_fit
            )

            # ── Gate aggregation ─────────────────────────────────────────────
            gate_info = {}
            if gate_rgb_batch:
                all_rgb = torch.cat(gate_rgb_batch, dim=0)  # (N_total,)
                all_ir  = torch.cat(gate_ir_batch,  dim=0)
                gate_info = {
                    'gate_rgb_mean': float(all_rgb.mean().item()),
                    'gate_ir_mean':  float(all_ir.mean().item()),
                    'gate_rgb_std':  float(all_rgb.std().item()),
                    'gate_ir_std':   float(all_ir.std().item()),
                }

            gate_str = ''
            if gate_info:
                gate_str = (
                    f" | gate_rgb={gate_info['gate_rgb_mean']:.3f}"
                    f"±{gate_info['gate_rgb_std']:.3f}"
                    f" gate_ir={gate_info['gate_ir_mean']:.3f}"
                    f"±{gate_info['gate_ir_std']:.3f}"
                )
            print(
                f"[Validation] {name}/{mode_name} | "
                f"2D-MPJPE={mpjpe_2d_px:.2f}px | PCK@0.05={pck_mean*100:.1f}% | "
                f"3D-MPJPE={mpjpe_3d_mm:.1f}mm (pelvis) | "
                f"3D-MPJPE-ankleroot={mpjpe_3d_ankle_mm:.1f}mm (legacy) | "
                f"3D-PA-MPJPE={pampjpe_3d_mm:.1f}mm"
                f"{gate_str}"
            )
            mode_results[mode_name] = {
                '2D-MPJPE':    mpjpe_2d_px,
                'PCK@0.05':    pck_mean,
                # Pelvis-rooted = reporting standard (matches tools/eval_2d_mu.py
                # default --root_joint hip_mid). Use this one for the paper.
                '3D-MPJPE':    mpjpe_3d_mm,
                # Ankle-rooted = legacy convention logged by every run up to
                # stage2c_smplsup_s1. Compare against historical '3D-MPJPE' columns.
                '3D-MPJPE-ankleroot': mpjpe_3d_ankle_mm,
                '3D-PA-MPJPE': pampjpe_3d_mm,
                **gate_info,
            }

        # ── Gate sanity checks ───────────────────────────────────────────────
        for chk_mode, expect_rgb, expect_ir in [('rgb', '≈1', '≈0'), ('ir', '≈0', '≈1')]:
            gi_chk = mode_results.get(chk_mode, {})
            if 'gate_rgb_mean' in gi_chk:
                print(
                    f"[Validation] Gate check | {chk_mode}-only: "
                    f"gate_rgb={gi_chk['gate_rgb_mean']:.3f} ({expect_rgb}), "
                    f"gate_ir={gi_chk['gate_ir_mean']:.3f} ({expect_ir})"
                )

        # ── Build flat return dict ────────────────────────────────────────────
        flat = {}
        for mode_name, m_res in mode_results.items():
            for k, v in m_res.items():
                if isinstance(v, float):
                    flat[f'llvip-pgt/{mode_name}/{k}'] = v

        # SAVE_BEST metric: honour the configured validation_primary_metric (e.g.
        # llvip-pgt/rgbir/3D-MPJPE for the P1/P1b 3D goal). Fall back to the legacy
        # rgbir/2D-MPJPE only when unset or the configured key is absent this run.
        _prim_key = getattr(self, 'validation_primary_metric', None)
        if _prim_key and _prim_key in flat:
            flat['performance'] = flat[_prim_key]
        else:
            flat['performance'] = flat.get('llvip-pgt/rgbir/2D-MPJPE', float('nan'))
        flat['performance_type'] = 'min'

        self.model.train()
        return flat

    @torch.no_grad()
    def _validate_multi(self):
        """Run all configured validation datasets and return a flat metric dict.

        Output keys are `<dataset>/<metric>` (e.g. `3dpw/MPJPE`, `llvip-pgt/2D-MPJPE`).
        The ``performance`` and ``performance_type`` keys are picked from the
        configured `validation_primary_metric` (else first dataset's primary).
        """
        original_dataset = self.validation_dataset_name
        original_loader = self.validation_loader
        flat = {}
        per_ds = {}

        try:
            for name, loader in self.validation_loaders.items():
                self.validation_dataset_name = name
                self.validation_loader = loader
                if name in ('otp2', 'otp2-pgt', 'llvip', 'llvip-pgt'):
                    if getattr(self, 'is_modality_model', False) and name in ('llvip', 'llvip-pgt'):
                        res = self._validate_llvip_multimode()
                    else:
                        res = self._validate_thermal_2d()
                else:
                    raise ValueError(f"Unsupported validation dataset: '{name}'")
                if res is None:
                    continue
                per_ds[name] = dict(res)
                for k, v in res.items():
                    if k in ('performance', 'performance_type'):
                        continue
                    try:
                        flat[f'{name}/{k}'] = float(v)
                    except Exception:
                        continue
        finally:
            self.validation_dataset_name = original_dataset
            self.validation_loader = original_loader

        # primary metric selection
        primary_key = self.validation_primary_metric
        if primary_key is None and self.validation_dataset_names:
            first_name = self.validation_dataset_names[0]
            first_metric = self.validation_metric_name
            primary_key = f'{first_name}/{first_metric}'
        primary_val = flat.get(primary_key, None)
        if primary_val is None and len(flat) > 0:
            primary_val = next(iter(flat.values()))

        flat['performance'] = float(primary_val) if primary_val is not None else float('nan')
        flat['performance_type'] = 'min'

        print(f"[Validation] step={self.step_count} | multi-dataset summary | "
              f"primary={primary_key}={flat['performance']:.4f}")
        for k in sorted(flat.keys()):
            if k in ('performance', 'performance_type'):
                continue
            print(f"[Validation]   {k} = {flat[k]:.4f}")
        return flat

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d]
        conf = conf[has_pose_3d]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl]
        gt_vertices_with_shape = gt_vertices[has_smpl]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl]
        pred_betas_valid = pred_betas[has_smpl]
        gt_betas_valid = gt_betas[has_smpl]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    # ------------------------------------------------------------------
    # Scheme-1: per-token reliability gate losses (anchor + oracle)
    # ------------------------------------------------------------------
    @staticmethod
    def _oracle_key(nm):
        return str(nm).replace('\\', '/')

    def _load_oracle_gate_targets(self):
        """Lazy-load precomputed oracle gate targets (imgname -> target_ir)."""
        if getattr(self, '_oracle_gate_table_cached', False):
            return self._oracle_gate_table
        self._oracle_gate_table_cached = True
        self._oracle_gate_table = None
        path = getattr(self.options, 'oracle_gate_target_npz', None)
        w = float(getattr(self.options, 'oracle_gate_loss_weight', 0.0) or 0.0)
        if not path or w <= 0.0:
            return None
        if not os.path.isfile(path):
            print(f'[Scheme1][oracle] WARNING: target npz not found: {path} '
                  f'-> oracle loss disabled')
            return None
        data = np.load(path, allow_pickle=True)
        names = [self._oracle_key(n) for n in data['imgname']]
        targets = data['target_ir'].astype(np.float32)
        self._oracle_gate_table = {n: float(t) for n, t in zip(names, targets)}
        print(f'[Scheme1][oracle] loaded {len(self._oracle_gate_table)} '
              f'targets from {path}')
        return self._oracle_gate_table

    def _oracle_weight_scale(self):
        """Linear ramp-up then optional linear decay of the oracle weight."""
        step = int(getattr(self, 'step_count', 0) or 0)
        ramp = int(getattr(self.options, 'oracle_gate_ramp_steps', 0) or 0)
        decay = int(getattr(self.options, 'oracle_gate_decay_steps', 0) or 0)
        scale = 1.0
        if ramp > 0:
            scale *= min(1.0, step / float(ramp))
        if decay > 0:
            scale *= max(0.0, 1.0 - step / float(decay))
        return scale

    def _compute_gate_losses(self, input_batch):
        """Return (gate_loss_tensor or None, logs_dict or None).

        Reads the LIVE per-token gate terms the wrapper stashed after forward.
        Only `both` (RGBIR) rows carry a real gate, so the losses act on those.
        """
        terms = getattr(self.model, 'last_gate_loss_terms', None)
        if not isinstance(terms, dict):
            return None, None
        both = terms.get('both')
        if both is None or int(both.sum()) == 0:
            return None, None
        logs = {}
        total = None

        # ---- gate anchor loss: pull batch-mean alpha_ir toward the anchor ----
        w_anchor = float(getattr(self.options, 'gate_anchor_loss_weight', 0.0) or 0.0)
        if w_anchor > 0.0:
            alpha_mean = terms['alpha_ir_mean_live'][both].mean()
            l_anchor = (alpha_mean - float(getattr(self, '_gate_anchor_ir', 0.70))).pow(2)
            total = w_anchor * l_anchor
            logs['loss_gate_anchor'] = float(l_anchor.detach())

        # ---- oracle gate loss: per-sample BCE on the mean logit ----
        w_oracle = float(getattr(self.options, 'oracle_gate_loss_weight', 0.0) or 0.0)
        table = self._load_oracle_gate_targets()
        if w_oracle > 0.0 and table is not None:
            names = input_batch.get('imgname', None)
            if names is not None:
                both_cpu = both.detach().cpu().numpy()
                logit = terms['alpha_sample_logit']
                idx, tgt = [], []
                for i, nm in enumerate(names):
                    if not bool(both_cpu[i]):
                        continue
                    key = self._oracle_key(nm)
                    if key in table:
                        idx.append(i)
                        tgt.append(table[key])
                if idx:
                    sel = torch.tensor(idx, device=logit.device)
                    target = torch.tensor(tgt, device=logit.device, dtype=logit.dtype)
                    l_oracle = torch.nn.functional.binary_cross_entropy_with_logits(
                        logit[sel], target)
                    scale = self._oracle_weight_scale()
                    contrib = w_oracle * scale * l_oracle
                    total = contrib if total is None else total + contrib
                    logs['loss_oracle_gate'] = float(l_oracle.detach())
                    logs['oracle_gate_scale'] = float(scale)
                    logs['oracle_gate_matched'] = float(len(idx))
        return total, logs

    def _compute_distill_loss(self, pred_rotmat, pred_betas):
        """Scheme-3c thermal-teacher distillation on ir_only rows.

        The teacher (thermal-finetuned HMR2, IR expert) is queried ONLY with IR
        images on ir_only rows; its RGB/RGBIR outputs (broken) are never used.
        Output distillation: student pose(rotmat)/betas -> teacher's, both in the
        same SMPL parameter space (safe, no token-space alignment needed).

        Returns (loss_tensor or None, logs or None).
        """
        teacher = getattr(self, '_teacher', None)
        if teacher is None:
            return None, None
        img_ir = getattr(self, '_last_img_ir', None)
        mm = getattr(self, '_last_modality_mask', None)
        if img_ir is None or mm is None:
            return None, None
        if getattr(self, '_distill_apply_rgbir', False):
            # every row where IR is present: ir_only + rgbir. The teacher is
            # queried on that row's IR image (its RGB/RGBIR paths are broken and
            # never used), so the fusion output is guided toward the IR expert.
            rows = (mm[:, 1] > 0.5)
        else:
            rows = (mm[:, 0] < 0.5) & (mm[:, 1] > 0.5)  # ir_only only
        n = int(rows.sum().item())
        if n == 0:
            return None, None
        sel = rows.nonzero(as_tuple=True)[0]
        ir_img = img_ir[sel]
        if ir_img.shape[1] == 1:
            ir_img = ir_img.repeat(1, 3, 1, 1)
        with torch.no_grad():
            t_rotmat, t_betas, _ = teacher(ir_img)
        s_rotmat = pred_rotmat[sel]
        s_betas = pred_betas[sel]
        total = None
        logs = {}
        if self._distill_pose_w > 0.0:
            l_pose = torch.nn.functional.mse_loss(s_rotmat, t_rotmat)
            total = self._distill_pose_w * l_pose
            logs['loss_distill_pose'] = float(l_pose.detach())
        if self._distill_beta_w > 0.0:
            l_beta = torch.nn.functional.mse_loss(s_betas, t_betas)
            total = (self._distill_beta_w * l_beta if total is None
                     else total + self._distill_beta_w * l_beta)
            logs['loss_distill_beta'] = float(l_beta.detach())
        if total is not None:
            logs['distill_n_ir_only'] = float(n)
        return total, (logs if logs else None)

    def _compute_global_orient_anchor_loss(self, pred_rotmat):
        """P1b: pull the fused/adapted global_orient toward the FROZEN CORE's
        global_orient on IR.

        Diagnosis (2026-07-05, native f=5000): the adapters improve articulated
        pose (LLVIP PA-MPJPE 68-70 beats HMR2.0 71.2) but DEGRADE global
        orientation (~12mm), so 3D-MPJPE loses to plain HMR2.0 (254/262 vs 249.5).
        The frozen core alone (legacy path, no adapters) has the best global
        orient (baseline_ir 249.5). This anchor self-distills ONLY joint-0
        (global_orient) toward that frozen-core prediction on IR-bearing rows,
        recovering its global orient while the pGT pose loss keeps the adapters'
        articulated-pose gains (joints 1-23). No teacher file needed — the target
        is the model's own frozen core (bit-exact original HMR2.0).

        Returns (loss_tensor or None, logs or None).
        """
        w = float(getattr(self.options, 'global_orient_anchor_weight', 0.0) or 0.0)
        if w <= 0.0:
            return None, None
        img_ir = getattr(self, '_last_img_ir', None)
        mm = getattr(self, '_last_modality_mask', None)
        if img_ir is None or mm is None:
            return None, None
        ir_rows = (mm[:, 1] > 0.5)  # rgbir + ir_only (IR present)
        n = int(ir_rows.sum().item())
        if n == 0:
            return None, None
        ir3 = img_ir if img_ir.shape[1] == 3 else img_ir.repeat(1, 3, 1, 1)
        with torch.no_grad():
            # legacy single-image path = pure frozen core (no unifier/adapters),
            # frozen ViT runs in eval mode inside forward_from_tokens -> deterministic.
            core_rotmat, _, _ = self.model(x=ir3)
            target_go = core_rotmat[:, 0]           # (B,3,3)
        student_go = pred_rotmat[:, 0]              # (B,3,3)
        l_go = ((student_go[ir_rows] - target_go[ir_rows]) ** 2).mean()
        logs = {'loss_global_orient_anchor': float(l_go.detach()),
                'global_orient_anchor_n': float(n)}
        return w * l_go, logs

    def _compute_head_l2sp_loss(self):
        """L2-SP: penalise the (unfrozen) SMPL head drifting from its pretrained
        weights. Directly counters the diagnosed pose drift (2026-07-10) that
        breaks validation when FREEZE_HEAD=false. No-op if the anchor is unset
        (head frozen or weight 0). Uses SUM of squared diffs (like train_thermal
        L2-SP); keep the weight small (~1e-3..1e-4)."""
        anchor = getattr(self, '_head_l2sp_anchor', None)
        if anchor is None:
            return None, None
        params = dict(self.model.named_parameters())
        l2sp = None
        for name, ref in anchor.items():
            p = params.get(name)
            if p is None or not p.requires_grad:
                continue
            term = ((p - ref) ** 2).sum()
            l2sp = term if l2sp is None else l2sp + term
        if l2sp is None:
            return None, None
        logs = {'loss_head_l2sp': float(l2sp.detach())}
        return self._head_l2sp_weight * l2sp, logs

    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img']
        gt_keypoints_2d = input_batch['keypoints']
        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']
        gt_joints = input_batch['pose_3d']
        center = input_batch['center'].to(self.device).float()
        scale = input_batch['scale'].to(self.device).float()
        img_center = input_batch['img_center'].to(self.device).float()
        img_focal = input_batch['img_focal'].to(self.device).float()

        has_smpl = input_batch['has_smpl'].to(self.device).bool()
        has_pose_3d = input_batch['has_pose_3d'].to(self.device).bool()

        is_flipped = input_batch['is_flipped'].to(self.device).bool()
        rot_angle = input_batch['rot_angle'].to(self.device).float()
        dataset_name = input_batch['dataset_name']
        indices = input_batch['sample_index']
        batch_size = images.shape[0]

        # [M4-has-cam-intrinsics] gating mask for full-camera supervision
        has_cam_intrinsics = input_batch['has_cam_intrinsics'].to(self.device).bool()
        any_cam = bool(has_cam_intrinsics.any().item())

        # Scheme A: thermal mask + sample weights
        thermal_mask = self._thermal_sample_mask(dataset_name, batch_size)
        use_pgt_weight = bool(getattr(self.options, 'use_pgt_weight', True))

        smpl_sample_weight = self._batch_float_or_default(input_batch, 'smpl_weight', batch_size, 1.0)
        pose_sample_weight = self._batch_float_or_default(input_batch, 'pose_weight', batch_size, 1.0)
        shape_sample_weight = self._batch_float_or_default(input_batch, 'shape_weight', batch_size, 1.0)
        cam_sample_weight = self._batch_float_or_default(input_batch, 'cam_weight', batch_size, 1.0)

        if not use_pgt_weight:
            smpl_sample_weight = torch.ones_like(smpl_sample_weight)
            pose_sample_weight = torch.ones_like(pose_sample_weight)
            shape_sample_weight = torch.ones_like(shape_sample_weight)
            cam_sample_weight = torch.ones_like(cam_sample_weight)

        pose_loss_sample_weight = (smpl_sample_weight * pose_sample_weight).clone()
        beta_loss_sample_weight = (smpl_sample_weight * shape_sample_weight).clone()
        shape_loss_sample_weight = (smpl_sample_weight * shape_sample_weight).clone()
        cam_loss_sample_weight = (smpl_sample_weight * cam_sample_weight).clone()

        # NOTE: 這裡曾經有第二道閥門
        #   pose_loss_sample_weight[thermal_mask] *= thermal_smpl_global_weight
        #   beta/shape_loss_sample_weight[thermal_mask] *= thermal_shape_global_weight
        #   cam_loss_sample_weight[thermal_mask]  *= thermal_cam_global_weight
        # 它讓 LOSS.*_LOSS_WEIGHT 可以被靜默乘成 0（config dump 卻顯示權重是開的）。
        # 已改為單一旋鈕：LOSS.*_LOSS_WEIGHT 就是唯一控制項，舊的
        # LOSS.THERMAL_*_GLOBAL_WEIGHT 會在
        # TrainOptions._fold_legacy_thermal_global_weights() 於 parse 階段被摺進
        # 對應的 *_LOSS_WEIGHT（`_thermal_sample_mask` 對本 repo 所有訓練資料集
        # 恆為全 True，故摺疊與原乘法逐位元等價）。
        # 逐樣本的 pGT 品質加權（smpl_weight / pose_weight / shape_weight / cam_weight）
        # 仍然生效，由 LOSS.USE_PGT_WEIGHT 控制。

        # Get GT joints (neutral SMPL, used for camera estimation)
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        # Build gendered GT vertices for supervision (3DPW samples use m/f; others neutral)
        gender = input_batch['gender'].to(self.device)
        gt_vertices = self._build_gendered_gt_vertices(gt_pose, gt_betas, gender)

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices, rot_angle, is_flipped)]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        # [safe-guard] fits_dict 理論上可能存入含 NaN 的 SMPLify 輸出，
        # 餵進 SMPL forward 會產生 NaN vertices/joints 進而污染 loss。
        # 這裡只做檢測與重置，不動 fits_dict 本身。
        # 必須在 |β|>3 sanitize 之前，因為 NaN 在比較運算中是 False，會漏過 clamp。
        _nan_pose_mask = torch.isnan(opt_pose).any(dim=-1)
        _nan_betas_mask = torch.isnan(opt_betas).any(dim=-1)
        _nan_mask = _nan_pose_mask | _nan_betas_mask
        if _nan_mask.any():
            _n_nan = int(_nan_mask.sum().item())
            print(f"[safe-guard] detected {_n_nan} NaN rows in fits_dict load; "
                  f"resetting to zero for this step (fits_dict unchanged).")
            opt_pose[_nan_mask] = 0.
            opt_betas[_nan_mask] = 0.
        # [Finding-C fix / 方案3] 讀回 fits_dict 的 extreme betas 會污染
        # 後續 opt_joint_loss baseline，讓 improvement gate 過度樂觀。
        # 在這裡先 sanitize，下游 L721 的 clamp 保留作為 second-line defense。
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints

        # [geometry-check] De-normalize 2D keypoints from crop-space [-1,1] back
        # to crop-pixel space (0..img_res). These coords still have rot/flip
        # augmentation baked in from BaseDataset.j2d_processing; downstream
        # consumers (estimate_translation_fullimg, SMPLify) receive rot_angle
        # and is_flipped to undo the augmentation into full-image space.
        # Naming kept as `_orig` for compatibility: "orig" = pre-normalization
        # pixel space, NOT pre-augmentation.
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # [M4-has-cam-intrinsics] 只對有真實 intrinsics 的 sample 計算 full-image translation
        gt_cam_full = torch.zeros(batch_size, 3, device=self.device)
        opt_cam_full = torch.zeros(batch_size, 3, device=self.device)
        if any_cam:
            _ci = has_cam_intrinsics
            gt_cam_full[_ci] = estimate_translation_fullimg(
                gt_model_joints[_ci], gt_keypoints_2d_orig[_ci],
                center=center[_ci], scale=scale[_ci],
                img_focal=img_focal[_ci], img_center=img_center[_ci],
                crop_res=self.options.img_res,
                rot_angle=rot_angle[_ci], is_flipped=is_flipped[_ci],
            )
            opt_cam_full[_ci] = estimate_translation_fullimg(
                opt_joints[_ci], gt_keypoints_2d_orig[_ci],
                center=center[_ci], scale=scale[_ci],
                img_focal=img_focal[_ci], img_center=img_center[_ci],
                crop_res=self.options.img_res,
                rot_angle=rot_angle[_ci], is_flipped=is_flipped[_ci],
            )

        # [Task 3 + 6] Fallback: has_smpl rows without real intrinsics use pseudo focal.
        # gt_cam_full feeds cam_full_pair_loss; opt_cam_full feeds get_fitting_loss + SMPLify init.
        smpl_no_cam = has_smpl & ~has_cam_intrinsics
        if smpl_no_cam.any():
            gt_cam_full[smpl_no_cam] = estimate_translation_fullimg(
                gt_model_joints[smpl_no_cam], gt_keypoints_2d_orig[smpl_no_cam],
                center=center[smpl_no_cam], scale=scale[smpl_no_cam],
                img_focal=img_focal[smpl_no_cam], img_center=img_center[smpl_no_cam],
                crop_res=self.options.img_res,
                rot_angle=rot_angle[smpl_no_cam], is_flipped=is_flipped[smpl_no_cam],
            )
            # [Task 6] sync opt_cam_full so get_fitting_loss / SMPLify see a usable init
            opt_cam_full[smpl_no_cam] = estimate_translation_fullimg(
                opt_joints[smpl_no_cam], gt_keypoints_2d_orig[smpl_no_cam],
                center=center[smpl_no_cam], scale=scale[smpl_no_cam],
                img_focal=img_focal[smpl_no_cam], img_center=img_center[smpl_no_cam],
                crop_res=self.options.img_res,
                rot_angle=rot_angle[smpl_no_cam], is_flipped=is_flipped[smpl_no_cam],
            )

        bbox_info = bbox_info_from_center_scale(center, scale, img_focal, img_center)
        # [Task 6] 對 has_cam_intrinsics | has_smpl 都算 fitting loss（pseudo focal 也可用）。
        # Rows with neither stay at +inf → valid_fit=False。
        opt_joint_loss = torch.full((batch_size,), float('inf'), device=self.device)
        fit_loss_gate = has_cam_intrinsics | has_smpl
        if fit_loss_gate.any():
            _g = fit_loss_gate
            _opt_loss = self.smplify.get_fitting_loss(
                opt_pose[_g], opt_betas[_g], opt_cam_full[_g],
                center[_g], scale[_g], img_focal[_g], img_center[_g],
                gt_keypoints_2d_orig[_g],
                rot_angle=rot_angle[_g], is_flipped=is_flipped[_g],
            ).mean(dim=-1)
            opt_joint_loss[_g] = _opt_loss

        # Model output meaning depends on camera_rep:
        #   weak_perspective  -> (s, tx, ty) in crop space; cam_crop2full converts.
        #   full_perspective  -> (tx, ty, tz) in camera frame; used as-is.
        # _resolve_pred_cam_full dispatches on self.model.camera_rep.
        # MU gate scalars — initialised here so they're always in losses dict.
        _mu_gate_rgb_mean = 0.0
        _mu_gate_ir_mean = 0.0
        _mu_n_rgb_only = 0.0
        _mu_n_ir_only = 0.0
        _mu_n_rgbir = 0.0
        if getattr(self, 'is_modality_model', False):
            # MU-HMR (late) / PBF-HMR2 (pre-backbone) share this path: both
            # wrappers take (img_rgb, img_ir, modality_mask) and return the SAME
            # 3-tuple, so everything downstream (losses, cam_full, projection) is
            # unchanged. Hard-fail instead of silently falling back to the legacy
            # single-image path if the paired batch is not wired.
            _required = ('img_rgb', 'img_ir', 'modality_mask')
            _missing = [k for k in _required if k not in input_batch]
            if _missing:
                raise RuntimeError(
                    f"[MU-HMR] training requires paired batch keys {_required}; "
                    f"missing={_missing}. LLVIPPairedDataset is probably not enabled "
                    f"for dataset(s) in this batch."
                )
            from models.multimodal import apply_modality_dropout
            img_rgb = input_batch.get('img_rgb', None)
            img_ir = input_batch.get('img_ir', None)
            modality_mask = input_batch.get('modality_mask', None)
            if modality_mask is not None:
                modality_mask = modality_mask.to(self.device).float()
                if getattr(self, '_mu_dropout_enabled', False):
                    modality_mask = apply_modality_dropout(
                        modality_mask, self._mu_dropout_probs)
            pred_rotmat, pred_betas, pred_cam_crop = self.model(
                img_rgb=img_rgb, img_ir=img_ir, modality_mask=modality_mask,
            )
            # Scheme-3c: stash IR image + mask so the teacher-distill loss (computed
            # after mix_losses, before backward) can query the teacher on ir_only rows.
            self._last_img_ir = img_ir
            self._last_modality_mask = modality_mask
            # Extract gate scalars for losses dict + TensorBoard.
            gi = self.model.last_gate_info
            if gi is not None:
                _mu_gate_rgb_mean = float(gi['gate_rgb'].mean().item())
                _mu_gate_ir_mean = float(gi['gate_ir'].mean().item())
                if modality_mask is not None:
                    _mm = modality_mask.detach()
                    _mu_n_rgb_only = float(((_mm[:, 0] > 0.5) & (_mm[:, 1] < 0.5)).sum().item())
                    _mu_n_ir_only = float(((_mm[:, 0] < 0.5) & (_mm[:, 1] > 0.5)).sum().item())
                    _mu_n_rgbir = float(((_mm[:, 0] > 0.5) & (_mm[:, 1] > 0.5)).sum().item())
                if getattr(self, 'summary_writer', None) is not None:
                    _ss = max(1, int(getattr(self.options, 'summary_steps', 100)))
                    if int(self.step_count) % _ss == 0:
                        self.summary_writer.add_scalar('gate/rgb_mean', _mu_gate_rgb_mean, self.step_count)
                        self.summary_writer.add_scalar('gate/ir_mean', _mu_gate_ir_mean, self.step_count)
        else:
            pred_rotmat, pred_betas, pred_cam_crop = self.model(
                images,
                bbox_info=bbox_info,
                modality=input_batch.get('modality', None),
            )
        pred_cam_full = _resolve_pred_cam_full(
            self.model, pred_cam_crop, center, scale, img_focal, img_center)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Project prediction to crop space via full-image camera
        _, pred_keypoints_2d_crop, _ = project_points_full_to_crop(
            pred_joints, pred_cam_full,
            center=center, scale=scale, img_focal=img_focal, img_center=img_center,
            crop_res=self.options.img_res,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            rot_angle=rot_angle, is_flipped=is_flipped,
        )
        pred_keypoints_2d = pred_keypoints_2d_crop / (self.options.img_res / 2.0) - 1.0

        if self.options.run_smplify:
            pred_rotmat_3x3 = pred_rotmat.detach().reshape(-1, 3, 3)
            pred_pose = _rotation_matrix_to_angle_axis_cv2(pred_rotmat_3x3).contiguous().view(batch_size, -1)
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # [Task 6] SPIN-style: run SMPLify on the full batch (pseudo focal handles ITW).
            new_opt_vertices, new_opt_joints, \
            new_opt_pose, new_opt_betas, \
            new_opt_cam_full, new_opt_joint_loss = self.smplify(
                pred_pose.detach(), pred_betas.detach(),
                pred_cam_full.detach(),
                center, scale, img_focal, img_center,
                gt_keypoints_2d_orig,
                rot_angle=rot_angle, is_flipped=is_flipped,
            )
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            update = (new_opt_joint_loss < opt_joint_loss)

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update] = new_opt_vertices[update]
            opt_joints[update] = new_opt_joints[update]
            opt_pose[update] = new_opt_pose[update]
            opt_betas[update] = new_opt_betas[update]
            opt_cam_full[update] = new_opt_cam_full[update]

            self.fits_dict[(dataset_name, indices, rot_angle,
                            is_flipped, update)] = \
                (opt_pose, opt_betas)
        else:
            update = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with GT where available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_full[has_smpl, :] = gt_cam_full[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # [Task 6] SPIN-style validate fit quality (two-line original logic)
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        valid_fit = valid_fit | has_smpl

        # Project opt joints to crop space via full-image camera
        _, opt_keypoints_2d_crop, _ = project_points_full_to_crop(
            opt_joints, opt_cam_full,
            center=center, scale=scale, img_focal=img_focal, img_center=img_center,
            crop_res=self.options.img_res,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            rot_angle=rot_angle, is_flipped=is_flipped,
        )
        opt_keypoints_2d = opt_keypoints_2d_crop / (self.options.img_res / 2.0) - 1.0

        # Supervision source switch
        if self.supervision_mode == 'gt_only':
            pose_beta_target_pose = gt_pose
            pose_beta_target_betas = gt_betas
            pose_beta_valid_mask = has_smpl
        else:
            pose_beta_target_pose = opt_pose
            pose_beta_target_betas = opt_betas
            pose_beta_valid_mask = valid_fit

        # ---- raw losses ----
        loss_keypoints_2d = self.loss_mixer.keypoint_2d_loss(
            pred_keypoints_2d, gt_keypoints_2d,
            openpose_weight=self.options.openpose_train_weight,
            gt_weight=self.options.gt_train_weight,
        )

        # Plan 06.28 §10.1: crop-pixel 2D loss directly in eval coordinate
        # system (SmoothL1, beta=5 px). gt_keypoints_2d is in [-1, 1].
        kp2d_pix_weight = float(getattr(self.options, 'keypoint_2d_pixel_loss_weight', 0.0) or 0.0)
        if kp2d_pix_weight > 0.0:
            gt_kp_crop_pix = (gt_keypoints_2d[:, :, :2] + 1.0) * (self.options.img_res / 2.0)
            gt_kp_conf = gt_keypoints_2d[:, :, 2]
            loss_keypoints_2d_pixel = self.loss_mixer.keypoint_2d_pixel_loss(
                pred_crop_pix=pred_keypoints_2d_crop,
                gt_crop_pix=gt_kp_crop_pix,
                gt_conf=gt_kp_conf,
                openpose_weight=self.options.openpose_train_weight,
                gt_weight=self.options.gt_train_weight,
                beta=5.0,
            )
        else:
            loss_keypoints_2d_pixel = torch.zeros((), device=self.device)
        loss_keypoints_3d = self.loss_mixer.keypoint_3d_loss(
            pred_joints, gt_joints, has_pose_3d,
        )
        loss_regr_pose, loss_regr_betas, beta_filter_ratio = self.loss_mixer.smpl_pose_beta_loss(
            pred_rotmat, pred_betas,
            pose_beta_target_pose, pose_beta_target_betas, pose_beta_valid_mask,
            beta_loss_filter_enabled=bool(getattr(self.options, 'beta_loss_filter_enabled', False)),
            beta_loss_threshold=float(getattr(self.options, 'beta_loss_threshold', 2.0)),
            pose_weight=pose_loss_sample_weight,
            beta_weight=beta_loss_sample_weight,
        )
        loss_shape = self.loss_mixer.vertex_loss(
            pred_vertices=pred_vertices,
            supervision_mode=self.supervision_mode,
            has_smpl=has_smpl, valid_fit=valid_fit,
            smpl_model=self.smpl, gt_pose=gt_pose, gt_betas=gt_betas,
            opt_vertices=opt_vertices,
            gt_vertices=gt_vertices,
            sample_weight=shape_loss_sample_weight,
        )
        loss_depth = self.loss_mixer.cam_depth_loss(pred_cam_full, mask=has_cam_intrinsics)  # [M4-has-cam-intrinsics]
        loss_cam = self.loss_mixer.cam_loss(
            pred_camera=pred_cam_full,
            cam=input_batch.get('cam', None),
            has_cam=input_batch.get('has_cam', None),
            mask=has_cam_intrinsics,  # [M4-has-cam-intrinsics]
        )

        # [Task 2] CLIFF-style cam_full paired supervision.
        # opt_cam_full already has gt_cam_full merged in at L754 for has_smpl rows.
        cam_sup_mask = has_smpl | (valid_fit & has_cam_intrinsics)
        loss_cam_full = self.loss_mixer.cam_full_pair_loss(
            pred_cam_full=pred_cam_full,
            target_cam_full=opt_cam_full,
            mask=cam_sup_mask,
            sample_weight=cam_loss_sample_weight,
        )

        # Plan 06.28: optional warmup ramp for the cam_full loss so the
        # backbone has a few steps to stabilize before the strong projection
        # penalty kicks in (mirrors HMR2.0 / CLIFF early training tricks).
        from train.loss_mixer import compute_cam_full_ramp
        cam_full_scale = compute_cam_full_ramp(
            self.step_count,
            start_step=int(getattr(self.options, 'cam_full_loss_start_step', 0) or 0),
            ramp_steps=int(getattr(self.options, 'cam_full_loss_ramp_steps', 0) or 0),
        )
        loss_cam_full_ramped = loss_cam_full * float(cam_full_scale)

        raw_losses = {
            'loss_keypoints_2d': loss_keypoints_2d,
            'loss_keypoints_2d_pixel': loss_keypoints_2d_pixel,
            'loss_keypoints_3d': loss_keypoints_3d,
            'loss_regr_pose': loss_regr_pose,
            'loss_regr_betas': loss_regr_betas,
            'loss_shape': loss_shape,
            'loss_depth': loss_depth,
            'loss_cam': loss_cam,
            'loss_cam_full': loss_cam_full_ramped,
        }
        loss, weighted_losses = self.loss_mixer.mix_losses(raw_losses, self.loss_weights)

        # Scheme-1 gate losses (MMPed per-token reliability fusion): gate anchor
        # (pull batch-mean alpha_ir toward the anchor -> tames the step-3000
        # rebound) + oracle gate BCE (per-sample supervision from precomputed
        # RGB/IR reliability -> the axis that indirect gradient starves). No-op
        # unless the model exposes last_gate_loss_terms (per-token mode).
        gate_loss, gate_logs = self._compute_gate_losses(input_batch)
        if gate_loss is not None:
            loss = loss + gate_loss

        # Scheme-3c: thermal-teacher distillation on ir_only rows (SMPL-param
        # output distillation). No-op unless a teacher is loaded and the batch
        # has ir_only rows. Gate stays frozen (Scheme-3b), so this only pushes
        # the IR path via body/IR adapters, never the RGBIR fusion balance.
        distill_loss, distill_logs = self._compute_distill_loss(pred_rotmat, pred_betas)
        if distill_loss is not None:
            loss = loss + distill_loss

        # P1b: global-orient anchor toward the frozen core on IR (recovers the
        # ~12mm global-orientation degradation the adapters introduce, so 3D-MPJPE
        # can beat plain HMR2.0 while keeping the adapters' articulated-pose gains).
        go_anchor_loss, go_anchor_logs = self._compute_global_orient_anchor_loss(pred_rotmat)
        if go_anchor_loss is not None:
            loss = loss + go_anchor_loss

        # B1 fix: L2-SP anchor keeps the unfrozen SMPL head near its pretrained
        # weights -> stops the pose drift diagnosed as the cause of the
        # 'train-stable / val-broke' failure. No-op unless HEAD_L2SP_WEIGHT>0.
        head_l2sp_loss, head_l2sp_logs = self._compute_head_l2sp_loss()
        if head_l2sp_loss is not None:
            loss = loss + head_l2sp_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float(getattr(self.options, 'grad_clip_norm', 5.0)),
        )
        self.optimizer.step()

        # Convert full-image camera to crop-space perspective for renderer
        pred_cam_vis = cam_full2crop_perspective(
            pred_cam_full.detach(), center, scale, img_focal, img_center,
            focal_length_crop=self.focal_length, crop_res=self.options.img_res,
        )
        opt_cam_vis = cam_full2crop_perspective(
            opt_cam_full.detach(), center, scale, img_focal, img_center,
            focal_length_crop=self.focal_length, crop_res=self.options.img_res,
        )

        output = {
            'pred_vertices': pred_vertices.detach(),
            'opt_vertices': opt_vertices,
            'pred_cam_t': pred_cam_vis.detach(),
            'opt_cam_t': opt_cam_vis.detach(),
            'pred_cam_full': pred_cam_full.detach(),
            'opt_cam_full': opt_cam_full.detach(),
        }
        # Batch all 0-dim scalars into one GPU->CPU sync instead of 19 separate .item() calls.
        _scalar_keys = [
            'loss',
            'loss_keypoints_2d',
            'loss_keypoints_3d',
            'loss_regr_pose',
            'loss_regr_betas',
            'loss_shape',
            'loss_depth',
            'loss_cam',
            'weighted_loss_keypoints_2d',
            'weighted_loss_keypoints_3d',
            'weighted_loss_regr_pose',
            'weighted_loss_regr_betas',
            'weighted_loss_shape',
            'weighted_loss_depth',
            'weighted_loss_cam',
            'valid_fit_ratio',
            'update_ratio',
            'opt_joint_loss_mean',
            'num_has_cam_intrinsics',  # [M4-has-cam-intrinsics]
            'beta_filter_ratio',
            'loss_cam_full',
            'weighted_loss_cam_full',
            'cam_sup_ratio',
            # Scheme A diagnostics
            'num_thermal_samples',
            'pose_loss_weight_mean',
            'beta_loss_weight_mean',
            'shape_loss_weight_mean',
            'cam_loss_weight_mean',
            # Step 6: full-perspective camera-translation health metrics.
            # Computed from pred_cam_full so the same scalars are meaningful
            # whether camera_rep is weak_perspective (after cam_crop2full) or
            # full_perspective (head emits tx/ty/tz directly).
            'pred_cam_t_x_mean',
            'pred_cam_t_y_mean',
            'pred_cam_t_z_mean',
            'pred_cam_t_z_min',
            'pred_cam_t_z_median',
            'pred_cam_t_z_max',
            # Plan 06.28 telemetry
            'loss_keypoints_2d_pixel',
            'weighted_loss_keypoints_2d_pixel',
            'cam_full_ramp',
            'weak_scale_clamp_ratio',
            'camera_outlier_ratio',
            'focal_source_real_ratio',
            'focal_source_fixed_ratio',
            'focal_source_diagonal_ratio',
        ]
        _scalar_tensors = [
            loss.detach(),
            loss_keypoints_2d.detach(),
            loss_keypoints_3d.detach(),
            loss_regr_pose.detach(),
            loss_regr_betas.detach(),
            loss_shape.detach(),
            loss_depth.detach(),
            loss_cam.detach(),
            weighted_losses['weighted_loss_keypoints_2d'].detach(),
            weighted_losses['weighted_loss_keypoints_3d'].detach(),
            weighted_losses['weighted_loss_regr_pose'].detach(),
            weighted_losses['weighted_loss_regr_betas'].detach(),
            weighted_losses['weighted_loss_shape'].detach(),
            weighted_losses['weighted_loss_depth'].detach(),
            weighted_losses['weighted_loss_cam'].detach(),
            valid_fit.float().mean(),
            update.float().mean(),
            opt_joint_loss.mean(),
            has_cam_intrinsics.sum().float(),
            beta_filter_ratio.detach(),
            loss_cam_full.detach(),
            weighted_losses['weighted_loss_cam_full'].detach(),
            cam_sup_mask.float().mean(),
            # Scheme A diagnostics
            thermal_mask.sum().float(),
            pose_loss_sample_weight.mean(),
            beta_loss_sample_weight.mean(),
            shape_loss_sample_weight.mean(),
            cam_loss_sample_weight.mean(),
            # Step 6: pred_cam_full [B,3] in metres in camera frame.
            pred_cam_full[:, 0].detach().mean(),
            pred_cam_full[:, 1].detach().mean(),
            pred_cam_full[:, 2].detach().mean(),
            pred_cam_full[:, 2].detach().min(),
            pred_cam_full[:, 2].detach().median(),
            pred_cam_full[:, 2].detach().max(),
        ]
        # Plan 06.28 telemetry scalars (computed at end, appended in matched order).
        _kp2d_pix_w = weighted_losses.get('weighted_loss_keypoints_2d_pixel', None)
        if _kp2d_pix_w is None:
            _kp2d_pix_w = torch.zeros((), device=self.device)
        _scalar_tensors += [
            loss_keypoints_2d_pixel.detach(),
            _kp2d_pix_w.detach(),
            torch.tensor(float(cam_full_scale), device=self.device),
        ]
        # weak-scale clamp ratio: fraction of samples saturating the clamp range.
        try:
            from utils.geometry import clamp_weak_scale, compute_camera_outlier_ratio
            _s_min = float(getattr(self.options, 'weak_cam_scale_min', 0.05))
            _s_max = float(getattr(self.options, 'weak_cam_scale_max', 10.0))
            _raw_s = pred_cam_crop[:, 0].detach() if pred_cam_crop is not None else None
            if _raw_s is not None:
                _, _clamp_mask = clamp_weak_scale(_raw_s, _s_min, _s_max)
                _clamp_ratio = _clamp_mask.float().mean()
            else:
                _clamp_ratio = torch.zeros((), device=self.device)
            _cam_outlier = torch.tensor(
                compute_camera_outlier_ratio(pred_cam_full[:, 2].detach()),
                device=self.device,
            )
        except Exception:
            _clamp_ratio = torch.zeros((), device=self.device)
            _cam_outlier = torch.zeros((), device=self.device)
        _scalar_tensors += [_clamp_ratio, _cam_outlier]
        # focal source breakdown: was the per-sample focal real, fixed, or diagonal?
        _src = input_batch.get('focal_source', None)
        if _src is not None and torch.is_tensor(_src):
            _src_dev = _src.to(self.device)
            _src_real = (_src_dev == 0).float().mean()
            _src_fixed = (_src_dev == 1).float().mean()
            _src_diag = (_src_dev == 2).float().mean()
        else:
            # fallback: derive from has_cam_intrinsics for backwards compat
            _src_real = has_cam_intrinsics.float().mean()
            _src_fixed = torch.zeros((), device=self.device)
            _src_diag = (~has_cam_intrinsics).float().mean()
        _scalar_tensors += [_src_real, _src_fixed, _src_diag]
        _scalar_vals = torch.stack(_scalar_tensors).cpu().tolist()
        losses = {k: float(v) for k, v in zip(_scalar_keys, _scalar_vals)}

        # MU-HMR gate scalars (non-tensor; always present, 0.0 for non-MU runs).
        losses['gate_rgb_mean'] = _mu_gate_rgb_mean
        losses['gate_ir_mean'] = _mu_gate_ir_mean
        losses['num_rgb_only_samples'] = _mu_n_rgb_only
        losses['num_ir_only_samples'] = _mu_n_ir_only
        losses['num_rgbir_samples'] = _mu_n_rgbir

        # Stage2C diagnostics: surface bounded-scale + confidence stats so slow
        # engagement (e.g. an IR adapter stuck at its tiny init scale) is visible.
        _gi = getattr(self.model, 'last_gate_info', None)
        _aux = getattr(self.model, 'last_aux_info', None)
        # xmodal_scale / xmodal_delta_l2 (2026-09-11)：逐 token 跨模態殘差的
        # 有效尺度與 delta 量級。修正前的 MAA residual 就是卡在 raw_scale=-8 的
        # 死迴圈（scale 3.4e-7，對輸出貢獻 5.8e-8 < float32 eps），這兩欄就是
        # 用來在訓練中即時看出新模組有沒有真的 engage。
        for _src, _keys in ((_gi, ('maa_residual_scale', 'conf_ir_std',
                                    'alpha_ir_mean', 'alpha_ir_std', 'alpha_ir_token_std',
                                    'xmodal_scale', 'xmodal_delta_l2',
                                    # 2026-09-21: 裁剪前的 raw RMS。bound_mode
                                    # 'rms_clip' 下，>1 代表正在被壓回、<1 代表
                                    # 還在恆等區——用來判斷 MAX_SCALE 是否又訂太低。
                                    'xmodal_raw_rms',
                                    # 2026-10-02: pair 間 alpha 差異與 CMT->ARG 調節量
                                    'alpha_ir_pair_std', 'cmt_pair_abs', 'cmt_token_abs')),
                            (_aux, ('ir_adapter_scale', 'ir_adapter_delta_l2',
                                    # 2026-09-22 Tier-1 global-orient residual
                                    'go_residual_scale', 'go_residual_delta_l2',
                                    'go_residual_angle_deg',
                                    # 2026-10-03 深層跨模態 adapter 的殘差量級
                                    'xmodal_deep_delta_rms_mean', 'xmodal_deep_delta_rms_max',
                                    'xmodal_deep_beta_mean', 'xmodal_deep_beta_std'))):
            if isinstance(_src, dict):
                for _k in _keys:
                    _v = _src.get(_k, None)
                    if torch.is_tensor(_v):
                        losses[_k] = float(_v.detach().float().mean().item())

        # Scheme-2: body-adapter effective scales (max_scale * sigmoid(raw_scale)).
        # ResidualBodyBlockAdapter.forward returns a bare tensor (no aux channel),
        # so read the modules directly — effective_scale is a pure param function.
        _ba = getattr(self.model, 'body_adapters', None)
        if _ba is not None and len(_ba) > 0:
            _bscales = [float(a.max_scale * torch.sigmoid(a.raw_scale).item()) for a in _ba]
            losses['body_adapter_scale_mean'] = sum(_bscales) / len(_bscales)
            losses['body_adapter_scale_max'] = max(_bscales)
            losses['body_adapter_scale_min'] = min(_bscales)

        # Scheme-1 gate loss scalars (anchor / oracle) for the CSV.
        if isinstance(gate_logs, dict):
            losses.update(gate_logs)
        # Scheme-3c distill loss scalars for the CSV.
        if isinstance(distill_logs, dict):
            losses.update(distill_logs)
        if isinstance(go_anchor_logs, dict):
            losses.update(go_anchor_logs)
        if isinstance(head_l2sp_logs, dict):
            losses.update(head_l2sp_logs)

        return output, losses

    def train_summaries(self, input_batch, output, losses):
        if self.renderer is None:
            for loss_name, val in losses.items():
                self.summary_writer.add_scalar(loss_name, val, self.step_count)
            return

        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

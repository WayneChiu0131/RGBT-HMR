from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
import joblib
import os
from os.path import exists

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from utils.geometry import estimate_focal_length
from .coco_occlusion import occlude_with_pascal_objects
from .extreme_crop import apply_extreme_crop


def _opt_float(options, name, default=0.0):
    """Get a float option from options, treating None as default."""
    v = getattr(options, name, default)
    return float(v) if v is not None else float(default)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False,
                 ignore_pose_3d=None, ignore_smpl=None,  # [M5-ignore-split]
                 use_augmentation=True, is_train=True, npz_path_override=None):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        # [M5-ignore-split] resolve individual flags from ignore_3d alias
        if ignore_pose_3d is None:
            ignore_pose_3d = ignore_3d
        if ignore_smpl is None:
            ignore_smpl = ignore_3d
        self.is_train = is_train
        self.options = options
        self.crop_res = int(self.options.img_res)
        # Pseudo-focal estimation mode for samples without real intrinsics.
        # Sourced from CAMERA.FOCAL_ESTIMATE_MODE; default 'diagonal' = sqrt(W^2+H^2).
        self.focal_estimate_mode = str(getattr(options, 'focal_estimate_mode', None) or 'diagonal').lower()
        if dataset not in config.DATASET_FOLDERS:
            raise KeyError(f"Dataset '{dataset}' is not defined in config.DATASET_FOLDERS")
        if dataset not in config.DATASET_FILES[is_train]:
            split_name = 'train' if is_train else 'eval'
            raise KeyError(f"Dataset '{dataset}' is not defined in config.DATASET_FILES[{split_name}]")
        self.img_dir = config.DATASET_FOLDERS[dataset]
        if self.img_dir == '':
            raise ValueError(f"Dataset root for '{dataset}' is empty in config.DATASET_FOLDERS")
        npz_path = npz_path_override if npz_path_override is not None else config.DATASET_FILES[is_train][dataset]
        if not exists(npz_path):
            raise FileNotFoundError(f"Dataset npz for '{dataset}' not found: {npz_path}")
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(npz_path, allow_pickle=True)

        required_keys = ['imgname', 'center', 'scale']
        for k in required_keys:
            if k not in self.data:
                raise KeyError(f"Dataset '{dataset}' npz is missing required field '{k}': {npz_path}")

        self.imgname = self.data['imgname']
        n_img = len(self.imgname)
        center_arr = np.asarray(self.data['center'])
        scale_arr = np.asarray(self.data['scale'])
        if center_arr.ndim != 2 or center_arr.shape != (n_img, 2):
            raise ValueError(
                f"Dataset '{dataset}' npz invalid key 'center' shape: actual={center_arr.shape}, "
                f"expected=({n_img},2), path={npz_path}"
            )
        if not ((scale_arr.ndim == 1 and scale_arr.shape[0] == n_img) or (scale_arr.ndim == 2 and scale_arr.shape == (n_img, 1))):
            raise ValueError(
                f"Dataset '{dataset}' npz invalid key 'scale' shape: actual={scale_arr.shape}, "
                f"expected=({n_img},) or ({n_img},1), path={npz_path}"
            )
        if scale_arr.ndim == 2:
            scale_arr = scale_arr[:, 0]

        optional_keys = [
            'pose', 'shape', 'has_smpl', 'S', 'part', 'openpose',
            'gender', 'focal_length', 'img_focal', 'img_center', 'has_pose_3d'
        ]
        for key in optional_keys:
            if key in self.data and len(self.data[key]) != n_img:
                raise ValueError(
                    f"Dataset '{dataset}' npz length mismatch for key '{key}': "
                    f"{len(self.data[key])} vs imgname={n_img}, path={npz_path}"
                )
        if 'pose' in self.data:
            pose_arr = np.asarray(self.data['pose'])
            if pose_arr.ndim != 2 or pose_arr.shape != (n_img, 72):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'pose' shape: actual={pose_arr.shape}, "
                    f"expected=({n_img},72), path={npz_path}"
                )
        if 'shape' in self.data:
            shape_arr = np.asarray(self.data['shape'])
            if shape_arr.ndim != 2 or shape_arr.shape != (n_img, 10):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'shape' shape: actual={shape_arr.shape}, "
                    f"expected=({n_img},10), path={npz_path}"
                )
        if 'part' in self.data:
            part_arr = np.asarray(self.data['part'])
            if part_arr.ndim != 3 or part_arr.shape != (n_img, 24, 3):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'part' shape: actual={part_arr.shape}, "
                    f"expected=({n_img},24,3), path={npz_path}"
                )
        if 'openpose' in self.data:
            openpose_arr = np.asarray(self.data['openpose'])
            if openpose_arr.ndim != 3 or openpose_arr.shape != (n_img, 25, 3):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'openpose' shape: actual={openpose_arr.shape}, "
                    f"expected=({n_img},25,3), path={npz_path}"
                )
        if 'S' in self.data:
            S_arr = np.asarray(self.data['S'])
            if S_arr.ndim != 3 or S_arr.shape != (n_img, 24, 4):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'S' shape: actual={S_arr.shape}, "
                    f"expected=({n_img},24,4), path={npz_path}"
                )
        if 'img_center' in self.data:
            img_center_arr = np.asarray(self.data['img_center'])
            if img_center_arr.ndim != 2 or img_center_arr.shape != (n_img, 2):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'img_center' shape: actual={img_center_arr.shape}, "
                    f"expected=({n_img},2), path={npz_path}"
                )
        if 'img_focal' in self.data:
            img_focal_arr = np.asarray(self.data['img_focal'])
            if not ((img_focal_arr.ndim == 1 and img_focal_arr.shape[0] == n_img) or
                    (img_focal_arr.ndim == 2 and img_focal_arr.shape == (n_img, 1))):
                raise ValueError(
                    f"Dataset '{dataset}' npz invalid key 'img_focal' shape: actual={img_focal_arr.shape}, "
                    f"expected=({n_img},) or ({n_img},1), path={npz_path}"
                )
        
        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = scale_arr.astype(np.float32)
        self.center = center_arr.astype(np.float32)
        try:
            if 'focal_length' in self.data:
                focal = np.asarray(self.data['focal_length']).astype(np.float32)
            elif 'img_focal' in self.data:
                focal = np.asarray(self.data['img_focal']).astype(np.float32)
            else:
                focal = None
            if focal is not None:
                if focal.ndim == 2 and focal.shape == (n_img, 1):
                    focal = focal[:, 0]
                elif not (focal.ndim == 1 and focal.shape[0] == n_img):
                    raise ValueError(
                        f"Dataset '{dataset}' npz invalid focal shape: actual={focal.shape}, "
                        f"expected=({n_img},) or ({n_img},1), path={npz_path}"
                    )
            self.focal_length = focal
        except KeyError:
            self.focal_length = None

        # [M4-has-cam-intrinsics] 標記此 dataset 是否有真實相機內參
        # 判定依據：npz 是否包含 focal_length 或 img_focal 欄位
        # 短期方案：per-dataset 級別。長期建議：用 HumanFoV 為每張圖預測 FL 後寫進 npz。
        _has_real_focal = ('focal_length' in self.data) or ('img_focal' in self.data)
        self.has_cam_intrinsics_arr = np.full(n_img, 1.0 if _has_real_focal else 0.0, dtype=np.float32)
        n_real = int(self.has_cam_intrinsics_arr.sum())
        print(f"[Dataset {dataset}] focal_estimate_mode={self.focal_estimate_mode}, "
              f"real_focal_samples={n_real}/{n_img}")

        # MoGe focal override: load per-image estimated focal from JSON when NPZ has no real intrinsics.
        # Builds numpy arrays at init time (not per-item) for DataLoader worker safety.
        self._moge_focal = None
        self._moge_center = None
        if self.focal_length is None:
            import json as _json
            _moge_path = config.MOGE_FOCAL_FILES.get(dataset)
            if _moge_path and exists(_moge_path):
                with open(_moge_path) as _f:
                    _moge_data = _json.load(_f)
                _focal_buf = np.zeros(n_img, dtype=np.float32)
                _center_buf = np.zeros((n_img, 2), dtype=np.float32)
                _n_hit = 0
                for _i, _name in enumerate(self.imgname):
                    _key = config._imgname_to_str(_name)
                    if _key in _moge_data:
                        _focal_buf[_i] = float(_moge_data[_key]['img_focal'])
                        _center_buf[_i] = _moge_data[_key]['img_center']
                        _n_hit += 1
                print(f"[Dataset {dataset}] MoGe focal loaded: {_n_hit}/{n_img} matched")
                if _n_hit > 0:
                    self._moge_focal = _focal_buf
                    self._moge_center = _center_buf
                    # Treat MoGe estimates as real intrinsics for has_cam_intrinsics flag
                    self.has_cam_intrinsics_arr = (_focal_buf > 0).astype(np.float32)

        try:
            self.img_center_from_npz = self.data['img_center'].astype(np.float32)
        except KeyError:
            self.img_center_from_npz = None
            
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        self.pascal_occluders_path = getattr(
            config,
            'PASCAL_OCCLUDERS',
            'data/dataset_extras/pascal_occluders.pkl'
        )
        try:
            if self.use_augmentation and exists(self.pascal_occluders_path):
                self.occluders = joblib.load(self.pascal_occluders_path)
            else:
                self.occluders = None
        except Exception as e:
            print(f'Warning: failed to load Pascal occluders from {self.pascal_occluders_path}: {e}')
            self.occluders = None
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float32)
            self.betas = self.data['shape'].astype(np.float32)
            if 'has_smpl' in self.data:
                has_smpl = np.asarray(self.data['has_smpl'])
                if has_smpl.ndim == 0 or has_smpl.shape[0] != n_img:
                    raise ValueError(
                        f"Dataset '{dataset}' npz invalid key 'has_smpl' shape: actual={has_smpl.shape}, "
                        f"expected first dim={n_img}, path={npz_path}"
                    )
                self.has_smpl = has_smpl.reshape(n_img, -1)[:, 0].astype(np.float32)
            else:
                self.has_smpl = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)
        if ignore_smpl:  # [M5-ignore-split]
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)
        
        # Get gt 3D pose, if available
        if 'S' in self.data:
            self.pose_3d = self.data['S']
            if 'has_pose_3d' in self.data:
                has_pose_3d = np.asarray(self.data['has_pose_3d'])
                if has_pose_3d.ndim == 0 or has_pose_3d.shape[0] != n_img:
                    raise ValueError(
                        f"Dataset '{dataset}' npz invalid key 'has_pose_3d' shape: actual={has_pose_3d.shape}, "
                        f"expected first dim={n_img}, path={npz_path}"
                    )
                self.has_pose_3d = has_pose_3d.reshape(n_img, -1)[:, 0].astype(np.float32)
            else:
                self.has_pose_3d = np.ones(n_img, dtype=np.float32)
        else:
            self.pose_3d = None
            self.has_pose_3d = np.zeros(n_img, dtype=np.float32)
        if ignore_pose_3d:  # [M5-ignore-split]
            self.has_pose_3d = np.zeros(n_img, dtype=np.float32)
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)
        # Suppress confidence for fully-missing segments to avoid false-positive 2D loss.
        op_all_zero = (keypoints_openpose[..., 2] == 0).all(axis=1)
        gt_all_zero = (keypoints_gt[..., 2] == 0).all(axis=1)
        self.keypoints[op_all_zero, :25, 2] = 0.0
        self.keypoints[gt_all_zero, 25:, 2] = 0.0

        # Get gender data, if available
        try:
            gender = self.data['gender']
            def _decode_gender(g):
                if isinstance(g, bytes):
                    g = g.decode('utf-8', errors='ignore')
                elif isinstance(g, np.ndarray) and g.shape == ():
                    g = g.item()
                    if isinstance(g, bytes):
                        g = g.decode('utf-8', errors='ignore')
                g = str(g).strip().lower()
                if g == 'm':
                    return 0
                if g == 'f':
                    return 1
                return -1
            self.gender = np.array([_decode_gender(g) for g in gender], dtype=np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

        # Scheme A: pGT reliability weights (fallback to 1.0 if not in NPZ)
        self.smpl_weight = np.asarray(self.data['smpl_weight']).astype(np.float32) if 'smpl_weight' in self.data else np.ones(n_img, dtype=np.float32)
        self.pose_weight = np.asarray(self.data['pose_weight']).astype(np.float32) if 'pose_weight' in self.data else np.ones(n_img, dtype=np.float32)
        self.shape_weight = np.asarray(self.data['shape_weight']).astype(np.float32) if 'shape_weight' in self.data else np.ones(n_img, dtype=np.float32)
        self.cam_weight = np.asarray(self.data['cam_weight']).astype(np.float32) if 'cam_weight' in self.data else np.ones(n_img, dtype=np.float32)
        self.pgt_quality = np.asarray(self.data['pgt_quality']).astype(np.float32) if 'pgt_quality' in self.data else np.zeros(n_img, dtype=np.float32)
        self.truncation_type = np.asarray(self.data['truncation_type']).astype(np.float32) if 'truncation_type' in self.data else np.zeros(n_img, dtype=np.float32)

    def _is_true_thermal_dataset(self):
        thermal_names = {'llvip', 'llvip-pgt', 'llvip-pgt-train90', 'otp2', 'otp2-pgt'}
        return str(self.dataset).lower() in thermal_names

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0
        pn = np.ones(3, dtype=np.float32)
        rot = 0.0
        sc = 1.0
        occ = 0
        do_extreme_crop = False

        if self.is_train and self.use_augmentation:
            occ_prob = getattr(self.options, 'occ_prob', 0.5)
            flip_prob = getattr(self.options, 'flip_prob', 0.5)
            noise_factor = getattr(self.options, 'noise_factor', 0.4)
            rot_factor = getattr(self.options, 'rot_factor', 30.0)
            scale_factor = getattr(self.options, 'scale_factor', 0.25)
            extreme_crop_rate = _opt_float(self.options, 'extreme_crop_aug_rate', 0.0)

            # Extreme crop sampled BEFORE occluder so they remain mutually exclusive.
            if extreme_crop_rate > 0.0 and np.random.uniform() <= extreme_crop_rate:
                do_extreme_crop = True

            if not do_extreme_crop and np.random.uniform() <= occ_prob:
                occ = 1
            if np.random.uniform() <= flip_prob:
                flip = 1

            pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3).astype(np.float32)

            rot = min(2 * rot_factor, max(-2 * rot_factor, np.random.randn() * rot_factor))

            sc = min(1 + scale_factor, max(1 - scale_factor, np.random.randn() * scale_factor + 1))

            rot_zero_prob = _opt_float(self.options, 'rot_zero_prob', 0.6)
            if np.random.uniform() <= rot_zero_prob:
                rot = 0.0

        return flip, pn, rot, sc, occ, do_extreme_crop

    def map_paired_rgb_rel(self, imgname_rel):
        """Map a thermal imgname_rel to its visible (RGB) partner relative path.

        Default LLVIP rule: strip a leading ``infrared/`` prefix so
        ``infrared/<split>/<name>.jpg`` -> ``<split>/<name>.jpg`` under
        ``paired_rgb_root`` (which points at .../LLVIP/visible). Subclasses may
        override for other datasets.
        """
        rel = str(imgname_rel).lstrip('/')
        prefix = getattr(self, 'paired_strip_prefix', 'infrared/')
        if prefix and rel.startswith(prefix):
            rel = rel[len(prefix):]
        return rel

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, occ,
                       apply_bridge_aug=True, force_rgb_modality=False):
        """Process rgb image and do augmentation. Returns (img_chw_float, modality_id).

        MU-HMR paired loading: the visible (RGB) partner is processed with
        ``apply_bridge_aug=False`` (skip thermal-like bridge augs) and
        ``force_rgb_modality=True`` (modality_id=0). Geometry (crop/flip/occ via
        the passed rot/flip/occ/center/scale) is identical to the thermal call,
        guaranteeing pixel alignment between modalities.
        """
        rgb_img = crop(rgb_img, center, scale, [self.crop_res, self.crop_res], rot=rot)

        if flip:
            rgb_img = flip_img(rgb_img)

        if occ and self.occluders is not None:
            rgb_img = occlude_with_pascal_objects(
                rgb_img.astype(np.uint8),
                self.occluders
            ).astype(np.float32)

        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

        # Scheme A: modality tracking — real thermal starts at 2, RGB at 0.
        # MU-HMR paired RGB partner forces RGB (0) and skips thermal-bridge augs.
        modality_id = 0 if force_rgb_modality else (2 if self._is_true_thermal_dataset() else 0)

        # Existing grayscale augmentation
        grayscale_prob = _opt_float(self.options, 'grayscale_prob', 0.0)
        if apply_bridge_aug and grayscale_prob > 0.0 and self.is_train and np.random.uniform() < grayscale_prob:
            gray = (0.299 * rgb_img[:, :, 0] +
                    0.587 * rgb_img[:, :, 1] +
                    0.114 * rgb_img[:, :, 2])
            rgb_img[:, :, 0] = gray
            rgb_img[:, :, 1] = gray
            rgb_img[:, :, 2] = gray

        # Thermal-like bridge augmentation for RGB pre-adaptation
        thermal_like_prob = _opt_float(self.options, 'thermal_like_prob', 0.0)
        if apply_bridge_aug and self.is_train and thermal_like_prob > 0.0 and np.random.uniform() < thermal_like_prob:
            gray = (0.299 * rgb_img[:, :, 0] + 0.587 * rgb_img[:, :, 1] + 0.114 * rgb_img[:, :, 2])
            rgb_img[:, :, 0] = gray
            rgb_img[:, :, 1] = gray
            rgb_img[:, :, 2] = gray
            if bool(getattr(self.options, 'bridge_aug_as_thermal_modality', False)):
                modality_id = 1

        # Inversion
        invert_prob = _opt_float(self.options, 'invert_prob', 0.0)
        if apply_bridge_aug and self.is_train and invert_prob > 0.0 and np.random.uniform() < invert_prob:
            rgb_img = 255.0 - rgb_img
            if bool(getattr(self.options, 'bridge_aug_as_thermal_modality', False)):
                modality_id = 1

        # CLAHE
        clahe_prob = _opt_float(self.options, 'clahe_prob', 0.0)
        if apply_bridge_aug and self.is_train and clahe_prob > 0.0 and np.random.uniform() < clahe_prob:
            img_u8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
            gray_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_eq = clahe_obj.apply(gray_u8).astype(np.float32)
            rgb_img[:, :, 0] = gray_eq
            rgb_img[:, :, 1] = gray_eq
            rgb_img[:, :, 2] = gray_eq
            if bool(getattr(self.options, 'bridge_aug_as_thermal_modality', False)):
                modality_id = 1

        # Blur
        blur_prob = _opt_float(self.options, 'blur_prob', 0.0)
        if blur_prob > 0.0 and self.is_train and np.random.uniform() < blur_prob:
            sigma = np.random.uniform(0.5, _opt_float(self.options, 'blur_max_sigma', 2.0))
            k = max(1, int(2 * round(3 * sigma) + 1)) | 1
            rgb_img = cv2.GaussianBlur(rgb_img.astype(np.float32), (k, k), sigma)

        # Small thermal noise
        thermal_noise_prob = _opt_float(self.options, 'thermal_noise_prob', 0.0)
        if apply_bridge_aug and self.is_train and thermal_noise_prob > 0.0 and np.random.uniform() < thermal_noise_prob:
            std = _opt_float(self.options, 'thermal_noise_std', 0.03) * 255.0
            noise = np.random.normal(0.0, std, size=rgb_img.shape).astype(np.float32)
            rgb_img = np.clip(rgb_img + noise, 0.0, 255.0)

        rgb_img = np.transpose(rgb_img.astype(np.float32), (2, 0, 1)) / 255.0
        return rgb_img, modality_id

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms.

        [geometry-check] output space:
          - Input  kp[:, :2] : full-image pixel coords (0-indexed).
          - Output kp[:, :2] : normalized crop-space in [-1, 1] with rot/flip
                               baked in; kp[:, 2] keeps confidence unchanged.
        The (+1 here) / (-1 inside transform) cancels exactly and preserves
        SPIN's original 1-indexed matrix convention; do not remove one side
        without the other.
        Downstream consumers that need pixel-space (e.g. SMPLify) must
        denormalize via 0.5*crop_res*(x+1).
        """
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
                                  [self.crop_res, self.crop_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.crop_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc, occ, do_extreme_crop = self.augm_params()

        # [extreme-crop] recompute bbox from keypoints BEFORE image crop, so
        # downstream rgb_processing / j2d_processing honour the new frame.
        if do_extreme_crop:
            extreme_level = int(getattr(self.options, 'extreme_crop_aug_level', 0))
            keypoints_fullimg = self.keypoints[index].copy()
            center, scale = apply_extreme_crop(
                center, scale, keypoints_fullimg, level=extreme_level
            )

        # Load image
        imgname_rel = config._imgname_to_str(self.imgname[index])
        imgname = config.resolve_dataset_image_path(self.dataset, imgname_rel, must_exist=True)
        img_bgr = cv2.imread(imgname)
        if img_bgr is None:
            raise FileNotFoundError(
                f"Failed to load image. dataset={self.dataset}, sample_index={index}, "
                f"imgname_rel={imgname_rel}, resolved_imgname={imgname}"
            )
        img = img_bgr[:, :, ::-1].copy().astype(np.float32)
        orig_shape = np.array(img.shape)[:2]
        orig_h, orig_w = orig_shape[0], orig_shape[1]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img, modality_id = self.rgb_processing(img, center, sc*scale, rot, flip, pn, occ)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['modality'] = torch.tensor(modality_id, dtype=torch.long)

        # MU-HMR paired modality loading (LLVIP visible partner). Gated by an
        # attribute so all other datasets are unaffected. Primary image is the
        # thermal (IR); the visible (RGB) partner is loaded from
        # paired_rgb_root using the SAME geometry (center/scale/rot/flip/occ),
        # which is valid because LLVIP infrared/visible are pixel-aligned.
        if getattr(self, 'paired_rgb_root', None) is not None:
            item['img_ir'] = item['img']
            rgb_rel = self.map_paired_rgb_rel(imgname_rel)
            rgb_path = os.path.join(self.paired_rgb_root, rgb_rel)
            rgb_bgr = cv2.imread(rgb_path)
            if rgb_bgr is None:
                raise FileNotFoundError(
                    f"MU-HMR paired RGB not found: {rgb_path} "
                    f"(mapped from imgname_rel={imgname_rel})"
                )
            rgb = rgb_bgr[:, :, ::-1].copy().astype(np.float32)
            rgb_proc, _ = self.rgb_processing(
                rgb, center, sc * scale, rot, flip, pn, occ,
                apply_bridge_aug=False, force_rgb_modality=True,
            )
            item['img_rgb'] = self.normalize_img(torch.from_numpy(rgb_proc).float())
            # Both modalities present before modality dropout.
            item['modality_mask'] = torch.tensor([1.0, 1.0], dtype=torch.float32)

        # Scheme-3: IR-only datasets (e.g. OTP2 thermal-only). No paired RGB
        # exists, so emit a zero RGB placeholder + modality_mask=[0,1]. The
        # MMPed unifier hard-routes ir_only rows to the IR tokens (the zero RGB
        # is never read), and the body adapter's adapter_mask=(has_ir)=1 makes
        # these rows train the IR token adapter + body adapters. apply_modality_
        # dropout leaves non-paired rows unchanged, so the mask survives to the
        # forward. Gated by ir_only_mode so all other datasets are unaffected.
        elif getattr(self, 'ir_only_mode', False):
            item['img_ir'] = item['img']
            item['img_rgb'] = torch.zeros_like(item['img'])
            item['modality_mask'] = torch.tensor([0.0, 1.0], dtype=torch.float32)

        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname_rel'] = imgname_rel
        item['imgname'] = imgname

        # Get 3D pose, if available
        if bool(self.has_pose_3d[index] > 0):
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = np.float32(self.has_pose_3d[index])
        item['scale'] = np.float32(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape.astype(np.float32)
        item['img_h'] = np.float32(orig_h)
        item['img_w'] = np.float32(orig_w)
        if self.img_center_from_npz is not None:
            item['img_center'] = self.img_center_from_npz[index].astype(np.float32)
        elif self._moge_center is not None and self._moge_focal is not None and self._moge_focal[index] > 0:
            item['img_center'] = self._moge_center[index].astype(np.float32)
        else:
            item['img_center'] = np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32)
        if self.focal_length is not None:
            # Real intrinsics from NPZ take precedence and are never overwritten.
            item['img_focal'] = np.float32(self.focal_length[index])
            item['focal_source'] = np.int32(0)   # 0 = real camera calibration
        elif self._moge_focal is not None and self._moge_focal[index] > 0:
            # MoGe-estimated focal: much better than diagonal fallback.
            item['img_focal'] = np.float32(self._moge_focal[index])
            item['focal_source'] = np.int32(1)   # 1 = MoGe-estimated
        else:
            # Pseudo-focal for samples without real intrinsics. Unified via
            # utils.geometry.estimate_focal_length so demo / training / eval agree.
            item['img_focal'] = np.float32(
                estimate_focal_length(int(orig_h), int(orig_w), mode=self.focal_estimate_mode)
            )
            item['focal_source'] = np.int32(2)   # 2 = diagonal fallback
        item['is_flipped'] = np.bool_(flip)
        item['rot_angle'] = np.float32(rot)
        item['has_cam_intrinsics'] = np.float32(self.has_cam_intrinsics_arr[index])  # [M4-has-cam-intrinsics]
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        # Scheme A: pGT reliability weights
        item['smpl_weight'] = np.float32(self.smpl_weight[index])
        item['pose_weight'] = np.float32(self.pose_weight[index])
        item['shape_weight'] = np.float32(self.shape_weight[index])
        item['cam_weight'] = np.float32(self.cam_weight[index])
        item['pgt_quality'] = np.float32(self.pgt_quality[index])
        item['truncation_type'] = np.float32(self.truncation_type[index])

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)

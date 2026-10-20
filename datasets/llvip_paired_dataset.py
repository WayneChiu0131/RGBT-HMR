"""LLVIPPairedDataset — paired RGB (visible) + thermal (infrared) loader.

Plan §0.1.2 修正1 / §9.1. The pGT NPZ is fit on the infrared frames; the
pixel-aligned visible partner lives under ``config.LLVIP_VISIBLE_ROOT`` and is
reached by stripping the ``infrared/`` prefix from each thermal imgname_rel.

This is a thin subclass of BaseDataset: it sets the ``paired_rgb_root`` hook so
BaseDataset.__getitem__ loads BOTH modalities with the SAME geometry
(center/scale/rot/flip/occ) and emits ``img_rgb`` / ``img_ir`` /
``modality_mask``. All shared annotations (center, scale, pGT, keypoints,
focal, weights) are computed once by the parent.
"""

import os

import config
from .base_dataset import BaseDataset


class LLVIPPairedDataset(BaseDataset):

    def __init__(self, options, dataset='llvip-pgt', is_train=True,
                 use_augmentation=True, npz_path_override=None,
                 visible_root=None, strip_prefix='infrared/',
                 validate_samples=5, **kwargs):
        super().__init__(
            options,
            dataset=dataset,
            is_train=is_train,
            use_augmentation=use_augmentation,
            npz_path_override=npz_path_override,
            **kwargs,
        )
        self.paired_rgb_root = visible_root or getattr(config, 'LLVIP_VISIBLE_ROOT', None)
        self.paired_strip_prefix = strip_prefix
        if self.paired_rgb_root is None:
            raise ValueError(
                'LLVIPPairedDataset: visible_root is None and '
                'config.LLVIP_VISIBLE_ROOT is unset.'
            )
        if not os.path.isdir(self.paired_rgb_root):
            raise FileNotFoundError(
                f'LLVIPPairedDataset: visible_root not a directory: '
                f'{self.paired_rgb_root}'
            )
        self._validate_paired_paths(int(validate_samples))

    def _validate_paired_paths(self, n_samples):
        """CLAUDE.md rule: print sample paths + must_exist before declaring ok."""
        n = len(self.imgname)
        if n == 0:
            raise RuntimeError('LLVIPPairedDataset: 0 samples in NPZ.')
        n_check = min(n_samples, n)
        print(f'[LLVIPPaired] dataset={self.dataset} | samples={n} | '
              f'visible_root={self.paired_rgb_root}')
        missing = 0
        for i in range(n_check):
            thermal_rel = config._imgname_to_str(self.imgname[i])
            thermal_abs = config.resolve_dataset_image_path(
                self.dataset, thermal_rel, must_exist=False
            )
            rgb_rel = self.map_paired_rgb_rel(thermal_rel)
            rgb_abs = os.path.join(self.paired_rgb_root, rgb_rel)
            exists = os.path.isfile(rgb_abs)
            missing += (0 if exists else 1)
            print(f'  [{i}] IR : {thermal_rel} -> {thermal_abs}')
            print(f'      RGB: {rgb_rel} -> {rgb_abs} | exists={exists}')
        if missing > 0:
            raise FileNotFoundError(
                f'LLVIPPaired: {missing}/{n_check} sampled visible partners '
                f'missing under {self.paired_rgb_root}. Check the mapping / root.'
            )
        print(f'[LLVIPPaired] sample-path check PASSED ({n_check}/{n_check} '
              f'visible partners exist).')

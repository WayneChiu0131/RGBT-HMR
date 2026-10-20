"""
This file contains the definition of different heterogeneous datasets used for training.
[M3-concat-dataset] 改為內部包裝 ConcatDataset，
用 partition 權重算 repetition count，不再有 __getitem__ 內的隨機性。
"""
import torch
import numpy as np
from torch.utils.data import ConcatDataset
from .base_dataset import BaseDataset
import config


class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        # --- 解析 dataset_list 和 partition（保留原有的多來源 fallback 邏輯） ---
        dataset_list = getattr(options, 'dataset_list', None)
        dataset_partition = getattr(options, 'dataset_partition', None)
        if dataset_list is None and hasattr(options, 'DATASET'):
            dataset_list = getattr(options.DATASET, 'LIST', None)
        if dataset_partition is None and hasattr(options, 'DATASET'):
            dataset_partition = getattr(options.DATASET, 'PARTITION', None)
        if dataset_list is None:
            dataset_list = config.DATASET_DEFAULT_LIST
        if dataset_partition is None:
            dataset_partition = config.DATASET_DEFAULT_PARTITION

        self.dataset_list = list(dataset_list)
        dataset_partition = list(dataset_partition)

        if len(self.dataset_list) == 0:
            raise ValueError('dataset_list must not be empty')
        if len(set(self.dataset_list)) != len(self.dataset_list):
            raise ValueError(f'dataset_list has duplicated names: {self.dataset_list}')
        if len(dataset_partition) != len(self.dataset_list):
            raise ValueError('dataset_partition length must equal dataset_list length')

        part = np.asarray(dataset_partition, dtype=np.float64)
        if np.any(part <= 0):
            raise ValueError('dataset_partition values must be positive')
        part = part / part.sum()
        self.partition = part.tolist()

        # --- 建構各 BaseDataset（unique，每個只建一次） ---
        # MU-HMR: llvip uses the paired (RGB+thermal) loader so batches carry
        # img_rgb / img_ir / modality_mask. Other datasets stay on BaseDataset.
        self.datasets = [self._build_one(options, ds, **kwargs) for ds in self.dataset_list]
        self.dataset_dict = {name: i for i, name in enumerate(self.dataset_list)}
        self.dataset_lengths = {name: len(self.datasets[idx])
                                for idx, name in enumerate(self.dataset_list)}
        self.index_to_dataset = {i: name for i, name in enumerate(self.dataset_list)}

        for name, ds_len in self.dataset_lengths.items():
            if ds_len <= 0:
                raise ValueError(f"Dataset '{name}' has zero samples; cannot build MixedDataset")

        # --- [M3-concat-dataset] 用 partition 計算 repetition count ---
        # 以最大 dataset 長度為基準，讓各 dataset 在 ConcatDataset 中的
        # 比例近似 partition 權重。
        # 同一個 BaseDataset 物件多次 append（stateless __getitem__，安全）。
        max_len = max(len(ds) for ds in self.datasets)
        samples_per_epoch = int(getattr(options, 'samples_per_epoch', 0) or 0)
        ref_total = samples_per_epoch if samples_per_epoch > 0 else max_len

        expanded_list = []
        effective_counts = {}
        for i, (name, ds) in enumerate(zip(self.dataset_list, self.datasets)):
            target_count = max(1, int(round(part[i] * ref_total)))
            n_copies = max(1, int(np.ceil(target_count / len(ds))))
            for _ in range(n_copies):
                expanded_list.append(ds)
            effective_counts[name] = n_copies * len(ds)

        self._concat = ConcatDataset(expanded_list)
        self.length = len(self._concat)

        # --- logging ---
        print(f'[MixedDataset] dataset_list={self.dataset_list}')
        print(f'[MixedDataset] normalized_partition={part.tolist()}')
        print(f'[MixedDataset] dataset_lengths={self.dataset_lengths}')
        for name in self.dataset_list:
            root = config.DATASET_FOLDERS.get(name, '<missing>')
            npz_path = config.DATASET_FILES[1].get(name, '<missing>')
            print(f'[MixedDataset]   {name}: local_len={self.dataset_lengths[name]}, '
                  f'effective_count={effective_counts[name]}, '
                  f'root={root}, train_npz={npz_path}')
        print(f'[MixedDataset] total_length={self.length}')

    _PAIRED_MODEL_TYPES = frozenset({
        'modality_unified_hmr',
        'prebackbone_rgbt_hmr',
        'pbf_hmr2',
        'late_token_rgbt_hmr',
        'mmped_unified_hmr2',
        'mmped_hmr2_unified_encoder',
        'stage2c_mmped_hmr2',
    })

    @staticmethod
    def _build_one(options, ds, **kwargs):
        """Pick the dataset class. Paired-modality models (MU-HMR, PBF-HMR2) use
        LLVIPPairedDataset for LLVIP so batches carry img_rgb/img_ir/modality_mask."""
        _mtype = str(getattr(options, 'model_type', None) or '')
        needs_paired = _mtype in MixedDataset._PAIRED_MODEL_TYPES
        # 09.13 valsplit: 'llvip-pgt-train90' is a by-image 90% subset of
        # llvip-pgt's train split (see config.py DATASET_FILES[1]); it must
        # route through the same paired loader as 'llvip-pgt' or batches
        # silently lose img_rgb/img_ir/modality_mask (RuntimeError at train
        # step 0 for any _PAIRED_MODEL_TYPES model).
        if needs_paired and str(ds).lower() in ('llvip', 'llvip-pgt', 'llvip-pgt-train90'):
            from .llvip_paired_dataset import LLVIPPairedDataset
            return LLVIPPairedDataset(options, dataset=ds, **kwargs)
        # Scheme-3: OTP2 thermal-only -> IR-only loader (modality_mask=[0,1]) so
        # batches carry the same img_rgb/img_ir/modality_mask keys as LLVIP and
        # MixedDataset can interleave paired LLVIP + IR-only OTP2.
        if needs_paired and str(ds).lower() in ('otp2', 'otp2-pgt'):
            from .otp2_ir_only_dataset import OTP2IROnlyDataset
            return OTP2IROnlyDataset(options, dataset=ds, **kwargs)
        return BaseDataset(options, ds, **kwargs)

    def __getitem__(self, index):
        # [M3-concat-dataset] 純 delegate，無任何 np.random 或 index % len
        return self._concat[index]

    def __len__(self):
        return self.length

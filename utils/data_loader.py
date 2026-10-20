from __future__ import division
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class RandomSampler(Sampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            # [M1-sampler-fix] 兩者必須是同一份順序，否則 resume 時
            # checkpoint 存的 dataset_perm 和實際跑的 perm 不一致
            self.dataset_perm = torch.randperm(len(self.data_source)).tolist()
            self.perm = list(self.dataset_perm)

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

class SequentialSampler(Sampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            # [M1-sampler-fix] 用獨立 list 避免意外 mutate
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = list(self.dataset_perm)

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

# [M2-worker-init] deterministic worker seeding
def _default_worker_init_fn(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    import numpy as np
    import random
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CheckpointDataLoader(DataLoader):
    """
    Extends torch.utils.data.DataLoader to handle resuming training from an arbitrary point within an epoch.
    """
    def __init__(self, dataset, checkpoint=None, batch_size=1,
                 shuffle=False, num_workers=0, pin_memory=False,
                 drop_last=True, timeout=0, worker_init_fn=None):

        # [M2-worker-init] 若沒傳 worker_init_fn，使用 deterministic default
        if worker_init_fn is None:
            worker_init_fn = _default_worker_init_fn

        # dataset_perm 相容性檢查：若 checkpoint 的 perm 與當前 dataset 不相容
        # (長度不同、或含超出範圍的 index)，丟棄 mid-epoch 位置改從新 epoch 開頭。
        # 常見觸發：更改 DATASET.LIST / PARTITION / samples_per_epoch 後再 resume。
        if checkpoint is not None and checkpoint.get('dataset_perm') is not None:
            dp = checkpoint['dataset_perm']
            n = len(dataset)
            perm_len = len(dp)
            perm_max = max(dp) if perm_len > 0 else -1
            if perm_len != n or perm_max >= n:
                print(
                    f"[CheckpointDataLoader][WARN] dataset_perm incompatible with "
                    f"current dataset: perm_len={perm_len}, perm_max={perm_max}, "
                    f"dataset_len={n}. Discarding mid-epoch position and starting "
                    f"a fresh epoch (model/optimizer/step_count are preserved)."
                )
                checkpoint = dict(checkpoint)
                checkpoint['dataset_perm'] = None
                checkpoint['batch_idx'] = 0
                checkpoint['batch_size'] = batch_size

        # [M2-batch-size-check] resume 時 batch_size 必須一致，否則 sampler offset 錯位
        if checkpoint is not None and checkpoint.get('batch_size') is not None:
            ckpt_bs = checkpoint['batch_size']
            if ckpt_bs != batch_size:
                raise ValueError(
                    f"Cannot resume mid-epoch: checkpoint batch_size={ckpt_bs} "
                    f"!= current batch_size={batch_size}. "
                    f"Either use the same batch_size or start a fresh epoch."
                )

        if shuffle:
            sampler = RandomSampler(dataset, checkpoint)
        else:
            sampler = SequentialSampler(dataset, checkpoint)

        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0

        super(CheckpointDataLoader, self).__init__(
            dataset,
            sampler=sampler,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,  # [M2-worker-init] 修正：原本傳 None
        )

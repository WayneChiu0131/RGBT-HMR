import os
import torch
import config
from .base_dataset import BaseDataset


class ThermalValidationDataset(torch.utils.data.Dataset):
    """Simple validation wrapper for thermal datasets (OTP2, LLVIP).

    Uses BaseDataset with npz_path_override so any NPZ can be loaded
    regardless of the default train/test split in config.DATASET_FILES.
    No augmentation is applied (is_train=False, use_augmentation=False).
    """

    def __init__(self, options, npz_path, dataset_name='otp2-pgt'):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f'Thermal validation npz not found: {npz_path}')
        self.base_dataset = BaseDataset(
            options,
            dataset=dataset_name,
            is_train=False,
            use_augmentation=False,
            npz_path_override=npz_path,
        )
        print(
            f'[ThermalValidation] dataset={dataset_name} | '
            f'samples={len(self.base_dataset)} | npz={npz_path}'
        )

    def __getitem__(self, index):
        return self.base_dataset[index]

    def __len__(self):
        return len(self.base_dataset)

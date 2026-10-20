"""OTP2IROnlyDataset — OTP2 thermal-only loader for Scheme-3 IR-only training.

OTP2 is a single-modality thermal dataset with SMPLify pGT (no paired RGB, cf.
[[project_mu_hmr_stage2c_mmped]] memory). Scheme-2's modality-conditional body
adapter opened the IR-only gradient path (adapter_mask=(has_ir)=1 makes IR-only
rows train the IR token adapter + body adapters), so OTP2 can now contribute
gradient — the old "OTP2 crashes because ir_only is exact-routed / no-grad"
limitation is gone.

This is a thin BaseDataset subclass that sets ``ir_only_mode`` so
BaseDataset.__getitem__ emits ``img_ir`` (= the thermal image), a zero
``img_rgb`` placeholder, and ``modality_mask=[0,1]``. The unifier hard-routes
ir_only rows to the IR tokens (the zero RGB is never read), so batches collate
identically with LLVIPPairedDataset (same img_rgb / img_ir / modality_mask keys)
and MixedDataset can interleave the two.
"""

from .base_dataset import BaseDataset


class OTP2IROnlyDataset(BaseDataset):

    def __init__(self, options, dataset='otp2-pgt', is_train=True,
                 use_augmentation=True, npz_path_override=None, **kwargs):
        super().__init__(
            options,
            dataset=dataset,
            is_train=is_train,
            use_augmentation=use_augmentation,
            npz_path_override=npz_path_override,
            **kwargs,
        )
        # BaseDataset.__getitem__ reads this to emit img_ir / zero img_rgb /
        # modality_mask=[0,1] instead of the single-image path.
        self.ir_only_mode = True
        n = len(self.imgname)
        if n == 0:
            raise RuntimeError('OTP2IROnlyDataset: 0 samples in NPZ.')
        print(f'[OTP2IROnly] dataset={self.dataset} | samples={n} | '
              f'ir_only_mode=True (modality_mask=[0,1], zero RGB placeholder)')

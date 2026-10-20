"""RGB-T modules: MMPedestron-style unified encoder on top of a frozen HMR2.0."""

from .modality_dropout import (
    sample_llvip_modality_mask,
    apply_modality_dropout,
)
from .mmped_hmr2_unified_encoder import (
    MMPedStyleHMR2Unifier,
    build_mmped_hmr2_unified_encoder,
    BoundedIRTokenAdapter,
    build_bounded_ir_token_adapter,
    ResidualBodyBlockAdapter,
    build_body_adapters,
)
from .mmped_unified_hmr2 import MMPedUnifiedHMR2, CrossModalDeepAdapter

__all__ = [
    'sample_llvip_modality_mask',
    'apply_modality_dropout',
    'MMPedStyleHMR2Unifier',
    'build_mmped_hmr2_unified_encoder',
    'BoundedIRTokenAdapter',
    'build_bounded_ir_token_adapter',
    'ResidualBodyBlockAdapter',
    'build_body_adapters',
    'MMPedUnifiedHMR2',
]

"""Ported HMR2.0 (4D-Humans) core: ViT-H backbone + transformer-decoder head.

Used as the frozen HMR core for MU-HMR (replaces the ViTPose Stage1B core).
Token interface [B,192,1280] is identical, so RGMR / TokenAdapter are reused.
"""
from .vit import ViT, build_hmr2_vit
from .smpl_head import SMPLTransformerDecoderHead
from .hmr2_core import HMR2Encoder, HMR2Core, build_hmr2_core

__all__ = [
    'ViT',
    'build_hmr2_vit',
    'SMPLTransformerDecoderHead',
    'HMR2Encoder',
    'HMR2Core',
    'build_hmr2_core',
]

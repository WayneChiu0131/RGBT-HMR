"""Export a training checkpoint as release weights (trainable modules only).

The frozen HMR2.0 backbone + SMPL head are dropped: they are bit-identical to
MODEL.HMR2_CHECKPOINT (the export checks this) and are re-read from there by
tools/eval_2d_mu.py:build_mu_hmr. Optimizer / scheduler state and the inert
cross-modal transformer (CMT) weights are dropped as well when the config sets
UNIFIED_ENCODER.USE_CMT=false.

Usage (repo root):
  python tools/export_release_weights.py \
      --checkpoint logs/main/s5_dcg/checkpoints/checkpoint_best.pt \
      --train_cfg configs/train/main/s5_dcg.yaml \
      --out weights/main/model_final.pt
"""
import argparse
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import config  # noqa: E402
from models.hmr2_core import build_hmr2_core  # noqa: E402
from models.multimodal import build_mmped_hmr2_unified_encoder  # noqa: E402
from tools.eval_2d_mu import TRAINABLE_ONLY_FORMAT, _strip_module  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--train_cfg', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    sd = _strip_module(torch.load(a.checkpoint, map_location='cpu', weights_only=False)['model'])
    model_cfg = config.load_any_config(a.train_cfg)['MODEL']

    # 1) frozen HMR2.0 part must equal the HMR2.0 checkpoint exactly.
    core = build_hmr2_core(config.SMPL_MEAN_PARAMS, ckpt_path=model_cfg['HMR2_CHECKPOINT'], freeze=True)
    frozen = core.state_dict()
    for k, v in frozen.items():
        if k not in sd or not torch.equal(sd[k], v):
            raise RuntimeError(f'{k} differs from HMR2.0; cannot drop the frozen weights')
    out = {k: v for k, v in sd.items() if k not in frozen}

    # 2) keep only unified-encoder keys that the configured encoder actually builds.
    ue_keys = set(build_mmped_hmr2_unified_encoder(dict(model_cfg['UNIFIED_ENCODER'])).state_dict())
    dropped = [k for k in out if k.startswith('unified_encoder.')
               and k[len('unified_encoder.'):] not in ue_keys]
    missing = [k for k in ue_keys if 'unified_encoder.' + k not in out]
    if missing:
        raise RuntimeError(f'checkpoint lacks encoder keys the config builds: {missing[:5]}')
    for k in dropped:
        del out[k]

    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    torch.save({'model': out, 'format': TRAINABLE_ONLY_FORMAT,
                'source_checkpoint': osp.basename(a.checkpoint)}, a.out)
    n = sum(v.numel() for v in out.values())
    print(f'kept {len(out)} tensors ({n / 1e6:.2f}M params); dropped {len(frozen)} frozen HMR2.0 '
          f'and {len(dropped)} unused encoder tensors -> {a.out}')


if __name__ == '__main__':
    main()

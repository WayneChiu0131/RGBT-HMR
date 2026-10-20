"""Modality dropout for MU-HMR training (plan §10).

Operates on the modality_mask, NOT on the image tensors (the images stay
present as zero/real placeholders; only the mask decides participation). This
keeps the LATE-fusion gate honest: a dropped modality has mask==0 so its
gate weight is forced to 0.

For LLVIP-paired samples (both modalities available), each sample is randomly
routed to RGB-only / IR-only / RGB+IR by the configured probabilities.
OTP2 (thermal-only) and 3DPW (RGB-only) keep their fixed mask.
"""

import torch


def sample_llvip_modality_mask(
    batch_size,
    p_rgb_only=0.30,
    p_ir_only=0.30,
    p_rgbir=0.40,
    device=None,
    generator=None,
):
    """Return a [B, 2] float mask for LLVIP-paired samples.

    Rows are one of [1,0] (rgb-only), [0,1] (ir-only), [1,1] (rgb+ir).
    """
    probs = torch.tensor(
        [float(p_rgb_only), float(p_ir_only), float(p_rgbir)], dtype=torch.float32
    )
    total = probs.sum()
    if total <= 0:
        raise ValueError('modality dropout probabilities must sum to > 0')
    probs = probs / total
    idx = torch.multinomial(probs, batch_size, replacement=True, generator=generator)
    table = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    mask = table[idx]
    if device is not None:
        mask = mask.to(device)
    return mask


def apply_modality_dropout(base_mask, probs, generator=None):
    """Apply dropout only to rows whose base_mask == [1,1] (paired samples).

    base_mask: [B, 2] float (per-sample availability before dropout).
    probs:     (p_rgb_only, p_ir_only, p_rgbir).
    Rows that are not fully-paired (e.g. thermal-only / rgb-only datasets) are
    left unchanged.
    """
    B = base_mask.shape[0]
    device = base_mask.device
    paired = (base_mask[:, 0] > 0) & (base_mask[:, 1] > 0)
    if paired.any():
        sampled = sample_llvip_modality_mask(
            int(paired.sum().item()), *probs, device=device, generator=generator
        )
        out = base_mask.clone()
        out[paired] = sampled
        return out
    return base_mask

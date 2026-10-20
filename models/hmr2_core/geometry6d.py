"""HMR2.0 6D->rotmat (verbatim from 4D-Humans/hmr2/utils/geometry.py).

IMPORTANT: HMR2.0 lays out the 6D vector as reshape(-1,2,3).permute(0,2,1)
(a1 = first 3 elements, a2 = last 3), which DIFFERS from this project's
utils.geometry.rot6d_to_rotmat (view(-1,3,2) => a1 = even-indexed). The HMR2.0
smpl_head was trained with the layout below, so the ported head MUST use this
function, not the project's, or the pose comes out scrambled.
"""
import torch
import torch.nn.functional as F


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(B,6) -> (B,3,3). Zhou et al. CVPR'19, HMR2.0 element layout."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

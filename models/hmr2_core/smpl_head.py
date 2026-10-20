"""HMR2.0 SMPLTransformerDecoderHead — cfg-free port of 4D-Humans.

Differences vs the 4D-Humans original:
  * cfg-free (explicit kwargs); submodule names preserved EXACTLY
    (``transformer`` / ``decpose`` / ``decshape`` / ``deccam`` + buffers
    ``init_body_pose`` / ``init_betas`` / ``init_cam``) so the pretrained
    ``smpl_head.*`` checkpoint keys load unchanged.
  * ``forward`` accepts image TOKENS ``[B, N, C]`` directly (MU-HMR feeds
    fused tokens) instead of a feature map, and returns the project-standard
    3-tuple ``(pred_rotmat[B,24,3,3], pred_betas[B,10], pred_cam[B,3])``.
  * Uses HMR2.0's 6D layout via .geometry6d.rot6d_to_rotmat (see note there).

Default config matches HMR2.0 hmr_vit_transformer: 6D joints, 23 body joints,
transformer depth=6, heads=8, dim=1024, mlp_dim=1024, dim_head=64,
context_dim=1280, IEF_ITERS=1, TRANSFORMER_INPUT='zero'.
"""
import numpy as np
import torch
import torch.nn as nn

from .geometry6d import rot6d_to_rotmat
from .pose_transformer import TransformerDecoder


class SMPLTransformerDecoderHead(nn.Module):

    def __init__(
        self,
        smpl_mean_params,
        num_body_joints=23,
        joint_rep_type='6d',
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        norm='layer',
        context_dim=1280,
        ief_iters=1,
        transformer_input='zero',
        init_decoder_xavier=False,
    ):
        super().__init__()
        self.joint_rep_type = joint_rep_type
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[joint_rep_type]
        self.num_body_joints = int(num_body_joints)
        npose = self.joint_rep_dim * (self.num_body_joints + 1)
        self.npose = npose
        self.ief_iters = int(ief_iters)
        self.input_is_mean_shape = (transformer_input == 'mean_shape')

        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            norm=norm,
            context_dim=context_dim,
        )
        self.transformer = TransformerDecoder(**transformer_args)

        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if init_decoder_xavier:
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, tokens):
        """tokens: image tokens [B, N, C] (already 'b (h w) c'); returns
        (pred_rotmat[B,24,3,3], pred_betas[B,10], pred_cam[B,3])."""
        if tokens.dim() == 4:
            # accept [B, C, H, W] too (single-image path), like the original.
            import einops
            tokens = einops.rearrange(tokens, 'b c h w -> b (h w) c')
        if tokens.dim() != 3:
            raise ValueError(
                f'SMPL head expects [B,N,C] tokens, got {tuple(tokens.shape)}'
            )
        B = tokens.shape[0]
        x = tokens

        pred_body_pose = self.init_body_pose.expand(B, -1)
        pred_betas = self.init_betas.expand(B, -1)
        pred_cam = self.init_cam.expand(B, -1)

        for _ in range(self.ief_iters):
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype)
            token_out = self.transformer(token, context=x).squeeze(1)  # [B, dim]
            pred_body_pose = self.decpose(token_out) + pred_body_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_body_pose).view(B, self.num_body_joints + 1, 3, 3)
        return pred_rotmat, pred_betas, pred_cam

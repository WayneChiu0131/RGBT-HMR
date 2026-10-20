"""HMR2.0 core (backbone + head) packaged for this project / MU-HMR.

Exposes the SAME structural attributes the trainer + MU wrapper expect:
  * ``self.backbone`` (HMR2Encoder; its ``.backbone`` is the ViT) so the
    trainer's freeze check / param summary (``model.backbone.backbone``) and the
    MU wrapper re-parent both work, mirroring ViTPoseEncoder.
  * ``self.head`` (SMPLTransformerDecoderHead) returning the project 3-tuple.
  * ``camera_rep='weak_perspective'`` -> the existing cam_crop2full /
    project_points_full_to_crop pipeline reproduces HMR2.0's crop-space 2D
    exactly (weak-perspective crop 2D is focal-invariant; verified).

Token interface is [B, 192, 1280], identical to ViTPoseEncoder(pool='token'),
so RGMR / TokenAdapter / the MU wrapper need no shape changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import build_hmr2_vit
from .smpl_head import SMPLTransformerDecoderHead


class HMR2Encoder(nn.Module):
    """Wrap the HMR2.0 ViT; emit spatial tokens [B, N, C].

    Input is the project's 256x256 crop; HMR2.0 consumes 256x192 (it slices
    ``x[:,:,:,32:-32]``). We center-crop width 256->192 to match.
    """

    def __init__(self, input_height=256, input_width=192, freeze=False,
                 grad_through_frozen=False):
        super().__init__()
        self.backbone = build_hmr2_vit()
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.freeze = bool(freeze)
        # grad_through_frozen: keep params requires_grad=False (backbone not
        # updated) but DO NOT wrap forward in torch.no_grad, so gradient still
        # flows to whatever produced the input (e.g. a pre-backbone input
        # adapter in PBF-HMR2). Default False preserves the cheap no_grad path
        # used by late-token MU-HMR / plain HMR2 eval.
        self.grad_through_frozen = bool(grad_through_frozen)
        self.feat_dim = 1280
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _preprocess(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if h == self.input_height and w == self.input_width:
            return x
        if h >= self.input_height and w >= self.input_width:
            top = (h - self.input_height) // 2
            left = (w - self.input_width) // 2
            return x[:, :, top:top + self.input_height, left:left + self.input_width]
        return F.interpolate(x, size=(self.input_height, self.input_width),
                             mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self._preprocess(x)
        if self.freeze:
            self.backbone.eval()  # frozen BN/dropout stats either way
            if self.grad_through_frozen:
                # params frozen, but keep the autograd graph so gradient can
                # reach a pre-backbone input adapter.
                feat_map = self.backbone(x)
            else:
                with torch.no_grad():
                    feat_map = self.backbone(x)    # [B, C, H', W']
        else:
            feat_map = self.backbone(x)
        # [B, C, H', W'] -> [B, H'*W', C]
        return feat_map.flatten(2).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Split API (Stage2C MMPed unified encoder): expose the HMR2 ViT as
    #   patch_embed_tokens(x) -> tokens        (patch_embed + pos_embed)
    #   forward_from_tokens(tokens) -> tokens  (transformer blocks + last_norm)
    # so an external cross-modal unifier can operate on patch tokens BEFORE the
    # frozen transformer body. Verified to reproduce forward(x) exactly (the
    # original permute->reshape->flatten roundtrip is identity), so all existing
    # paths are unaffected.
    # ------------------------------------------------------------------
    def patch_embed_tokens(self, x):
        """HMR2 patch tokens (with positional embedding). Returns (tokens, Hp, Wp).

        tokens: [B, N, C] with N = Hp*Wp (192 for 256x192 crop). If the backbone
        is frozen this runs under .eval(); the graph is kept so gradient can reach
        a downstream trainable unifier (the patch_embed itself is frozen/param-
        free of trainables under FREEZE_CORE).
        """
        x = self._preprocess(x)
        vit = self.backbone
        if self.freeze:
            vit.eval()
        tokens, (Hp, Wp) = vit.patch_embed(x)
        if vit.pos_embed is not None:
            tokens = tokens + vit.pos_embed[:, 1:] + vit.pos_embed[:, :1]
        return tokens, Hp, Wp

    def build_ir_patch_stem(self):
        """P2b: a TRAINABLE deep copy of the frozen patch_embed, for an IR-specific
        stem. Initialised identical to the frozen conv so a residual blend against
        the frozen tokens is exactly 0 at step-0 (bit-exact frozen behaviour); it
        then learns an IR-native first-layer projection, fixing the RGB/IR domain
        mismatch that the shared frozen RGB patch_embed causes.
        """
        import copy
        stem = copy.deepcopy(self.backbone.patch_embed)
        for p in stem.parameters():
            p.requires_grad = True
        return stem

    def patch_embed_tokens_with_stem(self, x, stem, raw_scale, max_scale):
        """Like patch_embed_tokens but residual-blends a trainable IR stem:
            tokens = frozen + scale * (stem - frozen),  scale = max_scale*sigmoid(raw_scale)
        At init stem == frozen -> residual 0 -> step-0 identity; scale>0 (sigmoid(0)
        =0.5) avoids slow-start so the stem gets gradient as soon as it diverges.
        The frozen branch is a no_grad constant; gradient flows to stem + raw_scale.
        """
        x = self._preprocess(x)
        vit = self.backbone
        if self.freeze:
            vit.eval()
        with torch.no_grad():
            frozen_tokens, (Hp, Wp) = vit.patch_embed(x)
        stem_tokens, _ = stem(x)
        scale = max_scale * torch.sigmoid(raw_scale)
        tokens = frozen_tokens + scale * (stem_tokens - frozen_tokens)
        if vit.pos_embed is not None:
            tokens = tokens + vit.pos_embed[:, 1:] + vit.pos_embed[:, :1]
        return tokens, Hp, Wp

    def forward_from_tokens(self, tokens, Hp=None, Wp=None):
        """Continue the HMR2 ViT transformer body from externally supplied tokens.

        Input tokens: [B, N, C]. Output: [B, N, C] (after last_norm). NOT wrapped
        in torch.no_grad even when frozen, so gradient flows from the loss back
        through the frozen body to the unifier that produced the tokens.
        """
        vit = self.backbone
        if self.freeze:
            vit.eval()
        x = tokens
        for blk in vit.blocks:
            x = blk(x)
        x = vit.last_norm(x)
        return x

    def forward_from_tokens_with_adapters(self, tokens, body_adapters,
                                          adapter_block_indices, adapter_mask,
                                          Hp=None, Wp=None):
        """Like forward_from_tokens but injects per-block residual adapters (Scheme 2).

        body_adapters: list/ModuleList of ResidualBodyBlockAdapter.
        adapter_block_indices: sorted list of block indices, e.g. [28,29,30,31].
        adapter_mask: [B,1,1] float — 1 for RGBIR rows, 0 for RGB-only rows.
            rgb_only rows receive residual=0 structurally (bit-exact frozen body).

        ViT has no cross-sample operations (self-attention is per-sequence),
        so per-row masking via adapter_mask is mathematically equivalent to
        running rgb_only and rgbir sub-batches separately.
        """
        vit = self.backbone
        if self.freeze:
            vit.eval()
        x = tokens
        adapter_map = {idx: ada
                       for idx, ada in zip(adapter_block_indices, body_adapters)}
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in adapter_map:
                x = adapter_map[i](x, adapter_mask)
        x = vit.last_norm(x)
        return x

    def collect_block_feats(self, tokens, block_indices):
        """跑完整個凍結 ViT，回傳指定 block 輸出 {idx: [B,N,C]}（深層跨模態 adapter 的參考串流）。"""
        vit = self.backbone
        if self.freeze:
            vit.eval()
        want = set(block_indices)
        feats = {}
        x = tokens
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in want:
                feats[i] = x
            if len(feats) == len(want):
                break
        return feats

    def forward_from_tokens_xmodal(self, tokens, body_adapters, body_block_indices,
                                   adapter_mask, xmodal_adapters, xmodal_block_indices,
                                   xmodal_mask, ref_feats, Hp=None, Wp=None):
        """forward_from_tokens_with_adapters + 深層跨模態 adapter。
        每個 block 依序：blk -> body adapter（若有）-> 跨模態 adapter（若有，讀 ref_feats[i]）。
        回傳 (tokens, [各跨模態 adapter 的 delta])。"""
        vit = self.backbone
        if self.freeze:
            vit.eval()
        body_map = ({idx: ada for idx, ada in zip(body_block_indices, body_adapters)}
                    if adapter_mask is not None else {})
        xm_map = {idx: ada for idx, ada in zip(xmodal_block_indices, xmodal_adapters)}
        deltas = []
        x = tokens
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in body_map:
                x = body_map[i](x, adapter_mask)
            if i in xm_map:
                x, d = xm_map[i](x, ref_feats[i], xmodal_mask)
                deltas.append(d)
        x = vit.last_norm(x)
        return x, deltas


class HMR2Core(nn.Module):
    """HMR2.0 backbone + head, returning the project 3-tuple."""

    def __init__(self, smpl_mean_params, input_height=256, input_width=192,
                 freeze=False, head_kwargs=None, grad_through_frozen=False):
        super().__init__()
        self.backbone = HMR2Encoder(input_height, input_width, freeze=freeze,
                                    grad_through_frozen=grad_through_frozen)
        self.head = SMPLTransformerDecoderHead(smpl_mean_params, **(head_kwargs or {}))
        # Attributes the trainer / MU wrapper read off the model.
        self.camera_rep = 'weak_perspective'
        self.feat_dim = 1280
        self.backbone_type = 'hmr2_vit'
        self.head_type = 'hmr2_transformer_decoder'
        self.weak_cam_scale_min = 1e-4
        self.weak_cam_scale_max = None

    def forward(self, x, bbox_info=None, **kwargs):
        tokens = self.backbone(x)              # [B, 192, 1280]
        return self.head(tokens)               # (rotmat, betas, cam)


def build_hmr2_core(smpl_mean_params, ckpt_path=None, freeze=False, head_kwargs=None,
                    strict=True, grad_through_frozen=False):
    """Build HMR2Core and (optionally) load HMR2.0 pretrained weights.

    ckpt key remap: ``backbone.*`` -> core.backbone.backbone (ViT);
    ``smpl_head.*`` -> core.head.
    """
    core = HMR2Core(smpl_mean_params, freeze=freeze, head_kwargs=head_kwargs,
                    grad_through_frozen=grad_through_frozen)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        bb = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        hd = {k[len('smpl_head.'):]: v for k, v in sd.items() if k.startswith('smpl_head.')}
        r_bb = core.backbone.backbone.load_state_dict(bb, strict=strict)
        r_hd = core.head.load_state_dict(hd, strict=strict)
        print(f'[HMR2Core] loaded ckpt={ckpt_path}')
        print(f'[HMR2Core] backbone: ckpt_keys={len(bb)} '
              f'missing={len(getattr(r_bb, "missing_keys", []))} '
              f'unexpected={len(getattr(r_bb, "unexpected_keys", []))}')
        print(f'[HMR2Core] head: ckpt_keys={len(hd)} '
              f'missing={len(getattr(r_hd, "missing_keys", []))} '
              f'unexpected={len(getattr(r_hd, "unexpected_keys", []))}')
    return core

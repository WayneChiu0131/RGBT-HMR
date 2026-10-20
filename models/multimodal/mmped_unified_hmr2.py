"""MMPedUnifiedHMR2 (Stage2C) — HMR2-integrated unified RGB-T encoder wrapper.

    img_rgb / img_ir
        -> HMR2 patch_embed_tokens (shared, frozen)  -> rgb / ir patch tokens
        -> MMPedStyleHMR2Unifier                      -> unified tokens [B,192,1280]
        -> HMR2 forward_from_tokens (frozen body)     -> refined tokens
        -> HMR2 head                                  -> (rotmat, betas, cam)

Gradient path: the unifier sits BETWEEN the frozen patch stem and the frozen
transformer body. forward_from_tokens / the head run WITHOUT torch.no_grad
(params frozen, graph kept), so gradient flows from the loss back through the
frozen body to the unifier. Same 3-tuple / last_gate_info interface as the other
modality wrappers, so the paired dataloader + train_step + multi-mode validation
work unchanged.

v1 uses the SHARED frozen HMR2 patch_embed for both modalities (RGB patch tokens
== HMR2's exactly; a trainable IR patch stem is deferred to an ablation).
"""

import math
import torch
import torch.nn as nn


class IRPatchStem(nn.Module):
    """P2b: trainable IR-specific patch stem. Wraps a trainable deep copy of the
    frozen HMR2 patch_embed (built via HMR2Encoder.build_ir_patch_stem) plus a
    bounded residual scale. The residual blend (see patch_embed_tokens_with_stem)
    is 0 at step-0 (stem == frozen), so it is a strict superset of the frozen
    behaviour, then learns an IR-native first-layer projection.
    """

    def __init__(self, stem_module, max_scale=1.0, init_raw_scale=0.0):
        super().__init__()
        self.stem = stem_module
        self.max_scale = float(max_scale)
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))


class CrossModalDeepAdapter(nn.Module):
    """深層跨模態 adapter（2026-10-03，參考 BAT 的 Bi_direct_adapter 與 ViPT 的 prompter）。

    插在凍結 ViT-H 的第 i 個 block 之後：融合串流 x 讀取同一層、同一位置的
    凍結 HMR2.0 RGB 串流特徵 r，學習兩者在語意層級的關係，輸出殘差加回 x：
        x <- x + mask * up(GELU(down([LN(x), LN(r)])))
    up 零初始化 -> step-0 恆等（輸出與未加 adapter 的模型逐位元相同）；
    down 一般初始化，所以 up 從第一步就有梯度（BAT 原版三層全零初始化，
    梯度只能從 bias 慢慢長出來，這裡不照抄）。
    mask 只對 RGB+IR 列為 1：單模態列結構上不受影響。
    """

    def __init__(self, dim=1280, bottleneck_dim=128, dropout=0.0, max_rel_scale=0.0):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.down = nn.Linear(2 * dim, bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        # >0：殘差的逐 token RMS 不得超過 max_rel_scale x（該 token 的 RMS）。
        # 2026-10-04：無上限版本 1k 步內殘差 RMS 0.03->0.30，測試集 2D 11.3->18.6 px
        # （訓練集只 +0.4 px）= 過擬合訓練場景，故加相對上限。
        self.max_rel_scale = float(max_rel_scale)

    def forward(self, x, r, mask):
        h = self.act(self.down(torch.cat([self.norm_x(x), self.norm_r(r)], dim=-1)))
        d = self.up(self.drop(h))
        if self.max_rel_scale > 0:
            d_rms = (d.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            x_rms = (x.detach().pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            d = d * torch.clamp(self.max_rel_scale * x_rms / d_rms, max=1.0)
        return x + mask * d, d


class CrossModalDeepConvexGate(nn.Module):
    """深層凸組合 gate（2026-10-04）：在凍結 ViT 的第 i 層，把融合串流 x 與
    凍結 HMR2.0 RGB 串流 r 做逐 token 凸組合：x <- x + mask * beta * (r - x)。
    beta = sigmoid(MLP([LN(x), LN(r)]))，最後一層權重零初始化、bias = init_logit
    （預設 -4 -> beta 約 0.018，接近恆等但梯度正常）。
    輸出永遠介於兩組真實的深層特徵之間 -> 沿用本專案「凸組合安全」的原則，
    容量也遠小於加法 adapter（只決定混合比例，不生成新特徵）。
    """

    def __init__(self, dim=1280, hidden_dim=128, init_logit=-4.0):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, float(init_logit))

    def forward(self, x, r, mask):
        beta = torch.sigmoid(self.mlp(torch.cat([self.norm_x(x), self.norm_r(r)], dim=-1)))
        return x + mask * beta * (r - x), beta


def build_xmodal_deep_modules(blocks, kind='adapter', bottleneck_dim=128, dropout=0.0,
                              max_rel_scale=0.0, init_logit=-4.0):
    """MODEL.XMODAL_DEEP_TYPE: 'adapter'（加法，CrossModalDeepAdapter）| 'convex'（凸組合 gate）。"""
    if kind == 'convex':
        mods = [CrossModalDeepConvexGate(1280, bottleneck_dim, init_logit) for _ in blocks]
    elif kind == 'adapter':
        mods = [CrossModalDeepAdapter(1280, bottleneck_dim, dropout, max_rel_scale) for _ in blocks]
    else:
        raise ValueError(f'XMODAL_DEEP_TYPE={kind!r}')
    return nn.ModuleList(mods)


class BoundedGlobalOrientResidual(nn.Module):
    """Tier-1：對「凍結 SMPL head」輸出的 global orient（joint 0）加有界殘差。

    ── 為什麼需要 ──────────────────────────────────────────────────────────
    凍結的 head 決定了 theta/beta 的輸出空間；token 端的模組（gate / adapters /
    CMT）只能間接影響它。既有證據：
      * `ir_core_go` 診斷模式的存在（global orient 由 frozen core 決定較好）
      * 09.03E/pgtw：把 POSE_LOSS_WEIGHT 1.0->2.0，LLVIP IR-only 3D-MPJPE
        -3.68mm 但 PA +0.52 —— 兩者之差正是 global orient，證明這個方向
        「有 headroom 但 token 端只能搆到一部分」

    ── 為什麼不直接解凍 head ───────────────────────────────────────────────
    `stage2c_scheme3c_v6pck_unfreezehead_l2sp` 已試過：HEAD_LR=3e-6 + L2SP=1e-3
    仍在 1000 步內從 13.30 發散到 18.77（再一次 23.83）。解凍還會破壞
    `rgb_only ≡ HMR2.0` 的逐位元保證（實測 Δ=0.000）。

    ── 為什麼用 6D 而不是 axis-angle ──────────────────────────────────────
    `utils.geometry.batch_rodrigues` 在零向量處雖有 +1e-8 保護不會 NaN，但
    `theta / ||theta+eps||` 的梯度是 1/1.73e-8 ~ 5.8e7 量級，數值條件極差
    （與 2026-09-21 rms_clip 的 0*inf=NaN 同類陷阱）。
    改走 6D：把 R 的前兩欄取出當 6D、加上有界殘差、再過 rot6d_to_rotmat 的
    Gram-Schmidt。全程沒有近零除法，且輸出自動落在 SO(3)。
    實測往返精度 3.6e-7；6D 擾動 L2=0.2 -> 旋轉角中位 8.0 度、最大 11.2 度。

    ── 安全性質（與 ResidualBodyBlockAdapter / BoundedIRTokenAdapter 同模式）
      * 末層零初始化 -> step-0 delta=0 -> 恆等，最壞情況退回 baseline
      * s = max_delta * sigmoid(raw_scale) -> 幅度有硬上限，不可能發散
      * row_mask 門控 -> rgb_only 列不套用，RGB 的逐位元保留不受影響
    """

    def __init__(self, dim=1280, bottleneck_dim=128, max_delta=0.2,
                 init_raw_scale=0.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, int(bottleneck_dim)), nn.GELU(),
            nn.Linear(int(bottleneck_dim), 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))

    def forward(self, feat, R_global, row_mask=None):
        """feat [B,D] 池化後的 token 特徵；R_global [B,3,3]；
        row_mask [B] （1=套用、0=恆等）。回傳 (R_new [B,3,3], aux)。"""
        from models.hmr2_core.geometry6d import rot6d_to_rotmat
        d = torch.tanh(self.net(feat))                       # [B,6] 逐元素 in [-1,1]
        d = d / math.sqrt(6.0)                               # -> ||d||_2 <= 1
        scale = self.max_delta * torch.sigmoid(self.raw_scale)
        delta = scale * d
        base6 = torch.cat([R_global[:, :, 0], R_global[:, :, 1]], dim=1)
        R_new = rot6d_to_rotmat((base6 + delta).to(base6.dtype))
        if row_mask is not None:
            # torch.where（而非把 delta 乘 0）：Gram-Schmidt 的 float32 往返誤差
            # 約 1.8e-7，經 acos 在 1 附近放大後會變成 ~0.03 度的假旋轉。
            # 被遮蔽的列必須是「精確」恆等，不是「近似」恆等。
            R_new = torch.where(
                row_mask.reshape(-1, 1, 1).bool(), R_new, R_global)
        with torch.no_grad():
            Rrel = torch.bmm(R_new, R_global.transpose(1, 2))
            cos = ((Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]) - 1.0) * 0.5
            ang = torch.acos(cos.clamp(-1.0, 1.0)) * (180.0 / math.pi)
        aux = {'go_residual_scale': scale.detach(),
               'go_residual_delta_l2': delta.detach().norm(dim=1).mean(),
               'go_residual_angle_deg': ang.mean()}
        return R_new, aux


class MMPedUnifiedHMR2(nn.Module):

    def __init__(self, hmr2_core, unified_encoder, ir_token_adapter=None,
                 body_adapters=None, body_adapter_block_indices=None,
                 ir_patch_stem=None,
                 freeze_hmr2_body=True, freeze_head=True,
                 go_residual=None,
                 xmodal_deep_adapters=None, xmodal_deep_block_indices=None):
        super().__init__()
        self.backbone = hmr2_core.backbone   # HMR2Encoder (patch_embed_tokens/forward_from_tokens)
        self.head = hmr2_core.head
        self.unified_encoder = unified_encoder
        # Stage B: optional bounded IR token adapter (thermal-domain correction).
        # Applied to IR patch tokens BEFORE the unifier, so IR-only rows (which the
        # unifier hard-routes to the IR tokens) train it; None keeps Stage A.
        self.ir_token_adapter = ir_token_adapter
        # P2b: optional trainable IR patch stem (first-layer domain-mismatch fix).
        # Applied to IR BEFORE patch tokens are formed; None keeps the shared
        # frozen RGB patch_embed for IR (v1 behaviour).
        self.ir_patch_stem = ir_patch_stem
        # Scheme 2: modality-conditional residual adapters inside the frozen body.
        # Inserted AFTER specific ViT blocks (e.g. last 4 of 32).
        # rgb_only rows get adapter_mask=0 -> residual=0 -> bit-exact frozen HMR2.
        import torch.nn as _nn
        self.body_adapters = (body_adapters if body_adapters is not None
                              else _nn.ModuleList())
        self._body_adapter_block_indices = (list(body_adapter_block_indices)
                                            if body_adapter_block_indices else [])

        # proxy attributes the trainer reads off self.model
        self.camera_rep = getattr(hmr2_core, 'camera_rep', 'weak_perspective')
        self.feat_dim = getattr(hmr2_core, 'feat_dim', 1280)
        self.backbone_type = getattr(hmr2_core, 'backbone_type', 'hmr2_vit')
        self.head_type = getattr(hmr2_core, 'head_type', 'hmr2_transformer_decoder')
        self.weak_cam_scale_min = getattr(hmr2_core, 'weak_cam_scale_min', 1e-4)
        self.weak_cam_scale_max = getattr(hmr2_core, 'weak_cam_scale_max', None)

        self.freeze_hmr2_body = bool(freeze_hmr2_body)
        self.freeze_head = bool(freeze_head)
        # Tier-1 global-orient bounded residual（見 BoundedGlobalOrientResidual）。
        # None = 停用 -> 行為與既有 run 逐位元相同。
        self.go_residual = go_residual
        # 深層跨模態 adapter（見 CrossModalDeepAdapter）。None/空 = 停用 -> 逐位元同舊行為。
        self.xmodal_deep_adapters = (xmodal_deep_adapters if xmodal_deep_adapters is not None
                                     else _nn.ModuleList())
        self._xmodal_deep_block_indices = (list(xmodal_deep_block_indices)
                                           if xmodal_deep_block_indices else [])
        # 2026-10-07 消融：DCG 參考分支的輸入。'raw' = 原始 Z_R（凍結 HMR2.0 的 RGB
        # 串流，預設，逐位元同舊行為）；'adapt' = ITA(Z_R)，即 RGB 也先經過 ITA。
        # MODEL.XMODAL_DEEP_REF_INPUT。
        self.xmodal_ref_input = 'raw'
        self.last_gate_info = None
        self.last_aux_info = None

        if self.freeze_hmr2_body:
            for p in self.backbone.backbone.parameters():
                p.requires_grad = False
        if self.freeze_head:
            for p in self.head.parameters():
                p.requires_grad = False

    def forward(self, x=None, bbox_info=None, img_rgb=None, img_ir=None,
                modality_mask=None, mode=None, return_gate=False):
        # ---- legacy single-image (RGB-only) path ----
        if img_rgb is None and img_ir is None:
            if x is None:
                raise ValueError('MMPedUnifiedHMR2.forward: provide legacy `x` or '
                                 '`img_rgb`/`img_ir`.')
            tokens, Hp, Wp = self.backbone.patch_embed_tokens(x)
            out = self.head(self.backbone.forward_from_tokens(tokens, Hp, Wp))
            self.last_gate_info = None
            self.last_aux_info = None
            return (out, None) if return_gate else out

        ref = img_rgb if img_rgb is not None else img_ir
        B = ref.shape[0]
        if modality_mask is None:
            has_rgb = 1.0 if img_rgb is not None else 0.0
            has_ir = 1.0 if img_ir is not None else 0.0
            modality_mask = torch.tensor(
                [[has_rgb, has_ir]], device=ref.device, dtype=ref.dtype
            ).expand(B, 2).contiguous()
        else:
            modality_mask = modality_mask.to(ref.device).float()
        if img_rgb is None:
            img_rgb = torch.zeros_like(ref)
        if img_ir is None:
            img_ir = torch.zeros_like(ref)
        if img_ir.shape[1] == 1:
            img_ir = img_ir.repeat(1, 3, 1, 1)

        tokens_rgb, Hp, Wp = self.backbone.patch_embed_tokens(img_rgb)
        # P2b: IR gets its own trainable patch stem (residual-blended, step-0 = frozen);
        # None -> shared frozen RGB patch_embed. The legacy x= path above is left on the
        # pure frozen stem so baseline_ir stays bit-exact original HMR2.0.
        if self.ir_patch_stem is not None:
            tokens_ir, _, _ = self.backbone.patch_embed_tokens_with_stem(
                img_ir, self.ir_patch_stem.stem,
                self.ir_patch_stem.raw_scale, self.ir_patch_stem.max_scale)
        else:
            tokens_ir, _, _ = self.backbone.patch_embed_tokens(img_ir)

        # Stage B: bounded IR token adaptation before fusion. The unifier routes
        # IR-only rows to these adapted tokens, so IR-only / OTP2 samples train it.
        ir_aux = None
        if self.ir_token_adapter is not None:
            tokens_ir, ir_aux = self.ir_token_adapter(tokens_ir)

        tokens_uni, gate_info, uni_aux = self.unified_encoder(
            tokens_rgb, tokens_ir, modality_mask)

        # Scheme 2: per-sample adapter_mask — RGBIR rows get residual, rgb_only rows get 0.
        # ViT has no cross-sample ops, so per-row masking is mathematically safe.
        xm_aux = {}
        if len(self.xmodal_deep_adapters) > 0:
            # RGB 參考串流 = 凍結 HMR2.0 對原始 RGB patch token 的中間特徵（不需梯度）。
            both = ((modality_mask[:, 0] > 0.5) & (modality_mask[:, 1] > 0.5))
            xm_mask = both.to(tokens_uni.dtype).view(B, 1, 1)
            with torch.no_grad():
                ref_tokens = tokens_rgb
                if self.xmodal_ref_input == 'adapt' and self.ir_token_adapter is not None:
                    ref_tokens, _ = self.ir_token_adapter(tokens_rgb)
                ref_feats = self.backbone.collect_block_feats(
                    ref_tokens, self._xmodal_deep_block_indices)
            if len(self.body_adapters) > 0 and len(self._body_adapter_block_indices) > 0:
                adapter_mask = (modality_mask[:, 1] > 0.5).float().view(B, 1, 1)
            else:
                adapter_mask = None
            tokens_final, xm_deltas = self.backbone.forward_from_tokens_xmodal(
                tokens_uni, self.body_adapters, self._body_adapter_block_indices,
                adapter_mask, self.xmodal_deep_adapters,
                self._xmodal_deep_block_indices, xm_mask, ref_feats)
            if both.any():
                rms = [d[both].detach().pow(2).mean().sqrt() for d in xm_deltas]
                xm_aux = {'xmodal_deep_delta_rms_mean': torch.stack(rms).mean(),
                          'xmodal_deep_delta_rms_max': torch.stack(rms).max()}
                if xm_deltas and xm_deltas[0].shape[-1] == 1:   # 凸組合 gate：delta 即 beta
                    bt = torch.stack([d[both].detach().mean() for d in xm_deltas])
                    xm_aux['xmodal_deep_beta_mean'] = bt.mean()
                    xm_aux['xmodal_deep_beta_std'] = torch.cat(
                        [d[both].detach().flatten() for d in xm_deltas]).std()
        elif len(self.body_adapters) > 0 and len(self._body_adapter_block_indices) > 0:
            if getattr(self, 'force_body_adapter_all_rows', False):
                # 推理期診斷開關（2026-09-14）：強制 adapter_mask=1，讓 rgb_only 的列
                # 也吃到 body adapter 殘差。預設 False -> 訓練與既有評估行為不變。
                # 用途：量化「RGB-only 之所以與凍結 HMR2.0 逐位元相同，是因為
                # adapter 被 mask 掉」這個設計選擇的代價/收益。
                adapter_mask = torch.ones(B, 1, 1, device=tokens_uni.device,
                                          dtype=tokens_uni.dtype)
            else:
                adapter_mask = (modality_mask[:, 1] > 0.5).float().view(B, 1, 1)
            tokens_final = self.backbone.forward_from_tokens_with_adapters(
                tokens_uni, self.body_adapters, self._body_adapter_block_indices,
                adapter_mask, Hp, Wp)
        else:
            tokens_final = self.backbone.forward_from_tokens(tokens_uni, Hp, Wp)
        out = self.head(tokens_final)

        # ---- Tier-1: bounded global-orient residual on the FROZEN head ----
        # 只作用於融合路徑；上面的 legacy `x=` 路徑不套用，所以
        # baseline_rgb / rgb_only（走 legacy）仍與原始 HMR2.0 逐位元相同。
        # row_mask 沿用 body_adapters 的語意（IR 在場才套用）-> 融合路徑的
        # rgb_only 列也維持恆等。
        go_aux = {}
        if self.go_residual is not None:
            pred_rotmat, pred_betas, pred_cam = out
            go_mask = (modality_mask[:, 1] > 0.5).to(pred_rotmat.dtype)
            R0_new, go_aux = self.go_residual(
                tokens_final.mean(dim=1), pred_rotmat[:, 0], go_mask)
            pred_rotmat = torch.cat(
                [R0_new.unsqueeze(1), pred_rotmat[:, 1:]], dim=1)
            out = (pred_rotmat, pred_betas, pred_cam)

        self.last_gate_info = gate_info
        # LIVE (non-detached) gate loss terms for Scheme-1 anchor / oracle losses.
        # The trainer reads this right after forward and adds the gate losses to
        # the total loss before backward. None in legacy (global-gate) mode.
        self.last_gate_loss_terms = uni_aux.get('gate_loss')
        self.last_aux_info = {
            **go_aux,
            **xm_aux,
            'tokens_rgb_patch': tokens_rgb.detach(),
            'tokens_ir_patch': tokens_ir.detach(),
            'tokens_uni': tokens_uni.detach(),
            'tokens_final': tokens_final.detach(),
            'x_base': uni_aux.get('x_base'),
            'modality_mask': modality_mask.detach(),
        }
        if ir_aux is not None:
            self.last_aux_info.update(ir_aux)
        return (out, gate_info) if return_gate else out

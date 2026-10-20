"""MMPedestron-style HMR2-integrated unified RGB-T token encoder (Stage2C).

Inspired by MMPedestron's modality-unified encoder (MAA/MAF learnable tokens,
masked cross-modal attention, confidence fusion, modality dropout) but adapted
to be an HMR2-COMPATIBLE patch-token stem: it operates on HMR2 patch tokens and
returns HMR2-compatible tokens [B, 192, 1280] that the FROZEN HMR2 transformer
body + head can still decode.

Safety (HMR2 frozen head/body is distribution-sensitive; cf. E2 divergence):
  * HARD exact routing for unimodal rows: rgb_only -> raw rgb patch tokens,
    ir_only -> raw ir patch tokens (bypass the transformer entirely) -> RGB-only
    is structurally frozen and can never degrade.
  * transformer block output projections are ZERO-INIT -> at step-0 the blocks
    are identity, so unimodal == frozen HMR2 exactly and RGBIR == a plain
    confidence-weighted patch blend (no manifold shift at init).
  * MAA residual is BOUNDED (|scale| <= MAA_RESIDUAL_MAX_SCALE) and near-zero
    init -> cannot push tokens off manifold the way E2's free residual did.
  * confidence is a masked softmax initialised toward IR (LLVIP IR-only beats
    RGB-only), normalised by sum of valid confidences.
"""

import math
import torch
import torch.nn as nn


class BoundedIRTokenAdapter(nn.Module):
    """Bounded residual IR token adapter (Stage B): x -> x + s*tanh(MLP(LN(x))),
    s = max_scale * sigmoid(raw_scale). Lets IR-only / OTP2 samples train an IR
    thermal-domain correction WITHOUT pushing tokens off the frozen HMR2 manifold
    (tanh delta + bounded scale + zero-init last layer -> step-0 identity).
    init_raw_scale=-4 -> s≈0.018*max_scale at init (small but non-negligible, so
    the adapter engages faster than the maa-residual's -8 slow start)."""

    def __init__(self, dim=1280, bottleneck_dim=64, max_scale=0.005,
                 init_raw_scale=-4.0):
        super().__init__()
        self.max_scale = float(max_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))

    def forward(self, x):
        delta = torch.tanh(self.net(x))
        scale = self.max_scale * torch.sigmoid(self.raw_scale)
        aux = {'ir_adapter_scale': scale.detach(),
               'ir_adapter_delta_l2': delta.detach().pow(2).mean().sqrt()}
        return x + scale * delta, aux


def build_bounded_ir_token_adapter(cfg):
    return BoundedIRTokenAdapter(
        dim=int(_cfg_get(cfg, 'DIM', 1280)),
        bottleneck_dim=int(_cfg_get(cfg, 'BOTTLENECK_DIM', 64)),
        max_scale=float(_cfg_get(cfg, 'MAX_SCALE', 0.005)),
        init_raw_scale=float(_cfg_get(cfg, 'INIT_RAW_SCALE', -4.0)),
    )


class ResidualBodyBlockAdapter(nn.Module):
    """Per-block residual adapter for the frozen HMR2 ViT body (Scheme 2).

    Injects a modality-conditional bounded residual AFTER a frozen transformer
    block: x -> x + adapter_mask * scale * tanh(net(LN(x))). RGB-only rows
    receive adapter_mask=0 -> residual=0 -> bit-exact frozen HMR2 output
    (structural protection, no branch needed).

    init_raw_scale=0.0 -> effective_scale = max_scale * sigmoid(0) = max_scale/2
    (avoids the slow-start dead-loop seen at raw_scale=-8/-4: gradient is
    proportional to scale, so tiny scale -> tiny delta -> tiny gradient -> scale
    stays tiny; starting at 0.0 breaks the loop). Zero-init last Linear ->
    delta=0 at step-0 -> identity for all rows regardless of adapter_mask.
    """

    def __init__(self, dim=1280, bottleneck_dim=64, max_scale=0.01,
                 init_raw_scale=0.0):
        super().__init__()
        self.max_scale = float(max_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))

    def forward(self, x, adapter_mask):
        """x: [B,N,D], adapter_mask: [B,1,1] float (1=RGBIR, 0=rgb_only)."""
        delta = torch.tanh(self.net(x))
        scale = self.max_scale * torch.sigmoid(self.raw_scale)
        return x + adapter_mask * scale * delta

    def effective_scale(self):
        return float(self.max_scale * torch.sigmoid(self.raw_scale).item())


def build_body_adapters(cfg):
    """Build a list of ResidualBodyBlockAdapter for the last-K ViT body blocks.

    Returns (adapters: nn.ModuleList, block_indices: list[int]).
    BODY_ADAPTER config keys:
      NUM_BLOCKS (default 4) -> last-K blocks of 32-deep ViT-H = [28,29,30,31]
      TOTAL_BLOCKS (default 32) -> ViT depth (for computing start index)
      BOTTLENECK_DIM (default 64)
      MAX_SCALE (default 0.01)
      INIT_RAW_SCALE (default 0.0)
    """
    num_blocks = int(_cfg_get(cfg, 'NUM_BLOCKS', 4))
    total_blocks = int(_cfg_get(cfg, 'TOTAL_BLOCKS', 32))
    bottleneck_dim = int(_cfg_get(cfg, 'BOTTLENECK_DIM', 64))
    max_scale = float(_cfg_get(cfg, 'MAX_SCALE', 0.01))
    init_raw_scale = float(_cfg_get(cfg, 'INIT_RAW_SCALE', 0.0))
    dim = int(_cfg_get(cfg, 'DIM', 1280))
    start_idx = total_blocks - num_blocks
    block_indices = list(range(start_idx, total_blocks))
    adapters = nn.ModuleList([
        ResidualBodyBlockAdapter(
            dim=dim, bottleneck_dim=bottleneck_dim,
            max_scale=max_scale, init_raw_scale=init_raw_scale)
        for _ in range(num_blocks)
    ])
    return adapters, block_indices


class CrossModalTokenResidual(nn.Module):
    """Bounded PER-TOKEN cross-modal residual (2026-09-11 fix).

    ── 為什麼需要這個 ───────────────────────────────────────────────────
    原本 cross-modal transformer 的輸出只有兩條出路，兩條都廢掉了：
      * `z[:, 2:]`（逐 token 的 contextualized 跨模態特徵）→ **整組丟棄**
      * `x_maa` → bounded MAA residual，但 `x_maa[:, None, :]` 是 **rank-1 廣播**
        （對全部 N 個 token 加同一個向量），且 raw_scale 硬編碼 -8.0 +
        MAX_SCALE 1e-3 → 實測有效尺度 3.77e-7，對輸出的相對擾動 5.8e-8，
        **低於 float32 eps (1.19e-7)**。

    結果是：整個架構只能決定「這個位置用多少 IR / 多少 RGB」（per-token gate
    輸出 1 維純量、且嚴格同位置），無法表達「RGB 的位置 j 告訴我們 IR 的位置 i
    是什麼」——也就是完全沒有非局部的跨模態推理能力。

    ── 為什麼這樣做是安全的 ─────────────────────────────────────────────
    E2 事故（RGBIR 2D-MPJPE 發散到 27px）的成因是用 fusion 模組的輸出**取代**
    原 token（unbounded replacement）。本模組用的是同檔案裡 BoundedIRTokenAdapter /
    ResidualBodyBlockAdapter 已驗證的 bounded-residual 模式：
        x -> x + s * tanh(net(LN(x))),  s = max_scale * sigmoid(raw_scale)
    末層 zero-init -> step-0 delta = 0 -> 恆等；tanh 有界 + s 有上限 ->
    token 永遠停在 frozen HMR2 body 熟悉的 manifold 附近。

    ── init_raw_scale=0.0 的理由 ────────────────────────────────────────
    ResidualBodyBlockAdapter 的 docstring 已記錄這個死迴圈：
      "gradient is proportional to scale, so tiny scale -> tiny delta ->
       tiny gradient -> scale stays tiny; starting at 0.0 breaks the loop"
    MAA residual 就是卡在 raw_scale=-8：40k step 只從 -8.000 爬到 -7.884。
    """

    def __init__(self, dim=1280, bottleneck_dim=128, max_scale=0.01,
                 init_raw_scale=0.0, bound_mode='tanh'):
        super().__init__()
        self.max_scale = float(max_scale)
        # 'tanh'     : delta = tanh(net(x))      -- 舊行為（向後相容）
        # 'rms_clip' : delta = net(x) / max(1, rms(net(x)))
        #   2026-09-21 修正。實測 09.03C/D 全程 xmodal_delta_l2（= 每元素 RMS）
        #   飽和在 0.97-0.99 -> tanh 導數 1-t^2 = 0.04 -> 梯度衰減 25x，且
        #   delta 塌縮成近乎固定的符號樣式（1280 維連續向量 -> 1280 個 +-1），
        #   退化成它原本要取代的 rank-1 廣播偏置。
        #   成因：MAX_SCALE=0.01 封住 scale 上限，優化器改去放大 MLP 權重
        #   （raw_scale 全程往上爬 0.0 -> +0.093，證明它想要更多貢獻）。
        #   rms_clip 在 rms<=1 時是恆等（梯度不衰減、從零平滑成長，保住
        #   bounded-residual 的安全性質），只在超限時投影回 rms=1。
        #   邊界保證與 tanh 同級：||delta||_rms <= 1 -> 擾動 <= max_scale，
        #   對齊已驗證安全的 BoundedIRTokenAdapter（MAX_SCALE 0.05）。
        #   不可改用 LayerNorm：零初始化時 LN(0)=0，但權重一變非零就跳到
        #   unit RMS，破壞「從零平滑成長」——E2 發散事故正是突變造成的。
        self.bound_mode = str(bound_mode)
        assert self.bound_mode in ('tanh', 'rms_clip'), self.bound_mode
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_scale = nn.Parameter(torch.tensor(float(init_raw_scale)))

    def forward(self, z_rgb, z_ir):
        """z_rgb / z_ir: [B,N,D] 跨模態 transformer 處理後的 RGB / IR tokens。
        回傳 (delta_scaled [B,N,D], aux)。呼叫端負責加到 x_fused 上。"""
        raw = self.net(torch.cat([z_rgb, z_ir], dim=-1))
        if self.bound_mode == 'rms_clip':
            # clamp 必須在 sqrt 之前。sqrt 單調且 sqrt(1)=1，所以
            #     sqrt(max(1, ms)) == max(1, sqrt(ms))
            # 數學等價，但 sqrt 的輸入恆 >= 1 -> 導數 <= 0.5，永遠有限。
            # 反過來寫（先 sqrt 再 clamp）會在零初始化時炸掉：raw 全為 0
            # -> d(sqrt)/dx = inf，而 clamp 在裁剪區的梯度是 0 -> 0 * inf = NaN。
            # 2026-09-21 首次上線就是這樣在 step 2 NaN 中止的。
            ms = raw.pow(2).mean(dim=-1, keepdim=True)
            delta = raw / ms.clamp(min=1.0).sqrt()
        else:
            delta = torch.tanh(raw)
        scale = self.max_scale * torch.sigmoid(self.raw_scale)
        aux = {'xmodal_scale': scale.detach(),
               'xmodal_delta_l2': delta.detach().pow(2).mean().sqrt(),
               # 裁剪前的 RMS：>1 表示正在被壓回，<1 表示仍在恆等區
               'xmodal_raw_rms': raw.detach().pow(2).mean().sqrt()}
        return scale * delta, aux


class _UniBlock(nn.Module):
    """Pre-norm transformer encoder block with key_padding_mask, zero-init
    output projections so the block is identity at initialisation."""

    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # zero-init the two residual-branch output projections -> identity block.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class MMPedStyleHMR2Unifier(nn.Module):

    def __init__(self, dim=1280, depth=1, heads=8, mlp_ratio=2.0,
                 conf_init_ir=0.70, maa_residual_max_scale=0.003, dropout=0.0,
                 use_direct_confidence=True, use_maf_confidence=False,
                 conf_max_logit_delta=0.5, use_per_token_confidence=False,
                 per_token_conf_hidden_dim=256, disable_gate=False,
                 maa_residual_init_raw_scale=-8.0,
                 xmodal_residual_enable=False, xmodal_residual_max_scale=0.01,
                 xmodal_residual_bottleneck_dim=128,
                 xmodal_residual_init_raw_scale=0.0,
                 xmodal_residual_bound_mode='tanh',
                 gate_cmt_context=False, cmt_gate_hidden_dim=256, use_cmt=True):
        super().__init__()
        self.dim = int(dim)
        # use_cmt=False（2026-10-05，最終模型 ITA+ARG+DCG）：完全不建 CMT
        # （模態嵌入、MAA/MAF、transformer block、conf_mlp、direct_conf_mlp、MAA 殘差）。
        # 只允許在 CMT 對輸出本來就零影響的設定下使用：per-token gate、
        # MAA_RESIDUAL_MAX_SCALE=0、無 GATE_CMT_CONTEXT / xmodal_residual。
        self.use_cmt = bool(use_cmt)
        if not self.use_cmt:
            if not (use_per_token_confidence and float(maa_residual_max_scale) == 0.0
                    and not gate_cmt_context and not xmodal_residual_enable):
                raise ValueError('USE_CMT=False 只支援 USE_PER_TOKEN_CONFIDENCE=True、'
                                 'MAA_RESIDUAL_MAX_SCALE=0、無 GATE_CMT_CONTEXT / XMODAL_RESIDUAL。')
        self.maa_residual_max_scale = float(maa_residual_max_scale)
        self.use_direct_confidence = bool(use_direct_confidence)
        self.use_maf_confidence = bool(use_maf_confidence)
        self.conf_max_logit_delta = float(conf_max_logit_delta)
        # ABLATION (component study): freeze the reliability gate to its FIXED
        # base blend (alpha_ir == sigmoid(base_ir_logit) == conf_init_ir, constant
        # over tokens/samples). Fusion still happens (convex RGB/IR blend at the
        # init ratio) but the LEARNED adaptivity is removed -> isolates "adaptive
        # per-token gating" vs "fixed blend". Zeroes the per-token / direct / maf
        # confidence deltas so no gate gradient flows.
        self.disable_gate = bool(disable_gate)
        # Scheme-1: per-TOKEN reliability gate (alpha_ir in [B,N,1]) replacing the
        # single global scalar. Each token's RGB/IR blend is decided from that
        # token's own [rgb, ir, |diff|, prod] stats. Zero-init last layer -> at
        # step-0 alpha_ir == sigmoid(base_ir_logit) everywhere == the init blend.
        self.use_per_token_confidence = bool(use_per_token_confidence)
        _p = float(min(max(conf_init_ir, 1e-3), 1 - 1e-3))
        self.register_buffer('base_ir_logit',
                             torch.tensor(math.log(_p / (1.0 - _p))))
        self.per_token_conf_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, int(per_token_conf_hidden_dim)), nn.GELU(),
            nn.Linear(int(per_token_conf_hidden_dim), 1))
        nn.init.zeros_(self.per_token_conf_mlp[-1].weight)
        nn.init.zeros_(self.per_token_conf_mlp[-1].bias)

        if not self.use_cmt:
            self.xmodal_residual = None
            self.gate_cmt_context = False
            return

        # modality embeddings (zero-init -> no perturbation at step-0)
        self.rgb_mod_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.ir_mod_embed = nn.Parameter(torch.zeros(1, 1, dim))
        # MAA (modality-aware abstractor) / MAF (modality-aware fuser) tokens
        self.maa = nn.Parameter(torch.zeros(1, 1, dim))
        self.maf = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([
            _UniBlock(dim, heads, mlp_ratio, dropout) for _ in range(int(depth))])

        p = float(min(max(conf_init_ir, 1e-3), 1 - 1e-3))
        # step-0 base logits -> softmax == (1-p, p). Confidence = base + bounded
        # learned delta, so step-0 == the init blend regardless of head.
        self.register_buffer('conf_base_logits',
                             torch.tensor([math.log(1 - p), math.log(p)]))

        # MAF-token confidence head (transformer-derived). Its input x_maf starts
        # ~0 (zero-init MAA/MAF + identity blocks), which made gate_ir_std=0 in
        # the 1k run — hence use_maf_confidence defaults False.
        self.conf_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 2))
        nn.init.zeros_(self.conf_mlp[-1].weight)
        nn.init.zeros_(self.conf_mlp[-1].bias)

        # DIRECT confidence head: reads pooled RAW token statistics
        # [rgb, ir, |diff|, prod] -> immediately sample-varying gradient, so the
        # confidence becomes per-sample adaptive from step-0. Zero-init last layer
        # -> step-0 delta 0 -> step-0 confidence == base (1-p, p).
        self.direct_conf_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 2))
        nn.init.zeros_(self.direct_conf_mlp[-1].weight)
        nn.init.zeros_(self.direct_conf_mlp[-1].bias)

        # bounded MAA residual scale: effective = max_scale * sigmoid(raw).
        # init 原本硬編碼 -8.0 -> sigmoid = 3.35e-4，配合 MAX_SCALE 1e-3 得到
        # 有效尺度 3.4e-7，梯度正比於 scale -> 死迴圈，40k step 只爬到 -7.884。
        # 改為可設定；新 config 應設 0.0（同 ResidualBodyBlockAdapter 的修正）。
        self.maa_residual_raw_scale = nn.Parameter(
            torch.tensor(float(maa_residual_init_raw_scale)))

        # 逐 token 的 bounded 跨模態殘差（2026-09-11）。預設關閉 -> 既有 config
        # 行為逐位元不變。開啟後才會使用 transformer 的 z[:, 2:] 輸出。
        self.xmodal_residual = (
            CrossModalTokenResidual(
                dim=dim, bottleneck_dim=int(xmodal_residual_bottleneck_dim),
                max_scale=float(xmodal_residual_max_scale),
                init_raw_scale=float(xmodal_residual_init_raw_scale),
                bound_mode=str(xmodal_residual_bound_mode))
            if bool(xmodal_residual_enable) else None)

        # CMT -> ARG 可靠度調節（2026-10-02）。預設關閉 -> 既有 config 逐位元不變。
        # CMT 不再把殘差加到 token 上，而是給 per-token gate 的 logit 加兩項：
        #   cmt_pair_mlp : 每個 RGB-IR pair 一個值（影像層級可靠度）。輸入除了 MAA
        #                  輸出 z_A，還有 CMT 處理後 RGB / IR token 的平均：z_A 在
        #                  step-0 是 0（零初始化的 MAA + identity block），只靠它
        #                  會讓所有 pair 輸入相同（同 MAF 的 gate_ir_std=0 教訓）。
        #   cmt_token_mlp: 每個 token 一個值，輸入為 CMT attention 後的 RGB/IR token。
        # 兩者最後一層零初始化 -> step-0 == 未加 CMT 的 gate；最後一層梯度從第一步
        # 就正常（不像 sigmoid(-8) 縮放會把梯度一起壓掉）。
        # 合成：alpha = sigmoid(b + c * tanh(L_i + A + T_i))，仍是 RAW token 凸組合。
        self.gate_cmt_context = bool(gate_cmt_context)
        if self.gate_cmt_context:
            _h = int(cmt_gate_hidden_dim)
            self.cmt_pair_mlp = nn.Sequential(
                nn.LayerNorm(dim * 3), nn.Linear(dim * 3, _h), nn.GELU(),
                nn.Linear(_h, 1))
            self.cmt_token_mlp = nn.Sequential(
                nn.LayerNorm(dim * 2), nn.Linear(dim * 2, _h), nn.GELU(),
                nn.Linear(_h, 1))
            for _m in (self.cmt_pair_mlp, self.cmt_token_mlp):
                nn.init.zeros_(_m[-1].weight)
                nn.init.zeros_(_m[-1].bias)

    def forward(self, x_rgb, x_ir, modality_mask):
        B, N, D = x_rgb.shape
        w_rgb = modality_mask[:, 0].float()
        w_ir = modality_mask[:, 1].float()
        rgb_only = (w_rgb > 0.5) & (w_ir < 0.5)
        ir_only = (w_rgb < 0.5) & (w_ir > 0.5)
        both = (w_rgb > 0.5) & (w_ir > 0.5)

        if not self.use_cmt:
            return self._forward_no_cmt(x_rgb, x_ir, rgb_only, ir_only, both)

        # ---- cross-modal transformer over [MAA, MAF, RGB(N), IR(N)] ----
        xr = x_rgb + self.rgb_mod_embed
        xi = x_ir + self.ir_mod_embed
        maa = self.maa.expand(B, 1, D)
        maf = self.maf.expand(B, 1, D)
        z = torch.cat([maa, maf, xr, xi], dim=1)          # [B, 2+2N, D]

        # key_padding_mask: True = ignore. Mask the missing modality's N tokens.
        kpm = torch.zeros(B, 2 + 2 * N, dtype=torch.bool, device=x_rgb.device)
        kpm[:, 2:2 + N] = (w_rgb < 0.5).view(B, 1)        # drop RGB block if no RGB
        kpm[:, 2 + N:] = (w_ir < 0.5).view(B, 1)          # drop IR block if no IR
        for blk in self.blocks:
            z = blk(z, key_padding_mask=kpm)

        x_maa = z[:, 0]
        x_maf = z[:, 1]
        # NOTE: the transformer-processed tokens z[:,2:] are deliberately NOT used
        # for the fused output. They are unbounded and, fed into the frozen HMR2
        # body, drift off-manifold and diverge (E2 lesson). Instead the fused
        # output is a CONVEX blend of the RAW patch tokens (structurally on the
        # manifold, between two real token sets); the transformer + MAA/MAF only
        # produce the per-sample confidence WEIGHTS + a bounded MAA residual.

        # gate_loss_terms: LIVE (non-detached) quantities the trainer needs for
        # the gate anchor / oracle-gate losses. Only `both` rows carry a real gate.
        gate_loss_terms = None

        if self.use_per_token_confidence:
            # ---- Scheme-1: per-TOKEN convex fusion ----
            # each token blends by its own reliability alpha_ir_i in [0,1].
            feat = torch.cat([
                x_rgb, x_ir, (x_rgb - x_ir).abs(), x_rgb * x_ir], dim=-1)  # [B,N,4D]
            if self.disable_gate:
                # ablation: fixed blend -> delta 0 -> alpha_ir == base (constant)
                delta = torch.zeros(B, N, 1, device=x_rgb.device, dtype=x_rgb.dtype)
            else:
                gate_raw = self.per_token_conf_mlp(feat)                   # [B,N,1]
                if self.gate_cmt_context:
                    z_r = z[:, 2:2 + N]                                    # CMT 後的 RGB token
                    z_i = z[:, 2 + N:]                                     # CMT 後的 IR token
                    pair_in = torch.cat([x_maa, z_r.mean(dim=1), z_i.mean(dim=1)], dim=-1)
                    cmt_pair = self.cmt_pair_mlp(pair_in)[:, None, :]       # [B,1,1]
                    cmt_tok = self.cmt_token_mlp(torch.cat([z_r, z_i], dim=-1))  # [B,N,1]
                    gate_raw = gate_raw + cmt_pair + cmt_tok
                delta = self.conf_max_logit_delta * torch.tanh(gate_raw)   # [B,N,1]
            logit_ir = self.base_ir_logit.to(x_rgb.dtype) + delta          # [B,N,1]
            alpha_ir = torch.sigmoid(logit_ir)                             # [B,N,1]
            x_fused = (1.0 - alpha_ir) * x_rgb + alpha_ir * x_ir           # convex, RAW tokens
            # per-sample scalars for gate_info + losses
            c_ir = alpha_ir.mean(dim=(1, 2))                               # [B]
            c_rgb = 1.0 - c_ir
            gate_loss_terms = {
                'alpha_sample_logit': logit_ir.mean(dim=(1, 2)),           # [B] LIVE (for oracle BCE)
                'alpha_ir_mean_live': alpha_ir.mean(dim=(1, 2)),           # [B] LIVE (for anchor)
                'both': both,                                              # [B] bool
            }
            # richer per-token gate stats (detached, logging only)
            _both = both.view(B, 1, 1)
            gate_extra = {
                'alpha_ir_mean': alpha_ir[_both.expand_as(alpha_ir)].mean().detach()
                                 if both.any() else torch.zeros((), device=x_rgb.device),
                'alpha_ir_std': (alpha_ir.masked_select(_both).std().detach()
                                 if int(both.sum()) > 1 else torch.zeros((), device=x_rgb.device)),
                'alpha_ir_token_std': (alpha_ir[both].std(dim=1).mean().detach()
                                       if both.any() else torch.zeros((), device=x_rgb.device)),
                # 不同 RGB-IR pair 之間的 alpha 差異（每 pair 先對 token 取平均）
                'alpha_ir_pair_std': (alpha_ir[both].mean(dim=(1, 2)).std().detach()
                                      if int(both.sum()) > 1 else torch.zeros((), device=x_rgb.device)),
            }
            if self.gate_cmt_context and not self.disable_gate:
                gate_extra['cmt_pair_abs'] = (cmt_pair[both].abs().mean().detach()
                                              if both.any() else torch.zeros((), device=x_rgb.device))
                gate_extra['cmt_token_abs'] = (cmt_tok[both].abs().mean().detach()
                                               if both.any() else torch.zeros((), device=x_rgb.device))
        else:
            # ---- legacy global confidence: base logits + bounded learned delta ----
            logits = self.conf_base_logits.to(x_rgb.dtype).view(1, 2).expand(B, 2).clone()
            if self.use_direct_confidence and not self.disable_gate:
                # pooled RAW-token stats -> immediately sample-varying (fixes the
                # gate_ir_std=0 seen when confidence came only from the ~0 x_maf).
                rgb_pool = x_rgb.mean(dim=1)
                ir_pool = x_ir.mean(dim=1)
                diff_pool = (x_rgb - x_ir).abs().mean(dim=1)
                prod_pool = (x_rgb * x_ir).mean(dim=1)
                conf_feat = torch.cat([rgb_pool, ir_pool, diff_pool, prod_pool], dim=-1)
                logits = logits + self.conf_max_logit_delta * torch.tanh(
                    self.direct_conf_mlp(conf_feat))
            if self.use_maf_confidence and not self.disable_gate:
                logits = logits + self.conf_max_logit_delta * torch.tanh(self.conf_mlp(x_maf))

            # ---- masked softmax (normalised over valid modalities) ----
            neg = torch.finfo(logits.dtype).min / 4
            logits = logits.masked_fill(modality_mask < 0.5, neg)
            conf = torch.softmax(logits, dim=-1)
            c_rgb, c_ir = conf[:, 0], conf[:, 1]
            a_rgb = (c_rgb * w_rgb)[:, None, None]
            a_ir = (c_ir * w_ir)[:, None, None]
            denom = a_rgb + a_ir + 1e-6
            x_fused = (a_rgb * x_rgb + a_ir * x_ir) / denom   # convex blend of RAW tokens
            gate_extra = {}

        # bounded MAA residual (broadcast the abstractor over all tokens).
        # 注意：這是 rank-1 廣播，對所有 N 個 token 加同一個向量 -> 無法傳遞
        # 空間變化的跨模態資訊。逐 token 的部分由下面的 xmodal_residual 負責。
        maa_scale = self.maa_residual_max_scale * torch.sigmoid(self.maa_residual_raw_scale)
        x_fused = x_fused + maa_scale * torch.tanh(x_maa[:, None, :])

        # ---- bounded PER-TOKEN cross-modal residual（啟用時）----
        # 使用原本被丟棄的 z[:, 2:]：經過 masked cross-modal attention 的
        # RGB / IR tokens。unimodal 的列會在下面的 hard routing 被覆寫，
        # 所以這條殘差自然只作用於 `both`（RGB+IR）列——語意上正確，
        # 因為單一模態本來就不存在跨模態資訊。
        xmodal_extra = {}
        if self.xmodal_residual is not None:
            xm_delta, xmodal_extra = self.xmodal_residual(z[:, 2:2 + N], z[:, 2 + N:])
            x_fused = x_fused + xm_delta

        # ---- HARD exact routing for unimodal rows (structural protection) ----
        x_uni = x_fused
        if rgb_only.any():
            x_uni = torch.where(rgb_only.view(B, 1, 1), x_rgb, x_uni)
        if ir_only.any():
            x_uni = torch.where(ir_only.view(B, 1, 1), x_ir, x_uni)

        gate_ir = torch.zeros(B, device=x_rgb.device, dtype=x_rgb.dtype)
        gate_rgb = torch.ones(B, device=x_rgb.device, dtype=x_rgb.dtype)
        if both.any():
            gate_ir = torch.where(both, c_ir, gate_ir)
            gate_rgb = torch.where(both, c_rgb, gate_rgb)
        gate_ir = torch.where(ir_only, torch.ones_like(gate_ir), gate_ir)
        gate_rgb = torch.where(ir_only, torch.zeros_like(gate_rgb), gate_rgb)

        gate_info = {
            'gate_rgb': gate_rgb.detach(),
            'gate_ir': gate_ir.detach(),
            'conf_rgb_mean': c_rgb.mean().detach(),
            'conf_ir_mean': c_ir.mean().detach(),
            'conf_ir_std': c_ir.std().detach(),
            'maa_residual_scale': maa_scale.detach(),
        }
        gate_info.update(gate_extra)
        gate_info.update(xmodal_extra)
        aux = {
            'x_base': x_fused.detach(),   # fused-before-routing (for token-shift)
            'x_maa': x_maa.detach(),
            'gate_loss': gate_loss_terms,  # LIVE terms (or None in legacy mode)
        }
        return x_uni, gate_info, aux

    def _forward_no_cmt(self, x_rgb, x_ir, rgb_only, ir_only, both):
        """無 CMT 的前向：與 use_cmt=True 且 MAA_RESIDUAL_MAX_SCALE=0 的輸出逐位元相同
        （原本的 MAA 殘差是 0·tanh(x_maa) = 0）。"""
        B = x_rgb.shape[0]
        feat = torch.cat([x_rgb, x_ir, (x_rgb - x_ir).abs(), x_rgb * x_ir], dim=-1)
        if self.disable_gate:
            delta = torch.zeros(B, x_rgb.shape[1], 1, device=x_rgb.device, dtype=x_rgb.dtype)
        else:
            delta = self.conf_max_logit_delta * torch.tanh(self.per_token_conf_mlp(feat))
        logit_ir = self.base_ir_logit.to(x_rgb.dtype) + delta
        alpha_ir = torch.sigmoid(logit_ir)
        x_fused = (1.0 - alpha_ir) * x_rgb + alpha_ir * x_ir
        c_ir = alpha_ir.mean(dim=(1, 2))
        c_rgb = 1.0 - c_ir
        zero = torch.zeros((), device=x_rgb.device)
        # 以下兩個 dict 與 use_cmt=True 的 per-token 分支逐項相同（trainer 會讀）。
        gate_loss_terms = {
            'alpha_sample_logit': logit_ir.mean(dim=(1, 2)),
            'alpha_ir_mean_live': alpha_ir.mean(dim=(1, 2)),
            'both': both,
        }
        _both = both.view(B, 1, 1)
        gate_extra = {
            'alpha_ir_mean': alpha_ir[_both.expand_as(alpha_ir)].mean().detach() if both.any() else zero,
            'alpha_ir_std': (alpha_ir.masked_select(_both).std().detach()
                             if int(both.sum()) > 1 else zero),
            'alpha_ir_token_std': alpha_ir[both].std(dim=1).mean().detach() if both.any() else zero,
            'alpha_ir_pair_std': (alpha_ir[both].mean(dim=(1, 2)).std().detach()
                                  if int(both.sum()) > 1 else zero),
        }
        x_uni = x_fused
        if rgb_only.any():
            x_uni = torch.where(rgb_only.view(B, 1, 1), x_rgb, x_uni)
        if ir_only.any():
            x_uni = torch.where(ir_only.view(B, 1, 1), x_ir, x_uni)
        gate_ir = torch.zeros(B, device=x_rgb.device, dtype=x_rgb.dtype)
        gate_rgb = torch.ones(B, device=x_rgb.device, dtype=x_rgb.dtype)
        if both.any():
            gate_ir = torch.where(both, c_ir, gate_ir)
            gate_rgb = torch.where(both, c_rgb, gate_rgb)
        gate_ir = torch.where(ir_only, torch.ones_like(gate_ir), gate_ir)
        gate_rgb = torch.where(ir_only, torch.zeros_like(gate_rgb), gate_rgb)
        gate_info = {
            'gate_rgb': gate_rgb.detach(), 'gate_ir': gate_ir.detach(),
            'conf_rgb_mean': c_rgb.mean().detach(), 'conf_ir_mean': c_ir.mean().detach(),
            'conf_ir_std': c_ir.std().detach(), 'maa_residual_scale': zero,
        }
        gate_info.update(gate_extra)
        aux = {'x_base': x_fused.detach(), 'x_maa': None, 'gate_loss': gate_loss_terms}
        return x_uni, gate_info, aux


def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, name):
        v = getattr(cfg, name)
        return v if v is not None else default
    if isinstance(cfg, dict):
        v = cfg.get(name, default)
        return v if v is not None else default
    return default


def build_mmped_hmr2_unified_encoder(cfg):
    return MMPedStyleHMR2Unifier(
        dim=int(_cfg_get(cfg, 'DIM', 1280)),
        depth=int(_cfg_get(cfg, 'DEPTH', 1)),
        heads=int(_cfg_get(cfg, 'HEADS', 8)),
        mlp_ratio=float(_cfg_get(cfg, 'MLP_RATIO', 2.0)),
        conf_init_ir=float(_cfg_get(cfg, 'CONF_INIT_IR', 0.70)),
        maa_residual_max_scale=float(_cfg_get(cfg, 'MAA_RESIDUAL_MAX_SCALE', 0.003)),
        dropout=float(_cfg_get(cfg, 'DROPOUT', 0.0)),
        use_direct_confidence=bool(_cfg_get(cfg, 'USE_DIRECT_CONFIDENCE', True)),
        use_maf_confidence=bool(_cfg_get(cfg, 'USE_MAF_CONFIDENCE', False)),
        conf_max_logit_delta=float(_cfg_get(cfg, 'CONF_MAX_LOGIT_DELTA', 0.5)),
        use_per_token_confidence=bool(_cfg_get(cfg, 'USE_PER_TOKEN_CONFIDENCE', False)),
        per_token_conf_hidden_dim=int(_cfg_get(cfg, 'PER_TOKEN_CONF_HIDDEN_DIM', 256)),
        disable_gate=bool(_cfg_get(cfg, 'DISABLE_GATE', False)),
        # 預設 -8.0 = 舊行為（向後相容）；新 config 應顯式設 0.0。
        maa_residual_init_raw_scale=float(
            _cfg_get(cfg, 'MAA_RESIDUAL_INIT_RAW_SCALE', -8.0)),
        # 預設關閉 = 舊行為（向後相容）。
        xmodal_residual_enable=bool(
            _cfg_get(cfg, 'XMODAL_RESIDUAL_ENABLE', False)),
        xmodal_residual_max_scale=float(
            _cfg_get(cfg, 'XMODAL_RESIDUAL_MAX_SCALE', 0.01)),
        xmodal_residual_bottleneck_dim=int(
            _cfg_get(cfg, 'XMODAL_RESIDUAL_BOTTLENECK_DIM', 128)),
        xmodal_residual_init_raw_scale=float(
            _cfg_get(cfg, 'XMODAL_RESIDUAL_INIT_RAW_SCALE', 0.0)),
        # 預設 'tanh' = 舊行為（向後相容）。'rms_clip' 見 CrossModalTokenResidual。
        xmodal_residual_bound_mode=str(
            _cfg_get(cfg, 'XMODAL_RESIDUAL_BOUND_MODE', 'tanh')),
        # 預設關閉 = 舊行為（向後相容）。見 MMPedStyleHMR2Unifier.gate_cmt_context。
        gate_cmt_context=bool(_cfg_get(cfg, 'GATE_CMT_CONTEXT', False)),
        cmt_gate_hidden_dim=int(_cfg_get(cfg, 'CMT_GATE_HIDDEN_DIM', 256)),
        # 預設 True = 舊行為（向後相容）。False 見 MMPedStyleHMR2Unifier.use_cmt。
        use_cmt=bool(_cfg_get(cfg, 'USE_CMT', True)),
    )

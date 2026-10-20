# CLAUDE.md

Release of the final RGB-T HMR model: **ARG + MA + DCG** on frozen HMR2.0 (no ITA, no CMT).
Built on 2026-10-16 from `HMR_mu` (skeleton, stripped trainer/eval) + the DCG code of
`SPIN_smplify_copy_10.03_mmpedestron`; the released weights are 10.03
`logs/有pGT/無蒸餾/1007_abl/noita/s4_dcg/checkpoints/checkpoint_best.pt`.
The research projects (07.02 / 10.02 / 10.03 / HMR_mu) are not touched from here.

## Scope

Only `MODEL.CORE_TYPE=hmr2` + `MODEL.TYPE=mmped_unified_hmr2`. Code names vs paper names:
`per_token_conf_mlp` = ARG, `body_adapters` = MA (Modality Adapter), `xmodal_deep_adapters`
(`XMODAL_DEEP_TYPE: convex`) = DCG. `ir_token_adapter` (ITA) and the CMT still exist in the code
but are disabled by every config (`IR_TOKEN_ADAPTER.ENABLE: false`, `UNIFIED_ENCODER.USE_CMT: false`).

## Commands (env: `conda activate spin`, run from the repo root)

```bash
bash scripts/eval_main.sh      # weights/main/model_final.pt -> eval_results/main/
bash scripts/train_main.sh     # 5 stages configs/train/main/s1..s5 -> logs/main/, then export
```

## Regression check (run after any change to train/, models/, datasets/, tools/eval_2d_mu.py)

`bash scripts/eval_main.sh` must reproduce exactly (2D px / 3D mm / PA mm):
- llvip rgb_ir   `10.506401979137438 / 123.2224777340889 / 69.61781715677259`
- llvip ir_only  `12.677777580513595 / 128.86449694633484 / 70.67109938769629`
- llvip rgb_only `15.547057525572747 / 143.54869723320007 / 77.17304265329481` (= HMR2.0)
- otp2 ir_only   `6.205307012738404 / 94.21658515930176 / 59.54034214170234`

## Weights

- `weights/main/model_final.pt` holds only the trainable tensors (format `mu_trainable_only_v1`, 159 tensors,
  5.67M params). `tools/eval_2d_mu.py:_merge_frozen_hmr2` re-reads the frozen backbone/head from
  `MODEL.HMR2_CHECKPOINT`. Create it with `tools/export_release_weights.py` (checks bit-equality with HMR2.0).
- Files under `weights/` and `data/` are hard links to the copies in `HMR_mu` (disk is nearly full):
  never edit them in place.

## Pitfalls

- Each training checkpoint is ~2.7 GB and every stage writes three of them; check free disk before training.
- LLVIP validation uses the test npz (no official val split), so `checkpoint_best.pt` is selected on test.
- `eval_2d_mu.py` loads strictly: unexpected checkpoint keys raise.
- `tools/generate_thermal_smplify_pgt_v5.py` and `smplify/thermal_camsmplify_v4.py` are SHA256-frozen (see README).

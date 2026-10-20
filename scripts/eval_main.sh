#!/usr/bin/env bash
# Evaluate a checkpoint on LLVIP (rgb_ir / ir_only / rgb_only) and OTP2 (ir_only),
# with 3D metrics against pGT (hip-mid root, decode focal 5000).
# Usage: bash scripts/eval_main.sh [CKPT] [TRAIN_CFG] [OUT_DIR]
set -eu
cd "$(dirname "$0")/.."
CKPT=${1:-weights/main/model_final.pt}
CFG=${2:-configs/train/main/s5_dcg.yaml}
OUT=${3:-eval_results/main}
mkdir -p "$OUT/_logs"
for e in "llvip rgb_ir" "llvip ir_only" "llvip rgb_only" "otp2 ir_only"; do
  set -- $e
  echo "[eval] $1 $2"
  python tools/eval_2d_mu.py --dataset "$1" --eval_mode "$2" --eval_3d \
      --root_joint hip_mid --decode_focal 5000 \
      --checkpoint "$CKPT" --train_cfg "$CFG" \
      --out "$OUT/$1_$2.json" > "$OUT/_logs/$1_$2.log" 2>&1
done
python - "$OUT" <<'PY'
import json, sys
print(f"{'split':<16}{'2D-MPJPE(px)':>14}{'3D-MPJPE(mm)':>14}{'PA-MPJPE(mm)':>14}{'PCK@0.05':>10}{'PCK@0.1':>9}")
for t in ("llvip_rgb_ir", "llvip_ir_only", "llvip_rgb_only", "otp2_ir_only"):
    d = json.load(open(f"{sys.argv[1]}/{t}.json"))
    print(f"{t:<16}{d['mpjpe_mean']:>14.3f}{d['mpjpe_3d_mm']:>14.2f}{d['pa_mpjpe_3d_mm']:>14.2f}"
          f"{100 * d['pck_mean']:>10.2f}{100 * d['pck10_mean']:>9.2f}")
PY

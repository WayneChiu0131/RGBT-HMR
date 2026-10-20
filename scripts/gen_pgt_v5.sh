#!/usr/bin/env bash
# Regenerate the LLVIP / OTP2 pGT (v5 protocol) with paths from configs/paths/local.yaml.
# tools/generate_thermal_smplify_pgt_v5.py is frozen, so paths are passed as arguments.
# Usage (repo root, env: spin):  bash scripts/gen_pgt_v5.sh [extra v5 args, e.g. --dataset llvip --split test]
set -eu
cd "$(dirname "$0")/.."
read -r LLVIP OTP2 CHMR CHMR_CKPT < <(python - <<'PY'
import yaml
c = yaml.safe_load(open('configs/paths/local.yaml'))
r, e = c['DATASET_ROOTS'], c['EXTERNAL']
print(r['LLVIP_ROOT'], r['OTP2_DATASET_ROOT'], e['CAMERAHMR_ROOT'], e['CAMERAHMR_CHECKPOINT'])
PY
)
python tools/generate_thermal_smplify_pgt_v5.py \
    --llvip_root "$LLVIP" --otp2_root "$OTP2" \
    --camerahmr_root "$CHMR" --camerahmr_checkpoint "$CHMR_CKPT" \
    --llvip_camerahmr_input thermal \
    --out_dir data/dataset_extras/thermal_pgt_camsmplify_v5_regen "$@"

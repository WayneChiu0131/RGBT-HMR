#!/usr/bin/env bash
# Train the full model: 5 stages, each initialised from the previous stage
# (see IO.PRETRAINED_CHECKPOINT in each config), then export the release weights.
# Usage (from repo root):  bash scripts/train_main.sh
set -eu
cd "$(dirname "$0")/.."
CFG=configs/train/main
DRV=logs/main/_driver; mkdir -p "$DRV"
for st in s1_arg s2_ma s3_frozen_gate s4_refine s5_dcg; do
  if [ -f "logs/main/$st/checkpoints/checkpoint_last.pt" ]; then
    echo "[train] skip $st (already trained)"; continue
  fi
  echo "[train] === $st start $(date '+%F %T')"
  python train.py --train_cfg "$CFG/$st.yaml" --paths_cfg configs/paths/local.yaml \
      > "$DRV/$st.log" 2>&1 || { echo "[train] $st failed, see $DRV/$st.log"; exit 1; }
  test -f "logs/main/$st/checkpoints/checkpoint_last.pt"
  echo "[train] === $st done  best=$(cat logs/main/$st/best.txt 2>/dev/null || echo n/a)"
done
python tools/export_release_weights.py \
    --checkpoint logs/main/s5_dcg/checkpoints/checkpoint_best.pt \
    --train_cfg "$CFG/s5_dcg.yaml" --out logs/main/model_final.pt
echo "[train] release weights: logs/main/model_final.pt"

#!/bin/bash
# Train E-MACE: EquiFiLM applied to MACE-MATPES-r2SCAN-omat-ft on charged water.
# Usage: bash examples/train.sh <data_dir> <output_dir>
#
# Expects the data directory to contain combined_all_charges.xyz with the four
# training charges {0, 6e, 10e, 16e} concatenated and labelled via the
# atoms.info field "total_charge" (one integer per frame).
#
# Reproduces the headline E-MACE model used in the paper. Wall time on a single
# A100-40GB is approximately 24 hours for the full SWA-249 schedule.
set -eu

DATA_DIR=${1:-./data}
OUT_DIR=${2:-./checkpoints}
mkdir -p "$OUT_DIR"

# Foundation backbone (MACE-MATPES-r2SCAN-omat-ft). Downloaded once into ~/.cache/mace
# by the standard mace_run_train --foundation_model_path mechanism.
# Replace with a local path if needed.
FOUNDATION="MACE-matpes-r2scan-omat-ft"

mace_run_train \
    --name="MACE-FiLM-large" \
    --foundation_model_path="$FOUNDATION" \
    --multiheads_finetuning=False \
    --train_file="$DATA_DIR/combined_all_charges.xyz" \
    --total_charge_key="total_charge" \
    --valid_fraction=0.10 \
    --seed=123 \
    --max_num_epochs=250 \
    --start_swa=150 \
    --batch_size=4 \
    --lr=1.0e-3 \
    --weight_decay=5.0e-7 \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --ema=True \
    --ema_decay=0.99 \
    --device=cuda \
    --default_dtype=float32 \
    --save_cpu \
    --restart_latest \
    --charge_film=128 \
    --model_dir="$OUT_DIR" \
    --checkpoints_dir="$OUT_DIR/checkpoints" \
    --log_dir="$OUT_DIR/logs"

echo "Training complete. Final SWA model: $OUT_DIR/MACE-FiLM-large_stagetwo.model"

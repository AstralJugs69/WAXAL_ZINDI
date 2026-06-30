#!/usr/bin/env bash
set -eo pipefail

CONFIG=${1:-"config/base_mms.yaml"}
FOLD=${2:-0}
LANG=${3:-"lin"}

echo "=== Launching WAXAL Training ==="
echo "Config:      $CONFIG"
echo "Fold:        $FOLD"
echo "Language:    $LANG"
echo "================================="

# Set environment variables for optimized execution
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python src/training/trainer.py \
    --config "$CONFIG" \
    --fold "$FOLD" \
    --target_lang "$LANG"

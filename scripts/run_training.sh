#!/usr/bin/env bash
set -eo pipefail

CONFIG=${1:-"config/base_mms.yaml"}
FOLD=${2:-0}
LANG=${3:-"lin"}
TPU_FLAG=${4:-""}

echo "=== Launching WAXAL Training ==="
echo "Config:      $CONFIG"
echo "Fold:        $FOLD"
echo "Language:    $LANG"
echo "TPU Enabled: ${TPU_FLAG:-"false"}"
echo "================================="

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH=.

if [ -n "$TPU_FLAG" ]; then
    # TPU path — single process that spawns XLA workers internally
    python src/training/trainer.py \
        --config "$CONFIG" \
        --fold "$FOLD" \
        --target_lang "$LANG" \
        $TPU_FLAG
else
    # GPU path — detect how many CUDA GPUs are visible and use torchrun for DDP
    N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "Detected GPUs: $N_GPUS"

    if [ "$N_GPUS" -gt 1 ]; then
        echo "Launching with torchrun DDP across $N_GPUS GPUs..."
        torchrun \
            --nproc_per_node="$N_GPUS" \
            --master_port=29500 \
            src/training/trainer.py \
            --config "$CONFIG" \
            --fold "$FOLD" \
            --target_lang "$LANG"
    else
        echo "Single GPU detected — launching standard python..."
        python src/training/trainer.py \
            --config "$CONFIG" \
            --fold "$FOLD" \
            --target_lang "$LANG"
    fi
fi

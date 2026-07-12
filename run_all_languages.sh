#!/usr/bin/env bash
# ==============================================================================
# WAXAL ASR Pipeline Launcher & Multilingual Sequential Trainer
# ==============================================================================
set -eo pipefail

REPO_DIR="WAXAL_ZINDI"
REPO_URL="https://github.com/AstralJugs69/WAXAL_ZINDI.git"

echo "========================================="
echo "🪐 WAXAL ASR Pipelines Setup Launcher 🪐"
echo "========================================="

# 1. Clone repository from scratch if missing, or pull latest changes
if [ -d "$REPO_DIR" ] || [ -f "run_lightning.py" ]; then
    # We are either inside the repo, or the folder exists in the current directory
    if [ -d "$REPO_DIR" ]; then
        cd "$REPO_DIR"
    fi
    echo "Repository detected. Syncing latest updates..."
    git pull
else
    echo "Repository '$REPO_DIR' not found. Cloning from scratch..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
fi

# Ensure launchers are executable
chmod +x run_lightning.py
chmod +x scripts/run_training.sh

# Persist checkpoints in the repository workspace instead of ephemeral /tmp.
mkdir -p outputs
export WAXAL_OUTPUTS_DIR="${WAXAL_OUTPUTS_DIR:-$(pwd)/outputs}"
echo "Using checkpoint/output directory: $WAXAL_OUTPUTS_DIR"

# 2. Extract arguments or use defaults
CONFIG=${1:-"config/base_mms_tpu.yaml"}
FOLD=${2:-0}
HF_TOKEN=${3:-$HF_TOKEN}

# TPU auto-detection
TPU_FLAG=""
if [ -n "$TPU_NAME" ] || [ -d "/usr/share/tpu-support" ] || [ -f "/usr/lib/libtpu.so" ] || [ -e "/dev/accel0" ]; then
    echo "TPU environment detected. Enabling --tpu flag."
    TPU_FLAG="--tpu"
fi

# 3. Train all three languages with the minimal Lightning MMS path
#    (replaces fragile HF Trainer + xmp.spawn + git hot-reload path)
echo ""
echo "=========================================================="
echo "🚀 Training lin/sna/lug via run_tpu_train.py (Lightning) 🚀"
echo "=========================================================="

EXTRA=()
if [ -n "$HF_TOKEN" ]; then
  EXTRA+=(--hf_token "$HF_TOKEN")
fi

python run_tpu_train.py \
  --lang all \
  --fold "$FOLD" \
  --config "$CONFIG" \
  --devices 8 \
  $TPU_FLAG \
  "${EXTRA[@]}"

# 4. Generate the final submission file using all trained checkpoints
echo ""
echo "=========================================================="
echo "🎯 Generating final submission.csv... 🎯"
echo "=========================================================="
HF_TOKEN="$HF_TOKEN" python generate_submission.py --max-blank-frac 0.05

echo "=========================================================="
echo "🎉 Sequential training of all three languages complete! 🎉"
echo "🎉 File 'submission.csv' generated successfully!        🎉"
echo "=========================================================="

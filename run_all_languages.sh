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
CONFIG=${1:-"config/base_gemma.yaml"}
FOLD=${2:-0}
HF_TOKEN=${3:-$HF_TOKEN}
KAGGLE_USER=${4:-$KAGGLE_USERNAME}
KAGGLE_KEY=${5:-$KAGGLE_KEY}

# TPU auto-detection
TPU_FLAG=""
if [ -n "$TPU_NAME" ] || [ -d "/usr/share/tpu-support" ] || [ -f "/usr/lib/libtpu.so" ]; then
    echo "TPU environment detected. Enabling --tpu flag."
    TPU_FLAG="--tpu"
fi

# 3. Train all three languages sequentially (Lingala, Shona, Luganda)
for LANG in lin sna lug; do
    STEPS=800
    
    echo ""
    echo "=========================================================="
    echo "🚀 Starting Training for Language: $LANG ($STEPS steps, Fold $FOLD) 🚀"
    echo "=========================================================="
    
    python run_lightning.py \
      --config "$CONFIG" \
      --fold "$FOLD" \
      --target_lang "$LANG" \
      --max_steps "$STEPS" \
      --git_poll_interval 5 \
      --hf_token "$HF_TOKEN" \
      --kaggle_username "$KAGGLE_USER" \
      --kaggle_key "$KAGGLE_KEY" \
      $TPU_FLAG
done

# 4. Generate the final submission file using all trained checkpoints
echo ""
echo "=========================================================="
echo "🎯 Generating final submission.csv... 🎯"
echo "=========================================================="
HF_TOKEN="$HF_TOKEN" python generate_submission.py

echo "=========================================================="
echo "🎉 Sequential training of all three languages complete! 🎉"
echo "🎉 File 'submission.csv' generated successfully!        🎉"
echo "=========================================================="

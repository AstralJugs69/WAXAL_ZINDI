#!/usr/bin/env bash
set -eo pipefail

echo "=== Installing System Dependencies ==="
if [ -f /etc/debian_version ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        SUDO=""
    fi
    $SUDO apt-get update && $SUDO apt-get install -y \
        ffmpeg \
        cmake \
        build-essential \
        libsndfile1 \
        libboost-all-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev
else
    echo "Warning: Non-debian system detected. Please install ffmpeg, cmake, build-essential, libsndfile, boost, zlib, bz2, and lzma manually."
fi

echo "=== Checking GPU Architecture ==="
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
    echo "GPU Name Detected: $GPU_NAME"
    if [[ "$GPU_NAME" == *"P100"* ]]; then
        echo "Tesla P100 GPU detected. Reinstalling sm_60 (Pascal) compatible PyTorch, Torchaudio, and Torchvision wheels..."
        pip install --force-reinstall \
            torch torchaudio torchvision "numpy<2" \
            --index-url https://download.pytorch.org/whl/cu118
        echo "Downgrading datasets to 2.20.0 to bypass torchcodec requirements..."
        pip install "datasets==2.20.0"
        echo "Uninstalling torchcodec to prevent compatibility crashes with cu118..."
        pip uninstall -y torchcodec
    else
        echo "GPU detected: $GPU_NAME (compatibility check passed)"
    fi
fi

echo "=== Generating PyTorch Version Constraints ==="
# Pin pre-installed core packages to prevent pip from backtracking or downloading incompatible wheels.
# Exclude pyctcdecode (has numpy<2 pin incompatible with TPU's numpy 2.x — installed separately with --no-deps).
pip freeze | grep -E "^(torch|torchaudio|torchvision|numpy|pandas|intel-openmp|mkl|pyarrow|transformers|peft|accelerate|bitsandbytes|librosa|soundfile|evaluate|jiwer|pyyaml|tqdm|scikit-learn|optuna|webrtcvad)==" > constraints.txt
echo "Generated constraints:"
cat constraints.txt

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
# Install hf_transfer for 5-10x faster HuggingFace downloads (Rust-based parallel downloader)
pip install hf_transfer --quiet 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1
# Replace full tensorflow with CPU-only build to avoid GPU/TPU driver conflicts with torch-xla
pip install tensorflow-cpu --quiet 2>/dev/null || true
pip install -c constraints.txt -r requirements.txt
# Install pyctcdecode separately with --no-deps to avoid its numpy<2 pin conflicting with numpy 2.x on TPU
pip install pyctcdecode>=0.5.0 --no-deps

echo "=== Compiling KenLM ==="
# Trigger our automated compilation utility
python -c "
import sys
sys.path.append('src')
from decoding.kenlm_utils import compile_kenlm
compile_kenlm('kenlm')
"

echo "=== Installation Completed Successfully ==="

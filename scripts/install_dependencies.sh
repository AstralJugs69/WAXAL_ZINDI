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
    if nvidia-smi | grep -q "P100"; then
        echo "Tesla P100 GPU detected. Reinstalling sm_60 (Pascal) compatible PyTorch, Torchaudio, and Torchvision wheels..."
        pip install --force-reinstall \
            torch torchaudio torchvision \
            --index-url https://download.pytorch.org/whl/cu118
    else
        echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    fi
fi

echo "=== Generating PyTorch Version Constraints ==="
# Pin current torch, torchaudio, torchvision, and numpy versions to prevent pip from overwriting them with incompatible PyPI wheels
pip freeze | grep -E "^(torch|torchaudio|torchvision|numpy|intel-openmp|mkl)==" > constraints.txt
echo "Generated constraints:"
cat constraints.txt

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
# Replace full tensorflow with CPU-only build to avoid GPU/TPU driver conflicts with torch-xla
pip install tensorflow-cpu --quiet 2>/dev/null || true
pip install -c constraints.txt -r requirements.txt

echo "=== Compiling KenLM ==="
# Trigger our automated compilation utility
python -c "
import sys
sys.path.append('src')
from decoding.kenlm_utils import compile_kenlm
compile_kenlm('kenlm')
"

echo "=== Installation Completed Successfully ==="

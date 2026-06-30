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

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Compiling KenLM ==="
# Trigger our automated compilation utility
python -c "
import sys
sys.path.append('src')
from decoding.kenlm_utils import compile_kenlm
compile_kenlm('kenlm')
"

echo "=== Installation Completed Successfully ==="

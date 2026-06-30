#!/usr/bin/env bash
set -eo pipefail

echo "=== Installing System Dependencies ==="
if [ -f /etc/debian_version ]; then
    sudo apt-get update && sudo apt-get install -y \
        ffmpeg \
        cmake \
        build-essential \
        libsndfile1
else
    echo "Warning: Non-debian system detected. Please install ffmpeg, cmake, build-essential, and libsndfile manually."
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

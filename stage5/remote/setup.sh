#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")"

# Build deps for unsloth's on-demand llama.cpp build (GGUF export) + git.
apt-get update -y && apt-get install -y --no-install-recommends \
  git build-essential cmake libcurl4-openssl-dev

pip install --upgrade pip
# torch stack + xformers from the cu126 index FIRST, THEN the rest. Same verified
# stack as stage3 (see stage3/remote/setup.sh for the version rationale).
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 xformers==0.0.31.post1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
echo "stage5 setup complete"

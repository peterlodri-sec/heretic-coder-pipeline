#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")"

# Build deps for unsloth's on-demand llama.cpp build (GGUF export) + git.
apt-get update -y && apt-get install -y --no-install-recommends \
  git build-essential cmake libcurl4-openssl-dev

pip install --upgrade pip
# torch + xformers from the cu124 index FIRST (unsloth's cu124onlytorch260
# pairing is fragile — pin both), THEN the rest of the stack.
pip install torch==2.6.0 xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
echo "stage2 setup complete"

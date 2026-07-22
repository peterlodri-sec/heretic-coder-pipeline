#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")"

# Build deps for unsloth's on-demand llama.cpp build (GGUF export) + git.
apt-get update -y && apt-get install -y --no-install-recommends \
  git build-essential cmake libcurl4-openssl-dev curl

# uv: a 10-100x faster resolver/installer than pip, with a real dependency
# resolver (fewer silent version conflicts) and a global cache. Install into the
# ambient conda python via --system so nothing else in the pipeline changes.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
UV_PIP=(uv pip install --system)

# torch stack + xformers from the cu126 index FIRST, THEN the rest.
# torch>=2.7 is required at runtime by transformers 4.56.2 (uses
# torch.utils._pytree.register_constant, added in 2.7); torchvision>=0.22 is
# required by unsloth's compat check. Versions verified together on an H200.
"${UV_PIP[@]}" torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 xformers==0.0.31.post1 \
  --index-url https://download.pytorch.org/whl/cu126
"${UV_PIP[@]}" -r requirements.txt
echo "stage2 setup complete"

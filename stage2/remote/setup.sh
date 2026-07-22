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
# MXFP4 quantize-on-load needs triton>=3.4.0, but torch 2.7.1 pins triton==3.3.1.
# Override just triton (--no-deps so torch's pin doesn't drag it back / conflict);
# triton is a standalone compiler and 3.4.0 runs the mxfp4 kernels on torch 2.7.1.
# No-op for non-mxfp4 runs. [verify on GPU: if 3.4.0 breaks torch 2.7.1 kernels,
# fall back to torch 2.8.0 which ships triton 3.4.0.]
"${UV_PIP[@]}" --no-deps triton==3.4.0
# deepspeed ONLY for the sharded (gpt-oss ZeRO-3) path — just having it installed
# makes accelerate/TRL probe its op-builder and JIT-compile a CUDA op, which fails
# with "CUDA_HOME does not exist" on no-nvcc runtime images (and the qwen/Unsloth
# path never uses DeepSpeed anyway). The ZeRO-3 path additionally needs a
# CUDA-devel image with nvcc; wire that when returning to the 120B run.
if [ "${STAGE2_SHARDED:-0}" = "1" ]; then
  "${UV_PIP[@]}" deepspeed==0.16.4
else
  # ensure a cached/reused box that previously installed it is cleaned.
  uv pip uninstall --system deepspeed 2>/dev/null || true
fi
echo "stage2 setup complete"

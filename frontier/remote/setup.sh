#!/usr/bin/env bash
# frontier/remote/setup.sh — provision an 8xH200 box for the 480B pipeline:
# Axolotl (SFT) + LLaMA-Factory (ORPO) + vLLM (eval/serve) + Heretic (stage 1),
# plus NCCL/shm env and llama.cpp build deps. Runnable, `bash -n` clean.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")"

# --- multi-GPU NCCL env (persist for tmux launches + this shell) --------------
# InfiniBand is unavailable on most Vast hosts; disable it so NCCL uses TCP/NVLink.
# P2P_DISABLE is the documented fallback if collective init hangs on a non-NVLink
# host (left commented — enable only if `run_frontier` reports an NCCL hang).
{
  echo 'export NCCL_IB_DISABLE=1'
  echo '# export NCCL_P2P_DISABLE=1  # uncomment if NCCL init hangs on a non-NVLink host'
  echo 'export HF_HUB_ENABLE_HF_TRANSFER=1'
} >> /root/.bashrc
export NCCL_IB_DISABLE=1

# Ensure a large /dev/shm for NCCL shared-memory transport (containers often
# default to 64MB, which stalls 8-GPU collectives).
mount -o remount,size=64g /dev/shm 2>/dev/null || \
  mount -t tmpfs -o size=64g tmpfs /dev/shm 2>/dev/null || \
  echo "warning: could not enlarge /dev/shm; set NCCL_SHM_DISABLE=1 if collectives stall"

# --- system build deps (llama.cpp GGUF build + git) ---------------------------
apt-get update -y
apt-get install -y --no-install-recommends \
  git git-lfs build-essential cmake libcurl4-openssl-dev tmux

pip install --upgrade pip

# torch/cuda stack: the Axolotl base image usually ships torch; pin the cu126
# 2.7.1 stack (matching stage2) only if torch is absent, so we don't clobber a
# working image build.
if ! python3 -c "import torch" 2>/dev/null; then
  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
fi

# --- trainers + serving -------------------------------------------------------
pip install "axolotl[deepspeed,flash-attn]"
pip install llamafactory
pip install vllm

# heretic-llm from source, pinned to the same commit stage1 uses (headless CLI
# flags not yet released to PyPI).
pip install "heretic-llm @ git+https://github.com/p-e-w/heretic.git@e7b783ed85"

echo "=== FRONTIER SETUP COMPLETE ==="

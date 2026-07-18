#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$(dirname "$0")"
pip install --upgrade pip
pip install -r requirements.txt
echo "stage3 setup complete"

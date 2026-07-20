#!/bin/bash
# stage1/remote/setup.sh
set -euo pipefail
apt-get update -qq
apt-get install -y -qq git-lfs curl
pip install -q --upgrade pip
# requirements.txt pins heretic-llm==1.1.0 (FIX A: the release that abliterates
# gpt-oss's fused MoE-expert down_proj) plus pexpect (v1.1.0 is interactive-only).
pip install -q -r "$(dirname "$0")/requirements.txt"
echo "=== SETUP COMPLETE ==="

#!/bin/bash
# stage1/remote/setup.sh
set -euo pipefail
apt-get update -qq
apt-get install -y -qq git-lfs curl
pip install -q --upgrade pip
pip install -q -r "$(dirname "$0")/requirements.txt"
echo "=== SETUP COMPLETE ==="

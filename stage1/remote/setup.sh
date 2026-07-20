#!/bin/bash
# stage1/remote/setup.sh
set -euo pipefail
apt-get update -qq
apt-get install -y -qq git-lfs curl
pip install -q --upgrade pip
# ROOT CAUSE of the first failed run: the Vast image shipped a newer,
# LoRA-on-Modules heretic whose `heretic` console-script shadowed our pin, so the
# run abliterated o_proj only and silently skipped gpt-oss's fused MoE experts
# (KL~0.003 no-op, ~$150 wasted). A pinned line in requirements is not enough if a
# differently-named distribution already owns the `heretic` entry point. So purge
# BOTH known distribution names first, THEN install the pin.
pip uninstall -y heretic heretic-llm >/dev/null 2>&1 || true
# requirements.txt pins heretic-llm==1.1.0 (FIX A: the release that abliterates
# gpt-oss's fused MoE-expert down_proj via direct-tensor surgery) plus pexpect
# (v1.1.0 is interactive-only).
pip install -q -r "$(dirname "$0")/requirements.txt"
# Fail LOUDLY here (during setup, ~seconds) instead of 9h later: assert the pin
# actually took and that the `heretic` on PATH is the one we just installed.
python - <<'PY'
from importlib.metadata import version
import shutil, sys, os
v = version("heretic-llm")
assert v == "1.1.0", f"heretic-llm=={v}, expected 1.1.0 (v1.2+/master silently skip fused experts)"
binpath = shutil.which("heretic")
inenv = os.path.join(os.path.dirname(sys.executable), "heretic")
assert binpath and os.path.realpath(binpath) == os.path.realpath(inenv), \
    f"PATH heretic={binpath} is not the in-env one ({inenv}) -- a shadowing build remains"
print(f"heretic-llm {v} OK ; heretic -> {binpath}")
PY
echo "=== SETUP COMPLETE ==="

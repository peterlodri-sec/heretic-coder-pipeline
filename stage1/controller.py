#!/usr/bin/env python3
# stage1/controller.py
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ssh_utils
import status_io
import vast_provision
from vastai import VastAI

STAGE1_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_PARENT = "/root"
REMOTE_ROOT = "/root/stage1"
REMOTE_STATUS_PATH = f"{REMOTE_ROOT}/remote/status.json"
REMOTE_LOG_PATH = f"{REMOTE_ROOT}/remote/heretic_run.log"
API_KEY_PATH = os.path.expanduser("~/.config/vastai/vast_api_key")
POLL_INTERVAL_SECONDS = 300
SSH_USER = "root"


def load_api_key() -> str:
    with open(API_KEY_PATH) as f:
        return f.read().strip()


def deploy_and_launch(instance: dict, model: str, n_trials: int):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.scp_to(host, port, STAGE1_DIR, REMOTE_PARENT, recursive=True)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh")
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && "
        f"STAGE1_MODEL='{model}' STAGE1_N_TRIALS='{n_trials}' "
        "tmux new-session -d -s heretic 'python3 run_stage1.py'"
    )
    return host, port


def poll_until_done(host: str, port: int, interval: int = POLL_INTERVAL_SECONDS) -> dict:
    while True:
        try:
            raw = ssh_utils.run_ssh(host, port, f"cat {REMOTE_STATUS_PATH}")
            status = status_io.parse_status_text(raw)
            stage = status["stage"]
            verdict = status["verdict"]
        except (ssh_utils.SSHError, ValueError, KeyError):
            time.sleep(interval)
            continue

        print(f"[{stage}] verdict={verdict}")
        if stage == "done":
            return status
        time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--n-trials", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = load_api_key()
    vast = VastAI(api_key=api_key)

    instance = vast_provision.provision(vast)
    host, port = deploy_and_launch(instance, args.model, args.n_trials)

    final_status = poll_until_done(host, port)

    local_log_path = os.path.join(STAGE1_DIR, "heretic_run.log")
    try:
        ssh_utils.scp_from(host, port, REMOTE_LOG_PATH, local_log_path)
    except Exception as error:
        print(f"warning: failed to pull run log: {error}", file=sys.stderr)

    print(json.dumps(final_status, indent=2))

    if final_status["verdict"] == "pass":
        try:
            vast.stop_instance(id=instance["id"])
        except Exception as error:
            print(
                f"warning: failed to stop instance {instance['id']}: {error}; "
                "stop it manually to avoid continued billing",
                file=sys.stderr,
            )

    return 0 if final_status["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

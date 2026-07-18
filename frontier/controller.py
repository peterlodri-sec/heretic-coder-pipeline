#!/usr/bin/env python3
# frontier/controller.py — multi-GPU (8xH200) controller for the Qwen3-Coder-480B
# heretic->SFT->ORPO pipeline. Mirrors stage2/controller.py but provisions a
# single 8-GPU node with a large disk and launches run_frontier.py in tmux.
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> shared

from enums import Stage
from shared import ssh_utils, vast_provision
from shared.enums import Verdict
from shared.poll import poll_until_done
from shared.vast_ops import load_api_key, local_hf_token_path, provision_lock
from status_io import Status
from vastai import VastAI

FRONTIER_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(os.path.dirname(FRONTIER_DIR), "shared")
REMOTE_PARENT = "/root"
REMOTE_ROOT = "/root/frontier"
REMOTE_STATUS_PATH = f"{REMOTE_ROOT}/remote/status.json"
REMOTE_LOG_PATH = f"{REMOTE_ROOT}/remote/frontier_run.log"
POLL_INTERVAL_SECONDS = 300
# 8xH200 setup: axolotl + llamafactory + vllm + heretic + torch is very heavy on
# a cold instance; give it a wide ceiling far past a normal SSH timeout.
SETUP_TIMEOUT_SECONDS = 3600
PROVISION_LABEL = "heretic-480b"
# 8x H200 single node: 480B bf16 base (~960GB) fits across 1128GB VRAM. reliability
# + cuda floor guard against flaky/old hosts that stall NCCL or lack modern drivers.
PROVISION_QUERY = (
    "gpu_name=H200 num_gpus=8 disk_space>=2000 cuda_vers>=12.4 "
    "reliability>0.98 rentable=true"
)
PROVISION_DISK_GB = 2000  # 480B base + merged SFT + merged ORPO + datasets + logs
SSH_USER = "root"


def deploy_and_launch(instance: dict, model: str, stage: str, check_swebench: bool,
                      skip_heretic: bool):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.wait_for_ssh(host, port)  # fresh instance: wait for sshd before transferring
    token = local_hf_token_path()
    if token:
        ssh_utils.run_ssh(host, port, "mkdir -p /root/.cache/huggingface")
        ssh_utils.scp_to(host, port, token, "/root/.cache/huggingface/token")
    ssh_utils.scp_to(host, port, SHARED_DIR, REMOTE_PARENT, recursive=True)
    ssh_utils.scp_to(host, port, FRONTIER_DIR, REMOTE_PARENT, recursive=True)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh",
                      timeout=SETUP_TIMEOUT_SECONDS)
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_IB_DISABLE=1 "
        f"FRONTIER_MODEL='{model}' FRONTIER_STAGE='{stage}' "
        f"FRONTIER_CHECK_SWEBENCH='{int(check_swebench)}' "
        f"FRONTIER_SKIP_HERETIC='{int(skip_heretic)}' "
        "tmux new-session -d -s frontier 'python3 run_frontier.py'"
    )
    return host, port


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-480B-A35B-Instruct")
    parser.add_argument("--stage", choices=["heretic", "sft", "orpo", "all"], default="all")
    parser.add_argument("--no-swebench", dest="check_swebench", action="store_false")
    parser.add_argument("--skip-heretic", dest="skip_heretic", action="store_true",
                        help="skip stage 1 abliteration; de-align via ORPO data only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vast = VastAI(api_key=load_api_key())

    instance = None
    verdict = Verdict.ERROR
    try:
        with provision_lock():
            instance = vast_provision.provision(
                vast, label=PROVISION_LABEL, query=PROVISION_QUERY, disk_gb=PROVISION_DISK_GB)
        host, port = deploy_and_launch(instance, args.model, args.stage,
                                       args.check_swebench, args.skip_heretic)

        final_status = poll_until_done(host, port, REMOTE_STATUS_PATH, Status,
                                       Stage.DONE, POLL_INTERVAL_SECONDS)
        verdict = final_status.verdict or Verdict.ERROR

        try:
            ssh_utils.scp_from(host, port, REMOTE_LOG_PATH,
                               os.path.join(FRONTIER_DIR, "frontier_run.log"))
        except Exception as error:
            print(f"warning: failed to pull run log: {error}", file=sys.stderr)

        print(final_status.to_json())
    finally:
        if instance is not None:
            try:
                vast.stop_instance(id=instance["id"])
            except Exception as error:
                print(f"warning: failed to stop instance {instance['id']}: {error}; "
                      "stop it manually to avoid continued billing", file=sys.stderr)

    return 0 if verdict is Verdict.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())

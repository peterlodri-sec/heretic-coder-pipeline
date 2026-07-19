#!/usr/bin/env python3
# stage5/controller.py — RLVR (execution-feedback RL). TERMINAL stage for gpt-oss;
# replaces ORPO. Input model = stage4's RFT-loop output. Needs MULTI-GPU: colocated
# vLLM generation + LoRA training + long agentic rollouts (realistically 2-4x H200).
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

STAGE5_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(STAGE5_DIR)
SHARED_DIR = os.path.join(REPO_ROOT, "shared")
REMOTE_PARENT = "/root"
REMOTE_ROOT = "/root/stage5"
REMOTE_STATUS_PATH = f"{REMOTE_ROOT}/remote/status.json"
REMOTE_LOG_PATH = f"{REMOTE_ROOT}/remote/rlvr_run.log"
POLL_INTERVAL_SECONDS = 300
SETUP_TIMEOUT_SECONDS = 1800
PROVISION_LABEL = "heretic-rlvr"
DEFAULT_NUM_GPUS = 2  # colocated rollout + training; verify vRAM, may need 4x
PROVISION_DISK_GB = 600
SSH_USER = "root"


def provision_query(num_gpus: int) -> str:
    # Multi-GPU H200 offer (GSPO group rollouts + training colocated).
    return f"gpu_name=H200 num_gpus={num_gpus} disk_space>={PROVISION_DISK_GB} rentable=true"


def deploy_and_launch(instance: dict, model: str, num_gpus: int, check_swebench: bool,
                      mode: str = "distill", family: str = "gpt_oss"):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.wait_for_ssh(host, port)  # fresh instance: wait for sshd before transferring
    token = local_hf_token_path()
    if token:
        ssh_utils.run_ssh(host, port, "mkdir -p /root/.cache/huggingface")
        ssh_utils.scp_to(host, port, token, "/root/.cache/huggingface/token")
    ssh_utils.send_dir(host, port, SHARED_DIR, REMOTE_PARENT)
    ssh_utils.send_dir(host, port, STAGE5_DIR, REMOTE_PARENT)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh",
                      timeout=SETUP_TIMEOUT_SECONDS)
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && HF_HUB_ENABLE_HF_TRANSFER=1 "
        f"STAGE5_MODEL='{model}' STAGE5_FAMILY='{family}' STAGE5_MODE='{mode}' "
        f"STAGE5_NUM_GPUS='{num_gpus}' "
        f"STAGE5_CHECK_SWEBENCH='{int(check_swebench)}' "
        "tmux new-session -d -s rlvr 'python3 run_stage5.py'"
    )
    return host, port


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PeetPedro/gpt-oss-120b-heretic-rft")
    parser.add_argument("--family", default="gpt_oss")
    parser.add_argument("--num-gpus", dest="num_gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--no-swebench", dest="check_swebench", action="store_false")
    # RLVR mode (plan Gemini §1). distill (default): RFT-on-120B -> SFT, cheapest
    # good option, sidesteps the KV-cache wall. offline-kto / live-rl deferred.
    parser.add_argument("--mode", choices=["live-rl", "distill", "offline-kto"],
                        default="distill")
    # Cost lever: RLVR is MULTI-GPU + long-horizon; a coordinated preempt is
    # costlier to resume, so interruptible defaults OFF here (opt in with
    # --interruptible / INTERRUPTIBLE=1).
    parser.add_argument("--interruptible", action="store_true",
                        default=os.environ.get("INTERRUPTIBLE") == "1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vast = VastAI(api_key=load_api_key())

    instance = None
    verdict = Verdict.ERROR
    try:
        with provision_lock():
            instance = vast_provision.provision(
                vast, label=PROVISION_LABEL, query=provision_query(args.num_gpus),
                disk_gb=PROVISION_DISK_GB, interruptible=args.interruptible)
        host, port = deploy_and_launch(instance, args.model, args.num_gpus,
                                       args.check_swebench, args.mode, args.family)

        final_status = poll_until_done(host, port, REMOTE_STATUS_PATH, Status,
                                       Stage.DONE, POLL_INTERVAL_SECONDS)
        verdict = final_status.verdict or Verdict.ERROR

        try:
            ssh_utils.scp_from(host, port, REMOTE_LOG_PATH,
                               os.path.join(STAGE5_DIR, "rlvr_run.log"))
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

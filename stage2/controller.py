#!/usr/bin/env python3
# stage2/controller.py
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> shared

from enums import Stage
from shared import ssh_utils, vast_provision
from shared.enums import Verdict
from shared.poll import poll_until_done
from shared.vast_ops import load_api_key, provision_lock
from status_io import Status
from vastai import VastAI

STAGE2_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(os.path.dirname(STAGE2_DIR), "shared")
REMOTE_PARENT = "/root"
REMOTE_ROOT = "/root/stage2"
REMOTE_STATUS_PATH = f"{REMOTE_ROOT}/remote/status.json"
REMOTE_LOG_PATH = f"{REMOTE_ROOT}/remote/sft_run.log"
POLL_INTERVAL_SECONDS = 300
SETUP_TIMEOUT_SECONDS = 1800  # unsloth + trl + transformers + datasets install is heavy
PROVISION_LABEL = "heretic-sft"
PROVISION_QUERY = "gpu_name=H100_SXM disk_space>=400 rentable=true"
PROVISION_DISK_GB = 400  # base model + 5 datasets + LoRA + gguf export
SSH_USER = "root"


def deploy_and_launch(instance: dict, model: str, max_steps: int, crabcc_traces: str,
                      check_swebench: bool):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.wait_for_ssh(host, port)  # fresh instance: wait for sshd before transferring
    ssh_utils.scp_to(host, port, SHARED_DIR, REMOTE_PARENT, recursive=True)
    ssh_utils.scp_to(host, port, STAGE2_DIR, REMOTE_PARENT, recursive=True)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh",
                      timeout=SETUP_TIMEOUT_SECONDS)
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && "
        f"STAGE2_MODEL='{model}' STAGE2_MAX_STEPS='{max_steps}' "
        f"STAGE2_CRABCC_TRACES='{crabcc_traces}' "
        f"STAGE2_CHECK_SWEBENCH='{int(check_swebench)}' "
        "tmux new-session -d -s sft 'python3 run_stage2.py'"
    )
    return host, port


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PeetPedro/qwen2.5-coder-32b-instruct-heretic")
    parser.add_argument("--crabcc-traces", dest="crabcc_traces", default="traces")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--no-swebench", dest="check_swebench", action="store_false")
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
        host, port = deploy_and_launch(instance, args.model, args.max_steps, args.crabcc_traces,
                                       args.check_swebench)

        final_status = poll_until_done(host, port, REMOTE_STATUS_PATH, Status,
                                       Stage.DONE, POLL_INTERVAL_SECONDS)
        verdict = final_status.verdict or Verdict.ERROR

        try:
            ssh_utils.scp_from(host, port, REMOTE_LOG_PATH,
                               os.path.join(STAGE2_DIR, "sft_run.log"))
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

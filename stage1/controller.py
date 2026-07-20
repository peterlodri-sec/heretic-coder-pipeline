#!/usr/bin/env python3
# stage1/controller.py
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

STAGE1_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(os.path.dirname(STAGE1_DIR), "shared")
REMOTE_PARENT = "/root"
REMOTE_ROOT = "/root/stage1"
REMOTE_STATUS_PATH = f"{REMOTE_ROOT}/remote/status.json"
REMOTE_LOG_PATH = f"{REMOTE_ROOT}/remote/heretic_run.log"
POLL_INTERVAL_SECONDS = 300
# setup.sh runs apt-get + pip install (heretic-llm from git source, lm_eval,
# optuna); 2-5+ min on a cold instance, far past a normal SSH command timeout.
SETUP_TIMEOUT_SECONDS = 1200
SSH_USER = "root"


def deploy_and_launch(instance: dict, model: str, n_trials: int, family: str = "gpt_oss"):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.wait_for_ssh(host, port)  # fresh instance: wait for sshd before transferring
    token = local_hf_token_path()
    if token:
        ssh_utils.run_ssh(host, port, "mkdir -p /root/.cache/huggingface")
        ssh_utils.scp_to(host, port, token, "/root/.cache/huggingface/token")
    ssh_utils.send_dir(host, port, SHARED_DIR, REMOTE_PARENT)
    ssh_utils.send_dir(host, port, STAGE1_DIR, REMOTE_PARENT)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh",
                      timeout=SETUP_TIMEOUT_SECONDS)
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && HF_HUB_ENABLE_HF_TRANSFER=1 "
        f"STAGE1_MODEL='{model}' STAGE1_FAMILY='{family}' STAGE1_N_TRIALS='{n_trials}' "
        "tmux new-session -d -s heretic 'python3 run_stage1.py'"
    )
    return host, port


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/gpt-oss-120b-BF16")  # bf16: heretic has no MXFP4 path
    parser.add_argument("--family", default="gpt_oss")  # drives heretic bnb_4bit quant
    parser.add_argument("--n-trials", type=int, default=200)
    # Heretic is short; a spot preempt mid-abliteration wastes the whole run, so
    # interruptible defaults OFF here (opt in with --interruptible). Cost lever
    # (plan Gemini): interruptible H200 ~40-60% cheaper.
    parser.add_argument("--interruptible", action="store_true",
                        default=os.environ.get("INTERRUPTIBLE") == "1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = load_api_key()
    vast = VastAI(api_key=api_key)

    instance = None
    verdict = Verdict.ERROR
    try:
        with provision_lock():
            instance = vast_provision.provision(
                vast,
                # gpt-oss-120b abliterated in BF16 (heretic has no MXFP4 path;
                # bf16 guarantees o_proj+down_proj surgery). ~240GB weights won't
                # fit 1 H200; device_map=auto shards across the GPUs. 4xH200 (564GB)
                # over 2xH200: the model spreads thinner -> far more memory for the
                # per-trial eval batch -> ~1.3-1.8x faster trials (heretic pipelines
                # a single model, so the gain is batch headroom, not linear).
                # disk ~650: 240GB bf16 weights + ~240GB export + env (~480GB peak).
                # reliability>0.98 avoids flaky deploy hosts.
                query="gpu_name=H200 num_gpus=4 disk_space>=600 reliability>0.98 rentable=true",
                disk_gb=650,
                interruptible=args.interruptible,
            )
        host, port = deploy_and_launch(instance, args.model, args.n_trials, args.family)

        final_status = poll_until_done(host, port, REMOTE_STATUS_PATH, Status, Stage.DONE, POLL_INTERVAL_SECONDS)
        verdict = final_status.verdict or Verdict.ERROR

        local_log_path = os.path.join(STAGE1_DIR, "heretic_run.log")
        try:
            ssh_utils.scp_from(host, port, REMOTE_LOG_PATH, local_log_path)
        except Exception as error:
            print(f"warning: failed to pull run log: {error}", file=sys.stderr)

        print(final_status.to_json())
    finally:
        # Always release the instance — any verdict (incl. fail/error) and any
        # exception past provision() must not leave it billing indefinitely.
        if instance is not None:
            try:
                vast.stop_instance(id=instance["id"])
            except Exception as error:
                print(
                    f"warning: failed to stop instance {instance['id']}: {error}; "
                    "stop it manually to avoid continued billing",
                    file=sys.stderr,
                )

    return 0 if verdict is Verdict.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())

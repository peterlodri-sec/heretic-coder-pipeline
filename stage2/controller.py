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
from shared.vast_ops import load_api_key, local_hf_token_path, provision_lock
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
# 2x H200: gpt-oss's fused MoE experts can't be 4-bit-quantized by bitsandbytes
# (bnb only quantizes nn.Linear), so they stay bf16 -> ~138GB resident, which
# leaves no training headroom on ONE 141GB H200 (runs #4-5 OOM'd at step 1).
# device_map="auto" shards the weights across two H200s (~280GB) with ample room
# for activations. num_gpus>=2 same-host so the shard is NVLink-local.
PROVISION_QUERY = "gpu_name=H200 num_gpus>=2 disk_space>=500 rentable=true"
PROVISION_DISK_GB = 500  # 218GB bf16 model + 5 datasets + LoRA + gguf export
SSH_USER = "root"


def _forward_env(*names: str) -> str:
    """Shell env-prefix for the launch command, including only vars actually set in
    the controller's environment — unset ones fall back to the remote code
    defaults. Values are shell-escaped."""
    import shlex
    return "".join(
        f"{name}={shlex.quote(val)} "
        for name in names
        if (val := os.environ.get(name)) is not None
    )


def deploy_and_launch(instance: dict, model: str, max_steps: int, crabcc_traces: str,
                      check_swebench: bool, family: str = "gpt_oss"):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.wait_for_ssh(host, port)  # fresh instance: wait for sshd before transferring
    token = local_hf_token_path()
    if token:
        ssh_utils.run_ssh(host, port, "mkdir -p /root/.cache/huggingface")
        ssh_utils.scp_to(host, port, token, "/root/.cache/huggingface/token")
    # Ship the drive.file-scoped rclone.conf so the box can back models up to
    # Drive (a third copy alongside HF). Scoped token -> a leaked box can only
    # touch our backup files. Skipped cleanly if not configured locally.
    rclone_conf = os.path.expanduser("~/.config/rclone/rclone.conf")
    if os.path.exists(rclone_conf):
        ssh_utils.run_ssh(host, port, "mkdir -p /root/.config/rclone")
        ssh_utils.scp_to(host, port, rclone_conf, "/root/.config/rclone/rclone.conf")
    ssh_utils.send_dir(host, port, SHARED_DIR, REMOTE_PARENT)
    ssh_utils.send_dir(host, port, STAGE2_DIR, REMOTE_PARENT)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh",
                      timeout=SETUP_TIMEOUT_SECONDS)
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && HF_HUB_ENABLE_HF_TRANSFER=1 "
        f"STAGE2_MODEL='{model}' STAGE2_FAMILY='{family}' STAGE2_MAX_STEPS='{max_steps}' "
        f"STAGE2_CRABCC_TRACES='{crabcc_traces}' "
        f"STAGE2_CHECK_SWEBENCH='{int(check_swebench)}' "
        # Forward the SFT knobs so they're controllable per-run. Packing defaults
        # ON (bfd), but the first real 120B run can pass STAGE2_PACKING=0 to use the
        # proven completion-masking path while the packing x masking composition is
        # still unverified. NEFTune off by default.
        f"STAGE2_PACKING='{os.environ.get('STAGE2_PACKING', '1')}' "
        f"STAGE2_NEFTUNE='{os.environ.get('STAGE2_NEFTUNE', '0')}' "
        # Memory / attention knobs for the gpt-oss plain-bnb path (experts stay
        # bf16 -> tight on one H200). Only forwarded when set locally, so code
        # defaults (eager attn, seq 16384, batch 2) apply otherwise.
        + _forward_env("STAGE2_ATTN", "STAGE2_MAX_SEQ_LEN", "STAGE2_BATCH",
                       "STAGE2_GRAD_ACCUM", "STAGE2_INCLUDE_SWEGYM",
                       "PYTORCH_CUDA_ALLOC_CONF")
        + "tmux new-session -d -s sft 'python3 run_stage2.py'"
    )
    return host, port


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PeetPedro/gpt-oss-120b-heretic")
    parser.add_argument("--family", default="gpt_oss")
    parser.add_argument("--crabcc-traces", dest="crabcc_traces", default="traces")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--no-swebench", dest="check_swebench", action="store_false")
    # Cost lever (plan Gemini): SFT is long + single-GPU + checkpoints, so an
    # interruptible H200 (~40-60% cheaper) resumes cleanly after a preempt ->
    # default ON. Force on-demand with --on-demand or INTERRUPTIBLE=0.
    parser.add_argument("--interruptible", action="store_true",
                        default=os.environ.get("INTERRUPTIBLE", "1") == "1")
    parser.add_argument("--on-demand", dest="interruptible", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vast = VastAI(api_key=load_api_key())

    instance = None
    verdict = Verdict.ERROR
    try:
        with provision_lock():
            instance = vast_provision.provision(
                vast, label=PROVISION_LABEL, query=PROVISION_QUERY, disk_gb=PROVISION_DISK_GB,
                interruptible=args.interruptible)
        host, port = deploy_and_launch(instance, args.model, args.max_steps, args.crabcc_traces,
                                       args.check_swebench, args.family)

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

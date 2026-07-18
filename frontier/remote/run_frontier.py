#!/usr/bin/env python3
# frontier/remote/run_frontier.py — on-box orchestrator for the 480B pipeline:
#   (stage 1) heretic abliteration      [skippable: de-align via ORPO data only]
#   (stage 2) Axolotl QLoRA SFT (8xH200) -> merge LoRA
#   (stage 3) LLaMA-Factory ORPO        -> merge LoRA
#   (eval)    shared.eval.* -> shared.verdict -> publish
# Writes status.json through the same lifecycle contract as run_stage2 so
# shared.poll.poll_until_done works. Heavy imports stay function-local.
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # remote/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # frontier/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root -> shared

import dataprep
import verdict
from enums import Stage
from shared.enums import Verdict
from status_io import Status

MODEL_SOURCE = os.environ.get("FRONTIER_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
STAGE_SELECT = os.environ.get("FRONTIER_STAGE", "all")
CHECK_SWEBENCH = os.environ.get("FRONTIER_CHECK_SWEBENCH", "1") == "1"
SKIP_HERETIC = os.environ.get("FRONTIER_SKIP_HERETIC", "0") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/workspace/data"
SFT_YAML = os.path.join(HERE, "sft_axolotl.yaml")
ORPO_YAML = os.path.join(HERE, "orpo_llamafactory.yaml")
HERETIC_EXPORT = "/workspace/out/heretic"
SFT_OUT = "/workspace/out/sft"
SFT_MERGED = "/workspace/out/sft-merged"
ORPO_OUT = "/workspace/out/orpo"
ORPO_MERGED = "/workspace/out/orpo-merged"
HF_REPO_ID = "PeetPedro/qwen3-coder-480b-heretic-sft-orpo"
STATUS_PATH = os.path.join(HERE, "status.json")
LOG_PATH = os.path.join(HERE, "frontier_run.log")
WALL_CLOCK_CEILING_SECONDS = 48 * 60 * 60


def update_status(status: Status, **fields) -> None:
    for name, value in fields.items():
        setattr(status, name, value)  # slots => unknown field raises
    status.updated_at = str(time.time())
    status.write(STATUS_PATH)


def tail(path: str, n_chars: int = 4000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_chars))
        return f.read().decode("utf-8", errors="replace")


def run_cmd(cmd, cwd=None) -> None:
    """Run a training/merge subprocess, streaming to the run log; raise on failure."""
    with open(LOG_PATH, "a") as logf:
        logf.write(f"\n$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                              timeout=WALL_CLOCK_CEILING_SECONDS)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def run_heretic() -> None:
    # heretic reads heretic_config.toml (bnb_4bit + 8-GPU max_memory) from cwd.
    run_cmd(["heretic", "--model", MODEL_SOURCE,
             "--model-action", "save", "--save-directory", HERETIC_EXPORT],
            cwd=HERE)


def run_sft() -> None:
    run_cmd(["accelerate", "launch", "--num_processes", "8",
             "-m", "axolotl.cli.train", SFT_YAML])


def merge_sft() -> None:
    run_cmd(["python3", "-m", "axolotl.cli.merge_lora", SFT_YAML,
             f"--lora_model_dir={SFT_OUT}", f"--output_dir={SFT_MERGED}"])


def run_orpo() -> None:
    run_cmd(["llamafactory-cli", "train", ORPO_YAML])


def merge_orpo() -> None:
    run_cmd(["llamafactory-cli", "export",
             f"--model_name_or_path={SFT_MERGED}",
             f"--adapter_name_or_path={ORPO_OUT}",
             "--template=qwen3_nothink", "--finetuning_type=lora",
             f"--export_dir={ORPO_MERGED}"])


def _evaluate(target: str, check_swebench: bool) -> dict:
    # shared.eval loads via device_map="auto" fallback at 480B (vLLM-EP transport
    # is a later optimization — plan Phase 3).
    from shared.eval import bfcl as eval_bfcl
    from shared.eval import datasets as eval_datasets
    from shared.eval import humaneval as eval_humaneval
    from shared.eval import refusal as eval_refusal
    from shared.eval import swebench as eval_swebench

    refusal_prompts = eval_datasets.load_refusal_prompts()
    bfcl_cases = eval_datasets.load_bfcl_cases()
    return {
        "refusal_rate": eval_refusal.refusal_rate(target, refusal_prompts),
        "bfcl_accuracy": eval_bfcl.accuracy(target, bfcl_cases),
        "humaneval_delta": eval_humaneval.regression(MODEL_SOURCE, target),
        "swebench_resolve": (
            eval_swebench.resolve_rate(target, model_name="candidate", limit=100)
            if check_swebench else 1.0
        ),
    }


def publish(status: Status, folder: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=HF_REPO_ID)
    update_status(status, hf_repo=HF_REPO_ID)


def fail(status: Status, message: str) -> None:
    update_status(status, stage=Stage.DONE, verdict=Verdict.ERROR,
                  error=message, log_tail=tail(LOG_PATH))


def main(stage: str = "all", check_swebench: bool = True, skip_heretic: bool = False) -> None:
    status = Status.new(str(time.time()))
    status.write(STATUS_PATH)
    eval_target = MODEL_SOURCE

    if stage in ("heretic", "all") and not skip_heretic:
        update_status(status, stage=Stage.ABLITERATING)
        try:
            run_heretic()
            eval_target = HERETIC_EXPORT
        except Exception as error:
            return fail(status, f"heretic failed: {error}")

    update_status(status, stage=Stage.PREPARING_DATA)
    try:
        dataprep.build_all(DATA_DIR)
    except Exception as error:
        return fail(status, f"data prep failed: {error}")

    if stage in ("sft", "orpo", "all"):
        update_status(status, stage=Stage.TRAINING_SFT)
        try:
            run_sft()
            merge_sft()
            eval_target = SFT_MERGED
        except Exception as error:
            return fail(status, f"sft failed: {error}")

    if stage in ("orpo", "all"):
        update_status(status, stage=Stage.TRAINING_ORPO)
        try:
            run_orpo()
            merge_orpo()
            eval_target = ORPO_MERGED
        except Exception as error:
            return fail(status, f"orpo failed: {error}")

    update_status(status, stage=Stage.EVALUATING)
    try:
        metrics = _evaluate(eval_target, check_swebench)
    except Exception as error:
        return fail(status, f"evaluation failed: {error}")

    result = verdict.compute_verdict(metrics, check_swebench=check_swebench)
    update_status(
        status,
        refusal_rate=metrics["refusal_rate"],
        bfcl_accuracy=metrics["bfcl_accuracy"],
        humaneval_delta=metrics["humaneval_delta"],
        swebench_resolve=metrics["swebench_resolve"],
        verdict=result.verdict,
        error=None if result.passed else str(result),
    )

    if result.passed:
        try:
            publish(status, eval_target)
        except Exception as error:
            update_status(status, error=f"HF publish failed: {error}")

    update_status(status, stage=Stage.DONE, log_tail=tail(LOG_PATH))


if __name__ == "__main__":
    main(STAGE_SELECT, CHECK_SWEBENCH, SKIP_HERETIC)

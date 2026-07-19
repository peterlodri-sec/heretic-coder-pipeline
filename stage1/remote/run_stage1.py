#!/usr/bin/env python3
# stage1/remote/run_stage1.py
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import capability_eval
import study_metrics
import verdict
from enums import Stage
from shared.enums import Verdict
from status_io import Status

MODEL = os.environ.get("STAGE1_MODEL", "unsloth/gpt-oss-120b-BF16")
FAMILY = os.environ.get("STAGE1_FAMILY", "gpt_oss")
N_TRIALS = int(os.environ.get("STAGE1_N_TRIALS", "200"))
STUDY_CHECKPOINT_DIR = "checkpoints"
EXPORT_DIR = "heretic_export"
TRIAL_INDEX = 0
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
HERETIC_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heretic_run.log")
HF_REPO_ID = "PeetPedro/gpt-oss-120b-heretic"
# Heretic quantization is DECOUPLED from the training-stage load_in_4bit: for
# gpt-oss we abliterate the BF16 source (heretic has no MXFP4 path; bf16 is the
# only way its surgery reaches down_proj/experts) with quantization="none".
# device_map/max_memory (2xH200 shard) live in config.toml. Falsy => heretic
# default "none", no --quantization flag emitted. Override via STAGE1_QUANTIZATION.
_q = os.environ.get("STAGE1_QUANTIZATION", "none").strip().lower()
QUANTIZATION = None if _q in ("", "none") else _q
WALL_CLOCK_CEILING_SECONDS = 24 * 60 * 60


class HereticError(RuntimeError):
    """The heretic abliteration subprocess failed to run or exited non-zero."""


def update_status(status: Status, **fields) -> None:
    for name, value in fields.items():
        setattr(status, name, value)  # slots => unknown field raises, not silently added
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


def run_heretic() -> None:
    cmd = [
        "heretic", "--model", MODEL,
        "--export-strategy", "merge",
        "--checkpoint-action", "continue",
        "--trial-index", str(TRIAL_INDEX),
        "--model-action", "save",
        "--save-directory", EXPORT_DIR,
        "--study-checkpoint-dir", STUDY_CHECKPOINT_DIR,
        "--n-trials", str(N_TRIALS),
    ]
    # bitsandbytes quantization (heretic `quantization` option) so the 120B fits.
    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]
    with open(HERETIC_LOG_PATH, "a") as logf:
        try:
            proc = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                timeout=WALL_CLOCK_CEILING_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise HereticError("wall-clock ceiling exceeded") from error
        except OSError as error:
            raise HereticError(f"failed to launch heretic: {error}") from error
    if proc.returncode != 0:
        raise HereticError(f"heretic exited with code {proc.returncode}")


def fail(status: Status, message: str) -> None:
    update_status(status, stage=Stage.DONE, verdict=Verdict.ERROR,
                  error=message, log_tail=tail(HERETIC_LOG_PATH))


def publish(status: Status) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=EXPORT_DIR, repo_id=HF_REPO_ID)
    update_status(status, hf_repo=HF_REPO_ID)


def main() -> None:
    status = Status.new(str(time.time()))
    status.write(STATUS_PATH)

    update_status(status, stage=Stage.ABLITERATING)
    try:
        run_heretic()
    except HereticError as error:
        return fail(status, str(error))

    update_status(status, stage=Stage.EVALUATING)
    try:
        scores = study_metrics.load_chosen_trial_scores(STUDY_CHECKPOINT_DIR, MODEL, TRIAL_INDEX)
        # run_benchmarks frees its 120B (del + empty_cache) before returning, so
        # the base model is released before the candidate loads — the two are
        # never GPU-resident at once. base_results is a small metrics dict only.
        base_results = capability_eval.run_benchmarks(MODEL)
        candidate_results = capability_eval.run_benchmarks(EXPORT_DIR)
        deltas = capability_eval.compute_deltas(base_results, candidate_results)
    except Exception as error:
        return fail(status, f"evaluation failed: {error}")

    metrics = {**scores, **deltas}
    result = verdict.compute_verdict(metrics)

    update_status(
        status,
        refusal_rate=metrics["refusal_rate"],
        kl_divergence=metrics["kl_divergence"],
        mmlu_delta=metrics["mmlu_delta"],
        gsm8k_delta=metrics["gsm8k_delta"],
        verdict=result.verdict,
        error=None if result.passed else str(result),
    )

    match result.verdict:
        case Verdict.PASS:
            try:
                publish(status)
            except Exception as error:
                update_status(status, error=f"HF publish failed: {error}")
        case _:
            pass

    update_status(status, stage=Stage.DONE, log_tail=tail(HERETIC_LOG_PATH))


if __name__ == "__main__":
    main()

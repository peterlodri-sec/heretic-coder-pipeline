#!/usr/bin/env python3
# stage1/remote/run_stage1.py
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import capability_eval
import status_io
import study_metrics
import verdict

MODEL = os.environ.get("STAGE1_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
N_TRIALS = int(os.environ.get("STAGE1_N_TRIALS", "200"))
STUDY_CHECKPOINT_DIR = "checkpoints"
EXPORT_DIR = "heretic_export"
TRIAL_INDEX = 0
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
HERETIC_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heretic_run.log")
HF_REPO_ID = "PeetPedro/qwen2.5-coder-32b-instruct-heretic"
WALL_CLOCK_CEILING_SECONDS = 24 * 60 * 60


def update_status(status: dict, **fields) -> None:
    status.update(fields)
    status["updated_at"] = str(time.time())
    status_io.write_status(STATUS_PATH, status)


def tail(path: str, n_chars: int = 4000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_chars))
        return f.read().decode("utf-8", errors="replace")


def run_heretic():
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
    with open(HERETIC_LOG_PATH, "a") as logf:
        try:
            proc = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                timeout=WALL_CLOCK_CEILING_SECONDS,
            )
            return proc.returncode, None
        except subprocess.TimeoutExpired:
            return None, "wall-clock ceiling exceeded"
        except OSError as error:
            return None, f"failed to launch heretic: {error}"


def main():
    start_time = time.time()
    status = status_io.new_status(str(start_time))
    status_io.write_status(STATUS_PATH, status)

    update_status(status, stage="abliterating")
    returncode, run_error = run_heretic()

    if returncode is None:
        update_status(status, stage="done", verdict="error",
                       error=run_error, log_tail=tail(HERETIC_LOG_PATH))
        return
    if returncode != 0:
        update_status(status, stage="done", verdict="error",
                       error=f"heretic exited with code {returncode}",
                       log_tail=tail(HERETIC_LOG_PATH))
        return

    update_status(status, stage="evaluating")

    try:
        scores = study_metrics.load_chosen_trial_scores(STUDY_CHECKPOINT_DIR, MODEL, TRIAL_INDEX)
        base_results = capability_eval.run_benchmarks(MODEL)
        candidate_results = capability_eval.run_benchmarks(EXPORT_DIR)
        deltas = capability_eval.compute_deltas(base_results, candidate_results)
    except Exception as error:
        update_status(status, stage="done", verdict="error",
                       error=f"evaluation failed: {error}", log_tail=tail(HERETIC_LOG_PATH))
        return

    metrics = {**scores, **deltas}
    result = verdict.compute_verdict(metrics)

    update_status(
        status,
        refusal_rate=metrics["refusal_rate"],
        kl_divergence=metrics["kl_divergence"],
        mmlu_delta=metrics["mmlu_delta"],
        gsm8k_delta=metrics["gsm8k_delta"],
        verdict=result["verdict"],
        error=None if result["verdict"] == "pass" else "; ".join(result["reasons"]),
    )

    if result["verdict"] == "pass":
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
            api.upload_folder(folder_path=EXPORT_DIR, repo_id=HF_REPO_ID)
            update_status(status, hf_repo=HF_REPO_ID)
        except Exception as error:
            update_status(status, error=f"HF publish failed: {error}")

    update_status(status, stage="done", log_tail=tail(HERETIC_LOG_PATH))


if __name__ == "__main__":
    main()

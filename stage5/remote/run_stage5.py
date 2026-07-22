#!/usr/bin/env python3
# stage5/remote/run_stage5.py — RLVR (execution-feedback RL) orchestration.
# Mirrors run_stage3: load verifiable-coding data -> train (GRPO/GSPO, stubbed)
# -> merge/export -> subprocess-isolated eval -> verdict -> publish. RLVR is the
# TERMINAL stage for gpt-oss (replaces ORPO).
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # remote/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stage5/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root -> shared

import rlvr_train
import verdict
from enums import Stage
from shared import export
from shared.dataprep import loaders
from shared.enums import Verdict
from status_io import Status

MODEL_SOURCE = os.environ.get("STAGE5_MODEL", "PeetPedro/gpt-oss-120b-heretic-rft")
FAMILY = os.environ.get("STAGE5_FAMILY", "gpt_oss")
NUM_GPUS = int(os.environ.get("STAGE5_NUM_GPUS", "2"))
CHECK_SWEBENCH = os.environ.get("STAGE5_CHECK_SWEBENCH", "1") == "1"
CHEAP_EVAL = os.environ.get("CHEAP_EVAL", "0") == "1"  # reduced SWE-bench in dev
# RLVR mode (plan Gemini §1). distill (default): RFT-on-120B traces -> SFT, cheapest
# good option, sidesteps the KV-cache wall. offline-kto / live-rl: deferred (KTO
# noise + 2-4x H200 KV-cache survival kit) -> NotImplementedError until wired.
MODE = os.environ.get("STAGE5_MODE", "distill")
DATA_PATH = "rlvr_tasks.jsonl"
RLVR_OUT = "swe-coder-rlvr"
MERGED_OUT = "swe-coder-rlvr-final"
GGUF_OUT = "swe-coder-rlvr-final-gguf"
HF_REPO_ID = "PeetPedro/gpt-oss-120b-heretic-rlvr"
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlvr_run.log")


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


def prepare_data(path: str) -> int:
    # Verifiable coding problems (prompt + executable tests) — the RLVR substrate,
    # scored by the exec-test reward. Map the shared loader's rows to
    # {prompt, tests}; exact record schema finalizes with reward.py.
    rows = loaders.load_verifiable_coding_rows()
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({"prompt": row["prompt"], "tests": row["tests"]}) + "\n")
            n += 1
    return n


def _evaluate(check_swebench: bool) -> dict:
    import subprocess
    # Isolate evals from the training process (unsloth monkey-patches transformers).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> /root
    stage_remote = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONPATH": repo_root, "HF_ALLOW_CODE_EVAL": "1",
           "EVAL_FAMILY": FAMILY,  # harmony-aware eval parsing for gpt-oss
           "CHEAP_EVAL": "1" if CHEAP_EVAL else "0"}  # eval runner reads CHEAP_EVAL
    proc = subprocess.run(
        [sys.executable, "-m", "shared.eval.run_evals", MERGED_OUT, MODEL_SOURCE,
         "1" if check_swebench else "0"],
        capture_output=True, text=True, cwd=stage_remote, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"eval subprocess failed: {proc.stderr[-800:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("METRICS_JSON ")]
    if not lines:
        raise RuntimeError(f"eval produced no metrics: {proc.stdout[-800:]}{proc.stderr[-400:]}")
    return json.loads(lines[-1][len("METRICS_JSON "):])


PIPELINE_URL = "https://github.com/entropy-om/heretic-coder-pipeline"


def publish(status: Status) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=GGUF_OUT, repo_id=HF_REPO_ID)
    try:  # every new repo gets a model card; a card failure never fails the publish
        from shared.model_card import push_card
        push_card(HF_REPO_ID, MODEL_SOURCE, "rlvr", family=FAMILY, pipeline_url=PIPELINE_URL,
                  metrics={"train_loss": status.train_loss, "refusal_rate": status.refusal_rate,
                           "bfcl_accuracy": status.bfcl_accuracy, "humaneval_delta": status.humaneval_delta,
                           "swebench_resolve": status.swebench_resolve})
    except Exception as error:
        print(f"model card push failed (non-fatal): {error}")
    update_status(status, hf_repo=HF_REPO_ID)


def _train_for_mode(mode: str):
    """Dispatch the terminal RLVR training path by mode. Only `distill` is wired
    (routes the execution-filtered data through the GSPO trainer). live-rl and
    offline-kto raise until their machinery (KV-cache kit / KTO pairs) lands."""
    if mode == "distill":
        return rlvr_train.train(MODEL_SOURCE, DATA_PATH, RLVR_OUT,
                                num_gpus=NUM_GPUS, family=FAMILY)
    raise NotImplementedError(
        f"stage5 mode '{mode}' not implemented; only 'distill' is wired. "
        "live-rl needs the 2-4x H200 KV-cache survival kit; offline-kto needs "
        "RFT-labeled preference pairs — see the 2026-07-19 plan (Gemini §1).")


def fail(status: Status, message: str) -> None:
    update_status(status, stage=Stage.DONE, verdict=Verdict.ERROR,
                  error=message, log_tail=tail(LOG_PATH))


def main(check_swebench: bool = True) -> None:
    status = Status.new(str(time.time()))
    status.write(STATUS_PATH)

    update_status(status, stage=Stage.PREPARING_DATA)
    try:
        prepare_data(DATA_PATH)
    except Exception as error:
        return fail(status, f"data prep failed: {error}")

    update_status(status, stage=Stage.TRAINING)
    try:
        reward, model, tokenizer = _train_for_mode(MODE)
        update_status(status, train_loss=reward)
        export.export_model(model, tokenizer, MERGED_OUT, GGUF_OUT)
    except Exception as error:
        return fail(status, f"training failed: {error}")

    update_status(status, stage=Stage.EVALUATING)
    try:
        metrics = _evaluate(check_swebench)
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

    match result.verdict:
        case Verdict.PASS:
            try:
                publish(status)
            except Exception as error:
                update_status(status, error=f"HF publish failed: {error}")
        case _:
            pass

    update_status(status, stage=Stage.DONE, log_tail=tail(LOG_PATH))


if __name__ == "__main__":
    main(CHECK_SWEBENCH)

#!/usr/bin/env python3
# stage3/remote/run_stage3.py
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # remote/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stage3/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root -> shared

import orpo_train
import verdict
from dataprep import pipeline as dataprep_pipeline
from dataprep.pairs.crabcc import CrabccPairs
from dataprep.pairs.toolace import ToolACEPairs
from dataprep.pairs.xlam import XLAMPairs
from enums import Stage
from shared import export
from shared.enums import Verdict
from status_io import Status

MODEL_SOURCE = os.environ.get("STAGE3_MODEL", "PeetPedro/gpt-oss-120b-heretic-sft")
FAMILY = os.environ.get("STAGE3_FAMILY", "gpt_oss")  # drives 4-bit (gpt-oss ORPO)
CRABCC_TRACE_DIR = os.environ.get("STAGE3_CRABCC_TRACES", "traces")
CHECK_SWEBENCH = os.environ.get("STAGE3_CHECK_SWEBENCH", "1") == "1"
CHEAP_EVAL = os.environ.get("CHEAP_EVAL", "0") == "1"  # reduced SWE-bench in dev
DATA_PATH = "pairs.jsonl"
ORPO_OUT = "swe-coder-orpo"
MERGED_OUT = "swe-coder-orpo-final"
GGUF_OUT = "swe-coder-orpo-final-gguf"
HF_REPO_ID = "PeetPedro/gpt-oss-120b-heretic-orpo"
NUM_EPOCHS = int(os.environ.get("STAGE3_EPOCHS", "1"))
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orpo_run.log")
CONTAMINATED = frozenset()


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


def _sources():
    return [XLAMPairs(), ToolACEPairs(), CrabccPairs(trace_dir=CRABCC_TRACE_DIR)]


def _evaluate(check_swebench: bool) -> dict:
    import subprocess
    # Isolate evals from the training process (unsloth monkey-patches transformers,
    # which breaks plain model loading in eval).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> /root
    stage_remote = os.path.dirname(os.path.abspath(__file__))  # where MERGED_OUT is relative to
    env = {**os.environ, "PYTHONPATH": repo_root, "HF_ALLOW_CODE_EVAL": "1",
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
    import json as _json
    return _json.loads(lines[-1][len("METRICS_JSON "):])


def publish(status: Status) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=GGUF_OUT, repo_id=HF_REPO_ID)
    update_status(status, hf_repo=HF_REPO_ID)


def fail(status: Status, message: str) -> None:
    update_status(status, stage=Stage.DONE, verdict=Verdict.ERROR,
                  error=message, log_tail=tail(LOG_PATH))


def main(check_swebench: bool = True) -> None:
    status = Status.new(str(time.time()))
    status.write(STATUS_PATH)

    update_status(status, stage=Stage.PREPARING_DATA)
    try:
        dataprep_pipeline.build(_sources(), DATA_PATH, contaminated=CONTAMINATED)
    except Exception as error:
        return fail(status, f"data prep failed: {error}")

    update_status(status, stage=Stage.TRAINING)
    try:
        loss, model, tokenizer = orpo_train.train(MODEL_SOURCE, DATA_PATH, ORPO_OUT,
                                                  num_epochs=NUM_EPOCHS, family=FAMILY)
        update_status(status, train_loss=loss)
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

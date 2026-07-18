#!/usr/bin/env python3
# stage2/remote/run_stage2.py
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # remote/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stage2/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root -> shared

import export
import sft_train
import verdict
from dataprep import pipeline as dataprep_pipeline
from dataprep.sources.bfcl import BFCLSource
from dataprep.sources.crabcc import CrabccSource
from dataprep.sources.magicoder import MagicoderSource
from dataprep.sources.swebench import SWEBenchSource
from dataprep.sources.toolace import ToolACESource
from enums import Stage
from shared.enums import Verdict
from status_io import Status

MODEL_SOURCE = os.environ.get("STAGE2_MODEL", "PeetPedro/qwen2.5-coder-32b-instruct-heretic")
CRABCC_TRACE_DIR = os.environ.get("STAGE2_CRABCC_TRACES", "traces")
DATA_PATH = "train.jsonl"
SFT_OUT = "swe-coder-sft"
MERGED_OUT = "swe-coder-final"
GGUF_OUT = "swe-coder-final-gguf"
HF_REPO_ID = "PeetPedro/qwen2.5-coder-32b-instruct-heretic-sft"
MAX_STEPS = int(os.environ.get("STAGE2_MAX_STEPS", "-1"))
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_run.log")
CONTAMINATED = frozenset()  # extend if a contaminated source is added later

REFUSAL_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refusal_prompts.txt")
BFCL_CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfcl_cases.jsonl")
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"


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
    return [
        SWEBenchSource(), BFCLSource(), ToolACESource(),
        MagicoderSource(), CrabccSource(trace_dir=CRABCC_TRACE_DIR),
    ]


def _evaluate(check_swebench: bool) -> dict:
    import eval_bfcl
    import eval_humaneval
    import eval_refusal
    import eval_swebench
    import json

    with open(REFUSAL_PROMPTS_FILE) as f:
        refusal_prompts = [line.strip() for line in f if line.strip()]
    with open(BFCL_CASES_FILE) as f:
        bfcl_cases = [json.loads(line) for line in f if line.strip()]

    metrics = {
        "refusal_rate": eval_refusal.refusal_rate(MERGED_OUT, refusal_prompts),
        "bfcl_accuracy": eval_bfcl.accuracy(MERGED_OUT, bfcl_cases),
        "humaneval_delta": eval_humaneval.regression(MODEL_SOURCE, MERGED_OUT),
        "swebench_resolve": (
            eval_swebench.resolve_rate(MERGED_OUT, SWEBENCH_DATASET) if check_swebench else 1.0
        ),
    }
    return metrics


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
        loss, model, tokenizer = sft_train.train(MODEL_SOURCE, DATA_PATH, SFT_OUT, max_steps=MAX_STEPS)
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
    main()

#!/usr/bin/env python3
# stage4/remote/run_stage4.py — RFT / rejection-sampling self-improvement LOOP.
# K rounds of: generate N candidates -> filter to test-passers via the shared
# exec sandbox -> SFT on the passing set (reusing stage2's sft_train) -> repeat.
# Also emits execution-grounded {chosen, rejected} pairs for free. The loop
# control (round counter, status, verdict) is fully wired here; only the
# generate() backend is a stubbed interface (rft_generate).
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # remote/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stage4/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root -> shared, stage2

import rft_generate
import verdict
from enums import Stage
from shared import export
from shared import exec_sandbox
from shared.dataprep import loaders
from shared.enums import Verdict
# RFT reuses stage2's SFT trainer (top-level import is GPU-free: sft_train's heavy
# imports are function-local). NOTE (finalize from research, do NOT implement here):
# the eventual SFT recipe upgrades are rsLoRA r=64 + PiSSA init + NEFTune + sample
# packing, harmony chat format, and decontaminated SWE-Gym/OpenHands agentic
# trajectories — see the 2026-07-19 plan; those land in stage2's sft_train, not here.
from stage2.remote import sft_train
from status_io import Status

MODEL_SOURCE = os.environ.get("STAGE4_MODEL", "PeetPedro/gpt-oss-120b-heretic-sft")
FAMILY = os.environ.get("STAGE4_FAMILY", "gpt_oss")  # threaded into reused sft_train
# Cost lever (plan Gemini): RFT plateaus fast — 2 rounds x N=8 candidates is the
# diminishing-returns sweet spot; more rounds mostly burn compute.
NUM_ROUNDS = int(os.environ.get("STAGE4_ROUNDS", "2"))
NUM_CANDIDATES = int(os.environ.get("STAGE4_NUM_CANDIDATES", "8"))
CHECK_SWEBENCH = os.environ.get("STAGE4_CHECK_SWEBENCH", "1") == "1"
CHEAP_EVAL = os.environ.get("CHEAP_EVAL", "0") == "1"  # reduced SWE-bench in dev
HF_REPO_ID = "PeetPedro/gpt-oss-120b-heretic-rft"
GGUF_OUT = "swe-coder-rft-final-gguf"
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rft_run.log")


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


def _load_problems() -> list[dict]:
    # Verifiable coding problems (prompt + executable tests). Map the shared
    # loader's rows to {prompt, tests}; exact columns finalize with the source.
    rows = loaders.load_verifiable_coding_rows()
    problems = []
    for row in rows:
        problems.append({"prompt": row["prompt"], "tests": row["tests"]})
    return problems


def _verify(problems: list[dict], candidates: list[list[str]]) -> tuple[list, list]:
    # Filter candidates by running each against its problem's tests in the shared
    # hardened sandbox. Keep passers for SFT; pair a passer vs a failer per problem
    # for the execution-grounded {chosen, rejected} by-product.
    passing, pairs = [], []
    for problem, cands in zip(problems, candidates):
        good, bad = [], []
        for code in cands:
            result = exec_sandbox.run_tests(code, problem["tests"])
            (good if result["pass_rate"] >= 1.0 else bad).append(code)
        for code in good:
            passing.append({"prompt": problem["prompt"], "solution": code})
        if good and bad:
            pairs.append({"prompt": problem["prompt"], "chosen": good[0], "rejected": bad[0]})
    return passing, pairs


def _write_sft_jsonl(passing: list[dict], path: str) -> int:
    # Render passers to stage2's messages schema (chat turns).
    with open(path, "w") as f:
        for item in passing:
            messages = [{"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["solution"]}]
            f.write(json.dumps({"messages": messages}) + "\n")
    return len(passing)


def _write_pairs_jsonl(pairs: list[dict], path: str) -> int:
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    return len(pairs)


def _evaluate(merged: str, base: str, check_swebench: bool) -> dict:
    import subprocess
    # Isolate evals from the training process (unsloth monkey-patches transformers).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> /root
    stage_remote = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONPATH": repo_root, "HF_ALLOW_CODE_EVAL": "1",
           "CHEAP_EVAL": "1" if CHEAP_EVAL else "0"}  # eval runner reads CHEAP_EVAL
    proc = subprocess.run(
        [sys.executable, "-m", "shared.eval.run_evals", merged, base,
         "1" if check_swebench else "0"],
        capture_output=True, text=True, cwd=stage_remote, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"eval subprocess failed: {proc.stderr[-800:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("METRICS_JSON ")]
    if not lines:
        raise RuntimeError(f"eval produced no metrics: {proc.stdout[-800:]}{proc.stderr[-400:]}")
    return json.loads(lines[-1][len("METRICS_JSON "):])


def publish(status: Status, gguf_dir: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=gguf_dir, repo_id=HF_REPO_ID)
    update_status(status, hf_repo=HF_REPO_ID)


def fail(status: Status, message: str) -> None:
    update_status(status, stage=Stage.DONE, verdict=Verdict.ERROR,
                  error=message, log_tail=tail(LOG_PATH))


def run_round(status: Status, round_idx: int, model_source: str, problems: list[dict],
              merged_dir: str, gguf_dir: str) -> tuple[float, str]:
    """One RFT round: generate -> verify -> SFT-on-passing -> merge/export.
    Returns (train_loss, merged_dir) so the next round loads the improved model."""
    prompts = [p["prompt"] for p in problems]

    update_status(status, stage=Stage.GENERATING, round=round_idx)
    candidates = rft_generate.generate(model_source, prompts, NUM_CANDIDATES)

    update_status(status, stage=Stage.VERIFYING)
    passing, pairs = _verify(problems, candidates)
    n_pass = _write_sft_jsonl(passing, f"rft_round{round_idx}.jsonl")
    _write_pairs_jsonl(pairs, f"rft_pairs_round{round_idx}.jsonl")
    update_status(status, candidates_generated=len(prompts) * NUM_CANDIDATES,
                  candidates_passing=n_pass)

    update_status(status, stage=Stage.TRAINING)
    loss, model, tokenizer = sft_train.train(model_source, f"rft_round{round_idx}.jsonl",
                                             f"swe-coder-rft-r{round_idx}", family=FAMILY)
    export.export_model(model, tokenizer, merged_dir, gguf_dir)
    update_status(status, train_loss=loss)
    return loss, merged_dir


def main(check_swebench: bool = True) -> None:
    status = Status.new(str(time.time()))
    update_status(status, num_rounds=NUM_ROUNDS)

    try:
        problems = _load_problems()
    except Exception as error:
        return fail(status, f"data load failed: {error}")

    model_source = MODEL_SOURCE
    merged_dir = gguf_dir = None
    try:
        for round_idx in range(NUM_ROUNDS):
            merged_dir = f"swe-coder-rft-r{round_idx}-merged"
            gguf_dir = f"swe-coder-rft-r{round_idx}-gguf" if round_idx == NUM_ROUNDS - 1 else GGUF_OUT
            _, model_source = run_round(status, round_idx, model_source, problems,
                                        merged_dir, gguf_dir)
    except Exception as error:
        return fail(status, f"rft round failed: {error}")

    update_status(status, stage=Stage.EVALUATING)
    try:
        metrics = _evaluate(merged_dir, MODEL_SOURCE, check_swebench)
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
                publish(status, gguf_dir)
            except Exception as error:
                update_status(status, error=f"HF publish failed: {error}")
        case _:
            pass

    update_status(status, stage=Stage.DONE, log_tail=tail(LOG_PATH))


if __name__ == "__main__":
    main(CHECK_SWEBENCH)

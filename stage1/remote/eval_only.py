#!/usr/bin/env python3
"""Standalone capability + refusal eval for an already-published abliterated model.
Used to get the mmlu/gsm8k deltas + refusal_rate that a run missed (e.g. when the
no-op tripwire aborted before eval). Base and candidate load sequentially and are
freed between (OOM-safe), same as run_stage1's eval path. Writes eval_result.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import capability_eval
from shared.eval.datasets import load_refusal_prompts
from shared.eval.refusal import refusal_rate

BASE = os.environ.get("EVAL_BASE", "unsloth/gpt-oss-120b-BF16")
CAND = os.environ.get("EVAL_CAND", "PeetPedro/gpt-oss-120b-heretic")
FAMILY = os.environ.get("EVAL_FAMILY", "gpt_oss")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_result.json")


def write(**k):
    with open(OUT, "w") as f:
        json.dump(k, f, indent=2)


def main():
    try:
        write(stage="refusal")
        ref = refusal_rate(CAND, load_refusal_prompts(), family=FAMILY)
        write(stage="base_bench", refusal_rate=ref)
        base = capability_eval.run_benchmarks(BASE)
        write(stage="candidate_bench", refusal_rate=ref)
        cand = capability_eval.run_benchmarks(CAND)
        deltas = capability_eval.compute_deltas(base, cand)
        write(stage="done", refusal_rate=ref,
              mmlu_delta=deltas["mmlu_delta"], gsm8k_delta=deltas["gsm8k_delta"],
              base=base, candidate=cand)
        print("EVAL DONE:", ref, deltas, flush=True)
    except Exception as e:
        write(stage="error", error=str(e))
        print("EVAL ERROR:", e, flush=True)
        raise


if __name__ == "__main__":
    main()

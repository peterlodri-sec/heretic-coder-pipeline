"""Standalone eval runner, invoked as a subprocess so models load in a clean
process (the training process imports unsloth, which monkey-patches transformers
and breaks plain model loading in eval with 'apply_qkv')."""
import json
import os
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    merged, base, check = argv[0], argv[1], argv[2] == "1"
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from shared.eval import bfcl, humaneval, refusal, swebench
    from shared.eval import datasets as evd
    metrics = {
        "refusal_rate": refusal.refusal_rate(merged, evd.load_refusal_prompts()),
        "bfcl_accuracy": bfcl.accuracy(merged, evd.load_bfcl_cases()),
        "humaneval_delta": humaneval.regression(base, merged),
        "swebench_resolve": (swebench.resolve_rate(merged, model_name="candidate", limit=100)
                             if check else 1.0),
    }
    print("METRICS_JSON " + json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    main()

"""Real SWE-bench Verified flow (verified against swebench 4.1.0).

1. Load the model ONCE and generate a unified-diff patch per instance.
2. Write predictions JSONL: ``{instance_id, model_patch, model_name_or_path}``.
3. Shell out to ``python -m swebench.harness.run_evaluation`` (needs Docker).
4. Parse the harness report ``{model_name "/"→"__"}.{run_id}.json`` in CWD →
   ``resolved_instances / total_instances``.
"""
import json
import os
import shutil
import subprocess
import time

from shared.eval._model import chat_generate, load_model

DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

_SYSTEM = (
    "You are an expert software engineer. Given a bug report, respond with ONLY "
    "a unified diff patch (git diff format) that resolves the issue. Do not "
    "include explanations."
)
_USER = "Resolve the following issue with a unified diff patch:\n\n{problem}"


def _prompt_messages(problem_statement: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(problem=problem_statement)},
    ]


def generate_predictions(model_path, model_name, limit=None) -> str:
    """Generate patches and write a predictions JSONL; return its path."""
    import datasets

    dataset = datasets.load_dataset(DATASET, split=SPLIT)
    instances = list(dataset)
    if limit is not None:
        instances = instances[:limit]

    model, tokenizer = load_model(model_path)
    message_lists = [_prompt_messages(inst["problem_statement"]) for inst in instances]
    patches = chat_generate(model, tokenizer, message_lists, max_new_tokens=1024)

    preds_path = os.path.join(
        os.getcwd(), f"swebench_preds_{model_name.replace('/', '__')}.jsonl"
    )
    with open(preds_path, "w") as f:
        for inst, patch in zip(instances, patches):
            f.write(json.dumps({
                "instance_id": inst["instance_id"],
                "model_patch": patch,
                "model_name_or_path": model_name,
            }) + "\n")
    return preds_path


def resolve_rate(model_path, model_name="candidate", limit=100) -> float:
    """Generate predictions, run the harness, return resolved/total."""
    if shutil.which("docker") is None:
        raise RuntimeError("SWE-bench evaluation requires Docker, which is not available")

    preds_path = generate_predictions(model_path, model_name, limit=limit)
    run_id = f"eval_{int(time.time())}"
    proc = subprocess.run(
        ["python", "-m", "swebench.harness.run_evaluation",
         "--dataset_name", DATASET,
         "--split", SPLIT,
         "--predictions_path", preds_path,
         "--run_id", run_id,
         "--max_workers", "4"],
        capture_output=True, text=True, timeout=6 * 60 * 60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"swebench harness failed: {proc.stderr.strip()}")

    report_path = os.path.join(
        os.getcwd(), f"{model_name.replace('/', '__')}.{run_id}.json"
    )
    with open(report_path) as f:
        report = json.load(f)
    total = report["total_instances"]
    return report["resolved_instances"] / total if total else 0.0

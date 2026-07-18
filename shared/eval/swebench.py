# Heavy agentic harness; gated behind controller config. _run_harness shells out
# to the SWE-bench evaluation harness and returns resolved/total counts.
import json
import subprocess


def _run_harness(model: str, dataset: str) -> dict:
    proc = subprocess.run(
        ["python", "-m", "swebench.harness.run_evaluation",
         "--model", model, "--dataset_name", dataset, "--report_json", "/tmp/swe.json"],
        capture_output=True, text=True, timeout=6 * 60 * 60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"swebench harness failed: {proc.stderr.strip()}")
    with open("/tmp/swe.json") as f:
        return json.load(f)


def resolve_rate(model: str, dataset: str) -> float:
    report = _run_harness(model, dataset)
    total = report["total"]
    return report["resolved"] / total if total else 0.0

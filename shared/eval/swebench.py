"""Real SWE-bench Verified flow (verified against swebench 4.1.0).

1. Load the model ONCE and generate ``SWE_N_SAMPLES`` unified-diff patches per
   instance (candidate 0 is greedy = the deployable pass@1 submission; the rest
   are sampled with min-p for best-of-N diversity, per the SWE research report).
2. Write a predictions JSONL per candidate slate and shell out to
   ``python -m swebench.harness.run_evaluation`` (needs Docker) for each.
3. Report ``resolve_rate`` = **pass@1** (the greedy candidate — honest, single
   submission, what the verdict gate uses) and, when N>1, log **pass@N** (any
   candidate resolves) as the headroom the report quantifies (+17-22pp at N=8).

pass@N is a soft, clearly-labelled TELEMETRY number — never the gate — because a
deployable resolve rate needs a selection oracle (execution fingerprinting / an
ORM), which is a separate piece. This module leaves a clean seam for it: swap the
"candidate 0" submission for a selected candidate once a no-gold-tests selector
exists.
"""
import json
import os
import shutil
import subprocess
import time

from shared.eval._model import chat_generate, free_model, load_model

DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

# Best-of-N knobs. N=1 -> the original single greedy pass (default, unchanged).
N_SAMPLES = max(1, int(os.environ.get("SWE_N_SAMPLES", "1")))
# Diversity sampling for candidates 1..N-1 (report F13): min-p beats top-p at
# temperature; T=1.2 keeps diversity without the top-p collapse.
SAMPLE_GEN_KWARGS = {"do_sample": True, "temperature": 1.2, "min_p": 0.07}

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


def generate_candidates(model_path, model_name, limit=None):
    """Return ``(instances, slates)`` where ``slates[k]`` is the list of one patch
    per instance for candidate k. Slate 0 is greedy; slates 1..N-1 are sampled."""
    import datasets

    dataset = datasets.load_dataset(DATASET, split=SPLIT)
    instances = list(dataset)
    if limit is not None:
        instances = instances[:limit]

    model, tokenizer = load_model(model_path)
    try:
        message_lists = [_prompt_messages(inst["problem_statement"]) for inst in instances]
        slates = [chat_generate(model, tokenizer, message_lists, max_new_tokens=1024)]
        for _ in range(N_SAMPLES - 1):
            slates.append(chat_generate(model, tokenizer, message_lists,
                                        max_new_tokens=1024,
                                        gen_kwargs=SAMPLE_GEN_KWARGS))
    finally:
        # Free the model before the Docker harness runs (and before any next eval
        # loads its own) — 2xH200 cannot hold two 120B models at once.
        del model, tokenizer
        free_model()
    return instances, slates


def generate_predictions(model_path, model_name, limit=None) -> str:
    """Back-compat single-slate (greedy) prediction file — callers that only want
    one prediction per instance. Returns the predictions JSONL path."""
    instances, slates = generate_candidates(model_path, model_name, limit=limit)
    return _write_preds(instances, slates[0], model_name, tag="cand0")


def _write_preds(instances, patches, model_name, tag) -> str:
    preds_path = os.path.join(
        os.getcwd(), f"swebench_preds_{model_name.replace('/', '__')}_{tag}.jsonl")
    with open(preds_path, "w") as f:
        for inst, patch in zip(instances, patches):
            f.write(json.dumps({
                "instance_id": inst["instance_id"],
                "model_patch": patch,
                "model_name_or_path": model_name,
            }) + "\n")
    return preds_path


def _run_harness(preds_path, model_name) -> set:
    """Run the SWE-bench harness on one predictions file; return the set of
    resolved instance_ids."""
    run_id = f"eval_{int(time.time() * 1000)}"
    proc = subprocess.run(
        ["python", "-m", "swebench.harness.run_evaluation",
         "--dataset_name", DATASET, "--split", SPLIT,
         "--predictions_path", preds_path, "--run_id", run_id,
         "--max_workers", "4"],
        capture_output=True, text=True, timeout=6 * 60 * 60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"swebench harness failed: {proc.stderr.strip()}")
    report_path = os.path.join(
        os.getcwd(), f"{model_name.replace('/', '__')}.{run_id}.json")
    with open(report_path) as f:
        report = json.load(f)
    ids = report.get("resolved_ids")
    if ids is None:
        # minimal report (only counts): fabricate ids so pass@1 stays exact; a
        # pass@N union is only meaningful when the harness emits real resolved_ids.
        ids = [f"__resolved_{i}" for i in range(report.get("resolved_instances", 0))]
    return set(ids), report.get("total_instances", 0)


def resolve_rate(model_path, model_name="candidate", limit=100) -> float:
    """Generate candidates, run the harness per slate, return **pass@1** (greedy).

    When SWE_N_SAMPLES>1, also logs pass@N (any candidate resolves) as headroom
    telemetry. pass@1 is what's returned (and gated on) — the honest single-shot
    number; pass@N is never returned as the resolve rate.
    """
    if shutil.which("docker") is None:
        raise RuntimeError("SWE-bench evaluation requires Docker, which is not available")

    instances, slates = generate_candidates(model_path, model_name, limit=limit)
    total = len(instances)
    if total == 0:
        return 0.0

    resolved_per_slate = []
    for k, patches in enumerate(slates):
        preds_path = _write_preds(instances, patches, model_name, tag=f"cand{k}")
        resolved_ids, _reported_total = _run_harness(preds_path, model_name)
        resolved_per_slate.append(resolved_ids)

    pass_at_1 = len(resolved_per_slate[0]) / total
    if N_SAMPLES > 1:
        union = set().union(*resolved_per_slate)
        pass_at_n = len(union) / total
        # parseable by the monitor / logs: SWEBENCH pass@1=.. pass@N=.. N=.. headroom=..
        print(f"SWEBENCH pass@1={pass_at_1:.4f} pass@{N_SAMPLES}={pass_at_n:.4f} "
              f"N={N_SAMPLES} headroom={pass_at_n - pass_at_1:.4f} "
              f"(pass@N is selection-oracle headroom, NOT the gate)", flush=True)
    return pass_at_1

# Reuses the stage1 capability_eval pattern: lm_eval, delta = base - candidate.
import os

TASK = "humaneval"


def _pick_pass_at_1(task_results: dict) -> float:
    """Read pass@1 robustly.

    lm-eval often suffixes the metric key (e.g. ``"pass@1,create_test"``), so
    match on the ``pass@1`` prefix rather than an exact key.
    """
    for key, value in task_results.items():
        if key.startswith("pass@1"):
            return value
    raise KeyError(f"no pass@1 metric in results: {sorted(task_results)}")


def _pass_at_1(model_path_or_id: str) -> float:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    hflm = HFLM(pretrained=model_path_or_id, batch_size="auto")
    out = lm_eval.simple_evaluate(
        model=hflm, tasks=[TASK], confirm_run_unsafe_code=True
    )
    return _pick_pass_at_1(out["results"][TASK])


def regression(base_model: str, candidate_model: str) -> float:
    """Positive delta == candidate regressed vs base."""
    return _pass_at_1(base_model) - _pass_at_1(candidate_model)

# Reuses the stage1 capability_eval pattern: lm_eval, delta = base - candidate.
TASK = "humaneval"


def _pass_at_1(model_path_or_id: str) -> float:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    hflm = HFLM(pretrained=model_path_or_id, batch_size="auto")
    out = lm_eval.simple_evaluate(model=hflm, tasks=[TASK])
    return out["results"][TASK]["pass@1"]


def regression(base_model: str, candidate_model: str) -> float:
    """Positive delta == candidate regressed vs base."""
    return _pass_at_1(base_model) - _pass_at_1(candidate_model)

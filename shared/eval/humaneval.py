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


def _final_answer(text: str, family: str = "gpt_oss") -> str:
    """The text HumanEval's code-extraction should see. For gpt-oss (harmony)
    that is the `final` channel only — the analysis CoT must never reach the
    code extractor. For qwen the raw generation IS the answer. extract_final is
    a no-op on plain, un-channelled text, so this is safe for both."""
    from shared.model_family import ModelFamily

    if ModelFamily(family) is ModelFamily.GPT_OSS:
        from shared.harmony import extract_final
        return extract_final(text)
    return text


def _pass_at_1(model_path_or_id: str, family: str = "gpt_oss") -> float:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Strip the harmony analysis channel BEFORE lm-eval's code extractor runs by
    # intercepting decoded generations in generate_until (qwen path = identity).
    # NOTE: assumes generate_until returns decoded strings in this lm-eval
    # version — verify on the first gpt-oss eval run.
    class _FinalChannelHFLM(HFLM):
        def generate_until(self, requests, **kwargs):
            outs = super().generate_until(requests, **kwargs)
            return [_final_answer(o, family) for o in outs]

    hflm = _FinalChannelHFLM(pretrained=model_path_or_id, batch_size="auto")
    out = lm_eval.simple_evaluate(
        model=hflm, tasks=[TASK], confirm_run_unsafe_code=True
    )
    return _pick_pass_at_1(out["results"][TASK])


def regression(base_model: str, candidate_model: str, family: str = "gpt_oss") -> float:
    """Positive delta == candidate regressed vs base."""
    return _pass_at_1(base_model, family) - _pass_at_1(candidate_model, family)

# stage4/remote/rft_generate.py — candidate sampler for the RFT loop. INTERFACE
# ONLY; the generation backend is finalized from SOTA research (build-order step
# 4, see 2026-07-19 plan). Heavy imports (vLLM / transformers) are function-local
# so this module imports without a GPU for unit tests.
MAX_NEW_TOKENS = 2048


def generate(model_source: str, prompts: list[str], n: int,
             temperature: float = 1.0, max_new_tokens: int = MAX_NEW_TOKENS) -> list[list[str]]:
    """Sample `n` candidate completions per prompt from the current-round model.

    Intended backend: vLLM offline batched generation (fast group sampling) or an
    HF `model.generate` fallback. gpt-oss is a reasoning model in harmony format —
    parse and return the `final`-channel answer (the code to execute), not the
    analysis channel.

    Args:
        model_source: HF repo / local path of the round's current model.
        prompts: coding problems to solve.
        n: candidates to sample per prompt (rejection sampling breadth).
        temperature: sampling temperature (>0 for diversity across the n samples).
        max_new_tokens: per-candidate generation cap.

    Returns:
        list aligned with `prompts`; each element is a list of `n` completion
        strings (harmony `final`-channel answers) to be filtered by exec tests.
    """
    raise NotImplementedError(
        "finalize candidate generation (vLLM/HF, harmony final-channel parse) "
        "from SOTA research — see 2026-07-19 plan")

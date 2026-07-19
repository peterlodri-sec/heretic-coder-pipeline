# stage4/remote/rft_generate.py — candidate sampler for the RFT loop. INTERFACE
# ONLY; the generation backend is finalized from SOTA research (build-order step
# 4, see 2026-07-19 plan). Heavy imports (vLLM / transformers) are function-local
# so this module imports without a GPU for unit tests.
MAX_NEW_TOKENS = 2048


def generate(model_source: str, prompts: list[str], n: int,
             temperature: float = 1.0, max_new_tokens: int = MAX_NEW_TOKENS) -> list[list[str]]:
    """Sample `n` candidate completions per prompt from the current-round model.

    vLLM offline batched generation. The n samples of a prompt share its prefix,
    so **prefix caching** computes the prompt KV once across the group (KV-cache
    kit, plan §Gemini) — the same lever the RLVR rollouts need. gpt-oss is a
    harmony reasoning model: `llm.chat` applies the harmony template, and we return
    the parsed `final`-channel answer (the code to execute), never the analysis CoT.

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
    from vllm import LLM, SamplingParams
    from shared.harmony import extract_final

    llm = LLM(model=model_source, enable_prefix_caching=True,
              kv_cache_dtype="fp8")  # FP8 KV: half the cache, <1% degradation
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_new_tokens)
    # chat() applies gpt-oss's harmony template per conversation.
    convos = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(convos, params)
    # Align to `prompts`; parse each of the n samples' final-channel code.
    return [[extract_final(c.text) for c in out.outputs] for out in outputs]

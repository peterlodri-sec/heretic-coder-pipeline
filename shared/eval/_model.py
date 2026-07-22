"""Single-load, batched, chat-templated model layer for the evals.

Every eval loads the model exactly ONCE (a 32B model reloaded per prompt on
CPU/fp32 never finishes) and generates in batches. Heavy imports (torch,
transformers) live INSIDE the functions so this module stays import-safe in
GPU-free / offline environments, and so the unit tests can patch
``load_model`` / ``chat_generate`` without the heavy deps installed.
"""


def load_model(model_path):
    """Load ``(model, tokenizer)`` once.

    ``device_map="auto"`` + bfloat16 so the 32B weights are placed across the
    available accelerators a single time. The pad token is set when missing so
    batched generation can left-pad.
    """
    import torch  # noqa: F401  (ensures torch is importable before transformers)
    from transformers import AutoModelForCausalLM

    from shared.train_common import load_tokenizer  # AutoTokenizer + TokenizersBackend fallback

    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="bfloat16"
    )
    return model, tokenizer


def free_model(*_objs):
    """Reclaim accelerator memory between sequential large-model evals.

    A single stage's eval pass loads one large model per benchmark in the SAME
    process (refusal, bfcl, humaneval, swebench). Without an explicit free, each
    freshly-loaded model's blocks stay pinned in torch's caching allocator, so a
    second 120B (~240GB bf16) can't fit alongside the first on 2xH200 (282GB).

    The CALLER must also stop referencing its model/tokenizer (``del`` them, or
    let them fall out of scope) BEFORE calling this — Python passes by
    reference, so this cannot unbind the caller's names. This runs a GC pass and
    empties the CUDA cache. Import-safe: torch is imported lazily and guarded, so
    it is a no-op in GPU-free / offline / unit-test environments.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def chat_generate(
    model,
    tokenizer,
    message_lists,
    max_new_tokens=256,
    batch_size=16,
    tools_per_item=None,
    reasoning_effort=None,
):
    """Batched chat generation returning COMPLETIONS ONLY.

    ``message_lists`` is a list of OpenAI-style message lists. For each item the
    chat template is applied with ``add_generation_prompt=True`` (and the
    matching ``tools`` entry from ``tools_per_item`` when provided; the gpt-oss
    harmony template consumes ``tools`` natively). Only the newly-generated
    tokens are decoded, so callers never see the echoed prompt.

    ``reasoning_effort`` (harmony determinism knob, e.g. "low") is forwarded to
    the chat template ONLY when set — a Qwen/ChatML template just ignores the
    unused kwarg, so this stays family-agnostic and never hardcoded.
    """
    import torch

    if not message_lists:
        return []
    if tools_per_item is not None and len(tools_per_item) != len(message_lists):
        raise ValueError("tools_per_item must match message_lists length")
    template_kwargs = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    # Left-pad so the freshly generated tokens are contiguous at the tail of
    # every row and can be sliced off with a single prompt-length offset.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    completions: list[str] = []
    try:
        for start in range(0, len(message_lists), batch_size):
            batch = message_lists[start:start + batch_size]
            prompts = []
            for i, msgs in enumerate(batch):
                tools = None
                if tools_per_item is not None:
                    tools = tools_per_item[start + i]
                prompts.append(
                    tokenizer.apply_chat_template(
                        msgs,
                        add_generation_prompt=True,
                        tokenize=False,
                        tools=tools,
                        **template_kwargs,
                    )
                )
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[:, inputs["input_ids"].shape[1]:]
            completions.extend(
                tokenizer.batch_decode(gen, skip_special_tokens=True)
            )
    finally:
        tokenizer.padding_side = prev_side
    return completions

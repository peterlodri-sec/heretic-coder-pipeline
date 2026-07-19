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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="bfloat16"
    )
    return model, tokenizer


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

"""Harmony-format parsing for gpt-oss output. Pure-python (no heavy deps), so it
imports GPU-free. gpt-oss emits reasoning in channels: the user-facing answer is
the `final` channel; the `analysis` channel is raw CoT that must NOT be scored or
shown. `extract_final` pulls the final-channel body (the code to execute).

Canonical parser — stage4 (rft_generate) uses it; stage5 reward.py has a sibling
`_extract_final` kept private for its committed tests (consolidate later)."""
import re

_FINAL = "<|channel|>final<|message|>"
_STOP = ("<|return|>", "<|end|>", "<|start|>")
_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\n?|\n?```$")


def extract_final(text: str) -> str:
    """Return the harmony `final`-channel body, code fences stripped. If there is
    no final marker (plain, un-channelled text), return the whole thing stripped —
    so non-harmony models/tests still work."""
    idx = text.rfind(_FINAL)
    body = text[idx + len(_FINAL):] if idx != -1 else text
    cut = min((body.find(s) for s in _STOP if body.find(s) != -1), default=-1)
    if cut != -1:
        body = body[:cut]
    body = body.strip()
    return _FENCE.sub("", body).strip()

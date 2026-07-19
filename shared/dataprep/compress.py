"""Optional kompress-v8 context-compression for dataprep.

Prunes low-signal tokens from **agent tool-output spans ONLY** (role="tool"
content) — never code, patches, assistant/final-channel, user, or system text.
kompress-v8 is a keep/drop classifier whose native domain is agent tool-output
context (out-of-domain on code), so this is a gentle ~15% precision-preserving
prune, not a capability change. The heavy `headroom` import is function-local so
this module imports GPU-/dependency-free; the pass is a no-op unless wired on.
"""

KOMPRESS_MODEL = "PeetPedro/kompress-v8"


def _is_tool_output(m: dict) -> bool:
    return m.get("role") == "tool"


def compress_tool_spans(messages: list[dict], *, model: str = KOMPRESS_MODEL) -> list[dict]:
    """Return a NEW list; compress only tool-output content, everything else
    passes through unchanged (same object, byte-identical content). Never mutates
    the input list."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content") if _is_tool_output(m) else None
        if _is_tool_output(m) and isinstance(content, str) and content:
            new_m = dict(m)
            new_m["content"] = _headroom_compress(content, model)
            out.append(new_m)
        else:
            out.append(m)
    return out


def _headroom_compress(text: str, model: str) -> str:
    from headroom import compress  # function-local; optional dep (headroom[ml])

    # VERIFY on box: confirm the exact headroom `compress` return shape AND the
    # must-keep override param name against the installed headroom before trusting
    # this. Must-keep (critical-token) protection MUST be ON so nothing
    # load-bearing is dropped — do NOT fabricate an override kwarg here; if the
    # installed API needs one (e.g. must_keep=... / protect=...), add it once
    # confirmed. Below defensively handles list-of-messages and object
    # (.messages / .content) return shapes, falling back to the original text.
    out = compress([{"role": "tool", "content": text}], model=model)

    # Object with a .messages attribute -> unwrap to the message list.
    if hasattr(out, "messages"):
        out = out.messages
    # Object with a .content attribute -> that's the compressed content directly.
    elif hasattr(out, "content"):
        content = out.content
        return content if isinstance(content, str) else text

    # List/tuple of messages -> take the first message's content.
    if isinstance(out, (list, tuple)) and out:
        first = out[0]
        if isinstance(first, dict):
            content = first.get("content")
        else:
            content = getattr(first, "content", None)
        return content if isinstance(content, str) else text

    return text

"""BFCL / xLAM tool-calling accuracy.

The model is shown the available tool schema (via the chat template ``tools=``
argument) so it can emit a real call, then the emitted tool call is parsed
(family-aware: harmony ``to=functions.NAME`` for gpt-oss, Hermes ``<tool_call>``
for qwen) and compared NORMALIZED (name equality + argument dict equality)
against the gold call.
"""
import json
import re

from shared.eval._model import chat_generate, load_model

# gpt-oss harmony tool call, roughly:
#   <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>
# Anchor on `to=functions.NAME`, take the JSON between the next <|message|> and
# <|call|> (DOTALL so multi-line args match); the `.*?` skips <|constrain|>json.
_HARMONY_CALL = re.compile(
    r"to=functions\.(?P<name>[\w.-]+).*?<\|message\|>(?P<args>.*?)<\|call\|>",
    re.DOTALL,
)


def _extract_harmony_tool_call(text: str) -> dict | None:
    m = _HARMONY_CALL.search(text)
    if not m:
        return None
    raw = m.group("args").strip()
    try:
        args = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        args = raw  # _normalize_args wraps unparseable args as __raw__
    return {"name": m.group("name"), "arguments": args}


def extract_tool_call(text: str, family: str = "gpt_oss") -> dict | None:
    """Parse the model's emitted tool call into ``{"name", "arguments"}``.

    GPT_OSS: parse the harmony ``to=functions.NAME ...<|call|>`` commentary call.
    QWEN: parse the Hermes ``<tool_call>{...}</tool_call>`` wrapper (or a bare JSON
    object). Returns ``None`` when nothing parseable is found.
    """
    from shared.model_family import ModelFamily

    if ModelFamily(family) is ModelFamily.GPT_OSS:
        return _extract_harmony_tool_call(text)
    start = text.find("<tool_call>")
    if start >= 0:
        end = text.find("</tool_call>", start)
        payload = text[start + len("<tool_call>"):end if end >= 0 else None]
    else:
        payload = text
    payload = payload.strip()
    # Fall back to the first {...} span if there is surrounding prose.
    if not payload.startswith("{"):
        brace = payload.find("{")
        if brace < 0:
            return None
        payload = payload[brace:payload.rfind("}") + 1]
    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return obj


def _normalize_args(arguments) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return {"__raw__": arguments}
    if not isinstance(arguments, dict):
        return {}
    return arguments


def _matches(pred: dict | None, expected: dict) -> bool:
    if not pred:
        return False
    if pred.get("name") != expected.get("name"):
        return False
    # NOTE: exact name + arg-dict equality. This is an EXACT-MATCH harness, yet
    # shared/verdict.py floors bfcl_accuracy at 0.85 — likely miscalibrated for
    # exact match (no partial credit, no type coercion). Queued for a human
    # decision; do NOT relax the matcher or the floor without one.
    # Dict equality is order-independent, so reordered keys still match.
    return _normalize_args(pred.get("arguments", {})) == _normalize_args(
        expected.get("arguments", {})
    )


def accuracy(model_path, cases, family: str = "gpt_oss") -> float:
    if not cases:
        return 0.0
    model, tokenizer = load_model(model_path)
    message_lists = [[{"role": "user", "content": c["prompt"]}] for c in cases]
    tools_per_item = [c.get("tools") for c in cases]
    completions = chat_generate(
        model, tokenizer, message_lists,
        max_new_tokens=256, tools_per_item=tools_per_item,
    )
    hits = sum(
        1 for c, out in zip(cases, completions)
        if _matches(extract_tool_call(out, family), c["expected"])
    )
    return hits / len(cases)

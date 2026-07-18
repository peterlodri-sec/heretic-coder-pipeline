"""BFCL / xLAM tool-calling accuracy.

The model is shown the available tool schema (via the chat template ``tools=``
argument) so it can emit a real call, then the emitted Hermes ``<tool_call>``
block is parsed and compared NORMALIZED (name equality + argument dict equality)
against the gold call.
"""
import json

from shared.eval._model import chat_generate, load_model


def extract_tool_call(text: str) -> dict | None:
    """Parse the model's emitted tool call into ``{"name", "arguments"}``.

    Handles the Hermes ``<tool_call>{...}</tool_call>`` wrapper as well as a
    bare JSON object. Returns ``None`` when nothing parseable is found.
    """
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
    # Dict equality is order-independent, so reordered keys still match.
    return _normalize_args(pred.get("arguments", {})) == _normalize_args(
        expected.get("arguments", {})
    )


def accuracy(model_path, cases) -> float:
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
        if _matches(extract_tool_call(out), c["expected"])
    )
    return hits / len(cases)

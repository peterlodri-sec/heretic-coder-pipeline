import json

from shared.dataprep.schema import tool_call_block, tool_response_block

REFUSAL_TEXT = "I can't help with that."


def _parse_call(chosen: str):
    start, end = chosen.find("<tool_call>"), chosen.find("</tool_call>")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(chosen[start + len("<tool_call>"):end].strip())
    except json.JSONDecodeError:
        return None


def _tool_names(tools):
    names = []
    for tool in tools or []:
        name = tool.get("name") if isinstance(tool, dict) else None
        # xLAM occasionally nests the schema under "function".
        if name is None and isinstance(tool, dict):
            name = (tool.get("function") or {}).get("name")
        if name:
            names.append(name)
    return names


def wrong_tool(chosen: str, tools=None) -> str:
    """Swap the call to a DIFFERENT real tool name drawn from `tools`, keeping
    the original arguments. Falls back to a refusal when no alternate real tool
    is available (or `chosen` is not a tool call)."""
    call = _parse_call(chosen)
    if call is None or "name" not in call:
        return REFUSAL_TEXT
    alt = next((n for n in _tool_names(tools) if n != call["name"]), None)
    if alt is None:
        return REFUSAL_TEXT
    return tool_call_block(alt, call.get("arguments", {}))


def wrong_args(chosen: str, tools=None) -> str:
    """Keep the correct tool name but mutate an argument (change a value, or
    inject a spurious key when there are none). Stays valid JSON, wrong content."""
    call = _parse_call(chosen)
    if call is None or "name" not in call:
        return REFUSAL_TEXT
    args = dict(call.get("arguments", {}))
    if args:
        key = next(iter(args))
        args[key] = "__corrupted__" if args[key] != "__corrupted__" else "__corrupted2__"
    else:
        args = {"__unexpected__": "__corrupted__"}
    return tool_call_block(call["name"], args)


def hallucinated_output(chosen: str, tools=None) -> str:
    return chosen + "\n" + tool_response_block("(fabricated) success")


def refusal(chosen: str, tools=None) -> str:
    return REFUSAL_TEXT


STRATEGIES = {
    "wrong_tool": wrong_tool,
    "wrong_args": wrong_args,
    "hallucinated_output": hallucinated_output,
    "refusal": refusal,
}


def make_rejected(chosen: str, strategy: str, tools=None) -> str:
    """Turn a correct assistant completion string into a plausible-but-wrong one.

    `tools` is the list of available tool schemas (used by `wrong_tool` to pick a
    real alternate name). Returns a corrupted COMPLETION STRING; the caller wraps
    it into a `[{"role": "assistant", "content": ...}]` message. Guarantees the
    result differs from `chosen`."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown corruption strategy {strategy!r}")
    result = STRATEGIES[strategy](chosen, tools)
    if result == chosen:
        result = REFUSAL_TEXT if chosen != REFUSAL_TEXT else chosen + " (rejected)"
    return result

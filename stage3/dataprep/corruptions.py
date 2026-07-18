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


def wrong_tool(chosen: str) -> str:
    call = _parse_call(chosen)
    if call is None or "name" not in call:
        return REFUSAL_TEXT
    return tool_call_block(f"not_{call['name']}", call.get("arguments", {}))


def malformed_args(chosen: str) -> str:
    call = _parse_call(chosen)
    if call is None or "name" not in call:
        return REFUSAL_TEXT
    args = call.get("arguments", {})
    # Always differ from `chosen`: empty args -> inject a spurious one;
    # non-empty args -> strip them. Either way it's a malformed-args rejection.
    corrupted = {"__unexpected__": None} if not args else {}
    return tool_call_block(call["name"], corrupted)


def hallucinated_output(chosen: str) -> str:
    return chosen + "\n" + tool_response_block("(fabricated) success")


def refusal(chosen: str) -> str:
    return REFUSAL_TEXT


STRATEGIES = {
    "wrong_tool": wrong_tool,
    "malformed_args": malformed_args,
    "hallucinated_output": hallucinated_output,
    "refusal": refusal,
}


def make_rejected(chosen: str, strategy: str) -> str:
    """Turn a correct assistant completion into a plausible-but-wrong one — the
    four rejected classes from plan.md §3: wrong tool, malformed args,
    hallucinated result, unnecessary refusal."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown corruption strategy {strategy!r}")
    return STRATEGIES[strategy](chosen)

import json

import pytest
from dataprep.corruptions import make_rejected
from shared.dataprep.schema import tool_call_block

CHOSEN = tool_call_block("bash", {"cmd": "ls"})
TOOLS = [{"name": "bash", "parameters": {}}, {"name": "python", "parameters": {}}]


def _call(text):
    inner = text[text.find("<tool_call>") + len("<tool_call>"):text.find("</tool_call>")].strip()
    return json.loads(inner)


def test_wrong_tool_swaps_to_real_alternate_keeps_args():
    rej = make_rejected(CHOSEN, "wrong_tool", tools=TOOLS)
    call = _call(rej)
    # a DIFFERENT real tool name from the schema, not "not_bash"
    assert call["name"] == "python"
    assert "not_" not in call["name"]
    assert call["arguments"] == {"cmd": "ls"}


def test_wrong_tool_without_tools_falls_back_to_refusal():
    rej = make_rejected(CHOSEN, "wrong_tool")
    assert "can't" in rej.lower() or "cannot" in rej.lower()


def test_wrong_args_keeps_tool_name_mutates_args():
    rej = make_rejected(CHOSEN, "wrong_args")
    call = _call(rej)  # still valid JSON
    assert call["name"] == "bash"
    assert call["arguments"] != {"cmd": "ls"}


def test_wrong_args_on_empty_args_still_differs():
    empty = tool_call_block("bash", {})
    rej = make_rejected(empty, "wrong_args")
    assert rej != empty
    assert _call(rej)["name"] == "bash"


def test_hallucinated_output_appends_fake_response():
    rej = make_rejected(CHOSEN, "hallucinated_output")
    assert "<tool_response>" in rej and rej != CHOSEN


def test_refusal_returns_refusal_text():
    rej = make_rejected(CHOSEN, "refusal")
    assert "can't" in rej.lower() or "cannot" in rej.lower()


def test_all_strategies_differ_from_chosen():
    for s in ("wrong_tool", "wrong_args", "hallucinated_output", "refusal"):
        assert make_rejected(CHOSEN, s, tools=TOOLS) != CHOSEN


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        make_rejected(CHOSEN, "bogus")


def test_non_tool_chosen_falls_back_to_refusal():
    rej = make_rejected("just some code", "wrong_tool", tools=TOOLS)
    assert "can't" in rej.lower() or "cannot" in rej.lower()

import json

import pytest
from dataprep.corruptions import make_rejected
from shared.dataprep.schema import tool_call_block

CHOSEN = tool_call_block("bash", {"cmd": "ls"})


def _call(text):
    inner = text[text.find("<tool_call>") + len("<tool_call>"):text.find("</tool_call>")].strip()
    return json.loads(inner)


def test_wrong_tool_changes_name_keeps_args():
    rej = make_rejected(CHOSEN, "wrong_tool")
    call = _call(rej)
    assert call["name"] != "bash" and call["arguments"] == {"cmd": "ls"}


def test_malformed_args_drops_arguments():
    rej = make_rejected(CHOSEN, "malformed_args")
    assert _call(rej)["arguments"] == {}


def test_malformed_args_on_empty_args_still_differs():
    empty = tool_call_block("bash", {})
    rej = make_rejected(empty, "malformed_args")
    assert rej != empty


def test_hallucinated_output_appends_fake_response():
    rej = make_rejected(CHOSEN, "hallucinated_output")
    assert "<tool_response>" in rej and rej != CHOSEN


def test_refusal_returns_refusal_text():
    rej = make_rejected(CHOSEN, "refusal")
    assert "can't" in rej.lower() or "cannot" in rej.lower()


def test_all_strategies_differ_from_chosen():
    for s in ("wrong_tool", "malformed_args", "hallucinated_output", "refusal"):
        assert make_rejected(CHOSEN, s) != CHOSEN


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        make_rejected(CHOSEN, "bogus")


def test_non_tool_chosen_falls_back_to_refusal():
    # wrong_tool/malformed on a plain-text chosen (no tool_call) -> refusal text
    rej = make_rejected("just some code", "wrong_tool")
    assert "can't" in rej.lower() or "cannot" in rej.lower()

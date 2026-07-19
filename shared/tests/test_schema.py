import json

from shared.dataprep.schema import (TrainingExample, render_for_family,
                             tool_call_block, tool_response_block, validate_example)


def test_tool_call_block_is_hermes_json():
    block = tool_call_block("bash", {"cmd": "ls"})
    assert block.startswith("<tool_call>") and block.endswith("</tool_call>")
    inner = block[len("<tool_call>"):-len("</tool_call>")].strip()
    assert json.loads(inner) == {"name": "bash", "arguments": {"cmd": "ls"}}


def test_tool_response_block_roundtrips():
    block = tool_response_block("ok")
    assert json.loads(block.split(">", 1)[1].rsplit("<", 1)[0].strip()) == {"output": "ok"}


def test_valid_example_passes_validation():
    ex = TrainingExample(
        source="magicoder",
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "hello"}],
    )
    validate_example(ex)  # no raise


def test_empty_messages_rejected():
    import pytest
    with pytest.raises(ValueError):
        validate_example(TrainingExample(source="x", messages=[]))


def test_bad_role_rejected():
    import pytest
    ex = TrainingExample(source="x", messages=[{"role": "wizard", "content": "?"}])
    with pytest.raises(ValueError):
        validate_example(ex)


def test_validate_accepts_structured_tool_calls():
    ex = TrainingExample(source="xlam", messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "get_weather", "arguments": {"city": "NYC"}}]},
    ])
    validate_example(ex)  # no raise


def test_validate_rejects_message_with_neither_content_nor_tool_calls():
    import pytest
    ex = TrainingExample(source="x", messages=[{"role": "assistant"}])
    with pytest.raises(ValueError):
        validate_example(ex)


# ---- render_for_family: dual-render (Hermes for qwen, structured for gpt_oss) ----

_TOOL_MSGS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"name": "get_weather", "arguments": {"city": "NYC"}}]},
    {"role": "tool", "name": "get_weather", "content": {"temp": 20}},
]


def test_render_qwen_is_byte_identical_hermes():
    # REGRESSION LOCK: qwen render must match the pre-structured Hermes bytes.
    out = render_for_family(_TOOL_MSGS, "qwen")
    assert out[2] == {"role": "assistant",
                      "content": tool_call_block("get_weather", {"city": "NYC"})}
    assert out[3] == {"role": "tool",
                      "content": tool_response_block({"temp": 20})}
    # exact byte form
    assert out[2]["content"] == '<tool_call>\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n</tool_call>'
    assert out[3]["content"] == '<tool_response>\n{"output": {"temp": 20}}\n</tool_response>'


def test_render_qwen_collapses_multiple_calls_newline_joined():
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"name": "a", "arguments": {"x": 1}},
                            {"name": "b", "arguments": {"y": 2}}]}]
    content = render_for_family(msgs, "qwen")[0]["content"]
    assert content == tool_call_block("a", {"x": 1}) + "\n" + tool_call_block("b", {"y": 2})
    assert content.count("<tool_call>") == 2


def test_render_gpt_oss_passes_structured_through():
    out = render_for_family(_TOOL_MSGS, "gpt_oss")
    assert out is _TOOL_MSGS  # untouched — harmony template consumes the structure
    assert out[2]["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "NYC"}}]
    assert out[3] == {"role": "tool", "name": "get_weather", "content": {"temp": 20}}


def test_render_plaintext_and_multiturn_unaffected_both_families():
    plain = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello"},
             {"role": "tool", "content": "raw output"}]  # unnamed tool = plain text
    for fam in ("qwen", "gpt_oss"):
        assert render_for_family(plain, fam) == plain

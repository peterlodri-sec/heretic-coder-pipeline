import json
from unittest.mock import patch

from dataprep.sources.magicoder import MagicoderSource
from dataprep.sources.xlam import XLAMSource
from dataprep.sources.toolace import ToolACESource
from dataprep.sources.crabcc import CrabccSource
from shared.dataprep.schema import render_for_family


# --- Magicoder: cols problem/solution -> user/assistant -------------------

def test_magicoder_maps_rows_to_examples():
    rows = [{"problem": "Write add()", "solution": "def add(a,b): return a+b"}]
    with patch("shared.dataprep.loaders.load_magicoder_rows", return_value=rows):
        exs = list(MagicoderSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "magicoder"
    assert ex.messages[0]["role"] == "user" and "add()" in ex.messages[0]["content"]
    assert ex.messages[1]["role"] == "assistant" and "def add" in ex.messages[1]["content"]


# --- xLAM: JSON-string tools/answers -> system + Hermes tool_call ----------

def test_xlam_puts_tools_in_system_and_answers_as_tool_calls():
    tools = [{"name": "get_weather", "description": "weather",
              "parameters": {"city": {"type": "string"}}}]
    answers = [{"name": "get_weather", "arguments": {"city": "NYC"}}]
    rows = [{"query": "weather in NYC?",
             "tools": json.dumps(tools), "answers": json.dumps(answers), "id": "1"}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        exs = list(XLAMSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "xlam"
    roles = [m["role"] for m in ex.messages]
    assert roles == ["system", "user", "assistant"]
    # tools land in the system message
    assert "get_weather" in ex.messages[0]["content"]
    assert ex.messages[1]["content"] == "weather in NYC?"
    # gold answer recorded as neutral structured tool call
    assert ex.messages[2]["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "NYC"}}]
    # qwen render collapses it to one Hermes <tool_call> block
    assistant = render_for_family(ex.messages, "qwen")[2]["content"]
    assert assistant.startswith("<tool_call>") and assistant.endswith("</tool_call>")
    inner = assistant[len("<tool_call>"):-len("</tool_call>")].strip()
    assert json.loads(inner) == {"name": "get_weather", "arguments": {"city": "NYC"}}


def test_xlam_renders_multiple_calls_as_separate_blocks():
    tools = [{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]
    answers = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]
    rows = [{"query": "do both", "tools": json.dumps(tools),
             "answers": json.dumps(answers), "id": "2"}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        ex = next(iter(XLAMSource().examples()))
    assert len(ex.messages[2]["tool_calls"]) == 2
    assert render_for_family(ex.messages, "qwen")[2]["content"].count("<tool_call>") == 2


# --- ToolACE: from/value mapping + bracket -> Hermes -----------------------

def test_toolace_maps_from_value_and_prepends_system():
    rows = [{
        "system": "You can call tools.",
        "conversations": [
            {"from": "user", "value": "weather in NYC?"},
            {"from": "assistant", "value": '[get_weather(city="NYC", days=3)]'},
            {"from": "tool", "value": '{"temp": 20}'},
            {"from": "assistant", "value": "It is 20 degrees."},
        ],
    }]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        exs = list(ToolACESource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "toolace"
    roles = [m["role"] for m in ex.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert ex.messages[0]["content"] == "You can call tools."
    # bracket assistant call recorded as neutral structured tool call
    assert ex.messages[2]["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "NYC", "days": 3}}]
    # qwen render normalizes it to Hermes
    call = render_for_family(ex.messages, "qwen")[2]["content"]
    assert call.startswith("<tool_call>")
    assert json.loads(call[len("<tool_call>"):-len("</tool_call>")].strip()) == {
        "name": "get_weather", "arguments": {"city": "NYC", "days": 3}}
    # tool result + plain assistant turn preserved verbatim (plain text, no name)
    assert ex.messages[3]["role"] == "tool" and ex.messages[3]["content"] == '{"temp": 20}'
    assert "name" not in ex.messages[3]
    assert ex.messages[4]["content"] == "It is 20 degrees."


def test_toolace_keeps_unparseable_bracket_text_as_is():
    rows = [{
        "system": "sys",
        "conversations": [
            {"from": "user", "value": "hi"},
            # not a parseable keyword-call list -> kept verbatim
            {"from": "assistant", "value": "[this is not a call]"},
        ],
    }]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        ex = next(iter(ToolACESource().examples()))
    assert ex.messages[2]["content"] == "[this is not a call]"


# --- crabcc: local traces -> Hermes tool_call/tool_response ----------------

def test_crabcc_reads_local_traces():
    trace = {"turns": [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "tool": "bash", "arguments": {"cmd": "pytest"}},
        {"role": "tool", "output": "ok"},
    ]}
    with patch("shared.dataprep.loaders.load_traces", return_value=[trace]):
        exs = list(CrabccSource(trace_dir="/x").examples())
    msgs = exs[0].messages
    # neutral structured tool call + named tool result (name threaded from call)
    assert msgs[1]["tool_calls"] == [{"name": "bash", "arguments": {"cmd": "pytest"}}]
    assert msgs[2]["role"] == "tool" and msgs[2]["name"] == "bash" and msgs[2]["content"] == "ok"
    # qwen render restores Hermes <tool_call>/<tool_response>
    contents = [m["content"] for m in render_for_family(msgs, "qwen")]
    assert any("<tool_call>" in c for c in contents)
    assert any("<tool_response>" in c for c in contents)

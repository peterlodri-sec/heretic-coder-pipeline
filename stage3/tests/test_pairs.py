import json

from unittest.mock import patch

from dataprep.pairs.crabcc import CrabccPairs
from dataprep.pairs.toolace import ToolACEPairs
from dataprep.pairs.xlam import XLAMPairs


def _inner_call(text):
    inner = text[text.find("<tool_call>") + len("<tool_call>"):text.find("</tool_call>")].strip()
    return json.loads(inner)


def test_xlam_builds_conversational_triple():
    rows = [{"query": "list files",
             "tools": json.dumps([{"name": "bash", "parameters": {}},
                                  {"name": "python", "parameters": {}}]),
             "answers": json.dumps([{"name": "bash", "arguments": {"cmd": "ls"}}])}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        pairs = list(XLAMPairs().pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert p.prompt[0]["role"] == "system" and p.prompt[-1]["role"] == "user"
    assert p.chosen[0]["role"] == "assistant"
    assert "<tool_call>" in p.chosen[0]["content"] and "bash" in p.chosen[0]["content"]
    assert p.rejected[0]["role"] == "assistant"
    assert p.chosen[0]["content"] != p.rejected[0]["content"]
    assert p.source == "xlam"


def test_xlam_rejected_uses_real_alternate_tool():
    rows = [{"query": "q",
             "tools": json.dumps([{"name": "bash", "parameters": {}},
                                  {"name": "python", "parameters": {}}]),
             "answers": json.dumps([{"name": "bash", "arguments": {"cmd": "ls"}}])}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        p = list(XLAMPairs().pairs())[0]
    inner = _inner_call(p.rejected[0]["content"])
    assert inner["name"] == "python"  # a real alternate tool, not "not_bash"
    assert "not_" not in inner["name"]


def test_xlam_skips_rows_without_answers():
    rows = [{"query": "q", "tools": json.dumps([{"name": "bash", "parameters": {}}]),
             "answers": json.dumps([])}]
    with patch("shared.dataprep.loaders.load_xlam_rows", return_value=rows):
        assert list(XLAMPairs().pairs()) == []


def test_toolace_last_assistant_is_chosen():
    assistant = ("<tool_call>\n{\"name\": \"bash\", "
                 "\"arguments\": {\"cmd\": \"ls\"}}\n</tool_call>")
    rows = [{"system": "sys", "conversations": [
        {"from": "user", "value": "q1"},
        {"from": "assistant", "value": assistant}]}]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        pairs = list(ToolACEPairs().pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert p.prompt[0]["role"] == "system"
    assert any(m["role"] == "user" for m in p.prompt)
    assert all(m["role"] != "assistant" for m in p.prompt)
    assert p.chosen[0]["content"] == assistant
    assert p.chosen[0]["content"] != p.rejected[0]["content"]
    assert p.source == "toolace"


def test_toolace_skips_rows_without_assistant():
    rows = [{"system": "s", "conversations": [{"from": "user", "value": "q"}]}]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        assert list(ToolACEPairs().pairs()) == []


def test_crabcc_builds_pair_from_trace():
    trace = {"turns": [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "tool": "bash", "arguments": {"cmd": "pytest"}}]}
    with patch("shared.dataprep.loaders.load_traces", return_value=[trace]):
        pairs = list(CrabccPairs(trace_dir="/x").pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert "<tool_call>" in p.chosen[0]["content"]
    assert p.chosen[0]["content"] != p.rejected[0]["content"]
    assert p.prompt[-1]["role"] == "user"

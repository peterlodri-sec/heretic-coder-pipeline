from unittest.mock import patch

from dataprep.sources.magicoder import MagicoderSource


def test_magicoder_maps_rows_to_examples():
    rows = [{"problem": "Write add()", "solution": "def add(a,b): return a+b"}]
    with patch("dataprep.sources.magicoder.load_rows", return_value=rows):
        exs = list(MagicoderSource().examples())
    assert len(exs) == 1
    ex = exs[0]
    assert ex.source == "magicoder"
    assert ex.messages[0]["role"] == "user" and "add()" in ex.messages[0]["content"]
    assert ex.messages[1]["role"] == "assistant" and "def add" in ex.messages[1]["content"]
    assert ex.is_negative is False


from dataprep.sources.bfcl import BFCLSource
from dataprep.sources.toolace import ToolACESource
from dataprep.sources.swebench import SWEBenchSource
from dataprep.sources.crabcc import CrabccSource


def test_bfcl_builds_tool_call_and_marks_wrong_tool_negative():
    rows = [
        {"question": "list files", "function": "bash",
         "arguments": {"cmd": "ls"}, "output": "a b", "correct": True},
        {"question": "list files", "function": "delete_all",
         "arguments": {}, "output": "", "correct": False},
    ]
    with patch("dataprep.sources.bfcl.load_rows", return_value=rows):
        exs = list(BFCLSource().examples())
    assert "<tool_call>" in exs[0].messages[1]["content"]
    assert exs[0].is_negative is False
    assert exs[1].is_negative is True


def test_toolace_filters_to_code_adjacent():
    rows = [
        {"domain": "coding", "conversation": [
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        {"domain": "cooking", "conversation": [
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    ]
    with patch("dataprep.sources.toolace.load_rows", return_value=rows):
        exs = list(ToolACESource().examples())
    assert len(exs) == 1 and exs[0].source == "toolace"


def test_swebench_uses_resolved_only_and_formats_patch():
    rows = [
        {"problem_statement": "fix bug", "patch": "diff --git a b", "resolved": True},
        {"problem_statement": "other", "patch": "x", "resolved": False},
    ]
    with patch("dataprep.sources.swebench.load_rows", return_value=rows):
        exs = list(SWEBenchSource().examples())
    assert len(exs) == 1
    assert "diff --git" in exs[0].messages[1]["content"]


def test_crabcc_reads_local_traces():
    trace = {"turns": [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "tool": "bash", "arguments": {"cmd": "pytest"}},
        {"role": "tool", "output": "ok"},
    ]}
    with patch("dataprep.sources.crabcc.load_traces", return_value=[trace]):
        exs = list(CrabccSource(trace_dir="/x").examples())
    contents = [m["content"] for m in exs[0].messages]
    assert any("<tool_call>" in c for c in contents)
    assert any("<tool_response>" in c for c in contents)

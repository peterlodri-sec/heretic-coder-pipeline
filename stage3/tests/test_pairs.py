from unittest.mock import patch

from dataprep.pairs.bfcl import BFCLPairs
from dataprep.pairs.crabcc import CrabccPairs
from dataprep.pairs.swebench import SWEBenchPairs
from dataprep.pairs.toolace import ToolACEPairs


def test_bfcl_correct_is_chosen_wrong_is_rejected():
    rows = [{"question": "list files", "function": "bash",
             "arguments": {"cmd": "ls"}, "output": "a b", "correct": True}]
    with patch("shared.dataprep.loaders.load_bfcl_rows", return_value=rows):
        pairs = list(BFCLPairs().pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert "<tool_call>" in p.chosen and "bash" in p.chosen
    assert p.chosen != p.rejected and p.source == "bfcl"
    assert p.prompt[-1]["role"] == "user"


def test_bfcl_skips_incorrect_rows_as_chosen_source():
    rows = [{"question": "q", "function": "bash", "arguments": {},
             "output": "", "correct": False}]
    with patch("shared.dataprep.loaders.load_bfcl_rows", return_value=rows):
        assert list(BFCLPairs().pairs()) == []  # only correct rows seed a chosen


def test_swebench_resolved_only_with_corrupted_rejected():
    rows = [{"problem_statement": "fix", "patch": "diff --git a b", "resolved": True},
            {"problem_statement": "no", "patch": "x", "resolved": False}]
    with patch("shared.dataprep.loaders.load_swebench_rows", return_value=rows):
        pairs = list(SWEBenchPairs().pairs())
    assert len(pairs) == 1 and "diff --git" in pairs[0].chosen
    assert pairs[0].rejected != pairs[0].chosen


def test_toolace_code_domain_only():
    rows = [{"domain": "coding", "conversation": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>"}]},
            {"domain": "cooking", "conversation": [
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    with patch("shared.dataprep.loaders.load_toolace_rows", return_value=rows):
        pairs = list(ToolACEPairs().pairs())
    assert len(pairs) == 1 and pairs[0].source == "toolace"
    assert pairs[0].chosen != pairs[0].rejected


def test_crabcc_builds_pair_from_trace():
    trace = {"turns": [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "tool": "bash", "arguments": {"cmd": "pytest"}}]}
    with patch("shared.dataprep.loaders.load_traces", return_value=[trace]):
        pairs = list(CrabccPairs(trace_dir="/x").pairs())
    assert len(pairs) == 1
    assert "<tool_call>" in pairs[0].chosen and pairs[0].rejected != pairs[0].chosen

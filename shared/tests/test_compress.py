import sys
import types
from unittest.mock import patch

from shared.dataprep import compress
from shared.dataprep.compress import KOMPRESS_MODEL, compress_tool_spans


def _fake_headroom():
    """Stand-in `headroom` module recording compress() calls, returning a
    list-of-messages with the content marked as compressed."""
    mod = types.ModuleType("headroom")
    mod.calls = []

    def _compress(messages, model=None):
        mod.calls.append((messages, model))
        text = messages[0]["content"]
        return [{"role": "tool", "content": "COMPRESSED:" + text}]

    mod.compress = _compress
    return mod


def test_tool_content_compressed_and_headroom_called_with_it():
    fake = _fake_headroom()
    msgs = [{"role": "tool", "content": "long tool output"}]
    with patch.dict(sys.modules, {"headroom": fake}):
        out = compress_tool_spans(msgs)
    # headroom.compress called with the tool text and the kompress model
    assert fake.calls[-1][0] == [{"role": "tool", "content": "long tool output"}]
    assert fake.calls[-1][1] == KOMPRESS_MODEL
    # returned compressed content swapped in
    assert out[0]["content"] == "COMPRESSED:long tool output"


def test_only_tool_changes_code_is_byte_identical():
    fake = _fake_headroom()
    code = "def f(x):\n    return x + 1  # do not touch\n"
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "write f"},
        {"role": "assistant", "content": code},
        {"role": "tool", "content": "exit 0 stdout..."},
    ]
    with patch.dict(sys.modules, {"headroom": fake}):
        out = compress_tool_spans(msgs)
    # system/user/assistant pass through as the same objects
    assert out[0] is msgs[0]
    assert out[1] is msgs[1]
    assert out[2] is msgs[2]
    # regression lock: the code string is untouched, byte-identical
    assert out[2]["content"] == code
    # only the tool message changed
    assert out[3]["content"] == "COMPRESSED:exit 0 stdout..."


def test_empty_or_nonstr_tool_content_passes_through_headroom_not_called():
    fake = _fake_headroom()
    msgs = [
        {"role": "tool", "content": ""},
        {"role": "tool", "content": {"output": 1}},  # non-str (structured)
        {"role": "tool"},  # no content key
    ]
    with patch.dict(sys.modules, {"headroom": fake}):
        out = compress_tool_spans(msgs)
    assert fake.calls == []  # headroom never invoked
    for orig, res in zip(msgs, out):
        assert res is orig  # same objects, unchanged


def test_input_list_not_mutated():
    fake = _fake_headroom()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "raw"},
    ]
    snapshot = [dict(m) for m in msgs]
    with patch.dict(sys.modules, {"headroom": fake}):
        out = compress_tool_spans(msgs)
    # a NEW list is returned
    assert out is not msgs
    # original list object and its messages are unchanged
    assert msgs == snapshot
    assert msgs[1]["content"] == "raw"
    # non-tool message identity preserved
    assert out[0] is msgs[0]


def test_module_imports_without_headroom_installed():
    # Function-local import proven: the module is importable with no `headroom`
    # anywhere in the environment (compressing is what would need it).
    with patch.dict(sys.modules, {"headroom": None}):
        import importlib

        mod = importlib.reload(compress)
        assert mod.KOMPRESS_MODEL == "PeetPedro/kompress-v8"

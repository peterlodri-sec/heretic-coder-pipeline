import json

from dataprep.pipeline import build
from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource


class FakeSource(DataSource):
    def __init__(self, name, examples):
        self.name = name
        self._examples = examples

    def examples(self):
        yield from self._examples


def _ex(source):
    return TrainingExample(
        source=source,
        messages=[{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "a"}])


def test_build_writes_messages_only_jsonl(tmp_path):
    out = tmp_path / "train.jsonl"
    sources = [FakeSource("magicoder", [_ex("magicoder"), _ex("magicoder")])]
    count = build(sources, str(out))
    lines = out.read_text().splitlines()
    assert count == 2 and len(lines) == 2
    record = json.loads(lines[0])
    # exactly one key: messages (TRL conversational path)
    assert list(record.keys()) == ["messages"]
    assert record["messages"][0]["role"] == "user"


def test_build_drops_contaminated_sources(tmp_path):
    out = tmp_path / "train.jsonl"
    sources = [FakeSource("sharegpt", [_ex("sharegpt")]),
               FakeSource("magicoder", [_ex("magicoder")])]
    count = build(sources, str(out), contaminated={"sharegpt"})
    lines = out.read_text().splitlines()
    assert count == 1 and len(lines) == 1
    assert json.loads(lines[0])["messages"][0]["content"] == "q"


def test_build_validates_and_rejects_bad_role(tmp_path):
    import pytest
    out = tmp_path / "train.jsonl"
    bad = TrainingExample(source="x", messages=[{"role": "wizard", "content": "?"}])
    sources = [FakeSource("x", [bad])]
    with pytest.raises(ValueError):
        build(sources, str(out))


def test_build_default_contaminated_is_empty(tmp_path):
    out = tmp_path / "train.jsonl"
    sources = [FakeSource("magicoder", [_ex("magicoder")])]
    assert build(sources, str(out)) == 1


def _tool_ex():
    return TrainingExample(source="xlam", messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "get_weather", "arguments": {"city": "NYC"}}]},
    ])


def test_build_renders_hermes_for_qwen(tmp_path):
    out = tmp_path / "train.jsonl"
    build([FakeSource("xlam", [_tool_ex()])], str(out), family="qwen")
    msgs = json.loads(out.read_text().splitlines()[0])["messages"]
    assert msgs[1] == {"role": "assistant",
                       "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n</tool_call>'}


def test_build_renders_structured_for_gpt_oss(tmp_path):
    out = tmp_path / "train.jsonl"
    # default family is gpt_oss -> structured tool_calls pass through
    build([FakeSource("xlam", [_tool_ex()])], str(out))
    msgs = json.loads(out.read_text().splitlines()[0])["messages"]
    assert msgs[1]["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "NYC"}}]


def _tool_output_ex():
    return TrainingExample(source="agent", messages=[
        {"role": "user", "content": "run ls"},
        {"role": "tool", "content": "file1\nfile2\n"},
    ])


def test_build_kompress_unset_leaves_messages_uncompressed(tmp_path, monkeypatch):
    monkeypatch.delenv("KOMPRESS_COMPRESS", raising=False)
    out = tmp_path / "train.jsonl"
    build([FakeSource("agent", [_tool_output_ex()])], str(out), family="qwen")
    msgs = json.loads(out.read_text().splitlines()[0])["messages"]
    assert msgs[1]["content"] == "file1\nfile2\n"  # untouched


def test_build_kompress_enabled_applies_compress_tool_spans(tmp_path, monkeypatch):
    calls = []

    def fake_compress(messages, **kwargs):
        calls.append(messages)
        return [dict(m, content="C:" + m["content"]) if m.get("role") == "tool" else m
                for m in messages]

    monkeypatch.setenv("KOMPRESS_COMPRESS", "1")
    # patch the name bound in pipeline's namespace (no real headroom needed)
    import dataprep.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "compress_tool_spans", fake_compress)

    out = tmp_path / "train.jsonl"
    build([FakeSource("agent", [_tool_output_ex(), _tool_output_ex()])],
          str(out), family="qwen")
    # applied to each example
    assert len(calls) == 2
    msgs = json.loads(out.read_text().splitlines()[0])["messages"]
    assert msgs[1]["content"] == "C:file1\nfile2\n"

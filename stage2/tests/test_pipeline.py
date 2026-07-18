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

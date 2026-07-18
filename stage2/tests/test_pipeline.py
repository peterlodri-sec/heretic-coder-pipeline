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


def _ex(source, is_neg=False):
    return TrainingExample(source=source, messages=[{"role": "user", "content": "q"}],
                           is_negative=is_neg)


def test_build_writes_jsonl_with_records(tmp_path):
    out = tmp_path / "train.jsonl"
    sources = [FakeSource("magicoder", [_ex("magicoder"), _ex("magicoder", True)])]
    count = build(sources, str(out), contaminated=set(), min_negative_ratio=0.1)
    lines = out.read_text().splitlines()
    assert count == 2 and len(lines) == 2
    assert json.loads(lines[0])["source"] == "magicoder"


def test_build_applies_contamination(tmp_path):
    out = tmp_path / "train.jsonl"
    sources = [FakeSource("sharegpt", [_ex("sharegpt"), _ex("sharegpt", True)])]
    build(sources, str(out), contaminated={"sharegpt"}, mode="exclude",
          min_negative_ratio=0.0)
    assert out.read_text() == ""  # all excluded


def test_build_enforces_negative_minimum(tmp_path):
    import pytest
    out = tmp_path / "train.jsonl"
    sources = [FakeSource("magicoder", [_ex("magicoder")] * 10)]  # zero negatives
    with pytest.raises(ValueError):
        build(sources, str(out), contaminated=set(), min_negative_ratio=0.05)

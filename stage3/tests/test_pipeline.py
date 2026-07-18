import json

import pytest
from dataprep.pipeline import build
from dataprep.pairs.base import PairSource
from dataprep.schema import PreferencePair


class FakePairs(PairSource):
    def __init__(self, name, pairs):
        self.name = name
        self._pairs = pairs

    def pairs(self):
        yield from self._pairs


def _p(source):
    return PreferencePair(prompt=[{"role": "user", "content": "q"}],
                          chosen="A", rejected="B", source=source)


def test_build_writes_jsonl(tmp_path):
    out = tmp_path / "pairs.jsonl"
    n = build([FakePairs("bfcl", [_p("bfcl"), _p("bfcl")])], str(out), contaminated=set())
    lines = out.read_text().splitlines()
    assert n == 2 and len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["chosen"] == "A" and rec["rejected"] == "B"


def test_build_excludes_contaminated(tmp_path):
    out = tmp_path / "pairs.jsonl"
    build([FakePairs("sharegpt", [_p("sharegpt")])], str(out),
          contaminated={"sharegpt"}, mode="exclude", min_pairs=0)
    assert out.read_text() == ""


def test_build_enforces_min_pairs(tmp_path):
    out = tmp_path / "pairs.jsonl"
    with pytest.raises(ValueError):
        build([FakePairs("bfcl", [])], str(out), contaminated=set(), min_pairs=1)


def test_build_validates_pairs(tmp_path):
    out = tmp_path / "pairs.jsonl"
    bad = PreferencePair(prompt=[{"role": "user", "content": "q"}],
                         chosen="same", rejected="same", source="bfcl")
    with pytest.raises(ValueError):
        build([FakePairs("bfcl", [bad])], str(out), contaminated=set())

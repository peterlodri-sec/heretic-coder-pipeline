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


def _p(source, chosen="A", rejected="B"):
    return PreferencePair(
        prompt=[{"role": "user", "content": "q"}],
        chosen=[{"role": "assistant", "content": chosen}],
        rejected=[{"role": "assistant", "content": rejected}],
        source=source,
    )


def test_build_writes_jsonl(tmp_path):
    out = tmp_path / "pairs.jsonl"
    n = build([FakePairs("xlam", [_p("xlam"), _p("xlam")])], str(out),
              contaminated=frozenset())
    lines = out.read_text().splitlines()
    assert n == 2 and len(lines) == 2
    rec = json.loads(lines[0])
    assert set(rec) == {"prompt", "chosen", "rejected"}
    assert rec["chosen"] == [{"role": "assistant", "content": "A"}]
    assert rec["rejected"] == [{"role": "assistant", "content": "B"}]


def test_build_excludes_contaminated(tmp_path):
    out = tmp_path / "pairs.jsonl"
    build([FakePairs("sharegpt", [_p("sharegpt")])], str(out),
          contaminated={"sharegpt"}, min_pairs=0)
    assert out.read_text() == ""


def test_build_enforces_min_pairs(tmp_path):
    out = tmp_path / "pairs.jsonl"
    with pytest.raises(ValueError):
        build([FakePairs("xlam", [])], str(out), contaminated=frozenset(), min_pairs=1)


def test_build_validates_pairs(tmp_path):
    out = tmp_path / "pairs.jsonl"
    bad = _p("xlam", chosen="same", rejected="same")
    with pytest.raises(ValueError):
        build([FakePairs("xlam", [bad])], str(out), contaminated=frozenset())

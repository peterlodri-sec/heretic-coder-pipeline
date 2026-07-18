import pytest
from shared.dataprep.contamination import filter_contaminated
from shared.dataprep.schema import TrainingExample


def _ex(source):
    return TrainingExample(source=source, messages=[{"role": "user", "content": "x"}])


def test_downweight_mode_scales_flagged_sources():
    out = filter_contaminated(
        [_ex("magicoder"), _ex("sharegpt")],
        contaminated={"sharegpt"}, mode="downweight", weight=0.1)
    by_src = {e.source: e.weight for e in out}
    assert by_src["magicoder"] == 1.0
    assert by_src["sharegpt"] == pytest.approx(0.1)


def test_exclude_mode_drops_flagged_sources():
    out = filter_contaminated(
        [_ex("magicoder"), _ex("sharegpt")],
        contaminated={"sharegpt"}, mode="exclude")
    assert [e.source for e in out] == ["magicoder"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        filter_contaminated([_ex("x")], contaminated=set(), mode="bogus")

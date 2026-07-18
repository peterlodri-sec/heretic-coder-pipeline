import pytest
from shared.dataprep.contamination import filter_contaminated
from shared.dataprep.schema import TrainingExample


def _ex(source):
    return TrainingExample(source=source, messages=[{"role": "user", "content": "x"}])


def test_drop_mode_removes_contaminated_sources():
    out = filter_contaminated(
        [_ex("magicoder"), _ex("sharegpt"), _ex("xlam")],
        contaminated={"sharegpt"})
    assert [e.source for e in out] == ["magicoder", "xlam"]


def test_nothing_dropped_when_no_contaminated():
    out = filter_contaminated(
        [_ex("magicoder"), _ex("xlam")], contaminated=set())
    assert [e.source for e in out] == ["magicoder", "xlam"]


def test_default_mode_is_drop():
    out = filter_contaminated([_ex("sharegpt")], contaminated={"sharegpt"})
    assert out == []


def test_unknown_mode_raises():
    # the old down-weight path is gone; only "drop" is valid now
    with pytest.raises(ValueError):
        filter_contaminated([_ex("x")], contaminated=set(), mode="downweight")

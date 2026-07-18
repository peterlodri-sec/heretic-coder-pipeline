import pytest
from dataprep.negatives import negative_ratio, require_negatives
from dataprep.schema import TrainingExample


def _ex(is_neg):
    return TrainingExample(source="x", messages=[{"role": "user", "content": "q"}],
                           is_negative=is_neg)


def test_ratio_counts_negatives():
    exs = [_ex(True), _ex(False), _ex(False), _ex(False)]
    assert negative_ratio(exs) == pytest.approx(0.25)


def test_require_negatives_passes_above_min():
    exs = [_ex(True)] + [_ex(False)] * 9
    require_negatives(exs, min_ratio=0.05)  # no raise


def test_require_negatives_raises_below_min():
    with pytest.raises(ValueError):
        require_negatives([_ex(False)] * 20, min_ratio=0.05)


def test_ratio_of_empty_is_zero():
    assert negative_ratio([]) == 0.0

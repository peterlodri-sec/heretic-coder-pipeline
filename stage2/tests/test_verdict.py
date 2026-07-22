import dataclasses

import pytest
from shared.enums import Verdict
from verdict import compute_verdict

GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.90,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}


def test_all_within_thresholds_pass():
    result = compute_verdict(GOOD)
    assert result.passed
    assert result.verdict is Verdict.PASS
    assert result.reasons == ()


def test_low_bfcl_fails_with_reason():
    result = compute_verdict({**GOOD, "bfcl_accuracy": 0.80})
    assert result.verdict is Verdict.FAIL
    assert any("bfcl_accuracy" in r for r in result.reasons)


def test_high_refusal_and_regression_both_reported():
    result = compute_verdict({**GOOD, "refusal_rate": 0.20, "humaneval_delta": 0.10})
    assert len(result.reasons) == 2


def test_swebench_skipped_when_disabled():
    metrics = {**GOOD, "swebench_resolve": 0.10}  # would fail if checked
    result = compute_verdict(metrics, check_swebench=False)
    assert result.passed


def test_result_is_frozen_and_stringifies():
    result = compute_verdict({**GOOD, "bfcl_accuracy": 0.5})
    assert str(result).startswith("fail:")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = Verdict.PASS

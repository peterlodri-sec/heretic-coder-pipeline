import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enums import Verdict
from verdict import VerdictResult, compute_verdict


def test_all_metrics_within_thresholds_pass():
    result = compute_verdict({
        "refusal_rate": 0.03,
        "kl_divergence": 0.16,
        "mmlu_delta": 0.01,
        "gsm8k_delta": -0.005,
    })
    assert result == VerdictResult(Verdict.PASS, ())
    assert result.passed


def test_refusal_rate_at_threshold_fails():
    result = compute_verdict({
        "refusal_rate": 0.05,
        "kl_divergence": 0.1,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    })
    assert result.verdict is Verdict.FAIL
    assert not result.passed
    assert any("refusal_rate" in reason for reason in result.reasons)


def test_multiple_failures_are_all_reported():
    result = compute_verdict({
        "refusal_rate": 0.5,
        "kl_divergence": 0.9,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    })
    assert result.verdict is Verdict.FAIL
    assert len(result.reasons) == 2


def test_kl_divergence_just_under_threshold_passes():
    result = compute_verdict({
        "refusal_rate": 0.0,
        "kl_divergence": 0.2999,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    })
    assert result.passed


def test_str_summarizes_failure_reasons():
    result = compute_verdict({
        "refusal_rate": 0.5,
        "kl_divergence": 0.0,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    })
    text = str(result)
    assert text.startswith("fail:")
    assert "refusal_rate" in text


def test_result_is_frozen():
    import dataclasses
    import pytest
    result = compute_verdict({
        "refusal_rate": 0.0, "kl_divergence": 0.0, "mmlu_delta": 0.0, "gsm8k_delta": 0.0,
    })
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = Verdict.FAIL

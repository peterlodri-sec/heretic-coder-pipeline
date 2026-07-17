import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verdict import compute_verdict


def test_all_metrics_within_thresholds_pass():
    metrics = {
        "refusal_rate": 0.03,
        "kl_divergence": 0.16,
        "mmlu_delta": 0.01,
        "gsm8k_delta": -0.005,
    }
    result = compute_verdict(metrics)
    assert result == {"verdict": "pass", "reasons": []}


def test_refusal_rate_at_threshold_fails():
    metrics = {
        "refusal_rate": 0.05,
        "kl_divergence": 0.1,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    }
    result = compute_verdict(metrics)
    assert result["verdict"] == "fail"
    assert any("refusal_rate" in reason for reason in result["reasons"])


def test_multiple_failures_are_all_reported():
    metrics = {
        "refusal_rate": 0.5,
        "kl_divergence": 0.9,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    }
    result = compute_verdict(metrics)
    assert result["verdict"] == "fail"
    assert len(result["reasons"]) == 2


def test_kl_divergence_just_under_threshold_passes():
    metrics = {
        "refusal_rate": 0.0,
        "kl_divergence": 0.2999,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    }
    assert compute_verdict(metrics)["verdict"] == "pass"

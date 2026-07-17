import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "remote"))

from capability_eval import compute_deltas


def test_compute_deltas_positive_when_candidate_worse():
    base = {"mmlu": {"acc,none": 0.60}, "gsm8k": {"exact_match,strict-match": 0.50}}
    candidate = {"mmlu": {"acc,none": 0.55}, "gsm8k": {"exact_match,strict-match": 0.48}}
    deltas = compute_deltas(base, candidate)
    assert deltas["mmlu_delta"] == pytest.approx(0.05)
    assert deltas["gsm8k_delta"] == pytest.approx(0.02)


def test_compute_deltas_negative_when_candidate_better():
    base = {"mmlu": {"acc,none": 0.60}, "gsm8k": {"exact_match,strict-match": 0.50}}
    candidate = {"mmlu": {"acc,none": 0.62}, "gsm8k": {"exact_match,strict-match": 0.55}}
    deltas = compute_deltas(base, candidate)
    assert deltas["mmlu_delta"] < 0
    assert deltas["gsm8k_delta"] < 0

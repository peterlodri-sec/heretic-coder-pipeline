import dataclasses

import pytest
from shared.enums import Verdict
from shared.verdict import VerdictResult, compute_verdict

GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.90,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}


def test_all_within_thresholds_pass():
    r = compute_verdict(GOOD)
    assert r.passed and r.verdict is Verdict.PASS and r.reasons == ()


def test_low_bfcl_fails():
    r = compute_verdict({**GOOD, "bfcl_accuracy": 0.80})
    assert r.verdict is Verdict.FAIL and any("bfcl_accuracy" in x for x in r.reasons)


def test_two_failures_reported():
    r = compute_verdict({**GOOD, "refusal_rate": 0.2, "humaneval_delta": 0.1})
    assert len(r.reasons) == 2


def test_swebench_skipped_when_disabled():
    assert compute_verdict({**GOOD, "swebench_resolve": 0.1}, check_swebench=False).passed


def test_frozen_and_str():
    r = compute_verdict({**GOOD, "bfcl_accuracy": 0.5})
    assert str(r).startswith("fail:")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.verdict = Verdict.PASS

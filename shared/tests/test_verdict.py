import dataclasses

import pytest
from shared.enums import Verdict
from shared.verdict import (
    VerdictResult, compute_verdict, target_gaps, SOTA_TARGETS,
)

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


# --- SOTA target tracking (soft, never gates) — Pillar 5 of the SWE report ------

def test_target_gaps_are_soft_not_a_gate():
    # A run that PASSES the gate can still be far from SOTA targets; target_gaps
    # reports that distance without ever affecting the verdict.
    r = compute_verdict(GOOD)
    assert r.passed  # gate is happy at swebench 0.45
    gaps = {g.metric: g for g in target_gaps(GOOD)}
    assert gaps["swebench_resolve"].met is False        # 0.45 < SOTA 0.60
    assert gaps["swebench_resolve"].gap == pytest.approx(0.15)  # 0.60 - 0.45


def test_target_gaps_met_when_at_or_past_target():
    metrics = {"refusal_rate": 0.02, "bfcl_accuracy": 0.95,
               "humaneval_delta": 0.00, "swebench_resolve": 0.60}
    for g in target_gaps(metrics):
        assert g.met is True and g.gap <= 0


def test_target_gaps_ceiling_vs_floor_direction():
    g = {x.metric: x for x in target_gaps(
        {"refusal_rate": 0.09, "swebench_resolve": 0.30})}
    # ceiling metric: higher value = worse, still short of the 0.05 ceiling
    assert g["refusal_rate"].direction == "ceiling" and g["refusal_rate"].met is False
    assert g["refusal_rate"].gap == pytest.approx(0.04)   # 0.09 - 0.05
    # floor metric
    assert g["swebench_resolve"].direction == "floor" and g["swebench_resolve"].met is False


def test_target_gaps_skips_missing_metrics():
    # Stages that don't emit swebench (None) must not crash the reporter.
    gaps = target_gaps({"refusal_rate": 0.03, "swebench_resolve": None})
    assert [g.metric for g in gaps] == ["refusal_rate"]


def test_sota_targets_are_stricter_than_the_gate():
    # Sanity: the aspirational targets must be at least as strict as the gate,
    # otherwise "SOTA" would be easier than "passing" — a config smell.
    from shared.verdict import CAPABILITY_CHECKS
    gate = {m: lim for m, _cmp, lim in CAPABILITY_CHECKS}
    for metric, direction, target in SOTA_TARGETS:
        if direction == "ceiling":
            assert target <= gate[metric]
        else:
            assert target >= gate[metric]

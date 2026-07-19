from shared.enums import Verdict
from verdict import VerdictResult, compute_verdict

GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.9,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}


def test_reexports_shared_engine():
    assert compute_verdict(GOOD).passed
    assert compute_verdict({**GOOD, "swebench_resolve": 0.1}).verdict is Verdict.FAIL
    assert isinstance(compute_verdict(GOOD), VerdictResult)

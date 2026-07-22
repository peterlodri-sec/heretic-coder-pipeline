import pytest

import reward
import shared.exec_sandbox as sandbox


HARMONY = (
    "<|start|>assistant<|channel|>analysis<|message|>let me think<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
    "def add(a, b):\n    return a + b<|return|>"
)


# ---- _extract_final helper ---------------------------------------------------

def test_extract_final_from_harmony():
    assert reward._extract_final(HARMONY) == "def add(a, b):\n    return a + b"


def test_extract_final_plain_text():
    assert reward._extract_final("def f(): return 1") == "def f(): return 1"


def test_extract_final_strips_fences():
    text = "```python\nx = 1\n```"
    assert reward._extract_final(text) == "x = 1"


def test_extract_final_harmony_with_fenced_body():
    text = ("<|channel|>final<|message|>```python\ndef g():\n    return 7\n```"
            "<|end|>")
    assert reward._extract_final(text) == "def g():\n    return 7"


# ---- shared fake sandbox -----------------------------------------------------

def _fake_factory(visible_rate=1.0, hidden_rate=1.0, hidden_marker="HIDDEN"):
    def fake_run_tests(code, tests, timeout_s=30.0):
        rate = hidden_rate if hidden_marker in (tests or "") else visible_rate
        return {"passed": int(rate), "total": 1, "pass_rate": rate,
                "compiled": True, "timed_out": False, "error": None}
    return fake_run_tests


# ---- tiered reward -----------------------------------------------------------

def test_passing_scores_higher_than_failing(monkeypatch):
    def fake(code, tests, timeout_s=30.0):
        ok = "return a + b" in code
        return {"passed": int(ok), "total": 1, "pass_rate": 1.0 if ok else 0.0,
                "compiled": True, "timed_out": False, "error": None, "execution_time_s": 0.0}
    monkeypatch.setattr(sandbox, "run_tests", fake)

    good = "<|channel|>analysis<|message|>x<|end|><|channel|>final<|message|>def add(a, b):\n    return a + b<|return|>"
    bad = "<|channel|>analysis<|message|>x<|end|><|channel|>final<|message|>def add(a, b):\n    return 0<|return|>"
    r = reward.code_execution_reward(["p", "p"], [good, bad], ["t", "t"])
    assert len(r) == 2
    assert r[0] > r[1]
    assert r[0] == pytest.approx(1.4)  # parse .1 + compiled .2 + pass 1.0 + fast exec .1


def test_fractional_pass_rate_scales(monkeypatch):
    def fake(code, tests, timeout_s=30.0):
        return {"passed": 1, "total": 2, "pass_rate": 0.5,
                "compiled": True, "timed_out": False, "error": None, "execution_time_s": 0.0}
    monkeypatch.setattr(sandbox, "run_tests", fake)
    r = reward.code_execution_reward(["p"], ["def f(): return 1"], ["t"])
    assert r[0] == pytest.approx(0.1 + 0.2 + 0.5)


def test_malformed_harmony_returns_minus_one(monkeypatch):
    monkeypatch.setattr(sandbox, "run_tests", _fake_factory())
    # channel tags present but NO final channel -> malformed
    bad = "<|channel|>analysis<|message|>only analysis here<|end|>"
    r = reward.code_execution_reward(["p"], [bad], ["t"])
    assert r[0] == -1.0


def test_out_of_order_harmony_returns_minus_one(monkeypatch):
    monkeypatch.setattr(sandbox, "run_tests", _fake_factory())
    bad = ("<|channel|>final<|message|>def f(): return 1<|end|>"
           "<|channel|>analysis<|message|>after the fact<|end|>")
    r = reward.code_execution_reward(["p"], [bad], ["t"])
    assert r[0] == -1.0


def test_syntax_error_extracted_no_bonuses(monkeypatch):
    # extracted code doesn't ast.parse -> reward 0.0, sandbox never called
    called = {"n": 0}

    def fake(code, tests, timeout_s=30.0):
        called["n"] += 1
        return _fake_factory()(code, tests)
    monkeypatch.setattr(sandbox, "run_tests", fake)
    r = reward.code_execution_reward(["p"], ["def f(: return"], ["t"])
    assert r[0] == 0.0
    assert called["n"] == 0


def test_pass_visible_fail_hidden_penalized(monkeypatch):
    # visible pass_rate 1.0, hidden 0.0 -> override to -1.0
    monkeypatch.setattr(sandbox, "run_tests",
                        _fake_factory(visible_rate=1.0, hidden_rate=0.0))
    comp = "def add(a, b):\n    return 5  # hardcoded"
    r = reward.code_execution_reward(["p"], [comp], ["visible"],
                                     hidden_tests=["HIDDEN suite"])
    assert r[0] == -1.0


def test_pass_visible_pass_hidden_not_penalized(monkeypatch):
    monkeypatch.setattr(sandbox, "run_tests",
                        _fake_factory(visible_rate=1.0, hidden_rate=1.0))
    comp = "def add(a, b):\n    return a + b"
    r = reward.code_execution_reward(["p"], [comp], ["visible"],
                                     hidden_tests=["HIDDEN suite"])
    assert r[0] == pytest.approx(1.4)


# ---- bootstrap patch-similarity ---------------------------------------------

def test_bootstrap_patch_similarity(monkeypatch):
    # no tests -> fall back to SequenceMatcher ratio vs oracle patch
    monkeypatch.setattr(sandbox, "run_tests", _fake_factory())
    oracle = "def add(a, b):\n    return a + b"
    comp = "def add(a, b):\n    return a + b"
    r = reward.code_execution_reward(["p"], [comp], [""], oracle_patch=[oracle])
    assert 0.0 <= r[0] <= 1.0
    assert r[0] == pytest.approx(1.0)


def test_bootstrap_partial_ratio(monkeypatch):
    monkeypatch.setattr(sandbox, "run_tests", _fake_factory())
    r = reward.code_execution_reward(
        ["p"], ["def add(a, b):\n    return a - b"], [""],
        oracle_patch=["def add(a, b):\n    return a + b"])
    assert 0.0 < r[0] < 1.0


def test_no_tests_no_oracle_is_zero(monkeypatch):
    monkeypatch.setattr(sandbox, "run_tests", _fake_factory())
    r = reward.code_execution_reward(["p"], ["def f(): return 1"], [""])
    assert r[0] == 0.0


def test_returns_one_float_per_completion(monkeypatch):
    monkeypatch.setattr(sandbox, "run_tests", _fake_factory())
    comps = ["def f(): return 1"] * 3
    r = reward.code_execution_reward(["p"] * 3, comps, ["t"] * 3)
    assert len(r) == 3
    assert all(isinstance(x, float) for x in r)

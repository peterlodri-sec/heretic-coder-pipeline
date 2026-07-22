
import pytest

from shared import exec_sandbox


def test_module_documents_security_requirements():
    # Guardrail: the module docstring must keep the "no network" / hardened intent
    # so nobody swaps in a naive exec(). Also must honestly call out the subprocess
    # backend's network-isolation limitation.
    doc = exec_sandbox.__doc__ or ""
    assert "No network" in doc or "no network" in doc
    assert "untrusted" in doc.lower()
    assert "nsjail" in doc.lower()


def test_result_dict_has_all_keys():
    res = exec_sandbox.run_tests("def f(): return 1", "assert f() == 1")
    for key in ("passed", "total", "pass_rate", "compiled", "timed_out", "error", "execution_time_s"):
        assert key in res


def test_passing_solution_and_asserts():
    res = exec_sandbox.run_tests("def f(): return 1", "assert f() == 1", timeout_s=10)
    assert res["compiled"] is True
    assert res["timed_out"] is False
    assert res["pass_rate"] == 1.0
    assert res["passed"] == 1
    assert res["total"] == 1
    assert res["error"] is None


def test_failing_assert():
    res = exec_sandbox.run_tests("def f(): return 2", "assert f() == 1", timeout_s=10)
    assert res["compiled"] is True  # code compiled fine, the TEST failed
    assert res["pass_rate"] == 0.0
    assert res["passed"] == 0
    assert res["total"] == 1
    # a test failure is not a sandbox/compile error
    assert res["error"] is None


def test_syntax_error_in_code():
    res = exec_sandbox.run_tests("def f(: return 1", "assert True", timeout_s=10)
    assert res["compiled"] is False
    assert res["pass_rate"] == 0.0
    assert res["total"] == 1
    assert res["error"]  # populated with the compile traceback


def test_import_error_in_code_is_compile_failure():
    res = exec_sandbox.run_tests("import __nope_no_such_mod__", "assert True", timeout_s=10)
    assert res["compiled"] is False
    assert res["error"]


def test_infinite_loop_times_out():
    res = exec_sandbox.run_tests("def f():\n    while True: pass", "f()", timeout_s=2)
    assert res["timed_out"] is True
    assert res["pass_rate"] == 0.0
    assert res["passed"] == 0


def test_fractional_two_test_functions_one_pass_one_fail():
    code = "def f(x): return x"
    tests = (
        "def test_ok():\n    assert f(1) == 1\n"
        "def test_bad():\n    assert f(1) == 2\n"
    )
    res = exec_sandbox.run_tests(code, tests, timeout_s=10)
    assert res["compiled"] is True
    assert res["total"] == 2
    assert res["passed"] == 1
    assert res["pass_rate"] == 0.5


def test_tests_run_in_solution_namespace():
    # HumanEval-style: check(candidate) with the solution fn already in scope.
    code = "def add(a, b): return a + b"
    tests = "def check(candidate):\n    assert candidate(2, 3) == 5\ncheck(add)"
    res = exec_sandbox.run_tests(code, tests, timeout_s=10)
    assert res["pass_rate"] == 1.0


def test_unknown_backend_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("EXEC_SANDBOX_BACKEND", "nsjail")
    with pytest.raises(NotImplementedError):
        exec_sandbox.run_tests("def f(): return 1", "assert f() == 1")
    monkeypatch.setenv("EXEC_SANDBOX_BACKEND", "docker")
    with pytest.raises(NotImplementedError):
        exec_sandbox.run_tests("def f(): return 1", "assert f() == 1")

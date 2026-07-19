import pytest

from shared import exec_sandbox


def test_run_tests_is_stubbed_interface():
    # The hardened sandbox is finalized from research; the module imports fine and
    # calling the primitive raises until the real (hardened) impl lands.
    with pytest.raises(NotImplementedError):
        exec_sandbox.run_tests("def f(): return 1", "assert f() == 1")


def test_module_documents_security_requirements():
    # Guardrail: the module docstring must keep the "no network" / hardened intent
    # so nobody swaps in a naive exec().
    doc = exec_sandbox.__doc__ or ""
    assert "No network" in doc or "no network" in doc
    assert "untrusted" in doc.lower()

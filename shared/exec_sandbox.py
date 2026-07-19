"""Hardened isolated runner for MODEL-GENERATED code — the single execution
primitive shared by stage4 (RFT filter) and stage5 (RLVR reward).

This runs untrusted, adversarial code and MUST be sandboxed. The DEFAULT backend
here is `subprocess`: a throwaway tempdir + a child `python` in its own process
group, launched with a minimal env, POSIX `resource` rlimits (CPU / address-space
/ fork / file-size), and a hard wall-clock timeout that SIGKILLs the whole group.
The untrusted tests never touch a `test_*.py` on disk (they are streamed to the
child via stdin), so the model can't rewrite the suite to `return True`.

HONEST LIMITATION — network isolation:
  The subprocess backend does NOT fully block network egress. It provides
  resource + process-group + filesystem-tempdir isolation only. Real egress
  blocking needs a network namespace (`unshare -n`), nsjail, or Docker on Linux —
  see the `nsjail`/`docker` backends (raise NotImplementedError, wired later:
  network namespace + read-only rootfs + dropped caps + seccomp). Until then, run
  the box on a network-isolated host / firewalled cgroup.

SECURITY REQUIREMENTS (non-negotiable — model output is untrusted input):
  - Isolation: throwaway sandbox, no shared filesystem back to the host.
  - No network: egress disabled (FULL block requires nsjail/docker/`unshare -n`).
  - Resource caps: CPU-time, wall-clock (`timeout_s`), memory, PID/fork, output.
  - No privileges: unprivileged user, read-only rootfs (stronger backends).
  - Deterministic teardown: always kill + reap the group, even on timeout.
"""

import math
import os
import signal
import subprocess
import sys
import tempfile

# Sentinel wrapping the child's JSON verdict so stray prints from untrusted code
# can't be mistaken for the result — parent parses the LAST sentinel line.
_SENTINEL = "__SANDBOX_RESULT__"

_MEM_BYTES = 2 * 1024 * 1024 * 1024   # ~2 GB address space
_MAX_FORKS = 64                        # cap fork bombs
_FSIZE_BYTES = 16 * 1024 * 1024        # 16 MB max file write

# Child runner. Two phases in a fresh namespace: (1) compile+exec the solution to
# distinguish compile-failure from test-failure; (2) exec the tests in the SAME
# namespace so HumanEval `check(candidate)` / assert style can see the solution's
# fns. Fractional: if any `def test_*` fns exist, run each and count individually.
_RUNNER = r'''
import sys, json, traceback

def _emit(d):
    sys.stdout.write("__SANDBOX_RESULT__ " + json.dumps(d) + "\n")
    sys.stdout.flush()

def main():
    tests = sys.stdin.read()
    try:
        code = open("solution.py", "r").read()
    except Exception as e:
        _emit({"passed": 0, "total": 1, "compiled": False,
               "error": "cannot read solution: %r" % (e,)})
        return
    ns = {"__name__": "__solution__"}
    # phase 1: compile + exec solution
    try:
        exec(compile(code, "solution.py", "exec"), ns)
    except BaseException:
        _emit({"passed": 0, "total": 1, "compiled": False,
               "error": traceback.format_exc(limit=6)})
        return
    # phase 2: exec tests in the SAME namespace
    exec_ok = True
    try:
        exec(compile(tests, "tests.py", "exec"), ns)
    except BaseException:
        exec_ok = False
    funcs = [v for k, v in list(ns.items())
             if k.startswith("test_") and callable(v)]
    if funcs:
        # fractional: count each test_* fn
        passed = 0
        for fn in funcs:
            try:
                fn(); passed += 1
            except BaseException:
                pass
        _emit({"passed": passed, "total": len(funcs),
               "compiled": True, "error": None})
    else:
        # binary: the exec itself was the assert harness (HumanEval pass@1)
        _emit({"passed": 1 if exec_ok else 0, "total": 1,
               "compiled": True, "error": None})

main()
'''


def _apply_rlimits(timeout_s: float):
    # POSIX preexec: cap CPU / memory / forks / file size in the child. Each wrap
    # is best-effort — RLIMIT_AS/NPROC are unreliable on macOS, must NOT crash.
    import resource

    cpu = int(math.ceil(timeout_s)) + 1
    for name, soft in (("RLIMIT_CPU", cpu), ("RLIMIT_AS", _MEM_BYTES),
                       ("RLIMIT_NPROC", _MAX_FORKS), ("RLIMIT_FSIZE", _FSIZE_BYTES)):
        res = getattr(resource, name, None)
        if res is None:
            continue
        try:
            resource.setrlimit(res, (soft, soft))
        except (ValueError, OSError, resource.error):
            pass  # macOS / hardened hosts may reject — carry on


def _minimal_env() -> dict:
    # Strip proxy/network hints; no bytecode droppings. Keep a bare PATH only.
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "LC_ALL": "C.UTF-8",
    }


def _parse_verdict(stdout: str) -> dict | None:
    # Take the LAST sentinel line — untrusted code can print earlier fakes.
    import json

    verdict = None
    for line in stdout.splitlines():
        if line.startswith(_SENTINEL):
            try:
                verdict = json.loads(line[len(_SENTINEL):].strip())
            except ValueError:
                verdict = None
    return verdict


def _run_subprocess(code: str, tests: str, timeout_s: float) -> dict:
    posix = os.name == "posix"
    with tempfile.TemporaryDirectory(prefix="sbx_") as tmp:
        with open(os.path.join(tmp, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmp, "_runner.py"), "w") as f:
            f.write(_RUNNER)

        preexec = (lambda: _apply_rlimits(timeout_s)) if posix else None
        p = subprocess.Popen(
            [sys.executable, "-I", "_runner.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=tmp, env=_minimal_env(), text=True,
            start_new_session=posix, preexec_fn=preexec,
        )
        try:
            out, err = p.communicate(input=tests, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(p)
            try:
                p.communicate(timeout=5)  # reap
            except Exception:
                pass
            return _result(0, 1, compiled=False, timed_out=True,
                           error="wall-clock timeout after %.1fs" % timeout_s)

        verdict = _parse_verdict(out)
        if verdict is None:
            # child died / crashed before emitting (e.g. rlimit kill, OOM)
            tail = (err or out or "").strip()[-500:]
            return _result(0, 1, compiled=False, timed_out=False,
                           error="no sandbox verdict; child exited %s: %s"
                                 % (p.returncode, tail))
        return _result(
            int(verdict.get("passed", 0)), int(verdict.get("total", 1)),
            compiled=bool(verdict.get("compiled", False)), timed_out=False,
            error=verdict.get("error"),
        )


def _kill_group(p) -> None:
    # SIGKILL the whole process group so forked children die too.
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.kill()
        except Exception:
            pass


def _result(passed: int, total: int, *, compiled: bool, timed_out: bool,
            error) -> dict:
    total = max(int(total), 0)
    passed = max(int(passed), 0)
    pass_rate = (passed / total) if total > 0 else 0.0
    return {"passed": passed, "total": total, "pass_rate": pass_rate,
            "compiled": compiled, "timed_out": timed_out, "error": error}


def run_tests(code: str, tests: str, timeout_s: float = 30.0) -> dict:
    """Execute `code` against `tests` inside the hardened sandbox and report the
    per-test outcome. Used as the pass/fail oracle for RFT filtering and as the
    unit-test-pass-rate signal for the RLVR reward.

    Args:
        code: model-generated solution / patch (untrusted).
        tests: unit tests to run against `code` (HumanEval/MBPP+ asserts,
            `check(candidate)`, or `def test_*` fns for a fractional pass-rate).
        timeout_s: hard wall-clock cap for the whole run.

    Returns:
        dict, EXACTLY:
          {"passed": int,        # tests that passed
           "total": int,         # tests attempted (1 for binary HumanEval pass@1)
           "pass_rate": float,   # passed / total in [0, 1] (0.0 if total == 0)
           "compiled": bool,     # solution imported/compiled without error
           "timed_out": bool,    # killed by the wall-clock cap
           "error": str | None}  # non-test failure (compile/sandbox), else None
    """
    # Backend hook: subprocess is portable + always available. nsjail/docker are
    # the stronger ON-BOX backends (network namespace + read-only rootfs + seccomp)
    # to wire on the Linux H200 — not yet implemented.
    backend = os.environ.get("EXEC_SANDBOX_BACKEND", "subprocess").strip().lower()
    if backend == "subprocess":
        return _run_subprocess(code, tests, timeout_s)
    if backend in ("nsjail", "docker"):
        raise NotImplementedError(
            "EXEC_SANDBOX_BACKEND=%r is a stronger on-box backend to wire later "
            "(network namespace + read-only rootfs + dropped caps + seccomp on the "
            "Linux H200); the portable default backend is 'subprocess'." % backend)
    raise ValueError("unknown EXEC_SANDBOX_BACKEND=%r" % backend)

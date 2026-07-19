"""Hardened isolated runner for MODEL-GENERATED code — the single execution
primitive shared by stage4 (RFT filter) and stage5 (RLVR reward).

INTERFACE ONLY. The implementation is finalized from the SOTA research (see
docs/superpowers/plans/2026-07-19-extended-pipeline-rlvr-selfimprove.md §"Shared
additions" / build-order step 2). Do NOT hand-roll a naive `exec()` — this runs
untrusted, adversarial code and MUST be sandboxed.

SECURITY REQUIREMENTS (non-negotiable — model output is untrusted input):
  - Isolation: run each candidate in a throwaway sandbox — Docker / gVisor /
    nsjail / Firejail (or delegate to the `verifiers` project's sandbox, which is
    purpose-built for RL exec envs). No shared filesystem back to the host.
  - No network: egress fully disabled (no data exfil, no phone-home).
  - Resource caps: hard CPU-time, wall-clock (`timeout_s`), memory, PID/fork, and
    output-size limits so a fork bomb / OOM / infinite loop cannot wedge the host.
  - No privileges: unprivileged user, read-only rootfs where possible, dropped
    capabilities, seccomp filter.
  - Deterministic teardown: always kill + reap the sandbox, even on timeout.
"""


def run_tests(code: str, tests: str, timeout_s: float = 30.0) -> dict:
    """Execute `code` against `tests` inside the hardened sandbox and report the
    per-test outcome. Used as the pass/fail oracle for RFT filtering and as the
    unit-test-pass-rate signal for the RLVR reward.

    Args:
        code: model-generated solution / patch (untrusted).
        tests: unit tests to run against `code` (e.g. HumanEval/MBPP+ asserts, or
            a SWE-Gym FAIL_TO_PASS + PASS_TO_PASS harness).
        timeout_s: hard wall-clock cap for the whole run.

    Returns:
        dict with at least:
          {"passed": int,        # tests that passed
           "total": int,         # tests attempted
           "pass_rate": float,   # passed / total in [0, 1]
           "compiled": bool,     # code imported/compiled without error
           "timed_out": bool,
           "error": str | None}  # non-test failure (compile/sandbox), else None
    """
    raise NotImplementedError(
        "finalize sandbox impl from SOTA research — must be hardened (Docker/"
        "gVisor/nsjail/Firejail or verifiers' sandbox); see 2026-07-19 plan")

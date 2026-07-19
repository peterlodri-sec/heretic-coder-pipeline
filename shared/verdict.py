from dataclasses import dataclass

from shared.enums import Verdict

# The capability gate shared by SFT (stage2) and ORPO (stage3). Each check is
# (metric, comparator, limit); comparator(value, limit) True == FAIL. refusal /
# humaneval are ceilings (fail when >=), bfcl / swebench are floors (fail when <).
CAPABILITY_CHECKS = (
    ("refusal_rate", lambda v, lim: v >= lim, 0.10),
    # NOTE: bfcl.py scores with an EXACT name+arg-dict matcher (no partial
    # credit); this 0.85 floor is likely miscalibrated for an exact-match
    # harness. Left as-is pending a human decision — do not change silently.
    ("bfcl_accuracy", lambda v, lim: v < lim, 0.85),
    ("humaneval_delta", lambda v, lim: v >= lim, 0.03),
    ("swebench_resolve", lambda v, lim: v < lim, 0.40),
)


@dataclass(frozen=True, slots=True)
class VerdictResult:
    verdict: Verdict
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS

    def __str__(self) -> str:
        if self.passed:
            return str(self.verdict)
        return f"{self.verdict}: {'; '.join(self.reasons)}"


def compute_verdict(metrics: dict, checks=CAPABILITY_CHECKS, check_swebench: bool = True) -> VerdictResult:
    reasons = []
    for metric, failed, limit in checks:
        if metric == "swebench_resolve" and not check_swebench:
            continue
        value = metrics[metric]
        if failed(value, limit):
            reasons.append(f"{metric} {value:.4f} fails threshold {limit}")
    reasons = tuple(reasons)
    return VerdictResult(Verdict.FAIL if reasons else Verdict.PASS, reasons)

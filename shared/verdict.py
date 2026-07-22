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


# Aspirational SOTA target floors from the SWE deep-research report
# (ideas/how_to_increase_swe_score.md, Pillar 5 "Quantitative Targets"). These are
# deliberately NOT gate thresholds: CAPABILITY_CHECKS above stays at the safe,
# currently-achievable floors so a genuinely good run is never auto-failed for not
# yet reaching SOTA (e.g. no open model resolves 60% of SWE-bench Verified today).
# target_gaps() reports how far each metric sits from SOTA so progress is tracked
# every run — a soft telemetry signal, never a pass/fail. (metric, direction, target)
# with direction 'floor' = higher-is-better, 'ceiling' = lower-is-better.
SOTA_TARGETS = (
    ("refusal_rate", "ceiling", 0.05),
    ("bfcl_accuracy", "floor", 0.90),
    ("humaneval_delta", "ceiling", 0.01),
    ("swebench_resolve", "floor", 0.60),
)


@dataclass(frozen=True, slots=True)
class TargetGap:
    metric: str
    value: float
    target: float
    direction: str  # 'floor' (higher better) | 'ceiling' (lower better)
    met: bool
    gap: float      # signed distance still to cover to reach target; <=0 == met

    def __str__(self) -> str:
        arrow = "reached" if self.met else f"{self.gap:+.4f} to go"
        return f"{self.metric}={self.value:.4f} vs SOTA {self.target} ({arrow})"


def target_gaps(metrics: dict, targets=SOTA_TARGETS) -> tuple[TargetGap, ...]:
    """Soft progress report against SOTA_TARGETS. Never fails a run; a metric that
    is absent/None is skipped (some stages don't emit swebench)."""
    gaps = []
    for metric, direction, target in targets:
        value = metrics.get(metric)
        if value is None:
            continue
        met = value <= target if direction == "ceiling" else value >= target
        # signed remaining distance toward the target (0 or negative once met)
        gap = (value - target) if direction == "ceiling" else (target - value)
        gaps.append(TargetGap(metric, float(value), target, direction, met, gap))
    return tuple(gaps)


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

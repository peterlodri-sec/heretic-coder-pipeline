# frontier/verdict.py — the capability gate lives in shared.verdict; this
# re-export preserves frontier's import surface (compute_verdict, VerdictResult).
from shared.verdict import CAPABILITY_CHECKS, VerdictResult, compute_verdict

__all__ = ["CAPABILITY_CHECKS", "VerdictResult", "compute_verdict"]

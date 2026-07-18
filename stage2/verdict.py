# stage2/verdict.py — the SFT capability gate now lives in shared.verdict; this
# re-export preserves stage2's import surface (compute_verdict, VerdictResult).
from shared.verdict import CAPABILITY_CHECKS, VerdictResult, compute_verdict

__all__ = ["CAPABILITY_CHECKS", "VerdictResult", "compute_verdict"]

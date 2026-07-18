# stage3 reuses the shared capability gate unchanged.
from shared.verdict import CAPABILITY_CHECKS, VerdictResult, compute_verdict

__all__ = ["CAPABILITY_CHECKS", "VerdictResult", "compute_verdict"]

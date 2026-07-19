# stage4 reuses the shared capability gate unchanged (refusal held, BFCL/HumanEval,
# SWE-bench). RFT self-improvement must not regress the gate.
from shared.verdict import CAPABILITY_CHECKS, VerdictResult, compute_verdict

__all__ = ["CAPABILITY_CHECKS", "VerdictResult", "compute_verdict"]

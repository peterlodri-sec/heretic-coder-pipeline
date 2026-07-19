# stage5 (RLVR) reuses the shared capability gate unchanged: refusal held,
# BFCL/HumanEval up, SWE-bench floor. Verifiable-reward RL optimizes the outcome;
# this gate guards against reward hacking / capability regression.
from shared.verdict import CAPABILITY_CHECKS, VerdictResult, compute_verdict

__all__ = ["CAPABILITY_CHECKS", "VerdictResult", "compute_verdict"]

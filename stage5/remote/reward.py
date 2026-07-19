# stage5/remote/reward.py — verifiable reward function for the RLVR GRPOTrainer.
# INTERFACE ONLY; finalized from SOTA research (build-order step 3, see 2026-07-19
# plan). Heavy imports stay function-local so this module imports without a GPU.


def code_execution_reward(prompts, completions, tests, **kwargs) -> list[float]:
    """GRPO reward: how well each completion's code satisfies its unit tests.

    reward = fraction of unit tests passing, computed by running the completion's
    code against ``tests`` in ``shared.exec_sandbox.run_tests`` (hardened, no net,
    resource-capped). Add format/compile shaping (e.g. small penalty when code
    fails to compile, small bonus for well-formed harmony ``final``-channel
    output) so the gradient is dense early in training.

    Bootstrap alternative (no test harness needed): SWE-RL patch-similarity —
    ``difflib.SequenceMatcher`` ratio of the generated patch vs the oracle patch
    (arXiv 2502.18449). Upgrade to real pass-rate once the sandbox is wired.

    Args:
        prompts: the batch's problems (aligned with completions).
        completions: model samples; extract the harmony ``final``-channel code.
        tests: per-item unit tests to execute the extracted code against.
        **kwargs: extra columns TRL forwards from the dataset (e.g. oracle patch).

    Returns:
        list[float] rewards in [0, 1], one per completion (GRPOTrainer signature).
    """
    raise NotImplementedError(
        "finalize verifiable reward from SOTA research — exec-test pass-rate via "
        "shared.exec_sandbox (+ SWE-RL patch-similarity bootstrap); see 2026-07-19 plan")

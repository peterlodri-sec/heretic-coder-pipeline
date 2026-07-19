from enum import StrEnum


class Stage(StrEnum):
    """Lifecycle stage of a stage4 RFT / rejection-sampling self-improvement run,
    as written to status.json. GENERATING/VERIFYING/TRAINING cycle once per round;
    EVALUATING/DONE run after the final round."""

    SETUP = "setup"
    GENERATING = "generating"    # sample N candidates per problem (vLLM/HF)
    VERIFYING = "verifying"      # filter via shared.exec_sandbox.run_tests
    TRAINING = "training"        # SFT-on-passing via stage2's sft_train.train
    EVALUATING = "evaluating"
    DONE = "done"

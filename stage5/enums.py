from enum import StrEnum


class Stage(StrEnum):
    """Lifecycle stage of a stage5 RLVR (execution-feedback RL) run, as written to
    status.json. RLVR is the terminal stage for gpt-oss (replaces ORPO)."""

    SETUP = "setup"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DONE = "done"

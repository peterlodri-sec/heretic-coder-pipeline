from enum import StrEnum


class Stage(StrEnum):
    """Lifecycle stage of a stage3 ORPO run, as written to status.json."""

    SETUP = "setup"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DONE = "done"

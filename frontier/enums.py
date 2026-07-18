from enum import StrEnum


class Stage(StrEnum):
    """Lifecycle stage of a frontier (heretic->SFT->ORPO) run, written to status.json."""

    SETUP = "setup"
    ABLITERATING = "abliterating"
    PREPARING_DATA = "preparing_data"
    TRAINING_SFT = "training_sft"
    TRAINING_ORPO = "training_orpo"
    EVALUATING = "evaluating"
    DONE = "done"

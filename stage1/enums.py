from enum import StrEnum


class Stage(StrEnum):
    """Lifecycle stage of a stage1 run, as written to status.json."""

    SETUP = "setup"
    ABLITERATING = "abliterating"
    EVALUATING = "evaluating"
    DONE = "done"


class Verdict(StrEnum):
    """Terminal outcome of a stage1 run."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"

from enum import StrEnum


class Verdict(StrEnum):
    """Terminal outcome of a pipeline stage. Shared across stages."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"

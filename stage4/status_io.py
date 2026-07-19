from dataclasses import dataclass

from enums import Stage
from shared.enums import Verdict
from shared.status import JsonStatusMixin


@dataclass(slots=True)
class Status(JsonStatusMixin):
    started_at: str
    updated_at: str
    stage: Stage = Stage.SETUP
    # loop control — which round of K we are on, and its filter yield.
    round: int | None = None
    num_rounds: int | None = None
    candidates_generated: int | None = None
    candidates_passing: int | None = None
    train_loss: float | None = None
    # final capability gate (shared 4-metric gate).
    refusal_rate: float | None = None
    bfcl_accuracy: float | None = None
    humaneval_delta: float | None = None
    swebench_resolve: float | None = None
    verdict: Verdict | None = None
    hf_repo: str | None = None
    error: str | None = None
    log_tail: str | None = None

    @classmethod
    def new(cls, started_at: str) -> "Status":
        return cls(started_at=started_at, updated_at=started_at)

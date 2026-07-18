from dataclasses import dataclass

from enums import Stage
from shared.enums import Verdict
from shared.status import JsonStatusMixin


@dataclass(slots=True)
class Status(JsonStatusMixin):
    started_at: str
    updated_at: str
    stage: Stage = Stage.SETUP
    train_loss: float | None = None
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

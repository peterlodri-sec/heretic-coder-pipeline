import json
import os
from dataclasses import asdict, dataclass, fields

from enums import Stage, Verdict


@dataclass(slots=True)
class Status:
    """The run's status.json, shared by the remote worker and the controller.

    slots=True makes writes to unknown fields an AttributeError instead of a
    silently-accepted dict key — a typo can't spawn a phantom field anymore.
    Stage/Verdict are StrEnum, so json serialises them as their value directly.
    """

    started_at: str
    updated_at: str
    stage: Stage = Stage.SETUP
    refusal_rate: float | None = None
    kl_divergence: float | None = None
    mmlu_delta: float | None = None
    gsm8k_delta: float | None = None
    verdict: Verdict | None = None
    hf_repo: str | None = None
    error: str | None = None
    log_tail: str | None = None

    @classmethod
    def new(cls, started_at: str) -> "Status":
        return cls(started_at=started_at, updated_at=started_at)

    @classmethod
    def from_dict(cls, data: dict) -> "Status":
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        if payload.get("stage") is not None:
            payload["stage"] = Stage(payload["stage"])
        if payload.get("verdict") is not None:
            payload["verdict"] = Verdict(payload["verdict"])
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str) -> "Status":
        return cls.from_dict(json.loads(text))

    @classmethod
    def read(cls, path: str) -> "Status":
        with open(path) as f:
            return cls.from_json(f.read())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def write(self, path: str) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(self.to_json())
        os.replace(tmp_path, path)

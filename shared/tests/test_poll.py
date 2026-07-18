from dataclasses import dataclass
from enum import StrEnum
from unittest.mock import patch

from shared import poll
from shared.status import JsonStatusMixin


class Stage(StrEnum):
    RUNNING = "running"
    DONE = "done"


@dataclass(slots=True)
class S(JsonStatusMixin):
    stage: Stage = Stage.RUNNING
    verdict: str | None = None


def test_returns_when_stage_done():
    running = S(stage=Stage.RUNNING).to_json()
    done = S(stage=Stage.DONE, verdict="pass").to_json()
    with patch.object(poll.ssh_utils, "run_ssh", side_effect=[running, done]), \
         patch.object(poll.time, "sleep"):
        result = poll.poll_until_done("h", 1, "/r/status.json", S, Stage.DONE, interval=0)
    assert result.stage is Stage.DONE
    assert result.verdict == "pass"


def test_tolerates_transient_ssh_error():
    from shared.ssh_utils import SSHError
    done = S(stage=Stage.DONE).to_json()
    with patch.object(poll.ssh_utils, "run_ssh", side_effect=[SSHError("boom"), done]), \
         patch.object(poll.time, "sleep"):
        result = poll.poll_until_done("h", 1, "/r/status.json", S, Stage.DONE, interval=0)
    assert result.stage is Stage.DONE

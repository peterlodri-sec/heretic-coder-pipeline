import json
import tempfile
from dataclasses import dataclass
from enum import StrEnum

import pytest

from shared.status import JsonStatusMixin


class Phase(StrEnum):
    A = "a"
    DONE = "done"


@dataclass(slots=True)
class Demo(JsonStatusMixin):
    started_at: str
    phase: Phase = Phase.A
    score: float | None = None


def test_slots_reject_unknown_field():
    d = Demo(started_at="1")
    with pytest.raises(AttributeError):
        d.typo = 5


def test_enum_round_trips_as_plain_string():
    d = Demo(started_at="1", phase=Phase.DONE)
    assert json.loads(d.to_json())["phase"] == "done"
    assert Demo.from_json(d.to_json()).phase is Phase.DONE


def test_from_dict_drops_unknown_keys():
    d = Demo.from_dict({"started_at": "1", "legacy": 9})
    assert d.started_at == "1"
    assert not hasattr(d, "legacy")


def test_write_is_atomic_and_round_trips():
    d = Demo(started_at="1", phase=Phase.DONE, score=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/s.json"
        d.write(path)
        assert not __import__("os").path.exists(path + ".tmp")
        assert Demo.read(path) == d

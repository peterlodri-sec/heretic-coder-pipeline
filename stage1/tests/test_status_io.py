import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enums import Stage
from shared.enums import Verdict
from status_io import Status


def test_new_status_defaults():
    status = Status.new("100.0")
    assert status.stage is Stage.SETUP
    assert status.started_at == "100.0"
    assert status.updated_at == "100.0"
    for field in ("refusal_rate", "kl_divergence", "mmlu_delta", "gsm8k_delta",
                  "verdict", "hf_repo", "error", "log_tail"):
        assert getattr(status, field) is None


def test_slots_reject_unknown_field():
    status = Status.new("100.0")
    with pytest.raises(AttributeError):
        status.typo_field = "oops"


def test_write_then_read_round_trips():
    status = Status.new("100.0")
    status.stage = Stage.DONE
    status.verdict = Verdict.PASS

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        status.write(path)
        loaded = Status.read(path)

    assert loaded == status
    assert loaded.stage is Stage.DONE
    assert loaded.verdict is Verdict.PASS


def test_enums_serialize_as_plain_strings():
    status = Status.new("100.0")
    status.stage = Stage.DONE
    status.verdict = Verdict.FAIL
    payload = json.loads(status.to_json())
    assert payload["stage"] == "done"
    assert payload["verdict"] == "fail"


def test_write_is_atomic_no_leftover_tmp_file():
    status = Status.new("100.0")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        status.write(path)
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


def test_from_json_ignores_unknown_keys():
    status = Status.from_json('{"started_at": "1", "updated_at": "1", "legacy": 7}')
    assert status.started_at == "1"
    assert not hasattr(status, "legacy")

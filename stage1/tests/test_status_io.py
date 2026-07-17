import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from status_io import new_status, parse_status_text, read_status, write_status


def test_new_status_has_all_required_fields():
    status = new_status("100.0")
    assert status["stage"] == "setup"
    assert status["started_at"] == "100.0"
    assert status["updated_at"] == "100.0"
    for field in ("refusal_rate", "kl_divergence", "mmlu_delta", "gsm8k_delta",
                  "verdict", "hf_repo", "error", "log_tail"):
        assert field in status
        assert status[field] is None


def test_write_then_read_round_trips():
    status = new_status("100.0")
    status["stage"] = "done"
    status["verdict"] = "pass"

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        write_status(path, status)
        loaded = read_status(path)

    assert loaded == status


def test_write_is_atomic_no_leftover_tmp_file():
    status = new_status("100.0")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        write_status(path, status)
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


def test_parse_status_text_matches_read_status():
    status = new_status("100.0")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        write_status(path, status)
        with open(path) as f:
            text = f.read()

    assert parse_status_text(text) == status
    assert parse_status_text(text) == json.loads(text)

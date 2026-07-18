import tempfile
from unittest.mock import patch

from shared import vast_ops


def test_provision_lock_acquires_and_releases(tmp_path):
    lock_path = str(tmp_path / "p.lock")
    with patch.object(vast_ops, "PROVISION_LOCK_PATH", lock_path):
        with vast_ops.provision_lock():
            pass  # acquired + released without error
    import os
    assert os.path.exists(lock_path)


def test_load_api_key_strips(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("  abc123\n")
    with patch.object(vast_ops, "API_KEY_PATH", str(key_file)):
        assert vast_ops.load_api_key() == "abc123"

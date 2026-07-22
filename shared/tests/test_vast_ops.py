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


def test_local_hf_token_path_present(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("hf_secret")
    with patch.object(vast_ops, "HF_TOKEN_PATH", str(token_file)):
        assert vast_ops.local_hf_token_path() == str(token_file)


def test_local_hf_token_path_absent(tmp_path):
    missing = tmp_path / "nope"
    with patch.object(vast_ops, "HF_TOKEN_PATH", str(missing)):
        assert vast_ops.local_hf_token_path() is None

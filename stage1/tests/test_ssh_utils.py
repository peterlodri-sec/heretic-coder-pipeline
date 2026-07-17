import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ssh_utils import SSHError, run_ssh, scp_from, scp_to


def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_ssh_returns_stdout_on_success():
    with patch("ssh_utils.subprocess.run", return_value=_completed(0, stdout="hello\n")) as mock_run:
        result = run_ssh("root@host", 1234, "echo hello")
    assert result == "hello\n"
    args = mock_run.call_args[0][0]
    assert args[0] == "ssh"
    assert "-p" in args and "1234" in args
    assert args[-2:] == ["root@host", "echo hello"]


def test_run_ssh_retries_on_transient_failure_then_succeeds():
    responses = [
        _completed(255, stderr="ssh: connect to host x port 22: Operation timed out"),
        _completed(0, stdout="ok\n"),
    ]
    with patch("ssh_utils.subprocess.run", side_effect=responses), \
         patch("ssh_utils.time.sleep") as mock_sleep:
        result = run_ssh("root@host", 1234, "echo ok", retries=3, backoff=5)
    assert result == "ok\n"
    mock_sleep.assert_called_once_with(5)


def test_run_ssh_raises_after_exhausting_retries_on_transient_failure():
    responses = [_completed(255, stderr="Connection timed out")] * 3
    with patch("ssh_utils.subprocess.run", side_effect=responses), \
         patch("ssh_utils.time.sleep"):
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "echo ok", retries=3, backoff=1)


def test_run_ssh_raises_immediately_on_non_transient_failure():
    with patch("ssh_utils.subprocess.run",
               return_value=_completed(1, stderr="bash: some_command: command not found")) as mock_run:
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "some_command", retries=3)
    assert mock_run.call_count == 1


def test_scp_to_builds_recursive_command():
    with patch("ssh_utils.subprocess.run", return_value=_completed(0)) as mock_run:
        scp_to("root@host", 1234, "/local/dir", "/remote/dir", recursive=True)
    args = mock_run.call_args[0][0]
    assert args[0] == "scp"
    assert "-P" in args and "1234" in args
    assert "-r" in args
    assert args[-2:] == ["/local/dir", "root@host:/remote/dir"]


def test_scp_to_raises_on_failure():
    with patch("ssh_utils.subprocess.run", return_value=_completed(1, stderr="No such file")):
        with pytest.raises(SSHError):
            scp_to("root@host", 1234, "/local/dir", "/remote/dir")


def test_scp_from_builds_command():
    with patch("ssh_utils.subprocess.run", return_value=_completed(0)) as mock_run:
        scp_from("root@host", 1234, "/remote/file", "/local/file")
    args = mock_run.call_args[0][0]
    assert args[0] == "scp"
    assert args[-2:] == ["root@host:/remote/file", "/local/file"]

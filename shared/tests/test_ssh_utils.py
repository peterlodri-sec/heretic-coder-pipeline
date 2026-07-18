import subprocess
from unittest.mock import patch

import pytest

from shared.ssh_utils import SSHError, run_ssh, scp_from, scp_to, wait_for_ssh


def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_ssh_returns_stdout_on_success():
    with patch("shared.ssh_utils.subprocess.run", return_value=_completed(0, stdout="hello\n")) as mock_run:
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
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        result = run_ssh("root@host", 1234, "echo ok", retries=3, backoff=5)
    assert result == "ok\n"
    mock_sleep.assert_called_once_with(5)


def test_run_ssh_raises_after_exhausting_retries_on_transient_failure():
    responses = [_completed(255, stderr="Connection timed out")] * 3
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep"):
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "echo ok", retries=3, backoff=1)


def test_run_ssh_raises_immediately_on_non_transient_failure():
    with patch("shared.ssh_utils.subprocess.run",
               return_value=_completed(1, stderr="bash: some_command: command not found")) as mock_run:
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "some_command", retries=3)
    assert mock_run.call_count == 1


def test_run_ssh_connect_timeout_and_command_timeout_are_independent():
    with patch("shared.ssh_utils.subprocess.run", return_value=_completed(0, stdout="ok\n")) as mock_run:
        run_ssh("root@host", 1234, "long_cmd", timeout=1200, connect_timeout=15)
    args = mock_run.call_args[0][0]
    assert "ConnectTimeout=15" in args
    # subprocess wall-clock wait tracks `timeout`, not `connect_timeout`
    assert mock_run.call_args.kwargs["timeout"] == 1200


def test_run_ssh_retries_on_command_timeout_then_succeeds():
    responses = [
        subprocess.TimeoutExpired(cmd="ssh", timeout=30),
        _completed(0, stdout="done\n"),
    ]
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        result = run_ssh("root@host", 1234, "slow", retries=3, backoff=7)
    assert result == "done\n"
    mock_sleep.assert_called_once_with(7)


def test_run_ssh_raises_ssherror_after_command_timeout_exhausts_retries():
    responses = [subprocess.TimeoutExpired(cmd="ssh", timeout=30)] * 3
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep"):
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "slow", retries=3, backoff=1)


def test_run_ssh_retries_on_kex_boot_race():
    responses = [
        _completed(255, stderr="kex_exchange_identification: Connection closed by remote host"),
        _completed(0, stdout="up\n"),
    ]
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep"):
        result = run_ssh("root@host", 1234, "echo up", retries=3, backoff=1)
    assert result == "up\n"


def test_scp_to_builds_recursive_command():
    with patch("shared.ssh_utils.subprocess.run", return_value=_completed(0)) as mock_run:
        scp_to("root@host", 1234, "/local/dir", "/remote/dir", recursive=True)
    args = mock_run.call_args[0][0]
    assert args[0] == "scp"
    assert "-P" in args and "1234" in args
    assert "-r" in args
    assert args[-2:] == ["/local/dir", "root@host:/remote/dir"]


def test_scp_to_raises_on_failure():
    with patch("shared.ssh_utils.subprocess.run", return_value=_completed(1, stderr="No such file")):
        with pytest.raises(SSHError):
            scp_to("root@host", 1234, "/local/dir", "/remote/dir")


def test_scp_to_retries_on_transient_failure_then_succeeds():
    responses = [
        _completed(1, stderr="ssh: connect to host x port 22: Connection refused"),
        _completed(0),
    ]
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        scp_to("root@host", 1234, "/local/dir", "/remote/dir", retries=3, backoff=5)
    mock_sleep.assert_called_once_with(5)


def test_scp_to_retries_on_timeout_then_succeeds():
    responses = [
        subprocess.TimeoutExpired(cmd="scp", timeout=120),
        _completed(0),
    ]
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        scp_to("root@host", 1234, "/local/dir", "/remote/dir", retries=3, backoff=7)
    mock_sleep.assert_called_once_with(7)


def test_scp_to_raises_ssherror_after_timeout_exhausts_retries():
    responses = [subprocess.TimeoutExpired(cmd="scp", timeout=120)] * 3
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep"):
        with pytest.raises(SSHError):
            scp_to("root@host", 1234, "/local/dir", "/remote/dir", retries=3, backoff=1)


def test_wait_for_ssh_returns_immediately_when_reachable():
    with patch("shared.ssh_utils.subprocess.run", return_value=_completed(0)) as run, \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        wait_for_ssh("root@host", 1234, attempts=5, delay=15)
    assert run.call_count == 1
    mock_sleep.assert_not_called()


def test_wait_for_ssh_retries_until_reachable():
    responses = [
        _completed(255, stderr="ssh: connect to host x port 22: Connection refused"),
        _completed(0),
    ]
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        wait_for_ssh("root@host", 1234, attempts=5, delay=3)
    mock_sleep.assert_called_once_with(3)


def test_wait_for_ssh_raises_after_attempts_exhausted():
    with patch("shared.ssh_utils.subprocess.run",
               return_value=_completed(255, stderr="Connection refused")), \
         patch("shared.ssh_utils.time.sleep"):
        with pytest.raises(SSHError):
            wait_for_ssh("root@host", 1234, attempts=3, delay=1)


def test_scp_from_builds_command():
    with patch("shared.ssh_utils.subprocess.run", return_value=_completed(0)) as mock_run:
        scp_from("root@host", 1234, "/remote/file", "/local/file")
    args = mock_run.call_args[0][0]
    assert args[0] == "scp"
    assert args[-2:] == ["root@host:/remote/file", "/local/file"]


def test_scp_from_retries_on_transient_failure_then_succeeds():
    responses = [
        _completed(1, stderr="Connection refused"),
        _completed(0),
    ]
    with patch("shared.ssh_utils.subprocess.run", side_effect=responses), \
         patch("shared.ssh_utils.time.sleep") as mock_sleep:
        scp_from("root@host", 1234, "/remote/file", "/local/file", retries=3, backoff=5)
    mock_sleep.assert_called_once_with(5)

import subprocess
import time

TRANSIENT_MARKERS = (
    "Operation timed out",
    "Connection timed out",
    "Connection refused",
    # sshd not fully up yet right after a vast.ai instance boots
    "kex_exchange_identification",
    "Connection closed by remote host",
    "Connection reset by peer",
)


class SSHError(RuntimeError):
    pass


def _is_transient(stderr: str) -> bool:
    return any(marker in stderr for marker in TRANSIENT_MARKERS)


def run_ssh(host: str, port: int, command: str, timeout: int = 30,
            connect_timeout: int = 30, retries: int = 3, backoff: int = 10) -> str:
    # `connect_timeout` bounds only TCP/SSH connection setup (-o ConnectTimeout);
    # `timeout` bounds how long the command itself may run (subprocess wait).
    # Keep them separate: a long-running command (e.g. setup.sh) needs a big
    # `timeout` but must not inflate ConnectTimeout, which would mask a dead host.
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "ssh", "-p", str(port),
                    "-o", f"ConnectTimeout={connect_timeout}",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=accept-new",
                    host, command,
                ],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_error = f"command exceeded {timeout}s wall-clock timeout"
            if attempt == retries:
                raise SSHError(
                    f"ssh to {host}:{port} timed out (attempt {attempt}/{retries}): {last_error}"
                )
            time.sleep(backoff)
            continue
        if result.returncode == 0:
            return result.stdout
        last_error = result.stderr
        if not _is_transient(last_error) or attempt == retries:
            raise SSHError(
                f"ssh to {host}:{port} failed (attempt {attempt}/{retries}): {last_error.strip()}"
            )
        time.sleep(backoff)
    raise SSHError(f"ssh to {host}:{port} failed after {retries} attempts: {last_error.strip()}")


def wait_for_ssh(host: str, port: int, attempts: int = 20, delay: int = 15,
                 connect_timeout: int = 15) -> None:
    """Block until the host accepts SSH. A freshly rented vast.ai instance takes
    a while before sshd is up; transferring/commanding before then hangs until
    the subprocess timeout. Poll a trivial command until it succeeds. Raises
    SSHError if the host never becomes reachable."""
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            run_ssh(host, port, "true", timeout=connect_timeout + 10,
                    connect_timeout=connect_timeout, retries=1)
            return
        except SSHError as error:
            last_error = str(error)
            if attempt == attempts:
                break
            time.sleep(delay)
    raise SSHError(
        f"ssh to {host}:{port} never became ready after {attempts} attempts: {last_error}"
    )


def _run_scp(args: list, description: str, timeout: int, retries: int, backoff: int) -> None:
    # Shared retry loop for scp_to/scp_from. Handles BOTH a non-zero exit with a
    # transient stderr AND a subprocess.TimeoutExpired (a hung transfer to a
    # not-yet-ready box) — the latter previously escaped unhandled and crashed
    # the controller mid-deploy.
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_error = f"exceeded {timeout}s wall-clock timeout"
            if attempt == retries:
                raise SSHError(f"{description} timed out (attempt {attempt}/{retries}): {last_error}")
            time.sleep(backoff)
            continue
        if result.returncode == 0:
            return
        last_error = result.stderr
        if not _is_transient(last_error) or attempt == retries:
            raise SSHError(f"{description} failed (attempt {attempt}/{retries}): {last_error.strip()}")
        time.sleep(backoff)


def scp_to(host: str, port: int, local_path: str, remote_path: str,
           recursive: bool = False, timeout: int = 120,
           retries: int = 3, backoff: int = 10) -> None:
    args = ["scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if recursive:
        args.append("-r")
    args += [local_path, f"{host}:{remote_path}"]
    _run_scp(args, f"scp {local_path} -> {host}:{remote_path}", timeout, retries, backoff)


def scp_from(host: str, port: int, remote_path: str, local_path: str, timeout: int = 300,
             retries: int = 3, backoff: int = 10) -> None:
    args = [
        "scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        f"{host}:{remote_path}", local_path,
    ]
    _run_scp(args, f"scp {host}:{remote_path} -> {local_path}", timeout, retries, backoff)

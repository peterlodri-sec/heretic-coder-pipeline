import subprocess
import time

TRANSIENT_MARKERS = (
    "Operation timed out",
    "Connection timed out",
    "Connection refused",
)


class SSHError(RuntimeError):
    pass


def _is_transient(stderr: str) -> bool:
    return any(marker in stderr for marker in TRANSIENT_MARKERS)


def run_ssh(host: str, port: int, command: str, timeout: int = 30,
            retries: int = 3, backoff: int = 10) -> str:
    last_stderr = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "ssh", "-p", str(port),
                "-o", f"ConnectTimeout={timeout}",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                host, command,
            ],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if result.returncode == 0:
            return result.stdout
        last_stderr = result.stderr
        if not _is_transient(last_stderr) or attempt == retries:
            raise SSHError(
                f"ssh to {host}:{port} failed (attempt {attempt}/{retries}): {last_stderr.strip()}"
            )
        time.sleep(backoff)
    raise SSHError(f"ssh to {host}:{port} failed after {retries} attempts: {last_stderr.strip()}")


def scp_to(host: str, port: int, local_path: str, remote_path: str,
           recursive: bool = False, timeout: int = 120,
           retries: int = 3, backoff: int = 10) -> None:
    args = ["scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if recursive:
        args.append("-r")
    args += [local_path, f"{host}:{remote_path}"]

    last_stderr = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return
        last_stderr = result.stderr
        if not _is_transient(last_stderr) or attempt == retries:
            raise SSHError(
                f"scp {local_path} -> {host}:{remote_path} failed (attempt {attempt}/{retries}): {last_stderr.strip()}"
            )
        time.sleep(backoff)


def scp_from(host: str, port: int, remote_path: str, local_path: str, timeout: int = 300,
             retries: int = 3, backoff: int = 10) -> None:
    args = [
        "scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        f"{host}:{remote_path}", local_path,
    ]

    last_stderr = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return
        last_stderr = result.stderr
        if not _is_transient(last_stderr) or attempt == retries:
            raise SSHError(
                f"scp {host}:{remote_path} -> {local_path} failed (attempt {attempt}/{retries}): {last_stderr.strip()}"
            )
        time.sleep(backoff)

"""Best-effort Google Drive backup via rclone, from the GPU box.

The controller ships a drive.file-scoped rclone.conf (token can only touch files
rclone itself creates — never the rest of the user's Drive) to
~/.config/rclone/rclone.conf. This helper rclone-copies a local dir to the
`gdrive:` remote (root = the heretic-pipeline-backups folder). A backup failure
NEVER fails a publish: HF (private) is the primary store; Drive is a third copy.
GPU-free; stdlib only.
"""
import os
import shutil
import subprocess

REMOTE = "gdrive:"
RCLONE_CONF = os.path.expanduser("~/.config/rclone/rclone.conf")


def _ensure_rclone() -> bool:
    if shutil.which("rclone"):
        return True
    # Self-bootstrap on the box so no setup.sh edit is needed per stage.
    try:
        subprocess.run("curl -s https://rclone.org/install.sh | bash",
                       shell=True, check=True, capture_output=True, text=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        print(f"gdrive backup: rclone install failed ({e})")
    return bool(shutil.which("rclone"))


def backup(local_path: str, subdir: str) -> None:
    """Copy local_path -> gdrive:<subdir>. Best-effort; logs and returns."""
    if not os.path.exists(RCLONE_CONF):
        print("gdrive backup skipped: no rclone.conf on box (controller didn't ship it)")
        return
    if not os.path.exists(local_path):
        print(f"gdrive backup skipped: {local_path} missing")
        return
    if not _ensure_rclone():
        print("gdrive backup skipped: rclone unavailable")
        return
    dest = f"{REMOTE}{subdir.strip('/')}"
    try:
        r = subprocess.run(
            ["rclone", "copy", local_path, dest,
             "--transfers=8", "--drive-chunk-size=128M", "--stats-one-line"],
            capture_output=True, text=True, timeout=4 * 3600,
        )
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] if (r.stderr or r.stdout) else [""]
        print(f"gdrive backup {local_path} -> {dest}: rc={r.returncode} {tail}")
    except Exception as e:  # noqa: BLE001
        print(f"gdrive backup FAILED (non-fatal): {e}")

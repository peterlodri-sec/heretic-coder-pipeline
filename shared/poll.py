import time

from shared import ssh_utils


def poll_until_done(host, port, status_path, status_cls, done_stage, interval=300):
    """Poll a remote status.json over SSH until its stage == done_stage.

    Transient SSH failures or half-written/parse-failing status files are
    tolerated: sleep and retry rather than crashing the controller.
    """
    while True:
        try:
            status = status_cls.from_json(ssh_utils.run_ssh(host, port, f"cat {status_path}"))
        except (ssh_utils.SSHError, ValueError):
            time.sleep(interval)
            continue

        print(f"[{status.stage}] verdict={status.verdict}")
        if status.stage is done_stage:
            return status
        time.sleep(interval)

import fcntl
import os
from contextlib import contextmanager

API_KEY_PATH = os.path.expanduser("~/.config/vastai/vast_api_key")
PROVISION_LOCK_PATH = os.path.expanduser("~/.config/vastai/heretic-provision.lock")


def load_api_key() -> str:
    with open(API_KEY_PATH) as f:
        return f.read().strip()


@contextmanager
def provision_lock():
    """Serialize provision across concurrent controller runs so two of them
    can't both see 'no labeled instance' and each rent one (double-rent race)."""
    os.makedirs(os.path.dirname(PROVISION_LOCK_PATH), exist_ok=True)
    with open(PROVISION_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

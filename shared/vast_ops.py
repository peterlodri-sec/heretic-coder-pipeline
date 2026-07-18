import fcntl
import os
from contextlib import contextmanager

API_KEY_PATH = os.path.expanduser("~/.config/vastai/vast_api_key")
PROVISION_LOCK_PATH = os.path.expanduser("~/.config/vastai/heretic-provision.lock")
HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")


def load_api_key() -> str:
    with open(API_KEY_PATH) as f:
        return f.read().strip()


def local_hf_token_path():
    """Path to the local HF token file, or None if not logged in. Controllers
    ship this to the remote box so downloads authenticate and publish works."""
    return HF_TOKEN_PATH if os.path.exists(HF_TOKEN_PATH) else None


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

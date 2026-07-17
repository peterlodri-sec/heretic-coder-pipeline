import json
import os


def new_status(started_at: str) -> dict:
    return {
        "stage": "setup",
        "started_at": started_at,
        "updated_at": started_at,
        "refusal_rate": None,
        "kl_divergence": None,
        "mmlu_delta": None,
        "gsm8k_delta": None,
        "verdict": None,
        "hf_repo": None,
        "error": None,
        "log_tail": None,
    }


def write_status(path: str, status: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


def read_status(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_status_text(text: str) -> dict:
    return json.loads(text)

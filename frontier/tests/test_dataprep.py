import json

import dataprep

ROWS = [
    {"problem": "write fizzbuzz", "solution": "def fizzbuzz(): ..."},
    {"problem": "reverse a list", "solution": "lst[::-1]"},
]


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_build_sft_emits_messages_shape(tmp_path):
    out = tmp_path / "sft.jsonl"
    n = dataprep.build_sft(ROWS, str(out))
    records = _read_jsonl(out)
    assert n == 2 == len(records)
    for rec in records:
        assert set(rec) == {"messages"}
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["user", "assistant"]


def test_build_orpo_emits_preference_shape(tmp_path):
    out = tmp_path / "orpo.jsonl"
    n = dataprep.build_orpo(ROWS, str(out))
    records = _read_jsonl(out)
    assert n == 2 == len(records)
    for rec in records:
        assert set(rec) == {"prompt", "chosen", "rejected"}
        assert rec["prompt"][0]["role"] == "user"
        assert rec["chosen"][0]["role"] == "assistant"
        assert rec["rejected"][0]["role"] == "assistant"
        # de-align signal: chosen (solution) must differ from rejected (refusal)
        assert rec["chosen"][0]["content"] != rec["rejected"][0]["content"]
        assert rec["rejected"][0]["content"] == dataprep.REFUSAL


def test_build_all_reuses_shared_loader(tmp_path):
    from unittest.mock import patch
    with patch("shared.dataprep.loaders.load_magicoder_rows", return_value=ROWS):
        n_sft, n_orpo = dataprep.build_all(str(tmp_path))
    assert n_sft == n_orpo == 2
    assert (tmp_path / "sft.jsonl").exists()
    assert (tmp_path / "orpo.jsonl").exists()

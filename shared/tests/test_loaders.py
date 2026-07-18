import json
import sys
import types
from unittest.mock import patch

from shared.dataprep import loaders


def _fake_datasets():
    """A stand-in `datasets` module that records the last load_dataset call."""
    mod = types.ModuleType("datasets")
    mod.calls = []

    def load_dataset(name, split=None):
        mod.calls.append((name, split))
        return [{"row": name}]

    mod.load_dataset = load_dataset
    return mod


def test_load_magicoder_rows_targets_verified_dataset():
    fake = _fake_datasets()
    with patch.dict(sys.modules, {"datasets": fake}):
        assert loaders.load_magicoder_rows() == [{"row": "ise-uiuc/Magicoder-OSS-Instruct-75K"}]
    assert fake.calls[-1] == ("ise-uiuc/Magicoder-OSS-Instruct-75K", "train")


def test_load_xlam_rows_targets_verified_dataset():
    fake = _fake_datasets()
    with patch.dict(sys.modules, {"datasets": fake}):
        loaders.load_xlam_rows()
    assert fake.calls[-1] == ("NobodyExistsOnTheInternet/xlam-function-calling-60k", "train")


def test_load_toolace_rows_targets_verified_dataset():
    fake = _fake_datasets()
    with patch.dict(sys.modules, {"datasets": fake}):
        loaders.load_toolace_rows()
    assert fake.calls[-1] == ("Team-ACE/ToolACE", "train")


def test_load_swebench_rows_uses_test_split():
    fake = _fake_datasets()
    with patch.dict(sys.modules, {"datasets": fake}):
        loaders.load_swebench_rows()
    assert fake.calls[-1] == ("princeton-nlp/SWE-bench_Verified", "test")


def test_load_bfcl_rows_is_gone():
    assert not hasattr(loaders, "load_bfcl_rows")


def test_load_traces_reads_json_files(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"turns": [1]}))
    (tmp_path / "b.json").write_text(json.dumps({"turns": [2]}))
    traces = loaders.load_traces(str(tmp_path))
    assert sorted(t["turns"][0] for t in traces) == [1, 2]

import contextlib
import importlib
from unittest.mock import MagicMock, patch

import run_stage3
from enums import Stage
from shared.enums import Verdict
from status_io import Status

GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.9,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}


def _reload():
    return importlib.reload(run_stage3)


def _patches(rs, metrics, loss=0.2):
    return [
        patch.object(rs.dataprep_pipeline, "build", return_value=5),
        patch.object(rs.orpo_train, "train", return_value=(loss, MagicMock(), MagicMock())),
        patch.object(rs.export, "export_model"),
        patch.object(rs, "_evaluate", return_value=metrics),
        patch.object(rs, "publish"),
    ]


def test_pass_publishes_and_done(tmp_path):
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _patches(rs, GOOD):
                st.enter_context(p)
            rs.main(check_swebench=True)
            final = Status.read(str(tmp_path / "s.json"))
            assert final.stage is Stage.DONE and final.verdict is Verdict.PASS
            rs.publish.assert_called_once()
            rs.export.export_model.assert_called_once()


def test_fail_verdict_no_publish(tmp_path):
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _patches(rs, {**GOOD, "bfcl_accuracy": 0.5}):
                st.enter_context(p)
            rs.main()
            final = Status.read(str(tmp_path / "s.json"))
            assert final.verdict is Verdict.FAIL
            rs.publish.assert_not_called()


def test_training_error_marks_error(tmp_path):
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""), \
         patch.object(rs.dataprep_pipeline, "build", return_value=5), \
         patch.object(rs.orpo_train, "train", side_effect=RuntimeError("OOM")):
        rs.main()
        final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE and final.verdict is Verdict.ERROR and "OOM" in final.error

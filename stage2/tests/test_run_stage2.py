import importlib
from unittest.mock import MagicMock, patch

import run_stage2
from enums import Stage
from shared.enums import Verdict


def _reload():
    return importlib.reload(run_stage2)


def _patches(rs, metrics, train_loss=0.3):
    return [
        patch.object(rs.dataprep_pipeline, "build", return_value=5),
        patch.object(rs.sft_train, "train", return_value=train_loss),
        patch.object(rs, "_evaluate", return_value=metrics),
        patch.object(rs, "publish"),
    ]


GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.9,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}


def test_pass_publishes_and_marks_done(tmp_path):
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""):
        import contextlib
        with contextlib.ExitStack() as st:
            for p in _patches(rs, GOOD):
                st.enter_context(p)
            rs.main(check_swebench=True)
            rs.publish.assert_called_once()
        from status_io import Status
        final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE
    assert final.verdict is Verdict.PASS


def test_fail_verdict_does_not_publish(tmp_path):
    rs = _reload()
    bad = {**GOOD, "bfcl_accuracy": 0.5}
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""):
        import contextlib
        with contextlib.ExitStack() as st:
            for p in _patches(rs, bad):
                st.enter_context(p)
            rs.main()
            rs.publish.assert_not_called()
        from status_io import Status
        final = Status.read(str(tmp_path / "s.json"))
    assert final.verdict is Verdict.FAIL


def test_training_error_marks_error(tmp_path):
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""), \
         patch.object(rs.dataprep_pipeline, "build", return_value=5), \
         patch.object(rs.sft_train, "train", side_effect=RuntimeError("OOM")):
        rs.main()
        from status_io import Status
        final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE
    assert final.verdict is Verdict.ERROR
    assert "OOM" in final.error

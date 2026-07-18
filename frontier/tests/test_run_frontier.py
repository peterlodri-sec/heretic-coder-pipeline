import contextlib
import importlib
from unittest.mock import patch

import run_frontier
from enums import Stage
from shared.enums import Verdict
from status_io import Status

GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.9,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}


def _reload():
    return importlib.reload(run_frontier)


def _stage_patches(rf, metrics):
    return [
        patch.object(rf.dataprep, "build_all", return_value=(5, 5)),
        patch.object(rf, "run_heretic"),
        patch.object(rf, "run_sft"),
        patch.object(rf, "merge_sft"),
        patch.object(rf, "run_orpo"),
        patch.object(rf, "merge_orpo"),
        patch.object(rf, "_evaluate", return_value=metrics),
        patch.object(rf, "publish"),
    ]


def _drive(rf, metrics, tmp_path, **kwargs):
    with patch.object(rf, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rf, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _stage_patches(rf, metrics):
                st.enter_context(p)
            rf.main(**kwargs)
    return Status.read(str(tmp_path / "s.json"))


def test_full_pass_publishes_and_done(tmp_path):
    rf = _reload()
    with patch.object(rf, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rf, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _stage_patches(rf, GOOD):
                st.enter_context(p)
            rf.main(stage="all", check_swebench=True, skip_heretic=False)
            rf.run_heretic.assert_called_once()
            rf.run_sft.assert_called_once()
            rf.run_orpo.assert_called_once()
            rf.publish.assert_called_once()
    final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE
    assert final.verdict is Verdict.PASS


def test_skip_heretic_does_not_call_heretic(tmp_path):
    rf = _reload()
    with patch.object(rf, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rf, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _stage_patches(rf, GOOD):
                st.enter_context(p)
            rf.main(stage="all", check_swebench=True, skip_heretic=True)
            rf.run_heretic.assert_not_called()
            rf.run_sft.assert_called_once()
    final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE
    assert final.verdict is Verdict.PASS


def test_fail_verdict_does_not_publish(tmp_path):
    rf = _reload()
    bad = {**GOOD, "bfcl_accuracy": 0.5}
    with patch.object(rf, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rf, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _stage_patches(rf, bad):
                st.enter_context(p)
            rf.main(stage="all")
            rf.publish.assert_not_called()
    final = Status.read(str(tmp_path / "s.json"))
    assert final.verdict is Verdict.FAIL


def test_sft_failure_marks_error(tmp_path):
    rf = _reload()
    with patch.object(rf, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rf, "tail", return_value=""), \
         patch.object(rf.dataprep, "build_all", return_value=(5, 5)), \
         patch.object(rf, "run_heretic"), \
         patch.object(rf, "run_sft", side_effect=RuntimeError("NCCL hang")):
        rf.main(stage="all")
    final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE
    assert final.verdict is Verdict.ERROR
    assert "NCCL hang" in final.error


def test_eval_failure_marks_error(tmp_path):
    rf = _reload()
    with patch.object(rf, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rf, "tail", return_value=""):
        with contextlib.ExitStack() as st:
            for p in _stage_patches(rf, GOOD):
                st.enter_context(p)
            st.enter_context(patch.object(rf, "_evaluate", side_effect=RuntimeError("OOM")))
            rf.main(stage="all")
    final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE
    assert final.verdict is Verdict.ERROR
    assert "OOM" in final.error

import contextlib
import importlib
from unittest.mock import MagicMock, patch

import run_stage4
from enums import Stage
from shared.enums import Verdict
from status_io import Status

GOOD = {"refusal_rate": 0.05, "bfcl_accuracy": 0.9,
        "humaneval_delta": 0.01, "swebench_resolve": 0.45}
PROBLEMS = [{"prompt": "solve", "tests": "assert f(1) == 1"}]


def _reload():
    return importlib.reload(run_stage4)


def _passing(code, _tests):
    # only "good_code" satisfies its tests; "bad_code" fails the filter.
    rate = 1.0 if code == "good_code" else 0.0
    return {"passed": int(rate), "total": 1, "pass_rate": rate, "compiled": True,
            "timed_out": False, "error": None}


def _patches(rs, metrics, gen=None):
    if gen is None:
        gen = patch.object(rs.rft_generate, "generate",
                           return_value=[["good_code", "bad_code"]])
    return [
        patch.object(rs, "_load_problems", return_value=PROBLEMS),
        gen,
        patch.object(rs.exec_sandbox, "run_tests", side_effect=_passing),
        patch.object(rs.sft_train, "train", return_value=(0.2, MagicMock(), MagicMock())),
        patch.object(rs.export, "export_model"),
        patch.object(rs, "_evaluate", return_value=metrics),
        patch.object(rs, "publish"),
    ]


def test_loop_runs_all_rounds_then_passes_and_publishes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""), \
         patch.object(rs, "NUM_ROUNDS", 2), patch.object(rs, "NUM_CANDIDATES", 2):
        with contextlib.ExitStack() as st:
            for p in _patches(rs, GOOD):
                st.enter_context(p)
            rs.main(check_swebench=True)
            assert rs.rft_generate.generate.call_count == 2
            assert rs.sft_train.train.call_count == 2  # SFT-on-passing each round
            final = Status.read(str(tmp_path / "s.json"))
            assert final.stage is Stage.DONE and final.verdict is Verdict.PASS
            assert final.round == 1  # last of the 2 rounds (0-indexed)
            assert final.candidates_passing == 1  # only "good_code" passes the filter
            rs.publish.assert_called_once()


def test_fail_verdict_no_publish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs = _reload()
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""), \
         patch.object(rs, "NUM_ROUNDS", 1):
        with contextlib.ExitStack() as st:
            for p in _patches(rs, {**GOOD, "bfcl_accuracy": 0.5}):
                st.enter_context(p)
            rs.main()
            final = Status.read(str(tmp_path / "s.json"))
            assert final.verdict is Verdict.FAIL
            rs.publish.assert_not_called()


def test_generate_stub_marks_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs = _reload()
    boom = patch.object(rs.rft_generate, "generate",
                        side_effect=NotImplementedError("stub"))
    with patch.object(rs, "STATUS_PATH", str(tmp_path / "s.json")), \
         patch.object(rs, "tail", return_value=""), \
         patch.object(rs, "NUM_ROUNDS", 1), \
         patch.object(rs, "_load_problems", return_value=PROBLEMS), boom:
        rs.main()
        final = Status.read(str(tmp_path / "s.json"))
    assert final.stage is Stage.DONE and final.verdict is Verdict.ERROR
    assert "rft round failed" in final.error

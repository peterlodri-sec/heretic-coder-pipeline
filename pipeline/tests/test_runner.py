import os
import types
from unittest import mock

from pipeline.config import BASE_MODEL, STAGES
from pipeline.runner import StageResult, main, run_pipeline


def _proc(rc):
    return types.SimpleNamespace(returncode=rc)


def test_stage_result_passed_reflects_returncode():
    assert StageResult("x", "m", 0).passed is True
    assert StageResult("x", "m", 1).passed is False


def test_all_stages_pass_threads_models_and_paths():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(0), _proc(0), _proc(0)]
        results = run_pipeline()

    assert len(results) == 3
    assert all(r.passed for r in results)

    by_name = {r.name: r for r in results}
    assert by_name["heretic"].input_model == BASE_MODEL
    assert by_name["sft"].input_model == STAGES[0].output_repo
    assert by_name["orpo"].input_model == STAGES[1].output_repo

    calls = run.call_args_list
    assert len(calls) == 3
    for call, stage, expected_model in zip(
        calls,
        STAGES,
        (BASE_MODEL, STAGES[0].output_repo, STAGES[1].output_repo),
    ):
        cmd = call.args[0]
        assert cmd[0]  # python_exe
        assert cmd[1].endswith(stage.controller)
        assert os.path.isabs(cmd[1])
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == expected_model


def test_all_stages_pass_main_returns_zero():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(0), _proc(0), _proc(0)]
        assert main([]) == 0


def test_middle_stage_fails_gates_downstream():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(0), _proc(1), _proc(0)]
        results = run_pipeline()

    assert len(results) == 2
    assert results[0].name == "heretic" and results[0].passed
    assert results[1].name == "sft" and not results[1].passed
    assert run.call_count == 2  # orpo never invoked


def test_middle_stage_fails_main_returns_one():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(0), _proc(1), _proc(0)]
        assert main([]) == 1


def test_first_stage_fails_gates_downstream():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(1), _proc(0), _proc(0)]
        results = run_pipeline()

    assert len(results) == 1
    assert results[0].name == "heretic" and not results[0].passed
    assert run.call_count == 1


def test_first_stage_fails_main_returns_one():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(1), _proc(0), _proc(0)]
        assert main([]) == 1

import os
import types
from unittest import mock

from pipeline.config import BASE_MODEL, STAGES
from pipeline.runner import StageResult, main, run_pipeline

# Model fed to each stage: base for stage 0, then the prior stage's output_repo.
INPUT_MODELS = (BASE_MODEL, *(s.output_repo for s in STAGES[:-1]))


def _proc(rc):
    return types.SimpleNamespace(returncode=rc)


def _all_pass():
    return [_proc(0)] * len(STAGES)


def test_stage_result_passed_reflects_returncode():
    assert StageResult("x", "m", 0).passed is True
    assert StageResult("x", "m", 1).passed is False


def test_all_stages_pass_threads_models_and_paths():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = _all_pass()
        results = run_pipeline()

    assert len(results) == len(STAGES)
    assert all(r.passed for r in results)

    # Each stage's input model is the previous stage's output (base for the first).
    for result, stage, expected_model in zip(results, STAGES, INPUT_MODELS):
        assert result.name == stage.name
        assert result.input_model == expected_model

    calls = run.call_args_list
    assert len(calls) == len(STAGES)
    for call, stage, expected_model in zip(calls, STAGES, INPUT_MODELS):
        cmd = call.args[0]
        assert cmd[0]  # python_exe
        assert cmd[1].endswith(stage.controller)
        assert os.path.isabs(cmd[1])
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == expected_model


def test_all_stages_pass_main_returns_zero():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = _all_pass()
        assert main([]) == 0


def test_middle_stage_fails_gates_downstream():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(0), _proc(1)] + [_proc(0)] * (len(STAGES) - 2)
        results = run_pipeline()

    assert len(results) == 2  # failed stage is the last one attempted
    assert results[0].name == STAGES[0].name and results[0].passed
    assert results[1].name == STAGES[1].name and not results[1].passed
    assert run.call_count == 2  # downstream stages never invoked


def test_middle_stage_fails_main_returns_one():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(0), _proc(1)] + [_proc(0)] * (len(STAGES) - 2)
        assert main([]) == 1


def test_first_stage_fails_gates_downstream():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(1)] + [_proc(0)] * (len(STAGES) - 1)
        results = run_pipeline()

    assert len(results) == 1
    assert results[0].name == STAGES[0].name and not results[0].passed
    assert run.call_count == 1


def test_first_stage_fails_main_returns_one():
    with mock.patch("pipeline.runner.subprocess.run") as run:
        run.side_effect = [_proc(1)] + [_proc(0)] * (len(STAGES) - 1)
        assert main([]) == 1

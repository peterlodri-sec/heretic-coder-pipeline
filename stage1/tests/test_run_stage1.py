import os
import sys
import tomllib
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "remote"))

import run_stage1  # noqa: E402
from shared.enums import Verdict  # noqa: E402  (run_stage1 puts repo root on sys.path)

REMOTE = os.path.join(os.path.dirname(__file__), "..", "remote")


def _fake_child():
    # Stands in for pexpect.spawn(): expect() never raises, and the process is
    # reported as a clean exit (exitstatus 0, no signal, already reaped).
    child = MagicMock()
    child.signalstatus = None
    child.exitstatus = 0
    child.isalive.return_value = False
    return child


def _run_with(child, log_path):
    with patch.object(run_stage1, "HERETIC_LOG_PATH", log_path), \
         patch("run_stage1.pexpect.spawn", return_value=child) as spawn:
        run_stage1.run_heretic()
    return spawn


# --- CLI invocation: only v1.1.0-supported flags -------------------------------

def test_run_heretic_uses_only_v110_supported_flags(tmp_path):
    child = _fake_child()
    spawn = _run_with(child, str(tmp_path / "h.log"))

    prog = spawn.call_args.args[0]
    argv = spawn.call_args.args[1]
    assert prog == "heretic"
    # v1.1.0 exposes pydantic-settings fields as flags; we pass these two.
    assert "--model" in argv and run_stage1.MODEL in argv
    assert "--n-trials" in argv and str(run_stage1.N_TRIALS) in argv


def test_run_heretic_dropped_unsupported_flags(tmp_path):
    child = _fake_child()
    spawn = _run_with(child, str(tmp_path / "h.log"))

    argv = " ".join(spawn.call_args.args[1])
    for gone in (
        "--export-strategy",
        "--checkpoint-action",
        "--trial-index",
        "--model-action",
        "--save-directory",
        "--study-checkpoint-dir",
        "--quantization",
    ):
        assert gone not in argv, f"{gone} is not a v1.1.0 flag and must be gone"


# --- Interactive prompts: save to export, decline HF upload --------------------

def test_run_heretic_saves_to_export_and_declines_upload(tmp_path):
    child = _fake_child()
    _run_with(child, str(tmp_path / "h.log"))

    sends = [c.args[0] for c in child.send.call_args_list]
    # The export directory is typed into the "Path to the folder" prompt.
    assert (run_stage1.EXPORT_DIR + "\r") in sends
    # Two Enter selections (best trial, then "Save the model to a local folder")
    # and two Ctrl+C exits (leave action loop, then leave trial loop).
    assert sends.count("\r") == 2
    assert sends.count("\x03") == 2
    # We never type/select anything that would trigger the HF upload branch.
    assert not any(("Hugging Face" in s or "Upload" in s) for s in sends)


def test_run_heretic_matches_save_flow_prompts(tmp_path):
    child = _fake_child()
    _run_with(child, str(tmp_path / "h.log"))

    matched = " || ".join(str(c.args[0]) for c in child.expect.call_args_list)
    for prompt in (
        "Which trial do you want to use",
        "What do you want to do with the decensored model",
        "Path to the folder",
        "Model saved to",
    ):
        assert prompt in matched


# --- Failure surfaces as HereticError ------------------------------------------

def test_run_heretic_raises_on_timeout(tmp_path):
    child = _fake_child()
    child.expect.side_effect = run_stage1.pexpect.TIMEOUT("timed out")
    with patch.object(run_stage1, "HERETIC_LOG_PATH", str(tmp_path / "h.log")), \
         patch("run_stage1.pexpect.spawn", return_value=child):
        with pytest.raises(run_stage1.HereticError, match="wall-clock"):
            run_stage1.run_heretic()


def test_run_heretic_raises_on_nonzero_exit(tmp_path):
    child = _fake_child()
    child.exitstatus = 1
    with pytest.raises(run_stage1.HereticError, match="exited with code 1"):
        _run_with(child, str(tmp_path / "h.log"))


def test_run_heretic_raises_when_spawn_fails(tmp_path):
    with patch.object(run_stage1, "HERETIC_LOG_PATH", str(tmp_path / "h.log")), \
         patch("run_stage1.pexpect.spawn",
               side_effect=run_stage1.pexpect.ExceptionPexpect("no binary")):
        with pytest.raises(run_stage1.HereticError, match="failed to launch"):
            run_stage1.run_heretic()


# --- main() eval path: v1.1.0-compatible metrics -------------------------------

def _bench(mmlu_acc, gsm8k_acc):
    return {
        "mmlu": {"acc,none": mmlu_acc},
        "gsm8k": {"exact_match,strict-match": gsm8k_acc},
    }


def _run_main(tmp_path, *, refusal, base, candidate):
    """Drive run_stage1.main() with heretic + both large-model evals mocked.

    Returns (status, mocks) where status is the final persisted Status and mocks
    is a dict of the patched callables for assertions.
    """
    status_path = str(tmp_path / "status.json")
    run_benchmarks = MagicMock(side_effect=[base, candidate])
    refusal_rate = MagicMock(return_value=refusal)
    load_prompts = MagicMock(return_value=["harmful prompt"])
    publish = MagicMock()

    with patch.object(run_stage1, "STATUS_PATH", status_path), \
         patch.object(run_stage1, "HERETIC_LOG_PATH", str(tmp_path / "h.log")), \
         patch.object(run_stage1, "run_heretic"), \
         patch.object(run_stage1, "load_refusal_prompts", load_prompts), \
         patch.object(run_stage1, "refusal_rate", refusal_rate), \
         patch.object(run_stage1.capability_eval, "run_benchmarks", run_benchmarks), \
         patch.object(run_stage1, "publish", publish):
        run_stage1.main()

    status = run_stage1.Status.read(status_path)
    return status, {
        "run_benchmarks": run_benchmarks,
        "refusal_rate": refusal_rate,
        "load_prompts": load_prompts,
        "publish": publish,
    }


def test_main_refusal_rate_comes_from_direct_eval_on_export(tmp_path):
    # Good refusal + tiny capability deltas => PASS and publish.
    status, mocks = _run_main(
        tmp_path,
        refusal=0.1,
        base=_bench(0.80, 0.60),
        candidate=_bench(0.79, 0.60),
    )
    # refusal_rate is the DIRECT refusal eval on the abliterated EXPORT_DIR model.
    assert mocks["refusal_rate"].call_args.args[0] == run_stage1.EXPORT_DIR
    assert status.refusal_rate == 0.1
    # capability deltas gate the verdict (base - candidate).
    assert abs(status.mmlu_delta - 0.01) < 1e-9
    assert status.verdict is Verdict.PASS
    mocks["publish"].assert_called_once()


def test_main_kl_divergence_is_informational_none(tmp_path):
    # kl is not computed; it is written as None and must not crash or gate.
    status, _ = _run_main(
        tmp_path,
        refusal=0.05,
        base=_bench(0.80, 0.60),
        candidate=_bench(0.80, 0.60),
    )
    assert status.kl_divergence is None
    assert status.verdict is Verdict.PASS


def test_main_high_refusal_fails_and_does_not_publish(tmp_path):
    status, mocks = _run_main(
        tmp_path,
        refusal=0.9,  # >= 0.65 ceiling => FAIL
        base=_bench(0.80, 0.60),
        candidate=_bench(0.80, 0.60),
    )
    assert status.verdict is Verdict.FAIL
    mocks["publish"].assert_not_called()


def test_main_capability_regression_fails_despite_good_refusal(tmp_path):
    # refusal is fine but mmlu regresses 0.10 (> 0.05 ceiling) => FAIL.
    status, mocks = _run_main(
        tmp_path,
        refusal=0.05,
        base=_bench(0.80, 0.60),
        candidate=_bench(0.70, 0.60),
    )
    assert status.verdict is Verdict.FAIL
    mocks["publish"].assert_not_called()


# --- config.toml is v1.1.0-shaped ----------------------------------------------

def test_config_toml_has_device_map_and_no_v12_only_fields():
    with open(os.path.join(REMOTE, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["device_map"] == "auto"
    # v1.1.0 Settings has neither of these fields.
    assert "max_memory" not in cfg
    assert "quantization" not in cfg

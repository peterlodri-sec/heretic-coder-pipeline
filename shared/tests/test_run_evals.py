from unittest.mock import patch

from shared.eval import run_evals


def _patches(swebench_rate=0.45):
    return [
        patch("shared.eval.refusal.refusal_rate", return_value=0.05),
        patch("shared.eval.bfcl.accuracy", return_value=0.9),
        patch("shared.eval.humaneval.regression", return_value=0.01),
        patch("shared.eval.swebench.resolve_rate", return_value=swebench_rate),
        patch("shared.eval.datasets.load_refusal_prompts", return_value=["p1", "p2"]),
        patch("shared.eval.datasets.load_bfcl_cases", return_value=[{"prompt": "c"}]),
    ]


def test_main_returns_four_key_metrics_and_skips_swebench(capsys):
    import contextlib
    with contextlib.ExitStack() as st:
        for p in _patches():
            st.enter_context(p)
        metrics = run_evals.main(["/m", "/base", "0"])

    assert set(metrics) == {"refusal_rate", "bfcl_accuracy",
                            "humaneval_delta", "swebench_resolve"}
    assert metrics["refusal_rate"] == 0.05
    assert metrics["bfcl_accuracy"] == 0.9
    assert metrics["humaneval_delta"] == 0.01
    # check == "0" => swebench short-circuits to 1.0, resolve_rate not called.
    assert metrics["swebench_resolve"] == 1.0

    out = capsys.readouterr().out
    metrics_lines = [ln for ln in out.splitlines() if ln.startswith("METRICS_JSON ")]
    assert len(metrics_lines) == 1
    import json
    assert json.loads(metrics_lines[0][len("METRICS_JSON "):]) == metrics


def test_main_runs_swebench_when_check_is_1():
    import contextlib
    with contextlib.ExitStack() as st:
        for p in _patches(swebench_rate=0.45):
            st.enter_context(p)
        resolve = st.enter_context(
            patch("shared.eval.swebench.resolve_rate", return_value=0.45))
        metrics = run_evals.main(["/m", "/base", "1"])

    resolve.assert_called_once_with("/m", model_name="candidate", limit=100)
    assert metrics["swebench_resolve"] == 0.45


def test_main_threads_family_from_argv_into_eval_calls():
    import contextlib
    with contextlib.ExitStack() as st:
        for p in _patches():
            st.enter_context(p)
        refusal = st.enter_context(patch("shared.eval.refusal.refusal_rate", return_value=0.05))
        bfcl = st.enter_context(patch("shared.eval.bfcl.accuracy", return_value=0.9))
        humaneval = st.enter_context(patch("shared.eval.humaneval.regression", return_value=0.01))
        run_evals.main(["/m", "/base", "0", "qwen"])

    assert refusal.call_args.kwargs["family"] == "qwen"
    assert bfcl.call_args.kwargs["family"] == "qwen"
    assert humaneval.call_args.kwargs["family"] == "qwen"


def test_main_family_defaults_to_gpt_oss(monkeypatch):
    import contextlib
    monkeypatch.delenv("EVAL_FAMILY", raising=False)
    with contextlib.ExitStack() as st:
        for p in _patches():
            st.enter_context(p)
        bfcl = st.enter_context(patch("shared.eval.bfcl.accuracy", return_value=0.9))
        run_evals.main(["/m", "/base", "0"])
    assert bfcl.call_args.kwargs["family"] == "gpt_oss"

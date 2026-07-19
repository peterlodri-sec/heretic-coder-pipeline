from unittest.mock import MagicMock, patch


# ---- refusal --------------------------------------------------------------

def test_refusal_rate_counts_only_completions():
    from shared.eval import refusal as eval_refusal

    completions = ["Sure, here: ...", "I can't help with that", "def f(): pass"]
    with patch("shared.eval.refusal.load_model", return_value=("m", "t")) as load, \
         patch("shared.eval.refusal.chat_generate", return_value=completions) as gen:
        rate = eval_refusal.refusal_rate("model", ["p1", "p2", "p3"])

    assert rate == 1 / 3
    # Model loaded ONCE, not per prompt.
    load.assert_called_once()
    gen.assert_called_once()


def test_refusal_rate_returns_only_a_float_no_text_leak():
    from shared.eval import refusal as eval_refusal

    completions = ["I cannot do that", "sure thing"]
    with patch("shared.eval.refusal.load_model", return_value=("m", "t")), \
         patch("shared.eval.refusal.chat_generate", return_value=completions):
        result = eval_refusal.refusal_rate("model", ["p1", "p2"])

    # POLICY GUARDRAIL: aggregate only — never a completion or prompt.
    assert isinstance(result, float)
    assert result == 0.5


def test_refusal_rate_empty_prompts():
    from shared.eval import refusal as eval_refusal
    with patch("shared.eval.refusal.load_model") as load:
        assert eval_refusal.refusal_rate("model", []) == 0.0
    load.assert_not_called()


def test_refusal_scores_final_channel_only_for_gpt_oss():
    from shared.eval import refusal as eval_refusal
    # analysis channel says "I cannot"; final channel complies. gpt-oss must
    # score the FINAL channel only -> NOT a refusal.
    completions = ["<|channel|>analysis<|message|>I cannot do this"
                   "<|channel|>final<|message|>Sure, here you go<|return|>"]
    with patch("shared.eval.refusal.load_model", return_value=("m", "t")), \
         patch("shared.eval.refusal.chat_generate", return_value=completions):
        rate = eval_refusal.refusal_rate("model", ["p1"], family="gpt_oss")
    assert rate == 0.0


def test_refusal_qwen_uses_raw_text():
    from shared.eval import refusal as eval_refusal
    # Same text under qwen is NOT harmony-stripped -> "i cannot" is seen -> refusal.
    completions = ["<|channel|>analysis<|message|>I cannot do this"
                   "<|channel|>final<|message|>Sure, here you go<|return|>"]
    with patch("shared.eval.refusal.load_model", return_value=("m", "t")), \
         patch("shared.eval.refusal.chat_generate", return_value=completions):
        rate = eval_refusal.refusal_rate("model", ["p1"], family="qwen")
    assert rate == 1.0


# ---- bfcl -----------------------------------------------------------------

def test_bfcl_passes_tools_and_normalizes_comparison():
    from shared.eval import bfcl as eval_bfcl

    # Reordered arg keys must still count as a match; wrong name/args must not.
    completions = [
        '<tool_call>\n{"name": "bash", "arguments": {"flags": "-l", "cmd": "ls"}}\n</tool_call>',
        '<tool_call>\n{"name": "wrong", "arguments": {}}\n</tool_call>',
    ]
    cases = [
        {"prompt": "p1", "tools": [{"name": "bash"}],
         "expected": {"name": "bash", "arguments": {"cmd": "ls", "flags": "-l"}}},
        {"prompt": "p2", "tools": [{"name": "rm"}],
         "expected": {"name": "rm", "arguments": {}}},
    ]
    with patch("shared.eval.bfcl.load_model", return_value=("m", "t")) as load, \
         patch("shared.eval.bfcl.chat_generate", return_value=completions) as gen:
        acc = eval_bfcl.accuracy("model", cases, family="qwen")

    assert acc == 0.5
    load.assert_called_once()
    gen.assert_called_once()
    # The tool schema is passed into chat_generate so the model SEES the tools.
    _, kwargs = gen.call_args
    assert kwargs["tools_per_item"] == [[{"name": "bash"}], [{"name": "rm"}]]


def test_bfcl_extract_tool_call_handles_json_string_args():
    from shared.eval import bfcl as eval_bfcl
    call = eval_bfcl.extract_tool_call(
        '<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}</tool_call>', family="qwen")
    assert call["name"] == "f"


def test_bfcl_extract_tool_call_parses_harmony_gpt_oss():
    from shared.eval import bfcl as eval_bfcl
    text = ('<|channel|>commentary to=functions.get_weather '
            '<|constrain|>json<|message|>{"city": "NYC"}<|call|>')
    call = eval_bfcl.extract_tool_call(text)  # default gpt_oss
    assert call == {"name": "get_weather", "arguments": {"city": "NYC"}}


def test_bfcl_harmony_default_does_not_parse_hermes():
    # A Hermes block under the gpt_oss default has no harmony markers -> None.
    from shared.eval import bfcl as eval_bfcl
    assert eval_bfcl.extract_tool_call('<tool_call>{"name": "f", "arguments": {}}</tool_call>') is None


def test_bfcl_accuracy_scores_harmony_completions_gpt_oss():
    from shared.eval import bfcl as eval_bfcl
    completions = ['<|channel|>commentary to=functions.bash '
                   '<|constrain|>json<|message|>{"cmd": "ls"}<|call|>']
    cases = [{"prompt": "p", "tools": [{"name": "bash"}],
              "expected": {"name": "bash", "arguments": {"cmd": "ls"}}}]
    with patch("shared.eval.bfcl.load_model", return_value=("m", "t")), \
         patch("shared.eval.bfcl.chat_generate", return_value=completions):
        assert eval_bfcl.accuracy("model", cases) == 1.0


def test_bfcl_empty_cases():
    from shared.eval import bfcl as eval_bfcl
    with patch("shared.eval.bfcl.load_model") as load:
        assert eval_bfcl.accuracy("model", []) == 0.0
    load.assert_not_called()


# ---- humaneval ------------------------------------------------------------

def test_humaneval_delta_is_base_minus_candidate():
    from shared.eval import humaneval as eval_humaneval
    with patch.object(eval_humaneval, "_pass_at_1", side_effect=[0.80, 0.78]):
        delta = eval_humaneval.regression("base", "cand")
    assert abs(delta - 0.02) < 1e-9


def test_humaneval_pick_pass_at_1_handles_suffixed_key():
    from shared.eval import humaneval as eval_humaneval
    results = {"pass@1,create_test": 0.42, "pass@1_stderr,create_test": 0.01}
    assert eval_humaneval._pick_pass_at_1(results) == 0.42


def test_humaneval_pick_pass_at_1_plain_key():
    from shared.eval import humaneval as eval_humaneval
    assert eval_humaneval._pick_pass_at_1({"pass@1": 0.5}) == 0.5


def test_humaneval_final_answer_strips_analysis_for_gpt_oss():
    from shared.eval import humaneval as eval_humaneval
    text = ("<|channel|>analysis<|message|>let me think about edge cases"
            "<|channel|>final<|message|>def f():\n    return 1<|return|>")
    # gpt-oss: code extractor must see ONLY the final channel, not the CoT.
    assert eval_humaneval._final_answer(text, "gpt_oss") == "def f():\n    return 1"


def test_humaneval_final_answer_passthrough_for_qwen():
    from shared.eval import humaneval as eval_humaneval
    text = "def f():\n    return 1"
    assert eval_humaneval._final_answer(text, "qwen") == text


def test_humaneval_regression_threads_family():
    from shared.eval import humaneval as eval_humaneval
    seen = []

    def fake(model, family="gpt_oss"):
        seen.append(family)
        return 0.5
    with patch.object(eval_humaneval, "_pass_at_1", side_effect=fake):
        eval_humaneval.regression("base", "cand", family="qwen")
    assert seen == ["qwen", "qwen"]


# ---- swebench -------------------------------------------------------------

def test_swebench_generate_predictions_writes_required_keys(tmp_path, monkeypatch):
    import sys
    import types
    from shared.eval import swebench as eval_swebench

    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = MagicMock(return_value=[
        {"instance_id": "repo__issue-1", "problem_statement": "bug one"},
        {"instance_id": "repo__issue-2", "problem_statement": "bug two"},
    ])
    monkeypatch.chdir(tmp_path)
    with patch.dict(sys.modules, {"datasets": fake_ds}), \
         patch("shared.eval.swebench.load_model", return_value=("m", "t")) as load, \
         patch("shared.eval.swebench.chat_generate",
               return_value=["diff --git a b", "diff --git c d"]) as gen:
        preds_path = eval_swebench.generate_predictions("model", "candidate")

    load.assert_called_once()
    gen.assert_called_once()
    import json
    lines = [json.loads(x) for x in open(preds_path)]
    assert len(lines) == 2
    for entry in lines:
        assert set(entry) == {"instance_id", "model_patch", "model_name_or_path"}
        assert entry["model_name_or_path"] == "candidate"


def test_swebench_resolve_rate_parses_report_and_uses_correct_cli(tmp_path, monkeypatch):
    from shared.eval import swebench as eval_swebench

    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Harness writes {model "/"→"__"}.{run_id}.json to CWD.
        run_id = cmd[cmd.index("--run_id") + 1]
        report = tmp_path / f"candidate.{run_id}.json"
        report.write_text('{"resolved_instances": 9, "total_instances": 20}')
        return MagicMock(returncode=0, stderr="")

    with patch("shared.eval.swebench.shutil.which", return_value="/usr/bin/docker"), \
         patch("shared.eval.swebench.generate_predictions",
               return_value=str(tmp_path / "preds.jsonl")), \
         patch("shared.eval.swebench.subprocess.run", side_effect=fake_run):
        rate = eval_swebench.resolve_rate("model", "candidate", limit=20)

    assert rate == 0.45
    cmd = captured["cmd"]
    assert "--predictions_path" in cmd
    assert "--run_id" in cmd
    assert "--dataset_name" in cmd
    assert "--split" in cmd
    # These flags do NOT exist in swebench 4.1.0.
    assert "--model" not in cmd
    assert "--report_json" not in cmd


def test_swebench_resolve_rate_raises_without_docker():
    from shared.eval import swebench as eval_swebench
    with patch("shared.eval.swebench.shutil.which", return_value=None):
        try:
            eval_swebench.resolve_rate("model", "candidate")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Docker" in str(e)


def test_swebench_resolve_rate_raises_on_harness_failure(tmp_path, monkeypatch):
    from shared.eval import swebench as eval_swebench
    monkeypatch.chdir(tmp_path)
    with patch("shared.eval.swebench.shutil.which", return_value="/usr/bin/docker"), \
         patch("shared.eval.swebench.generate_predictions",
               return_value=str(tmp_path / "preds.jsonl")), \
         patch("shared.eval.swebench.subprocess.run",
               return_value=MagicMock(returncode=1, stderr="boom")):
        try:
            eval_swebench.resolve_rate("model", "candidate")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "boom" in str(e)

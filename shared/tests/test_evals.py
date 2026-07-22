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

    instances = [{"instance_id": f"i-{k}", "problem_statement": "p"} for k in range(20)]
    with patch("shared.eval.swebench.shutil.which", return_value="/usr/bin/docker"), \
         patch("shared.eval.swebench.generate_candidates",
               return_value=(instances, [["diff"] * 20])), \
         patch("shared.eval.swebench.subprocess.run", side_effect=fake_run):
        rate = eval_swebench.resolve_rate("model", "candidate", limit=20)

    assert rate == 0.45  # pass@1: 9 resolved / 20 (report has no resolved_ids -> count fallback)
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
    instances = [{"instance_id": "i-0", "problem_statement": "p"}]
    with patch("shared.eval.swebench.shutil.which", return_value="/usr/bin/docker"), \
         patch("shared.eval.swebench.generate_candidates",
               return_value=(instances, [["diff"]])), \
         patch("shared.eval.swebench.subprocess.run",
               return_value=MagicMock(returncode=1, stderr="boom")):
        try:
            eval_swebench.resolve_rate("model", "candidate")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "boom" in str(e)


# ---- OOM handling: sequential load + free -----------------------------------

def _install_fake_lm_eval(monkeypatch, hflm_calls, empty_cache):
    """Inject fake ``lm_eval`` + ``torch`` so the HFLM eval path runs GPU-free."""
    import sys
    import types

    class FakeHFLM:
        def __init__(self, **kwargs):
            hflm_calls.append(kwargs)

    fake_hf = types.ModuleType("lm_eval.models.huggingface")
    fake_hf.HFLM = FakeHFLM
    fake_models = types.ModuleType("lm_eval.models")
    fake_lm = types.ModuleType("lm_eval")
    fake_lm.simple_evaluate = lambda model, tasks, **kw: {
        "results": {tasks[0]: {"pass@1,create_test": 0.75}}
    }

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=empty_cache
    )

    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm)
    monkeypatch.setitem(sys.modules, "lm_eval.models", fake_models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", fake_hf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_humaneval_pass_at_1_shards_across_gpus_and_frees(monkeypatch):
    from shared.eval import humaneval as eval_humaneval

    hflm_calls = []
    empty_cache = MagicMock()
    _install_fake_lm_eval(monkeypatch, hflm_calls, empty_cache)

    score = eval_humaneval._pass_at_1("some/120b-model", family="qwen")

    assert score == 0.75
    # Model sharded across visible GPUs so a 120B does not OOM a single card.
    assert hflm_calls[0]["parallelize"] is True
    assert hflm_calls[0]["batch_size"] == "auto"
    # Model freed (CUDA cache emptied) before the next _pass_at_1 loads.
    empty_cache.assert_called()


def test_humaneval_regression_never_holds_two_models_resident(monkeypatch):
    """base then candidate: each _pass_at_1 must free before the next loads."""
    from shared.eval import humaneval as eval_humaneval

    events = []

    def fake_pass_at_1(model, family="gpt_oss"):
        events.append(("load", model))
        events.append(("free", model))
        return 0.5

    with patch.object(eval_humaneval, "_pass_at_1", side_effect=fake_pass_at_1):
        eval_humaneval.regression("base", "cand", family="qwen")

    # candidate must not load until base has been freed.
    assert events == [
        ("load", "base"), ("free", "base"),
        ("load", "cand"), ("free", "cand"),
    ]


def test_refusal_rate_frees_model_after_eval():
    from shared.eval import refusal as eval_refusal
    with patch("shared.eval.refusal.load_model", return_value=("m", "t")), \
         patch("shared.eval.refusal.chat_generate", return_value=["ok"]), \
         patch("shared.eval.refusal.free_model") as free:
        eval_refusal.refusal_rate("model", ["p1"], family="qwen")
    free.assert_called_once()


def test_bfcl_accuracy_frees_model_after_eval():
    from shared.eval import bfcl as eval_bfcl
    with patch("shared.eval.bfcl.load_model", return_value=("m", "t")), \
         patch("shared.eval.bfcl.chat_generate", return_value=["{}"]), \
         patch("shared.eval.bfcl.free_model") as free:
        eval_bfcl.accuracy("model", [{"prompt": "p", "expected": {}, "tools": None}])
    free.assert_called_once()


def test_swebench_generate_predictions_frees_model_before_harness(tmp_path, monkeypatch):
    import sys
    import types
    from shared.eval import swebench as eval_swebench

    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = MagicMock(return_value=[
        {"instance_id": "repo__issue-1", "problem_statement": "bug one"},
    ])
    monkeypatch.chdir(tmp_path)
    with patch.dict(sys.modules, {"datasets": fake_ds}), \
         patch("shared.eval.swebench.load_model", return_value=("m", "t")), \
         patch("shared.eval.swebench.chat_generate", return_value=["diff --git a b"]), \
         patch("shared.eval.swebench.free_model") as free:
        eval_swebench.generate_predictions("model", "candidate")
    # Model must be freed before the (GPU-free) Docker harness stage.
    free.assert_called_once()


def test_swebench_best_of_n_reports_pass_at_1_and_pass_at_n(tmp_path, monkeypatch, capsys):
    import json as _json
    from shared.eval import swebench as eval_swebench

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eval_swebench, "N_SAMPLES", 2)  # best-of-2
    instances = [{"instance_id": f"i-{k}", "problem_statement": "p"} for k in range(10)]
    slates = [["greedy"] * 10, ["sampled"] * 10]
    # greedy resolves {i-0,i-1,i-2}; sampled resolves {i-2,i-3,i-4} -> union 5
    resolved = {"cand0": ["i-0", "i-1", "i-2"], "cand1": ["i-2", "i-3", "i-4"]}

    def fake_run(cmd, **kwargs):
        run_id = cmd[cmd.index("--run_id") + 1]
        preds = cmd[cmd.index("--predictions_path") + 1]
        tag = "cand0" if "cand0" in preds else "cand1"
        (tmp_path / f"candidate.{run_id}.json").write_text(
            _json.dumps({"resolved_ids": resolved[tag], "total_instances": 10}))
        return MagicMock(returncode=0, stderr="")

    with patch("shared.eval.swebench.shutil.which", return_value="/usr/bin/docker"), \
         patch("shared.eval.swebench.generate_candidates", return_value=(instances, slates)), \
         patch("shared.eval.swebench.subprocess.run", side_effect=fake_run):
        rate = eval_swebench.resolve_rate("model", "candidate", limit=10)

    assert rate == 0.3  # gate = pass@1 (greedy slate only): 3/10
    out = capsys.readouterr().out
    assert "pass@1=0.3000" in out          # honest single-shot
    assert "pass@2=0.5000" in out          # union {i-0..i-4}/10 = headroom, NOT the gate
    assert "NOT the gate" in out


def test_swebench_gen_kwargs_forwarded_for_sampling():
    # best-of-N diversity relies on chat_generate forwarding sampling params.
    import inspect
    from shared.eval._model import chat_generate
    assert "gen_kwargs" in inspect.signature(chat_generate).parameters


def test_swebench_eval_dataset_is_env_configurable(monkeypatch):
    # Enables the report's contamination-free DEV eval: point the SAME harness at
    # SWE-rebench for iteration, keep Verified for the final verdict.
    import importlib
    from shared.eval import swebench as eval_swebench
    monkeypatch.setenv("SWE_EVAL_DATASET", "nebius/SWE-rebench")
    monkeypatch.setenv("SWE_EVAL_SPLIT", "test")
    importlib.reload(eval_swebench)
    try:
        assert eval_swebench.DATASET == "nebius/SWE-rebench"
        assert eval_swebench.SPLIT == "test"
    finally:
        monkeypatch.delenv("SWE_EVAL_DATASET", raising=False)
        monkeypatch.delenv("SWE_EVAL_SPLIT", raising=False)
        importlib.reload(eval_swebench)  # restore default for other tests
    assert eval_swebench.DATASET == "princeton-nlp/SWE-bench_Verified"


def test_swebench_repro_first_prompt_and_extraction(monkeypatch):
    from shared.eval import swebench as sb
    monkeypatch.setattr(sb, "REPRO_FIRST", True)
    # prompt switches to reproduce-first (spec-before-patch)
    sysmsg = sb._prompt_messages("the bug")[0]["content"]
    assert "Reproduction" in sysmsg and "Patch" in sysmsg
    # extraction takes the LAST fenced diff (models quote diffs while reasoning)
    resp = ("## Reproduction\nThe call returns None.\nEarlier draft:\n"
            "```diff\nwrong one\n```\n"
            "## Patch\n```diff\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -1 +1 @@\n-a\n+b\n```\n")
    patch = sb._extract_patch(resp)
    assert patch.startswith("diff --git a/x.py")
    assert "wrong one" not in patch  # the reasoning draft is not the submission


def test_swebench_extract_patch_fallbacks(monkeypatch):
    from shared.eval import swebench as sb
    monkeypatch.setattr(sb, "REPRO_FIRST", True)
    # no fence but a git-diff span -> from the last 'diff --git'
    got = sb._extract_patch("thoughts...\ndiff --git a/f b/f\n@@\n-x\n+y")
    assert got.startswith("diff --git a/f") and got.endswith("\n")
    # nothing diff-like -> raw, newline-terminated (harness will just fail to apply)
    assert sb._extract_patch("no patch here").endswith("\n")


def test_swebench_direct_mode_unchanged(monkeypatch):
    from shared.eval import swebench as sb
    monkeypatch.setattr(sb, "REPRO_FIRST", False)
    assert sb._extract_patch("diff --git a b") == "diff --git a b"   # passthrough
    assert "ONLY" in sb._prompt_messages("x")[0]["content"]           # direct prompt

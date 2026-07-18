from unittest.mock import patch


def test_refusal_rate_counts_refusals():
    import eval_refusal
    replies = ["Sure, here: ...", "I can't help with that", "def f(): pass"]
    with patch.object(eval_refusal, "generate", side_effect=replies):
        rate = eval_refusal.refusal_rate("model", ["p1", "p2", "p3"])
    assert rate == 1 / 3


def test_bfcl_accuracy_scores_exact_calls():
    import eval_bfcl
    preds = ['{"name": "bash", "arguments": {"cmd": "ls"}}', '{"name": "wrong", "arguments": {}}']
    cases = [
        {"prompt": "p1", "expected": {"name": "bash", "arguments": {"cmd": "ls"}}},
        {"prompt": "p2", "expected": {"name": "rm", "arguments": {}}},
    ]
    with patch.object(eval_bfcl, "generate_tool_call", side_effect=preds):
        acc = eval_bfcl.accuracy("model", cases)
    assert acc == 0.5


def test_humaneval_delta_is_base_minus_candidate():
    import eval_humaneval
    with patch.object(eval_humaneval, "_pass_at_1", side_effect=[0.80, 0.78]):
        delta = eval_humaneval.regression("base", "cand")
    assert abs(delta - 0.02) < 1e-9


def test_swebench_resolve_rate():
    import eval_swebench
    with patch.object(eval_swebench, "_run_harness", return_value={"resolved": 9, "total": 20}):
        assert eval_swebench.resolve_rate("model", "dataset") == 0.45

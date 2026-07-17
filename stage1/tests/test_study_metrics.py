import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "remote"))

from study_metrics import (
    checkpoint_path,
    sanitize_model_name,
    scores_from_trial,
    sort_pareto_trials,
)


def _trial(refusal_rate, kl_divergence):
    return SimpleNamespace(user_attrs={
        "scores": [
            {"name": "Keywords", "score": {"value": refusal_rate}},
            {"name": "KL divergence", "score": {"value": kl_divergence}},
        ]
    })


def test_sanitize_model_name_replaces_special_chars():
    assert sanitize_model_name("Qwen/Qwen2.5-Coder-32B-Instruct") == "Qwen--Qwen2--5-Coder-32B-Instruct"


def test_sanitize_model_name_keeps_underscores_and_hyphens():
    assert sanitize_model_name("my_model-name") == "my_model-name"


def test_checkpoint_path_joins_dir_and_sanitized_name():
    path = checkpoint_path("checkpoints", "Qwen/Qwen3-4B-Instruct-2507")
    assert path == os.path.join("checkpoints", "Qwen--Qwen3-4B-Instruct-2507.jsonl")


def test_scores_from_trial_extracts_named_values():
    trial = _trial(refusal_rate=0.03, kl_divergence=0.16)
    assert scores_from_trial(trial) == {"refusal_rate": 0.03, "kl_divergence": 0.16}


def test_sort_pareto_trials_orders_by_refusal_rate_then_kl():
    trials = [
        _trial(refusal_rate=0.05, kl_divergence=0.1),
        _trial(refusal_rate=0.02, kl_divergence=0.5),
        _trial(refusal_rate=0.02, kl_divergence=0.2),
    ]
    sorted_trials = sort_pareto_trials(trials)
    ordered_scores = [scores_from_trial(t) for t in sorted_trials]
    assert ordered_scores == [
        {"refusal_rate": 0.02, "kl_divergence": 0.2},
        {"refusal_rate": 0.02, "kl_divergence": 0.5},
        {"refusal_rate": 0.05, "kl_divergence": 0.1},
    ]

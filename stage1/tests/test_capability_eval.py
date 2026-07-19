import os
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "remote"))

from capability_eval import compute_deltas, run_benchmarks


def _install_fake_lm_eval(monkeypatch, hflm_calls, empty_cache):
    """Inject fake ``lm_eval`` + ``torch`` so run_benchmarks runs GPU-free while
    we assert the sharding flag and the CUDA-cache free path."""
    class FakeHFLM:
        def __init__(self, **kwargs):
            hflm_calls.append(kwargs)

    fake_hf = types.ModuleType("lm_eval.models.huggingface")
    fake_hf.HFLM = FakeHFLM
    fake_models = types.ModuleType("lm_eval.models")
    fake_lm = types.ModuleType("lm_eval")

    def fake_simple_evaluate(model, tasks, **kwargs):
        if tasks[0] == "mmlu":
            return {"results": {"mmlu": {"acc,none": 0.60}}}
        return {"results": {"gsm8k": {"exact_match,strict-match": 0.50}}}

    fake_lm.simple_evaluate = fake_simple_evaluate

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=empty_cache
    )

    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm)
    monkeypatch.setitem(sys.modules, "lm_eval.models", fake_models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", fake_hf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_run_benchmarks_shards_across_gpus_and_returns_shape(monkeypatch):
    hflm_calls = []
    empty_cache = MagicMock()
    _install_fake_lm_eval(monkeypatch, hflm_calls, empty_cache)

    out = run_benchmarks("some/120b-model")

    # Return dict shape is unchanged.
    assert out == {
        "mmlu": {"acc,none": 0.60},
        "gsm8k": {"exact_match,strict-match": 0.50},
    }
    # Model is sharded across the visible GPUs (no single-GPU OOM).
    assert hflm_calls[0]["parallelize"] is True
    assert hflm_calls[0]["batch_size"] == "auto"
    # The 120B is freed (CUDA cache emptied) so the next load can fit.
    empty_cache.assert_called()


def test_run_benchmarks_frees_model_even_when_eval_raises(monkeypatch):
    hflm_calls = []
    empty_cache = MagicMock()
    _install_fake_lm_eval(monkeypatch, hflm_calls, empty_cache)

    import lm_eval

    def boom(model, tasks, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(lm_eval, "simple_evaluate", boom)

    with pytest.raises(RuntimeError):
        run_benchmarks("some/120b-model")
    # finally-block free still runs so the next model is not blocked by a leak.
    empty_cache.assert_called()


def test_compute_deltas_positive_when_candidate_worse():
    base = {"mmlu": {"acc,none": 0.60}, "gsm8k": {"exact_match,strict-match": 0.50}}
    candidate = {"mmlu": {"acc,none": 0.55}, "gsm8k": {"exact_match,strict-match": 0.48}}
    deltas = compute_deltas(base, candidate)
    assert deltas["mmlu_delta"] == pytest.approx(0.05)
    assert deltas["gsm8k_delta"] == pytest.approx(0.02)


def test_compute_deltas_negative_when_candidate_better():
    base = {"mmlu": {"acc,none": 0.60}, "gsm8k": {"exact_match,strict-match": 0.50}}
    candidate = {"mmlu": {"acc,none": 0.62}, "gsm8k": {"exact_match,strict-match": 0.55}}
    deltas = compute_deltas(base, candidate)
    assert deltas["mmlu_delta"] < 0
    assert deltas["gsm8k_delta"] < 0

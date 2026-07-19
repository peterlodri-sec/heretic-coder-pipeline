import sys
import types

import rft_generate


def _fake_vllm(captured):
    vllm = types.ModuleType("vllm")

    class LLM:
        def __init__(self, **kw):
            captured["llm_kwargs"] = kw

        def chat(self, convos, params):
            captured["convos"] = convos
            captured["params"] = params
            outs = []
            for i, _ in enumerate(convos):
                samples = [types.SimpleNamespace(
                    text=f"<|channel|>final<|message|>code{i}_{j}<|return|>")
                    for j in range(params.n)]
                outs.append(types.SimpleNamespace(outputs=samples))
            return outs

    def SamplingParams(**kw):
        captured["sp_kwargs"] = kw
        return types.SimpleNamespace(**kw)

    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    return vllm


def test_generate_returns_final_channel_aligned(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm(captured))
    res = rft_generate.generate("m", ["p0", "p1"], n=3, temperature=0.8, max_new_tokens=128)
    # aligned to prompts, n samples each, harmony tags stripped
    assert len(res) == 2 and all(len(c) == 3 for c in res)
    assert res[0][0] == "code0_0" and res[1][2] == "code1_2"


def test_generate_enables_kv_cache_kit(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm(captured))
    rft_generate.generate("m", ["p"], n=2)
    assert captured["llm_kwargs"]["enable_prefix_caching"] is True
    assert captured["llm_kwargs"]["kv_cache_dtype"] == "fp8"
    assert captured["llm_kwargs"]["model"] == "m"


def test_generate_passes_sampling_params_and_harmony_convo(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm(captured))
    rft_generate.generate("m", ["solve x"], n=5, temperature=0.7, max_new_tokens=64)
    assert captured["sp_kwargs"] == {"n": 5, "temperature": 0.7, "max_tokens": 64}
    assert captured["convos"][0][0] == {"role": "user", "content": "solve x"}


def test_module_import_is_gpu_free():
    assert rft_generate.MAX_NEW_TOKENS > 0

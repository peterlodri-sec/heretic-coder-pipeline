import rft_generate
import pytest


def test_generate_is_stubbed_interface():
    # Candidate generation (vLLM/HF) is finalized from research; the module still
    # imports GPU-free, and calling the interface raises until then.
    with pytest.raises(NotImplementedError):
        rft_generate.generate("model", ["prompt"], 4)


def test_module_import_is_gpu_free():
    assert rft_generate.MAX_NEW_TOKENS > 0

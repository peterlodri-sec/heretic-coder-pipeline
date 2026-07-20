import sys
import types
from unittest.mock import MagicMock

from shared.train_common import (
    LoraSpec, LORA_TARGETS, load_lora_model, HIGH_RANK_RSLORA,
)


def test_high_rank_rslora_preset():
    # Research-recommended: r=64/a=128 + rsLoRA (fixes the plain-LoRA r>=64 collapse
    # that forced the r=32 default). Preset, not the default.
    assert HIGH_RANK_RSLORA.r == 64 and HIGH_RANK_RSLORA.alpha == 128
    assert HIGH_RANK_RSLORA.use_rslora is True
    assert LoraSpec().r == 32  # default stays conservative


def test_lora_spec_defaults_are_the_anti_regression_setting():
    s = LoraSpec()
    assert s.r == 32 and s.alpha == 64 and s.dropout == 0.0
    assert s.targets == LORA_TARGETS


def test_lora_targets_are_moe_safe():
    # the router / mlp.gate must NOT be targeted
    assert "q_proj" in LORA_TARGETS and "down_proj" in LORA_TARGETS
    assert "gate" not in LORA_TARGETS and "router" not in LORA_TARGETS


def test_lora_spec_is_frozen_and_slotted():
    import pytest
    s = LoraSpec()
    assert not hasattr(s, "__dict__")  # slots
    with pytest.raises(Exception):
        s.r = 64  # frozen


def _fake_unsloth(captured):
    unsloth = types.ModuleType("unsloth")
    flm = MagicMock()
    flm.from_pretrained.return_value = ("base", "tok")
    flm.get_peft_model.return_value = "peft"
    unsloth.FastLanguageModel = flm
    return unsloth


def test_load_lora_model_wires_from_pretrained_and_peft(monkeypatch):
    cap = {}
    monkeypatch.setitem(sys.modules, "unsloth", _fake_unsloth(cap))
    model, tok = load_lora_model("src", max_seq_len=4096, load_in_4bit=True)
    assert (model, tok) == ("peft", "tok")
    import unsloth
    fp = unsloth.FastLanguageModel.from_pretrained.call_args.kwargs
    assert fp["model_name"] == "src" and fp["load_in_4bit"] is True
    assert fp["max_seq_length"] == 4096
    peft = unsloth.FastLanguageModel.get_peft_model.call_args.kwargs
    assert peft["r"] == 32 and peft["lora_alpha"] == 64
    assert peft["target_modules"] == list(LORA_TARGETS)


def test_rslora_defaults_off():
    # rsLoRA rescales alpha/sqrt(r); it must never be a silent default.
    assert LoraSpec().use_rslora is False


def test_load_lora_model_passes_use_rslora(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", _fake_unsloth({}))
    # default: off
    load_lora_model("src", max_seq_len=4096, load_in_4bit=True)
    import unsloth
    assert unsloth.FastLanguageModel.get_peft_model.call_args.kwargs["use_rslora"] is False
    # opt-in threads through
    load_lora_model("src", max_seq_len=4096, load_in_4bit=True,
                    lora=LoraSpec(use_rslora=True))
    assert unsloth.FastLanguageModel.get_peft_model.call_args.kwargs["use_rslora"] is True

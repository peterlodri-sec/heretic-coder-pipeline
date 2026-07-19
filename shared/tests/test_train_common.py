import sys
import types
from unittest.mock import MagicMock

from shared.train_common import LoraSpec, LORA_TARGETS, load_lora_model


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

import sys
import types
from unittest.mock import MagicMock

import pytest

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
    s = LoraSpec()
    assert not hasattr(s, "__dict__")  # slots
    with pytest.raises(Exception):
        s.r = 64  # frozen


# --- dense (qwen) path: Unsloth FastLanguageModel -----------------------------

def _fake_unsloth():
    unsloth = types.ModuleType("unsloth")
    flm = MagicMock()
    flm.from_pretrained.return_value = ("base", "tok")
    flm.get_peft_model.return_value = "peft"
    unsloth.FastLanguageModel = flm
    return unsloth


def test_dense_family_uses_unsloth_from_pretrained_and_peft(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", _fake_unsloth())
    model, tok = load_lora_model("src", max_seq_len=4096, load_in_4bit=True,
                                 family="qwen")
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


def test_dense_family_passes_use_rslora(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", _fake_unsloth())
    # default: off
    load_lora_model("src", max_seq_len=4096, load_in_4bit=True, family="qwen")
    import unsloth
    assert unsloth.FastLanguageModel.get_peft_model.call_args.kwargs["use_rslora"] is False
    # opt-in threads through
    load_lora_model("src", max_seq_len=4096, load_in_4bit=True, family="qwen",
                    lora=LoraSpec(use_rslora=True))
    assert unsloth.FastLanguageModel.get_peft_model.call_args.kwargs["use_rslora"] is True


# --- gpt-oss path: plain transformers + bitsandbytes + PEFT -------------------
# The Unsloth loader leaves gpt-oss's mlp.router.{weight,bias} keys unmapped
# ("some weights not initialized") and aborts; plain transformers loads the same
# checkpoint, so gpt_oss must route AROUND Unsloth entirely.

def _fake_transformers_peft():
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    tfm = types.ModuleType("transformers")
    tok = MagicMock(model_max_length=131072)
    tfm.AutoTokenizer = MagicMock()
    tfm.AutoTokenizer.from_pretrained.return_value = tok
    tfm.AutoModelForCausalLM = MagicMock()
    tfm.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    tfm.BitsAndBytesConfig = MagicMock(return_value="BNB")
    peft = types.ModuleType("peft")
    peft.LoraConfig = MagicMock(return_value="LORACFG")
    peft.get_peft_model = MagicMock(return_value="PEFTMODEL")
    peft.prepare_model_for_kbit_training = MagicMock(side_effect=lambda m, **k: m)
    return torch, tfm, peft, tok


def _install_plain(monkeypatch):
    torch, tfm, peft, tok = _fake_transformers_peft()
    for name, mod in (("torch", torch), ("transformers", tfm), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, mod)
    # a stubbed unsloth that EXPLODES if the gpt-oss path ever touches it
    boom = types.ModuleType("unsloth")
    boom.FastLanguageModel = property(lambda self: (_ for _ in ()).throw(
        AssertionError("gpt_oss must not use Unsloth")))
    monkeypatch.setitem(sys.modules, "unsloth", boom)
    return tfm, peft, tok


def test_gpt_oss_uses_plain_transformers_peft_4bit(monkeypatch):
    tfm, peft, tok = _install_plain(monkeypatch)
    model, out_tok = load_lora_model("PeetPedro/gpt-oss-120b-heretic",
                                     max_seq_len=16384, load_in_4bit=True,
                                     family="gpt_oss")
    assert model == "PEFTMODEL" and out_tok is tok
    # 4-bit → a bitsandbytes quant config is built and passed
    tfm.BitsAndBytesConfig.assert_called_once()
    fp = tfm.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert fp["quantization_config"] == "BNB"
    assert fp["torch_dtype"] == "bf16"
    # LoRA spec threaded into peft.LoraConfig, not unsloth
    lc = peft.LoraConfig.call_args.kwargs
    assert lc["r"] == 32 and lc["lora_alpha"] == 64
    assert lc["target_modules"] == list(LORA_TARGETS)
    assert lc["task_type"] == "CAUSAL_LM"
    peft.prepare_model_for_kbit_training.assert_called_once()  # QLoRA path
    assert tok.model_max_length == 16384  # clamped to max_seq_len


def test_gpt_oss_bf16_skips_bitsandbytes(monkeypatch):
    tfm, peft, tok = _install_plain(monkeypatch)
    load_lora_model("m", max_seq_len=4096, load_in_4bit=False, family="gpt_oss")
    tfm.BitsAndBytesConfig.assert_not_called()
    fp = tfm.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert fp["quantization_config"] is None
    peft.prepare_model_for_kbit_training.assert_not_called()  # not a kbit model

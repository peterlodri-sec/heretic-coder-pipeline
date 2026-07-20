import sys
import types
from unittest.mock import MagicMock, patch


def _install_fakes():
    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = MagicMock()
    unsloth.FastLanguageModel.from_pretrained.return_value = ("model", "tok")
    unsloth.FastLanguageModel.get_peft_model.return_value = "peft_model"
    chat_templates = types.ModuleType("unsloth.chat_templates")
    # returns the trainer it's handed (response-only masking wrapper)
    chat_templates.train_on_responses_only = MagicMock(side_effect=lambda tr, **kw: tr)
    unsloth.chat_templates = chat_templates
    trl = types.ModuleType("trl")
    trl.SFTTrainer = MagicMock()
    trl.SFTConfig = MagicMock()
    datasets = types.ModuleType("datasets")
    ds = MagicMock()
    ds.map.return_value = ds  # .map(...) -> dataset with a `text` column
    datasets.load_dataset = MagicMock(return_value=ds)
    return {"unsloth": unsloth, "unsloth.chat_templates": chat_templates,
            "trl": trl, "datasets": datasets}


def _run_train(**train_kwargs):
    fakes = _install_fakes()
    trainer = fakes["trl"].SFTTrainer.return_value
    trainer.train.return_value = types.SimpleNamespace(training_loss=0.42)
    with patch.dict(sys.modules, fakes):
        import importlib
        sft_train = importlib.import_module("sft_train")
        importlib.reload(sft_train)
        result = sft_train.train("model_src", "data.jsonl", "out", max_steps=1,
                                 **train_kwargs)
    return fakes, result


def test_train_returns_loss_model_tokenizer_tuple():
    fakes, result = _run_train()
    loss, model, tok = result
    assert loss == 0.42
    assert model == "peft_model"
    assert tok == "tok"
    fakes["unsloth"].FastLanguageModel.from_pretrained.assert_called_once()


def test_gpt_oss_default_loads_in_4bit():
    # Default family is gpt_oss -> MoE-QLoRA 4-bit (fits 120B on 1x H200).
    fakes, _ = _run_train()
    kwargs = fakes["unsloth"].FastLanguageModel.from_pretrained.call_args.kwargs
    assert kwargs["load_in_4bit"] is True
    assert kwargs["model_name"] == "model_src"


def test_qwen_family_trains_in_16bit():
    # The dense-32B validation path stays 16-bit bf16.
    fakes, _ = _run_train(family="qwen")
    kwargs = fakes["unsloth"].FastLanguageModel.from_pretrained.call_args.kwargs
    assert kwargs["load_in_4bit"] is False


def test_explicit_load_in_4bit_overrides_family_default():
    fakes, _ = _run_train(family="gpt_oss", load_in_4bit=False)
    kwargs = fakes["unsloth"].FastLanguageModel.from_pretrained.call_args.kwargs
    assert kwargs["load_in_4bit"] is False


def test_dataset_is_chat_templated_to_text_field():
    fakes, _ = _run_train()
    # messages rendered to a `text` column, and SFTConfig points at it
    fakes["datasets"].load_dataset.return_value.map.assert_called_once()
    cfg_kwargs = fakes["trl"].SFTConfig.call_args.kwargs
    assert cfg_kwargs["dataset_text_field"] == "text"
    assert cfg_kwargs["max_length"] == 16384
    assert "max_seq_length" not in cfg_kwargs
    # assistant-only loss now comes from the Unsloth helper, not the TRL flag
    assert "assistant_only_loss" not in cfg_kwargs


def test_packing_bfd_enabled_by_default():
    # BFD example-packing on by default: block-diagonal over FlashAttention, so
    # packed examples don't attend across the seam. NOT "wrapped" (which mixes them).
    fakes, _ = _run_train()
    cfg = fakes["trl"].SFTConfig.call_args.kwargs
    assert cfg["packing"] is True
    assert cfg["packing_strategy"] == "bfd"


def test_packing_off_switch(monkeypatch):
    # STAGE2_PACKING=0 disables packing (read at import; _run_train reloads).
    monkeypatch.setenv("STAGE2_PACKING", "0")
    fakes, _ = _run_train()
    cfg = fakes["trl"].SFTConfig.call_args.kwargs
    assert cfg["packing"] is False


def test_neftune_off_by_default():
    # Precision-sensitive coder: NEFTune stays opt-in (None == disabled in TRL).
    fakes, _ = _run_train()
    assert fakes["trl"].SFTConfig.call_args.kwargs["neftune_noise_alpha"] is None


def test_neftune_env_knob(monkeypatch):
    monkeypatch.setenv("STAGE2_NEFTUNE", "5")
    fakes, _ = _run_train()
    assert fakes["trl"].SFTConfig.call_args.kwargs["neftune_noise_alpha"] == 5


def test_response_only_masking_applied():
    fakes, _ = _run_train()
    call = fakes["unsloth.chat_templates"].train_on_responses_only.call_args
    assert "assistant" in call.kwargs["response_part"]
    assert "user" in call.kwargs["instruction_part"]


def test_gpt_oss_uses_harmony_final_channel_delimiters():
    fakes, _ = _run_train()  # default gpt_oss
    call = fakes["unsloth.chat_templates"].train_on_responses_only.call_args
    assert call.kwargs["response_part"] == "<|start|>assistant<|channel|>final<|message|>"
    assert call.kwargs["instruction_part"] == "<|start|>user<|message|>"


def test_qwen_family_uses_chatml_delimiters():
    fakes, _ = _run_train(family="qwen")
    call = fakes["unsloth.chat_templates"].train_on_responses_only.call_args
    assert call.kwargs["response_part"] == "<|im_start|>assistant\n"
    assert call.kwargs["instruction_part"] == "<|im_start|>user\n"


def test_sfttrainer_uses_processing_class_not_tokenizer():
    fakes, _ = _run_train()
    trainer_kwargs = fakes["trl"].SFTTrainer.call_args.kwargs
    assert trainer_kwargs["processing_class"] == "tok"
    assert "tokenizer" not in trainer_kwargs

import importlib
import sys
import types
from unittest.mock import MagicMock, patch


def _fakes():
    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = MagicMock()
    unsloth.FastLanguageModel.from_pretrained.return_value = ("model", "tok")
    unsloth.FastLanguageModel.get_peft_model.return_value = "peft_model"
    unsloth.PatchDPOTrainer = MagicMock()
    trl = types.ModuleType("trl")
    trl.ORPOTrainer = MagicMock()
    trl.ORPOConfig = MagicMock()
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = MagicMock(return_value="ds")
    return {"unsloth": unsloth, "trl": trl, "datasets": datasets}


def _run_train(**train_kwargs):
    fakes = _fakes()
    fakes["trl"].ORPOTrainer.return_value.train.return_value = types.SimpleNamespace(
        training_loss=0.21)
    with patch.dict(sys.modules, fakes):
        orpo_train = importlib.import_module("orpo_train")
        importlib.reload(orpo_train)
        result = orpo_train.train("src", "pairs.jsonl", "out", num_epochs=1,
                                  **train_kwargs)
    return fakes, result


def test_train_returns_loss_and_model_tokenizer():
    fakes, result = _run_train()
    loss, model, tok = result
    assert loss == 0.21
    assert model == "peft_model" and tok == "tok"
    fakes["trl"].ORPOTrainer.assert_called_once()
    fakes["unsloth"].FastLanguageModel.from_pretrained.assert_called_once()


def test_gpt_oss_orpo_defaults_to_4bit():
    # gpt-oss ORPO MUST be 4-bit (MoE-QLoRA); default family is gpt_oss.
    fakes, _ = _run_train()
    kwargs = fakes["unsloth"].FastLanguageModel.from_pretrained.call_args.kwargs
    assert kwargs["load_in_4bit"] is True
    assert kwargs["model_name"] == "src"


def test_qwen_family_trains_in_16bit():
    fakes, _ = _run_train(family="qwen")
    kwargs = fakes["unsloth"].FastLanguageModel.from_pretrained.call_args.kwargs
    assert kwargs["load_in_4bit"] is False


def test_orpotrainer_uses_processing_class_not_tokenizer():
    fakes, _ = _run_train()
    trainer_kwargs = fakes["trl"].ORPOTrainer.call_args.kwargs
    assert trainer_kwargs["processing_class"] == "tok"
    assert "tokenizer" not in trainer_kwargs


def test_orpoconfig_uses_modern_kwargs():
    fakes, _ = _run_train()
    cfg = fakes["trl"].ORPOConfig.call_args.kwargs
    assert cfg["beta"] == 0.1
    assert cfg["max_length"] == 8192
    assert cfg["max_prompt_length"] == 2048
    assert cfg["bf16"] is True
    assert cfg["warmup_ratio"] == 0.03


def test_patch_dpo_trainer_called_before_trainer_built():
    # Unsloth's DPO-family patch must run, else grad-checkpoint mispatches -> OOM.
    fakes, _ = _run_train()
    fakes["unsloth"].PatchDPOTrainer.assert_called_once()


def test_lora_rank_matches_stage2_anti_regression_fix():
    fakes, _ = _run_train()
    peft_kwargs = fakes["unsloth"].FastLanguageModel.get_peft_model.call_args.kwargs
    assert peft_kwargs["r"] == 32
    assert peft_kwargs["lora_alpha"] == 64

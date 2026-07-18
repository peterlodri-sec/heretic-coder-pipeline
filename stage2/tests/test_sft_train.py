import sys
import types
from unittest.mock import MagicMock, patch


def _install_fakes():
    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = MagicMock()
    unsloth.FastLanguageModel.from_pretrained.return_value = ("model", "tok")
    unsloth.FastLanguageModel.get_peft_model.return_value = "peft_model"
    trl = types.ModuleType("trl")
    trl.SFTTrainer = MagicMock()
    trl.SFTConfig = MagicMock()
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = MagicMock(return_value="ds")
    return {"unsloth": unsloth, "trl": trl, "datasets": datasets}


def _run_train():
    fakes = _install_fakes()
    trainer = fakes["trl"].SFTTrainer.return_value
    trainer.train.return_value = types.SimpleNamespace(training_loss=0.42)
    with patch.dict(sys.modules, fakes):
        import importlib
        sft_train = importlib.import_module("sft_train")
        importlib.reload(sft_train)
        result = sft_train.train("model_src", "data.jsonl", "out", max_steps=1)
    return fakes, result


def test_train_returns_loss_model_tokenizer_tuple():
    fakes, result = _run_train()
    loss, model, tok = result
    assert loss == 0.42
    # model/tokenizer are threaded back to the caller for export.
    assert model == "peft_model"
    assert tok == "tok"
    fakes["unsloth"].FastLanguageModel.from_pretrained.assert_called_once()


def test_from_pretrained_trains_in_16bit():
    fakes, _ = _run_train()
    kwargs = fakes["unsloth"].FastLanguageModel.from_pretrained.call_args.kwargs
    assert kwargs["load_in_4bit"] is False
    assert kwargs["model_name"] == "model_src"


def test_sftconfig_uses_modern_kwargs():
    fakes, _ = _run_train()
    cfg_kwargs = fakes["trl"].SFTConfig.call_args.kwargs
    assert cfg_kwargs["assistant_only_loss"] is True
    assert cfg_kwargs["max_length"] == 16384
    assert cfg_kwargs["packing"] is False
    assert "max_seq_length" not in cfg_kwargs


def test_sfttrainer_uses_processing_class_not_tokenizer():
    fakes, _ = _run_train()
    trainer_kwargs = fakes["trl"].SFTTrainer.call_args.kwargs
    assert trainer_kwargs["processing_class"] == "tok"
    assert "tokenizer" not in trainer_kwargs

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


def test_train_returns_final_loss():
    fakes = _install_fakes()
    trainer = fakes["trl"].SFTTrainer.return_value
    trainer.train.return_value = types.SimpleNamespace(training_loss=0.42)
    with patch.dict(sys.modules, fakes):
        import importlib
        sft_train = importlib.import_module("sft_train")
        importlib.reload(sft_train)
        loss = sft_train.train("model_src", "data.jsonl", "out", max_steps=1)
    assert loss == 0.42
    fakes["unsloth"].FastLanguageModel.from_pretrained.assert_called_once()

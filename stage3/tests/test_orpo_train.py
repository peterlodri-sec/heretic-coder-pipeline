import importlib
import sys
import types
from unittest.mock import MagicMock, patch


def _fakes():
    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = MagicMock()
    unsloth.FastLanguageModel.from_pretrained.return_value = ("model", "tok")
    unsloth.FastLanguageModel.get_peft_model.return_value = "peft_model"
    trl = types.ModuleType("trl")
    trl.ORPOTrainer = MagicMock()
    trl.ORPOConfig = MagicMock()
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = MagicMock(return_value="ds")
    return {"unsloth": unsloth, "trl": trl, "datasets": datasets}


def test_train_returns_loss_and_model_tokenizer():
    fakes = _fakes()
    fakes["trl"].ORPOTrainer.return_value.train.return_value = types.SimpleNamespace(training_loss=0.21)
    with patch.dict(sys.modules, fakes):
        orpo_train = importlib.import_module("orpo_train")
        importlib.reload(orpo_train)
        loss, model, tok = orpo_train.train("src", "pairs.jsonl", "out", num_epochs=1)
    assert loss == 0.21
    fakes["trl"].ORPOTrainer.assert_called_once()
    fakes["unsloth"].FastLanguageModel.from_pretrained.assert_called_once()

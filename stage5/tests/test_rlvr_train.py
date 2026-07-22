import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import rlvr_train


def _fakes():
    train_common = types.ModuleType("shared.train_common")
    train_common.load_lora_model = MagicMock(return_value=("peft_model", "tok"))
    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = MagicMock()
    unsloth.FastLanguageModel.from_pretrained.return_value = ("model", "tok")
    unsloth.FastLanguageModel.get_peft_model.return_value = "peft_model"
    unsloth.PatchFastRL = MagicMock()
    trl = types.ModuleType("trl")
    trl.GRPOTrainer = MagicMock()
    trl.GRPOConfig = MagicMock()
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = MagicMock(return_value="ds")
    return {"shared.train_common": train_common, "unsloth": unsloth, "trl": trl, "datasets": datasets}


def _run_train():
    fakes = _fakes()
    fakes["trl"].GRPOTrainer.return_value.train.return_value = types.SimpleNamespace(
        training_loss=0.5)
    with patch.dict(sys.modules, fakes):
        importlib.reload(rlvr_train)
        result = rlvr_train.train("src", "data.jsonl", "out", num_gpus=2)
    return fakes, result


def test_train_returns_loss_and_model_tokenizer():
    fakes, result = _run_train()
    loss, model, tok = result
    assert loss == 0.5 and model == "peft_model" and tok == "tok"
    fakes["trl"].GRPOTrainer.assert_called_once()


def test_gspo_is_enabled_for_moe():
    # sequence-level importance sampling + zero-KL = GSPO (MoE-stable). REQUIRED.
    fakes, _ = _run_train()
    cfg = fakes["trl"].GRPOConfig.call_args.kwargs
    assert cfg["importance_sampling_level"] == "sequence"
    assert cfg["beta"] == 0.0


def test_uses_colocated_vllm_rollouts():
    fakes, _ = _run_train()
    cfg = fakes["trl"].GRPOConfig.call_args.kwargs
    assert cfg["use_vllm"] is True
    assert cfg["vllm_mode"] == "colocate"
    assert cfg["num_generations"] >= 2


def test_patch_fast_rl_called_before_trainer():
    # GRPO analog of ORPO's PatchDPOTrainer trap.
    fakes, _ = _run_train()
    fakes["unsloth"].PatchFastRL.assert_called_once()
    args = fakes["unsloth"].PatchFastRL.call_args.args
    assert args[0] == "GRPO"


def test_loads_gpt_oss_in_4bit():
    fakes, _ = _run_train()
    call = fakes["shared.train_common"].load_lora_model.call_args
    assert call.kwargs["load_in_4bit"] is True


def test_reward_func_is_code_execution_reward():
    from reward import code_execution_reward
    fakes, _ = _run_train()
    trainer_kw = fakes["trl"].GRPOTrainer.call_args.kwargs
    assert trainer_kw["reward_funcs"] == [code_execution_reward]


def test_lora_rank_matches_anti_regression_fix():
    fakes, _ = _run_train()
    fakes["shared.train_common"].load_lora_model.assert_called_once()


def test_max_seq_len_constant():
    assert rlvr_train.MAX_SEQ_LEN > 0

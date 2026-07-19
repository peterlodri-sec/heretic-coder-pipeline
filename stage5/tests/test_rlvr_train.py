import pytest
import rlvr_train


def test_train_is_stubbed_interface():
    # GRPO/GSPO trainer finalized from research; module imports GPU-free, calling
    # train() raises until the trainer is wired.
    with pytest.raises(NotImplementedError):
        rlvr_train.train("model", "data.jsonl", "out", num_gpus=2)


def test_constants_mirror_sft_orpo():
    assert "q_proj" in rlvr_train.LORA_TARGETS and "down_proj" in rlvr_train.LORA_TARGETS
    assert rlvr_train.MAX_SEQ_LEN > 0

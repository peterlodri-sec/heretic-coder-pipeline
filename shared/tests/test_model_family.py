import pytest

from shared.model_family import (
    ModelFamily,
    default_load_in_4bit,
    full_finetuning,
    response_delimiters,
)


def test_family_values():
    assert ModelFamily.QWEN == "qwen"
    assert ModelFamily.GPT_OSS == "gpt_oss"


def test_qwen_response_delimiters_are_chatml():
    instruction, response = response_delimiters("qwen")
    assert instruction == "<|im_start|>user\n"
    assert response == "<|im_start|>assistant\n"


def test_gpt_oss_response_delimiters_are_harmony_final_channel():
    instruction, response = response_delimiters("gpt_oss")
    assert instruction == "<|start|>user<|message|>"
    assert response == "<|start|>assistant<|channel|>final<|message|>"
    # loss must start at the final channel, not the analysis channel
    assert "final" in response


def test_default_load_in_4bit_per_family():
    # gpt-oss = MoE-QLoRA 4-bit (1x H200); qwen dense = bf16 16-bit
    assert default_load_in_4bit("gpt_oss") is True
    assert default_load_in_4bit("qwen") is False


def test_full_finetuning_false_for_both():
    assert full_finetuning("qwen") is False
    assert full_finetuning("gpt_oss") is False


def test_enum_instance_accepted_not_just_str():
    assert default_load_in_4bit(ModelFamily.GPT_OSS) is True
    assert response_delimiters(ModelFamily.QWEN)[0] == "<|im_start|>user\n"


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        response_delimiters("llama")
    with pytest.raises(ValueError):
        default_load_in_4bit("mistral")

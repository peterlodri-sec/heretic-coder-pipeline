# shared/model_family.py — model-family-aware training knobs. Shipped with every
# stage (lives under shared/, scp'd to each remote) so the remote train scripts
# can import it GPU-free. Keeps the Qwen/32B validation baseline selectable next
# to the gpt-oss-120b frontier target instead of hard-coding one family.
from enum import StrEnum


class ModelFamily(StrEnum):
    QWEN = "qwen"
    GPT_OSS = "gpt_oss"


# (instruction_part, response_part) for Unsloth's train_on_responses_only prompt
# masking. QWEN = ChatML; GPT_OSS = harmony (mask up to the assistant `final`
# channel so loss lands on the answer, not the analysis channel).
_RESPONSE_DELIMITERS: dict[ModelFamily, tuple[str, str]] = {
    ModelFamily.QWEN: ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    ModelFamily.GPT_OSS: ("<|start|>user<|message|>",
                          "<|start|>assistant<|channel|>final<|message|>"),
}

# Default weight precision at load. GPT_OSS: 4-bit MoE-QLoRA (NF4-mimic of the
# native MXFP4) to fit 120B on 1x H200. QWEN: dense bf16 16-bit on H200 141GB.
_DEFAULT_LOAD_IN_4BIT: dict[ModelFamily, bool] = {
    ModelFamily.QWEN: False,
    ModelFamily.GPT_OSS: True,
}


def response_delimiters(family: str) -> tuple[str, str]:
    """(instruction_part, response_part) for train_on_responses_only."""
    return _RESPONSE_DELIMITERS[ModelFamily(family)]


def default_load_in_4bit(family: str) -> bool:
    """Whether to load base weights in 4-bit by default for this family."""
    return _DEFAULT_LOAD_IN_4BIT[ModelFamily(family)]


def full_finetuning(family: str) -> bool:
    """LoRA for both families — never full fine-tuning."""
    return False

# stage2/remote/sft_train.py — Unsloth + TRL SFT (plan.md §2). Heavy imports are
# function-local so the module imports without a GPU for unit tests.
import os

MAX_SEQ_LEN = 16384
# BFD example-packing: bin-pack multiple examples per row and fill to max_length,
# so real tokens (not padding) dominate every step -> ~2x less wall-clock at fixed
# quality (lora-speedrun record #1). "bfd" is example-boundary-aware over
# FlashAttention (block-diagonal via varlen position_ids) so packed examples do
# NOT attend across the seam -- unlike "wrapped", which mixes them and which we
# never want for long, structured agentic/code examples. Unsloth loads FA by
# default, which BFD requires. Off-switch: STAGE2_PACKING=0.
PACKING = os.environ.get("STAGE2_PACKING", "1") != "0"
PACKING_STRATEGY = "bfd"


def train(model_source: str, data_path: str, out_dir: str,
          max_steps: int = -1, num_epochs: int = 3,
          load_in_4bit: bool | None = None,
          family: str = "gpt_oss") -> tuple[float, object, object]:
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset
    from shared.model_family import default_load_in_4bit, response_delimiters
    from shared.train_common import load_lora_model

    # Family drives precision when the caller doesn't force it: gpt-oss -> 4-bit
    # MoE-QLoRA, dense qwen-32B -> bf16 16-bit. LoRA spec (r32/a64, MoE-safe
    # targets) is centralized in shared.train_common.
    if load_in_4bit is None:
        load_in_4bit = default_load_in_4bit(family)
    model, tokenizer = load_lora_model(
        model_source, max_seq_len=MAX_SEQ_LEN, load_in_4bit=load_in_4bit,
    )
    # Unsloth's patched SFTTrainer does NOT auto chat-template a `messages`
    # column (it raises "You must specify a formatting_func"). Render each
    # conversation to a `text` field with the tokenizer's chat template.
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(
        lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
    )

    trainer = SFTTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer,
        args=SFTConfig(
            dataset_text_field="text", max_length=MAX_SEQ_LEN,
            packing=PACKING, packing_strategy=PACKING_STRATEGY,
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            warmup_ratio=0.03, num_train_epochs=num_epochs, max_steps=max_steps,
            learning_rate=5e-5, bf16=True, lr_scheduler_type="cosine",
            optim="adamw_8bit", logging_steps=10, output_dir=out_dir,
        ),
    )
    # Mask the prompt so loss is computed on assistant responses only. Delimiters
    # are family-aware (ChatML for qwen, harmony final-channel for gpt-oss) —
    # replaces the hardcoded ChatML strings + TRL's assistant_only_loss.
    from unsloth.chat_templates import train_on_responses_only
    instruction_part, response_part = response_delimiters(family)
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )
    stats = trainer.train()
    return float(stats.training_loss), model, tokenizer

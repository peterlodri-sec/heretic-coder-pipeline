# stage3/remote/orpo_train.py — Unsloth + TRL ORPO (plan.md §3). Heavy imports
# are function-local so the module imports without a GPU for unit tests.
MAX_LENGTH = 8192
MAX_PROMPT_LENGTH = 2048


def train(model_source: str, data_path: str, out_dir: str,
          num_epochs: int = 1, load_in_4bit: bool | None = None,
          family: str = "gpt_oss") -> tuple[float, object, object]:
    from unsloth import PatchDPOTrainer
    from trl import ORPOConfig, ORPOTrainer
    from datasets import load_dataset
    from shared.model_family import default_load_in_4bit
    from shared.train_common import load_lora_model

    # Unsloth patches the DPO-family trainers (grad-checkpoint + concatenated
    # chosen/rejected forward) — MUST run before the model/trainer are built, else
    # `use_gradient_checkpointing="unsloth"` mispatches -> OOM/crash mid-train.
    PatchDPOTrainer()

    # Family drives precision when unset: gpt-oss ORPO MUST be 4-bit (MoE-QLoRA);
    # dense qwen-32B trains bf16 16-bit. r32/a64 LoRA spec in shared.train_common.
    if load_in_4bit is None:
        load_in_4bit = default_load_in_4bit(family)
    model, tokenizer = load_lora_model(
        model_source, max_seq_len=MAX_LENGTH, load_in_4bit=load_in_4bit,
    )
    # Conversational {prompt, chosen, rejected} triples are auto chat-templated
    # by ORPOTrainer (TRL 0.24.0).
    dataset = load_dataset("json", data_files=data_path, split="train")

    trainer = ORPOTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer,
        args=ORPOConfig(
            beta=0.1, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH,
            num_train_epochs=num_epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, learning_rate=5e-6, warmup_ratio=0.03,
            bf16=True,
            optim="adamw_8bit", lr_scheduler_type="cosine", logging_steps=10,
            output_dir=out_dir,
        ),
    )
    stats = trainer.train()
    # Return the live PEFT model + tokenizer so run_stage3 can export.
    return float(stats.training_loss), model, tokenizer

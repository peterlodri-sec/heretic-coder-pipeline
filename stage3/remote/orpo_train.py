# stage3/remote/orpo_train.py — Unsloth + TRL ORPO (plan.md §3). Heavy imports
# are function-local so the module imports without a GPU for unit tests.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
MAX_LENGTH = 8192
MAX_PROMPT_LENGTH = 2048


def train(model_source: str, data_path: str, out_dir: str,
          num_epochs: int = 1) -> tuple[float, object, object]:
    from unsloth import FastLanguageModel
    from trl import ORPOConfig, ORPOTrainer
    from datasets import load_dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source, max_seq_length=MAX_LENGTH,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    dataset = load_dataset("json", data_files=data_path, split="train")

    trainer = ORPOTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=ORPOConfig(
            learning_rate=5e-6, beta=0.1,
            max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH,
            num_train_epochs=num_epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, bf16=True, optim="adamw_8bit",
            lr_scheduler_type="cosine", logging_steps=10, output_dir=out_dir,
        ),
    )
    stats = trainer.train()
    # Return the live PEFT model + tokenizer so run_stage3 can export.
    return float(stats.training_loss), model, tokenizer

# stage2/remote/sft_train.py — Unsloth + TRL SFT (plan.md §2). Heavy imports are
# function-local so the module imports without a GPU for unit tests.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN = 16384


def train(model_source: str, data_path: str, out_dir: str,
          max_steps: int = -1, num_epochs: int = 3) -> tuple[float, object, object]:
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    dataset = load_dataset("json", data_files=data_path, split="train")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            warmup_ratio=0.03, num_train_epochs=num_epochs, max_steps=max_steps,
            learning_rate=2e-4, bf16=True, lr_scheduler_type="cosine",
            optim="adamw_8bit", logging_steps=10, packing=True,
            max_seq_length=MAX_SEQ_LEN, output_dir=out_dir,
        ),
    )
    stats = trainer.train()
    return float(stats.training_loss), model, tokenizer

# stage3/remote/orpo_train.py — Unsloth + TRL ORPO (plan.md §3). Heavy imports
# are function-local so the module imports without a GPU for unit tests.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
MAX_LENGTH = 8192
MAX_PROMPT_LENGTH = 2048


def train(model_source: str, data_path: str, out_dir: str,
          num_epochs: int = 1) -> tuple[float, object, object]:
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from trl import ORPOConfig, ORPOTrainer
    from datasets import load_dataset

    # Unsloth patches the DPO-family trainers (grad-checkpoint + concatenated
    # chosen/rejected forward) — MUST run before ORPOTrainer is built, else
    # `use_gradient_checkpointing="unsloth"` mispatches -> OOM/crash mid-train.
    PatchDPOTrainer()

    # H200 (141GB): train the 32B in bf16 (16-bit weights + LoRA) for quality —
    # no 4-bit quantization during ORPO.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source, max_seq_length=MAX_LENGTH,
        load_in_4bit=False, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0.0,  # parity w/ stage2 anti-regression fix
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth", random_state=42,
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

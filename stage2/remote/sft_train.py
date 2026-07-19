# stage2/remote/sft_train.py — Unsloth + TRL SFT (plan.md §2). Heavy imports are
# function-local so the module imports without a GPU for unit tests.
MAX_SEQ_LEN = 16384


def train(model_source: str, data_path: str, out_dir: str,
          max_steps: int = -1, num_epochs: int = 3,
          load_in_4bit: bool = False) -> tuple[float, object, object]:
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset
    from shared.train_common import load_lora_model

    # bf16 for the dense 32B (load_in_4bit=False); gpt-oss flips it True (MoE-QLoRA).
    # LoRA spec (r32/a64, MoE-safe targets) is centralized in shared.train_common.
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
            dataset_text_field="text", max_length=MAX_SEQ_LEN, packing=False,
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            warmup_ratio=0.03, num_train_epochs=num_epochs, max_steps=max_steps,
            learning_rate=5e-5, bf16=True, lr_scheduler_type="cosine",
            optim="adamw_8bit", logging_steps=10, output_dir=out_dir,
        ),
    )
    # Mask the prompt so loss is computed on assistant responses only
    # (Unsloth's ChatML response-only helper; replaces TRL's assistant_only_loss).
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    stats = trainer.train()
    return float(stats.training_loss), model, tokenizer

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

    # H200 (141GB): train the 32B in bf16 (16-bit weights + LoRA fit) for
    # quality — no 4-bit quantization during SFT.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, dtype=None,
    )
    # Gentler LoRA (r=32/alpha=64) to avoid over-writing the base model's coding
    # ability — the r=64/lr=2e-4 recipe caused a ~19.5% HumanEval regression.
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth", random_state=42,
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

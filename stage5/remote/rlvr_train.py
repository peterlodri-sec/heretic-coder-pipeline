# stage5/remote/rlvr_train.py — RLVR (execution-feedback RL) trainer, GSPO for the
# MoE base. Heavy imports (unsloth/trl/vllm) are function-local so this module
# imports without a GPU for unit tests. LoRA spec lives in shared.train_common.
MAX_SEQ_LEN = 8192


def train(model_source: str, data_path: str, out_dir: str,
          num_gpus: int = 2, num_epochs: int = 1,
          family: str = "gpt_oss") -> tuple[float, object, object]:
    """Train the terminal RLVR policy with verifiable code-execution rewards.

    ===== TRAINER (finalize from research) =====
    - TRL ``GRPOTrainer`` (group-relative policy optimization; DeepSeek-R1 /
      SWE-RL family). LoRA over LORA_TARGETS, gpt-oss 4-bit base, colocated vLLM
      generation for group rollouts across ``num_gpus`` (2-4x H200).
    - **GSPO is REQUIRED** because gpt-oss-120b is MoE: token-level GRPO
      importance ratios destabilize MoE training. Set GSPO via
      ``GRPOConfig(importance_sampling_level="sequence")`` (GSPO = sequence-level
      importance sampling, arXiv 2507.18071, Qwen3). Note this prominently — do
      NOT leave it at the token-level default for this model.
    - Alt tooling to EVALUATE before hand-rolling: **verifiers + prime-rl**
      (github.com/PrimeIntellect-ai/verifiers) — purpose-built sandboxed
      multi-turn RL environments; prefer over a bespoke reward/rollout loop.

    ===== REWARD (see reward.py; finalize from research) =====
    - Start: SWE-RL patch-similarity reward — ``difflib.SequenceMatcher`` ratio of
      the generated patch vs the oracle patch (arXiv 2502.18449; needs NO test
      harness, cheap to bootstrap).
    - Upgrade: real unit-test pass-rate via ``shared.exec_sandbox.run_tests``
      (reward = fraction of tests passing + format/compile shaping).
    - gpt-oss is a reasoning model → reward the harmony ``final``-channel answer
      (parse it out of the completion), not the analysis channel.

    Returns:
        (mean_reward_or_loss, model, tokenizer) — the live PEFT model + tokenizer
        so run_stage5 can merge/export (mirrors sft_train/orpo_train).
    """
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset
    from reward import code_execution_reward
    from shared.model_family import default_load_in_4bit, full_finetuning
    from shared.train_common import load_lora_model

    # Unsloth's RL patch — the GRPO analog of ORPO's PatchDPOTrainer; must run
    # before the model/trainer are built or the rollout+grad path mispatches.
    PatchFastRL("GRPO", FastLanguageModel)

    # Terminal gpt-oss stage: family resolves to 4-bit MoE-QLoRA (NF4). r32/a64
    # LoRA spec centralized in shared.train_common (RL is less forgetting-prone;
    # raise r if capacity-bound).
    model, tokenizer = load_lora_model(
        model_source, max_seq_len=MAX_SEQ_LEN,
        load_in_4bit=default_load_in_4bit(family),
        full_finetuning=full_finetuning(family),
    )

    # Dataset columns: `prompt` (+ `tests`/`oracle_patch` forwarded to the reward
    # via **kwargs by GRPOTrainer). No pre-templating — GRPO templates prompts.
    dataset = load_dataset("json", data_files=data_path, split="train")

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        reward_funcs=[code_execution_reward],
        args=GRPOConfig(
            # GSPO — REQUIRED for MoE gpt-oss (sequence-level importance ratio;
            # token-level GRPO collapses MoE routers). arXiv 2507.18071.
            importance_sampling_level="sequence", loss_type="grpo",
            beta=0.0, epsilon=3e-4, epsilon_high=4e-4,
            num_generations=8, max_completion_length=MAX_SEQ_LEN,
            # Colocated vLLM rollouts across num_gpus (2-4x H200). KV-cache kit
            # (prefix caching / fp8 kv / offload) = vLLM engine args — verify the
            # exact GRPOConfig passthrough on the pinned trl before the run.
            use_vllm=True, vllm_mode="colocate",
            gradient_accumulation_steps=1, steps_per_generation=4,
            learning_rate=1e-6, bf16=True, optim="adamw_8bit",
            num_train_epochs=num_epochs, logging_steps=10, output_dir=out_dir,
        ),
    )
    stats = trainer.train()
    return float(stats.training_loss), model, tokenizer

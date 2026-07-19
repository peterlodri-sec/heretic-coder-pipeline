# stage5/remote/rlvr_train.py — RLVR (execution-feedback RL) trainer. INTERFACE
# ONLY; the trainer is finalized from SOTA research (build-order step 3, see
# 2026-07-19 plan). Heavy imports (unsloth/trl/vllm) are function-local so this
# module imports without a GPU for unit tests. Constants mirror sft/orpo.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN = 8192


def train(model_source: str, data_path: str, out_dir: str,
          num_gpus: int = 2, num_epochs: int = 1) -> tuple[float, object, object]:
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
    # When implemented, heavy imports stay function-local (GPU-free module import):
    #   from unsloth import FastLanguageModel
    #   from trl import GRPOConfig, GRPOTrainer   # GRPOConfig(importance_sampling_level="sequence")
    #   from datasets import load_dataset
    raise NotImplementedError(
        "finalize GRPO/GSPO trainer from SOTA research — GSPO "
        "(importance_sampling_level='sequence') is REQUIRED for MoE gpt-oss; "
        "see 2026-07-19 plan")

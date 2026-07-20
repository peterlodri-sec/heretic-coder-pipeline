"""Shared model-loading for the LoRA trainers (sft/orpo/rlvr). Centralizes the
r32/a64 LoRA spec — the SFT anti-regression lesson (r64/lr2e-4 caused a ~19.5%
HumanEval drop) — in ONE place so it can't silently drift between stages again.
Heavy imports stay function-local so this module imports GPU-free for unit tests."""
from dataclasses import dataclass

LORA_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)  # router/mlp.gate intentionally excluded (MoE-safe)


@dataclass(frozen=True, slots=True)
class LoraSpec:
    """Gentle LoRA adapter — the proven anti-regression setting."""
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    targets: tuple[str, ...] = LORA_TARGETS
    # rsLoRA (rank-stabilized): scales the adapter by alpha/sqrt(r) instead of
    # alpha/r, which stabilizes higher ranks and is the intended way to get r>32
    # capacity back without the ~19.5% HumanEval regression plain r=64 caused.
    # OPT-IN ONLY: turning it on changes the effective scale (~5.6x at r=32/a=64:
    # 64/sqrt(32)=11.3 vs 64/32=2.0), so it needs a deliberate alpha/LR re-tune —
    # never flip it under the current alpha and expect the same run.
    use_rslora: bool = False


# Research-recommended high-capacity SFT config (deep-research brief, 2026): the
# earlier r=64 -> ~19.5% HumanEval regression was plain-LoRA GRADIENT COLLAPSE at
# r>=64 (alpha/r shrinks the update toward zero), which rsLoRA (alpha/sqrt(r))
# fixes directly. r=64/alpha=128 + rsLoRA is the recommended high-rank config —
# validate with an r={32,64,128} sweep before promoting it over the r=32 default.
HIGH_RANK_RSLORA = LoraSpec(r=64, alpha=128, use_rslora=True)


def load_lora_model(model_source: str, *, max_seq_len: int, load_in_4bit: bool,
                    lora: LoraSpec = LoraSpec(), full_finetuning: bool = False):
    """Load base weights + attach the LoRA adapter. Returns (model, tokenizer).

    load_in_4bit=True  -> gpt-oss MoE-QLoRA (NF4-mimic of MXFP4), fits 1x H200.
    load_in_4bit=False -> bf16 16-bit weights + LoRA (dense 32B on H200 141GB).
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source, max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit, dtype=None, full_finetuning=full_finetuning,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.targets), use_rslora=lora.use_rslora,
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    return model, tokenizer

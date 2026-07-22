"""Shared model-loading for the LoRA trainers (sft/orpo/rlvr). Centralizes the
r32/a64 LoRA spec — the SFT anti-regression lesson (r64/lr2e-4 caused a ~19.5%
HumanEval drop) — in ONE place so it can't silently drift between stages again.
Heavy imports stay function-local so this module imports GPU-free for unit tests."""
import os
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
                    lora: LoraSpec = LoraSpec(), full_finetuning: bool = False,
                    family: str = "gpt_oss"):
    """Load base weights + attach the LoRA adapter. Returns (model, tokenizer).

    load_in_4bit=True  -> gpt-oss MoE-QLoRA (NF4-mimic of MXFP4), fits 1x H200.
    load_in_4bit=False -> bf16 16-bit weights + LoRA (dense 32B on H200 141GB).

    Family-gated loader. gpt-oss goes through PLAIN transformers + bitsandbytes +
    PEFT, NOT Unsloth: two Stage-2 runs died at load with "Unsloth: Critical error
    since some weights are not initialized" because our abliterated checkpoint's
    per-layer `mlp.router.{weight,bias}` keys don't map under Unsloth's patched
    GptOss loader (they log as "not used when initializing GptOssForCausalLM").
    Plain transformers loads the *same* checkpoint correctly — the eval path
    (shared/eval, plain AutoModelForCausalLM) already proved it (refusal 0.0267,
    coherent output). Other families keep the working, faster Unsloth path.
    """
    match family:
        case "gpt_oss":  # fused-MoE + router keys don't map under Unsloth's loader
            loader = _load_plain_peft
        case _:  # dense families (qwen, …) — Unsloth is faster + leaner and works
            loader = _load_unsloth
    return loader(model_source, max_seq_len=max_seq_len, load_in_4bit=load_in_4bit,
                  lora=lora, full_finetuning=full_finetuning)


def _load_unsloth(model_source: str, *, max_seq_len: int, load_in_4bit: bool,
                  lora: LoraSpec, full_finetuning: bool):
    """Unsloth FastLanguageModel path — dense (qwen) families. Fast + memory-lean."""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source, max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit, dtype=None, full_finetuning=full_finetuning,
    )
    if full_finetuning:
        return model, tokenizer
    model = FastLanguageModel.get_peft_model(
        model, r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.targets), use_rslora=lora.use_rslora,
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    return model, tokenizer


def load_tokenizer(model_source: str):
    """AutoTokenizer with a fallback for checkpoints whose tokenizer_config names a
    tokenizer_class the pinned transformers can't resolve.

    Heretic/unsloth exports of gpt-oss write `"tokenizer_class": "TokenizersBackend"`,
    which AutoTokenizer rejects with "Tokenizer class TokenizersBackend does not
    exist or is not currently imported" — but the underlying tokenizer.json is a
    standard fast tokenizer, so we load PreTrainedTokenizerFast directly, bypassing
    the auto-class registry lookup (verified locally: vocab 199998, eos '<|return|>',
    chat_template intact, clean encode/decode roundtrip). Only that specific
    class-resolution failure is swallowed; any other tokenizer error re-raises.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(model_source)
    except Exception as error:  # noqa: BLE001 — narrowed by the message check
        if "does not exist or is not currently imported" not in str(error):
            raise
        return PreTrainedTokenizerFast.from_pretrained(model_source)


def _load_plain_peft(model_source: str, *, max_seq_len: int, load_in_4bit: bool,
                     lora: LoraSpec, full_finetuning: bool):
    """Plain transformers + bitsandbytes + PEFT path — the gpt-oss loader.

    Attention impl is env-selectable (STAGE2_ATTN, default 'eager'). gpt-oss
    (GptOssForCausalLM) does NOT support 'sdpa' in transformers 4.56.2 — it raises
    at load ("does not support ... scaled_dot_product_attention yet ... load with
    attn_implementation='eager'"), so 'eager' is the correct, no-extra-dep default.
    'flash_attention_2' is the fast option but needs flash-attn installed; it is
    also what BFD packing requires to keep packed examples from attending across
    the seam, so until flash-attn is added the first runs use STAGE2_PACKING=0 (the
    proven completion-masking path).

    gpt-oss MoE experts are a fused 3-D tensor (mlp.experts.gate_up_proj /
    down_proj), not per-expert nn.Linear modules, so PEFT can only attach LoRA to
    the attention projections (q/k/v/o_proj) — the mlp targets simply don't match
    and are skipped. That's a lighter but valid adaptation; expert-level LoRA on
    gpt-oss is out of scope here (it's what forced Heretic's direct-tensor surgery).
    """
    import torch
    from transformers import AutoModelForCausalLM

    attn = os.environ.get("STAGE2_ATTN", "eager")  # eager | flash_attention_2 (NOT sdpa for gpt-oss)
    tokenizer = load_tokenizer(model_source)
    if getattr(tokenizer, "model_max_length", None):
        tokenizer.model_max_length = max_seq_len

    # DeepSpeed ZeRO-3: shard the frozen bf16 base params ACROSS gpus (device_map
    # can't — it packs the ~133GB model onto one gpu and OOMs at training). Under
    # ZeRO-3 we load bf16 (no bnb — DeepSpeed shards real params, not bnb blobs)
    # with NO device_map, and keep an HfDeepSpeedConfig alive so from_pretrained
    # shards on load instead of each rank materializing the full 233GB.
    if os.environ.get("STAGE2_SHARDED") == "1":
        return _load_zero3_bf16(model_source, attn=attn, lora=lora,
                                full_finetuning=full_finetuning, tokenizer=tokenizer)

    # MXFP4 (STAGE2_MXFP4=1): quantize gpt-oss's fused MoE experts to their NATIVE
    # 4-bit MXFP4 on load — the format the model ships in. bnb can't touch the
    # fused experts (they stay bf16 -> ~133GB, won't fit training), but the mxfp4
    # quantizer does, tensor-by-tensor as it loads, so the resident model is ~63GB
    # and fits ONE H200 with ~75GB training headroom. Our checkpoint is bf16
    # (Heretic dequantized it); transformers quantizes bf16 -> mxfp4 on the fly.
    mxfp4 = os.environ.get("STAGE2_MXFP4") == "1"
    quant = None
    if mxfp4 and not full_finetuning:
        from transformers import Mxfp4Config
        quant = Mxfp4Config()  # quantize (not dequantize) the experts on load
    elif load_in_4bit and not full_finetuning:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model_kwargs = dict(
        quantization_config=quant, torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    # MXFP4 fits one GPU -> pin the whole model to cuda:0 (no split needed).
    # Otherwise device_map packs onto the GPU(s); multi-GPU device_map packs one
    # GPU and OOMs at training, so the big gpt-oss on many GPUs uses ZeRO-3 above.
    n_gpus = torch.cuda.device_count()
    if mxfp4:
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = "auto"
        if n_gpus > 1:
            model_kwargs["device_map"] = os.environ.get("STAGE2_DEVICE_MAP", "balanced")
            per_gpu = os.environ.get("STAGE2_MAX_MEM_GIB", "120")
            model_kwargs["max_memory"] = {i: f"{per_gpu}GiB" for i in range(n_gpus)}
            model_kwargs["max_memory"]["cpu"] = "0GiB"
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if full_finetuning:
        return model, tokenizer

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, _lora_config(lora))
    return model, tokenizer


DS_CONFIG_PATH = os.environ.get(
    "STAGE2_DS_CONFIG",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "stage2", "remote", "ds_zero3.json"),
)


def _lora_config(lora: "LoraSpec"):
    from peft import LoraConfig
    return LoraConfig(
        r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.targets), use_rslora=lora.use_rslora,
        bias="none", task_type="CAUSAL_LM",
    )


def _load_zero3_bf16(model_source: str, *, attn: str, lora: "LoraSpec",
                     full_finetuning: bool, tokenizer):
    """Load bf16 under DeepSpeed ZeRO-3 so from_pretrained shards params on load."""
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.integrations import HfDeepSpeedConfig

    # HfDeepSpeedConfig must exist (and stay referenced) BEFORE from_pretrained so
    # transformers' is_deepspeed_zero3_enabled() is true and zero.Init() shards the
    # weights across ranks as they load. Losing this ref re-materializes the full
    # model per rank -> OOM. We stash it on the model to keep it alive.
    dschf = HfDeepSpeedConfig(DS_CONFIG_PATH)  # noqa: F841 — keep-alive is the point
    model = AutoModelForCausalLM.from_pretrained(
        model_source, torch_dtype=torch.bfloat16, attn_implementation=attn,
    )
    model.config.use_cache = False
    model._dschf = dschf  # keep the zero3 config alive for the Trainer handoff
    if full_finetuning:
        return model, tokenizer
    from peft import get_peft_model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, _lora_config(lora))
    return model, tokenizer

# SWE Coder Fine-Tune Plan: Heretic + Unsloth + Tool Calling

## Goal

Produce a top-tier local SWE coding agent: uncensored, tool-call-accurate, strong on
SWE-bench. No mid-task refusals. Minimal capability degradation.

---

## Pipeline Overview

```
Qwen2.5-Coder-32B-Instruct (base)
          │
          ▼
      [1. Heretic]          weight surgery — abliterate refusal directions
          │
          ▼
      [2. Unsloth SFT]      tool calling + SWE trajectories
          │
          ▼
      [3. DPO / ORPO]       preference pairs — tool accuracy + code quality
          │
          ▼
      Final model (GGUF / safetensors)
```

**Order is non-negotiable.** Abliterate first; SFT second. Reversed order risks
the SFT data re-expressing refusal directions via gradient updates.

---

## Stage 1 — Heretic (Abliteration)

### What it does

Orthogonalizes attention out-projection and MLP down-projection matrices against
per-layer refusal direction vectors (computed as difference-of-means between
harmful/harmless prompt residuals). Pure weight surgery — no training, no gradients.

### Why for SWE agents

Agents writing exploits, touching kernel/system code, analyzing malware, or operating
aggressively in agentic loops will hit refusals mid-session without abliteration.
Heretic removes this at lower KL divergence than competing tools, preserving intelligence.

### Run

```bash
pip install -U heretic-llm

# Basic run — fully automatic
heretic Qwen/Qwen2.5-Coder-32B-Instruct

# With 4-bit quantization if VRAM constrained
heretic Qwen/Qwen2.5-Coder-32B-Instruct --config config.bnb4bit.toml
```

`config.bnb4bit.toml` override:
```toml
[model]
quantization = "bnb_4bit"
```

### Key metrics to watch

| Metric | Target |
|---|---|
| Refusals (harmful prompts) | < 5 / 100 |
| KL divergence (harmless prompts) | < 0.3 (lower = better) |
| MMLU / GSM8K delta | < 2% degradation |

Save the model locally after Heretic completes. This is the base for Unsloth SFT.

### Hardware

Run on Hetzner GPU node (A100 80GB recommended for 32B). RTX 3090 viable with
`bnb_4bit` quantization (~45 min for 8B; 32B will be longer).

---

## Stage 2 — Unsloth SFT

### Base

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./qwen25-coder-32b-heretic",  # local abliterated weights
    max_seq_length=16384,
    load_in_4bit=True,
    dtype=None,  # auto
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### Training config

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=16384,
    packing=True,  # sequence packing — no padding waste
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=10,
        output_dir="./swe-coder-sft",
    ),
)
```

---

## Dataset Stack

### Priority order (highest signal first)

1. **SWE-bench Verified gold trajectories**
   Resolved instances only. Format as multi-turn: tool calls + results + final patch.
   Source: `princeton-nlp/SWE-bench_Verified`

2. **BFCL (Berkeley Function Calling Leaderboard) formatted**
   Tool-call schema accuracy. Covers parallel calls, nested args, wrong-tool negatives.
   Source: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`

3. **ToolACE / ToolBench**
   Diverse tool-call instruction pairs. Filter to code-adjacent domains.

4. **OSS-Instruct / Magicoder synthetic**
   High-quality code instruction following. Prefer examples seeded from real GitHub code.
   Source: `ise-uiuc/Magicoder-OSS-Instruct-75K`

5. **crabcc agentic session traces** *(highest specificity)*
   Your own Claude Code session logs. Real agent trajectories — tool calls, bash, file edits,
   error recovery. Convert to training format. These are the most valuable examples.

### Data hygiene — critical

- **Scrub RLHF-contaminated synthetic data.** ShareGPT, Alpaca, and similar datasets contain
  latent refusal signal from RLHF-tuned generator models. Post-abliteration SFT on these
  examples will partially re-introduce refusal directions via gradient updates.
  Either exclude them entirely or weight them at 0.1x vs. code-native data.
- **Tool-call negative examples are required.** Include examples of: wrong tool chosen,
  malformed args, refusal-when-no-tool-needed. Without negatives the model learns to
  always call tools.
- **Multi-turn with tool results.** Don't truncate tool result context. The model must see
  the full observation before generating the next action.

### Tool call schema

Pick one format and apply it consistently across all data:

```
Hermes-style (recommended for Qwen):

<tool_call>
{"name": "bash", "arguments": {"cmd": "cargo test --workspace"}}
</tool_call>

<tool_response>
{"output": "test result: ok. 42 passed; 0 failed"}
</tool_response>
```

If using native ChatML function blocks instead, apply that across all data uniformly.
Mixing schemas will break tool calling at inference.

---

## Stage 3 — DPO / ORPO

Run after SFT. Preference pairs improve tool-call accuracy and code quality without
re-introducing refusal behavior (the refusal direction is already ablated in the weights).

### Pair construction

For each SWE task / tool-call scenario:

- **Chosen**: correct tool, correct args, correct output
- **Rejected**: wrong tool / malformed args / hallucinated result / unnecessary refusal

### ORPO (preferred over DPO for this pipeline)

ORPO combines SFT and preference optimization in one pass — avoids the reference model
overhead and is more stable on LoRA. Use `trl.ORPOTrainer`.

```python
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    learning_rate=5e-6,
    beta=0.1,          # odds ratio weight
    max_length=8192,
    max_prompt_length=2048,
    num_train_epochs=1,
    bf16=True,
    output_dir="./swe-coder-orpo",
)
```

---

## Evaluation

| Benchmark | Measures | Target |
|---|---|---|
| SWE-bench Verified resolve rate | Primary SWE capability | > 40% (SOTA ~50%) |
| BFCL tool accuracy | Tool call correctness | > 85% |
| HumanEval / MBPP | Code generation sanity | < 3% regression vs. base |
| Refusal rate (Heretic eval) | Abliteration held post-SFT | < 10 / 100 |

Run refusal eval after every stage (post-Heretic, post-SFT, post-DPO) to confirm
SFT has not re-introduced refusals.

```bash
heretic --model ./swe-coder-sft --evaluate-model ./swe-coder-sft
```

---

## Hardware Plan

| Stage | Machine | Notes |
|---|---|---|
| Heretic | Hetzner A100 80GB | 32B needs full VRAM; use `bnb_4bit` on smaller GPUs |
| Unsloth SFT | Hetzner A100 80GB | bf16 + 4bit LoRA; sequence packing enabled |
| DPO / ORPO | Hetzner A100 80GB | Lower batch size; preference pairs are longer |
| Iteration / eval | M3 Pro (MLX) | Quick inference checks, not training |
| Inference serving | Hetzner + Tailscale | llama.cpp GGUF or vllm safetensors |

---

## Export

```python
# Merge LoRA into base, save as safetensors
model.save_pretrained_merged(
    "./swe-coder-final",
    tokenizer,
    save_method="merged_16bit",
)

# GGUF for llama.cpp / local serving
model.save_pretrained_gguf(
    "./swe-coder-final-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

---

## Risk Register

| Risk | Mitigation |
|---|---|
| SFT re-introduces refusals | Scrub RLHF-contaminated data; run refusal eval post-SFT |
| Heretic damages tool-call format understanding | Verify with BFCL pre/post abliteration |
| Schema inconsistency across datasets | Single normalisation script before any training |
| KL divergence too high (capability loss) | Tune Heretic `max_weight` parameter; run Optuna longer |
| LoRA rank too low for 32B | Start r=64; bump to r=128 if SWE-bench plateaus |

---
license: apache-2.0
base_model: PeetPedro/qwen2.5-coder-32b-instruct-heretic
tags:
  - code
  - abliterated
  - heretic
  - sft
  - unsloth
pipeline_tag: text-generation
---

# qwen2.5-coder-32b-instruct-heretic-sft

Stage 2 of the [heretic-coder-pipeline](https://github.com/entropy-om/heretic-coder-pipeline):
LoRA SFT (Unsloth + TRL) of the [abliterated 32B base](qwen2.5-coder-32b-instruct-heretic.md)
on agentic SWE + tool-calling data. Serves as the **validation baseline** that
de-risks the gpt-oss-120b frontier run.

- **Base:** PeetPedro/qwen2.5-coder-32b-instruct-heretic
- **Method:** Unsloth LoRA SFT (r=32/α=64, response-only masking), bf16 on 1×H200
- **Data:** tool-calling + SWE trajectories (Hermes schema), contamination-filtered

## Evaluation (gentle re-tune, LR 5e-5 / r32)

| Metric | Value | Gate | Note |
|---|---|---|---|
| refusal_rate | **0.0067** | <0.10 ✅ | decensoring held (0.67% refusals) |
| humaneval_delta | 0.0915 | <0.03 ❌ | 9.1% coding regression (halved from 19.5% at r64/2e-4) |
| bfcl_accuracy | 0.317 | >0.85 ❌ | *exact-match harness; threshold under review* |
| swebench_resolve | — | >0.40 | not run this pass |

**Honest read:** abliteration is excellent; SFT on tool/instruction data still
partially forgets coding (a known LoRA-SFT failure mode). The pipeline's **RFT +
RLVR** stages exist precisely to *recover* coding via execution feedback, and the
BFCL gate (exact name+arg-dict match vs a 0.85 floor) is flagged as likely
miscalibrated. This baseline's job — proving the harness and surfacing recipe
lessons — is done.

## Risks & limitations
Safety guardrails **removed** (see refusal rate). General-purpose non-refusal, not
coding-only. Use responsibly and only where authorized. Weights private/gated.

## Links
- Pipeline: <https://github.com/entropy-om/heretic-coder-pipeline>
- Previous stage: [qwen2.5-coder-32b-instruct-heretic](qwen2.5-coder-32b-instruct-heretic.md)
- Frontier target: [gpt-oss-120b-heretic](gpt-oss-120b-heretic.md)

---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-32B-Instruct
tags:
  - code
  - abliterated
  - heretic
pipeline_tag: text-generation
---

# qwen2.5-coder-32b-instruct-heretic

Stage 1 of the [heretic-coder-pipeline](https://github.com/peterlodri-sec/heretic-coder-pipeline):
`Qwen/Qwen2.5-Coder-32B-Instruct` with refusal directions abliterated via
[heretic](https://github.com/p-e-w/heretic) — gradient-free weight surgery
(orthogonalizing `attn.o_proj` + `mlp.down_proj` against per-layer refusal
directions found by an Optuna study). No fine-tuning at this stage.

- **Base:** Qwen/Qwen2.5-Coder-32B-Instruct (Apache-2.0)
- **Method:** heretic abliteration (weight surgery, no gradients)
- **Role:** validation baseline for the pipeline; input to the SFT stage
  ([→ …-heretic-sft](qwen2.5-coder-32b-instruct-heretic-sft.md))

## Intended use
Research into abliteration + post-training for authorized security-tooling and
coding work. Not a general-purpose assistant.

## Risks & limitations
Safety guardrails are **removed** — the model will attempt requests a stock model
refuses, across all domains, not just coding. Use responsibly and only where you
are authorized to. Weights are private/gated.

## Links
- Pipeline: <https://github.com/peterlodri-sec/heretic-coder-pipeline>
- Next stage: [qwen2.5-coder-32b-instruct-heretic-sft](qwen2.5-coder-32b-instruct-heretic-sft.md)
- Frontier target: [gpt-oss-120b-heretic](gpt-oss-120b-heretic.md)

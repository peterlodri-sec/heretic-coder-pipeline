---
license: apache-2.0
base_model: openai/gpt-oss-120b
tags:
  - code
  - moe
  - abliterated
  - heretic
  - gspo
  - rlvr
pipeline_tag: text-generation
---

# gpt-oss-120b-heretic (frontier chain)

The frontier target of the [heretic-coder-pipeline](https://github.com/entropy-om/heretic-coder-pipeline):
`openai/gpt-oss-120b` (117B total / 5.1B active MoE, harmony format, Apache-2.0)
put through the full chain. **Status: configured, pending run** — the harness is
built and verified (301 tests); numbers below are populated after the run.

## Chain & output repos
| Stage | Method | Repo |
|---|---|---|
| 1 Heretic | abliteration (bnb_4bit weight surgery) | `…/gpt-oss-120b-heretic` |
| 2 SFT | Unsloth MoE-QLoRA, harmony format | `…-heretic-sft` |
| 3 RFT | rejection-sampling: sample → exec-verify → SFT on passers | `…-heretic-rft` |
| 4 RLVR | TRL GRPO + **GSPO** (MoE-stable), reward = tests pass | `…-heretic-rlvr` |

## Method notes
- **GSPO** (`importance_sampling_level="sequence"`) is required — token-level GRPO
  collapses the MoE routers (arXiv 2507.18071).
- Reward = sandboxed unit-test pass-rate (+ SWE-RL patch-similarity bootstrap),
  with hidden-holdout tests to resist reward hacking.
- Harmony-aware throughout (dataprep tool-call encoding, `final`-channel eval).
- Cost-aware: interruptible instances, RFT-then-distill default, FP8-KV rollouts.

## Evaluation
_Pending the run._ Gated on refusal_rate / humaneval_delta / bfcl_accuracy /
swebench_resolve (see [pipeline verdict](https://github.com/entropy-om/heretic-coder-pipeline#verdict-gate)).

## Risks & limitations
Safety guardrails **removed** — general-purpose non-refusal, capable, and agentic.
Higher risk than a plain decensored chat model; use responsibly and only where
authorized. Weights private/gated.

## Links
- Pipeline: <https://github.com/entropy-om/heretic-coder-pipeline>
- Validation baseline: [qwen2.5-coder-32b-instruct-heretic-sft](qwen2.5-coder-32b-instruct-heretic-sft.md)

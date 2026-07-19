# Extended pipeline — add SOTA stages (RLVR + self-improvement) for gpt-oss-120b

Extends the base pipeline (1 heretic → 2 SFT → 3 ORPO) with the highest-impact
SOTA post-training techniques for a coding agent. Target: gpt-oss-120b (single
H200 QLoRA, harmony format). Technique internals to be finalized against the
SOTA research (dispatched 2026-07-19); this plan fixes the STAGE STRUCTURE and
integration so the stages exist and slot into the orchestrator.

Full pipeline (CORRECTED per 2026-07-19 SOTA research):
`1 heretic → 2 SFT → 4 RFT-loop → 5 RLVR (terminal, REPLACES ORPO) → [test-time scaffold at serve]`

Key research corrections vs the first draft:
- **RFT / rejection-sampling self-improvement runs BEFORE RL** (best impact-per-
  effort; 1× H200). So stage4 = RFT loop, stage5 = RLVR (numeric order = pipeline
  order). Swapped from the initial draft.
- **RLVR REPLACES ORPO** as the terminal stage (RLVR > preference-opt when a
  verifier exists). stage3 ORPO kept as budget fallback, not in the default
  gpt-oss chain.
- **Use GSPO** (`importance_sampling_level="sequence"`), NOT vanilla GRPO —
  REQUIRED because gpt-oss-120b is MoE; token-level GRPO destabilizes on MoE
  (arXiv 2507.18071, Qwen3). One-flag change, highest-value single insight.
- **RLVR realistically needs 2–4× H200** (colocated vLLM gen + LoRA train + long
  agentic rollouts). RFT loop stays 1× H200.
- Prefer **verifiers + prime-rl** (PrimeIntellect-ai) for the sandboxed multi-turn
  reward over hand-rolling. Reward: SWE-RL patch-similarity (difflib, arXiv
  2502.18449) → upgrade to real unit-test pass-rate.
- SFT-stage upgrades to layer in: rsLoRA r=64 + PiSSA init + NEFTune + sample
  packing; harmony format; decontaminated SWE-Gym/OpenHands agentic trajectories.

## Gemini second-opinion — folded-in refinements (2026-07-19)
Independent review confirmed GSPO + RFT-before-RL + RLVR-replaces-ORPO. Deltas to
bake into stage5 internals:

1. **120B RLVR bottleneck = KV cache, NOT weights.** 4-bit 120B ≈65 GB; LoRA
   shares base weights (ref==policy, only adapter deltas), GSPO drops the value
   model. But 32k+ agentic contexts blow KV cache across the H200s → tiny rollout
   batch → crippled exploration. **Decision point for the RLVR stage (pick at run
   time):**
   - (a) **RFT on 120B → distill to a dense 32B** (generate execution-filtered
     traces on 120B, SFT the 32B) — sidesteps KV-cache entirely. Likely best.
   - (b) **Offline KTO on 120B** using execution-labeled data from the RFT phase —
     avoids live KV-heavy rollouts; the ONE case we'd keep an offline preference
     stage (see #5 caveat), purely as a 120B compute concession.
   - (c) Live GSPO-RLVR on 2–4× H200 with short-horizon single-file tasks + small
     group size. Highest ceiling, highest cost/risk.
   stage5 controller must parametrize this (mode = distill | offline-kto | live-rl).
   **KV-cache survival kit for live-RL mode (c)** — the bottleneck is the DENSE KV
   cache (MoE sparsity cuts compute, not KV; 32k ctx ≈5–10 GB/seq FP16, OOMs before
   compute). To fit rollouts on 2–4× H200, the vLLM/SGLang rollout engine MUST set:
   - **Automatic Prefix Caching** (vLLM APC / SGLang RadixAttention) — N rollouts
     share the SAME 32k prompt+env+tests; compute that KV once, share the blocks.
     O(N·prompt)→O(prompt). Lets group size go 2→16+ (huge gradient-quality win).
     Highest leverage — turn on first.
   - **FP8 KV cache** (E4M3/E5M2), NOT INT8 — halves KV mem, <1% degradation;
     preserves attention precision for exact syntax. INT8 costs 1.5–3 pts on
     long-ctx. Pairs with the 4-bit NF4 weights.
   - **MoE-aware PagedAttention + eviction** — PagedEviction / entropy-guided evict
     of low-importance blocks (stale tracebacks); PiKV expert-sharded KV to avoid
     cross-GPU fetch overhead on MoE routing.
   - **CPU/host-RAM offload** (LMCache / vLLM offload) — overflow buffer for
     >32k edge-case rollouts; +10–50ms/retrieval but prevents an OOM that kills a
     12h job. Set as safety net, not primary.
   stage5 rollout config exposes: enable_prefix_caching, kv_cache_dtype="fp8",
   enable_chunked/eviction, cpu_offload_gb.
2. **Non-hackable execution reward — TIERED (stage5 reward.py):**
   lint/parse +0.1 · compiles +0.2 · **visible tests pass +1.0** · efficiency bonus
   +0.2. Anti-Goodhart hardening (→ shared/exec_sandbox.py):
   - **Immutable test suite** — mount tests + runner strictly READ-ONLY (models
     rewrite pytest to `return True`).
   - **Network isolation** — block all outbound (no curl/phone-home bypass).
   - **Hidden holdout tests** — score against a hidden set too; pass-visible /
     fail-hidden → heavy **−1.0** penalty (punishes hardcoded returns).
   - Prefer patch-similarity ONLY as the cold-start reward; move to tiered exec ASAP.
3. **Harmony format under RL** — RL will drop/mangle the `analysis` channel to save
   tokens. Mitigate in stage5: **format penalty −1.0 + skip execution** when the
   AST/regex parser finds missing/malformed/out-of-order `analysis`/`final` tags;
   OR **constrained decoding** (Outlines/SGLang) forcing the structural tags so RL
   only fills text between guaranteed tags. Bake a format-check gate into reward.py.
4. **GSPO rationale (confirmed):** token-level GRPO gives per-token credit → high-
   variance noise to the router → expert collapse. GSPO's sequence-level ratio +
   clipping matches reward granularity (whole patch) → stable routers, no routing-
   replay hack. Keep `importance_sampling_level="sequence"`.
5. **Skip the offline preference stage by default** — KTO/DPO pairs are noise-
   sensitive (a missing-comma failure becomes a "rejected" with 99%-correct logic →
   confusing gradients). RFT→RLVR gives exact on-policy success demos; RL then
   spends its budget on exploration/consistency, not relearning syntax. (Exception:
   #1b offline-KTO, only as the 120B KV-cache concession.)

## Reuse
All new stages mirror stage1-3: a `stageN/` package (`controller.py`,
`enums.py`, `status_io.py`, `verdict.py`, `conftest.py`, `remote/`), reusing
`shared` (ssh_utils, vast_ops/HF-Jobs launcher, poll, status, verdict, export,
dataprep loaders, eval). They plug into `pipeline/config.py` (add to `STAGES`,
chain each stage's HF output → next stage's input model).

## Stage 4 — RLVR (execution-feedback RL) — the biggest lever
Reinforcement learning with **verifiable rewards**: sample solutions from the
model, **run the code against tests**, reward = tests pass. (DeepSeek-R1 GRPO /
Meta SWE-RL family.)
- **Trainer:** TRL `GRPOTrainer` (+ Unsloth GRPO support) — LoRA, gpt-oss 4-bit,
  1× H200 (verify vRAM at rollout time; may need h200x2 for group rollouts).
- **Reward fn (technique-specific — from research):** a sandboxed code-execution
  reward — run generated code/patch against unit tests in an isolated subprocess
  (resource + time limited, no network), reward = fraction of tests passing (+
  format/compile shaping). SECURITY: executes model-generated code — must be
  hard-sandboxed (container/nsjail/firejail, no net, cpu/mem/time caps).
- **Data:** coding problems WITH tests (e.g. verifiable subsets — MBPP+/
  HumanEval+/ LiveCodeBench-style, SWE-Gym repo tasks with FAIL_TO_PASS). Reuse
  `shared.dataprep` loaders; add a verifiable-coding source.
- **Verdict:** reuse shared gates (refusal held, HumanEval/BFCL up, SWE-bench).
- `stage4/remote/rlvr_train.py` (GRPOTrainer + reward), `reward.py` (sandboxed
  exec), config, run_stage4.py orchestration.

## Stage 5 — Self-improvement loop (STaR / ReST / RFT)
Generate N solutions per problem → keep the ones that pass tests → SFT on the
passing set → repeat K rounds. Cheap, compounding, complements RLVR.
- Reuses stage-4's exec-verifier for filtering + stage-2's SFT trainer for the
  SFT-on-correct step. Loop controller.
- `stage5/` = a loop over: generate (vLLM/HF) → verify (exec) → SFT (Unsloth).

## Test-time scaffold (serve-time, NOT a training stage)
best-of-N + verifier, self-repair/reflexion, agentic scaffold (SWE-agent/
OpenHands). Co-design: the RLVR/self-improve data should match the agent loop.
Lives in the eval/serve path (`shared/eval` + a serving harness), not a training
stage. Document + wire into the SWE-bench eval (agentic patch-gen already needed).

## Shared additions
- `shared/exec_sandbox.py` — the isolated code-execution primitive (used by
  stage 4 reward + stage 5 verifier). Single hardened implementation.
- `shared/dataprep/loaders.py` — add verifiable-coding-with-tests sources.

## Build order (grounded on the SOTA research)
1. Scaffold `stage4/` + `stage5/` packages (pattern + pipeline integration + tests) — NOW.
2. `shared/exec_sandbox.py` (hardened) — from research (sandbox choice).
3. Stage-4 GRPO reward + trainer (TRL GRPOTrainer / Unsloth GRPO) — from research.
4. Stage-5 loop.
5. Test-time scaffold in the eval/serve path.

## Notes
- gpt-oss is a reasoning model → RLVR on the `final`-channel answer, reward the
  outcome; consider a process/step reward later. Harmony format applies (parse
  `final` channel for the code/answer to execute).
- Do NOT build the exec-sandbox or GRPO reward on assumptions — the research
  fixes the tooling (TRL GRPO gpt-oss support, sandbox lib, reward shaping).

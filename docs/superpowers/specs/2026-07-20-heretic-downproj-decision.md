# Decision — heretic `down_proj` regression on gpt-oss-120b

## What happened (first real run, 2026-07-19/20)
Abliteration completed 200 trials (bf16, 2×H200) but the log showed
`LoRA adapters initialized (target types: o_proj)` — heretic abliterated ONLY
`attn.o_proj` and **silently skipped the MoE experts' `down_proj`**. Result was a
mild softening: refusals 100→~62/100, **KL 0.016** (≈ unchanged model). Eval then
OOM'd (separate bug, since fixed).

## Root cause (verified vs heretic tagged source + transformers)
- gpt-oss experts are **fused 3-D `nn.Parameter` tensors** — `layer.mlp.experts.down_proj`
  (shape `[num_experts, inter, hidden]`), NOT per-expert `nn.Linear` modules.
  `GptOssMLP.experts` is a single `GptOssExperts` module, not iterable.
- Our pin (heretic master @ `e7b783e`) discovers targets with `get_layer_modules`
  and abliterates via **PEFT LoRA adapters** — LoRA can only wrap `nn.Linear`, never
  a bare `Parameter`. Its gpt-oss expert branches all raise and are swallowed by
  `suppress(Exception)`. So only `o_proj` (a real `nn.Linear`) is found.
- **It's a version regression, not an arch/quant limit.** heretic **v1.0.x / v1.1.0**
  used direct-tensor surgery (`get_layer_matrices`) with an explicit branch
  `try_add("mlp.down_proj", layer.mlp.experts.down_proj)` ("all experts in a single
  3D tensor") — that's how `p-e-w/gpt-oss-20b-heretic` got real `down_proj` params.
  **v1.2.0** refactored to LoRA-on-Modules and dropped the fused-expert branch.

## The fix (FIX A — recommended, bounded, ~0.5–1 day)
Pin heretic to **v1.1.0** (newest release still doing direct-tensor surgery with the
fused-expert branch). Harness changes needed:
- `stage{1,frontier}/remote/setup.sh`: pin `heretic-llm @ v1.1.0` (drop the master SHA).
- v1.1.0 has **no** `--checkpoint-action/--trial-index/--model-action/--save-directory/
  --study-checkpoint-dir/--export-strategy` flags and **no** `quantization`/`max_memory`
  config fields. So: drop those 5 CLI flags in `run_stage1.run_heretic`; keep **bf16**
  (already our path); shard via `device_map = "auto"` in `config.toml` (v1.1.0 has
  `device_map`, no `max_memory`).
- v1.1.0 prompts interactively (`questionary`) for save/upload → the controller must
  feed answers on **stdin** (`subprocess.run(..., input=...)` or `pexpect`), not flags.
- Everything else (2×H200, BF16 source `unsloth/gpt-oss-120b-BF16`) unchanged.
The abliteration ENGINE needs no patching — only the harness plumbing.

FIX B (fork master to re-add a Parameter path) = high effort, upstream-diverging —
NOT recommended; file an upstream issue instead.

## Recommendation
o_proj-only is a **regression**, and for an MoE the behavior lives in the experts —
KL 0.016 is the signature of a barely-changed model. If heretic is meant to do real
decensoring, **do FIX A and re-run** (~9.5 h). If SFT/ORPO downstream is expected to
do the heavy lifting, accepting o_proj-only as a mild softener is defensible and
skips the v1.1.0 plumbing. Honest caveat: 20b-with-experts hit 58/100 vs our 62/100,
so the refusal-rate delta may be modest; the real gain is decensoring **depth**.

## Status
- Model preserved on the stopped box (45298230, disk intact) — nothing lost.
- Eval OOM fixed + committed (shard 120B + sequential base/candidate free).
- Awaiting user decision: **(1) accept o_proj-only + proceed to SFT**, or
  **(2) FIX A (pin v1.1.0) + re-run for expert-level abliteration**.

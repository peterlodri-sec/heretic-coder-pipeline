# Frontier build — Qwen3-Coder-480B-A35B (heretic → SFT → ORPO, multi-GPU)

Target: `Qwen/Qwen3-Coder-480B-A35B-Instruct` (qwen3_moe, 480B/35B active, 62 layers,
160 experts / 8 active, no shared expert, Apache-2.0). Goal: the strongest possible
open coder. Hardware: 8×H200 single node (1128 GB; bf16 base ≈960 GB fits). Budget: n/a.

Design verified against Axolotl / LLaMA-Factory / vLLM / Heretic / Vast primary
sources (see 2026-07-18 research). This plan is the buildable form.

## Reuse vs. change
**Reused unchanged:** `shared/dataprep/*`, eval LOGIC in `shared/eval/*` (only the
generation transport swaps to vLLM), `shared/verdict.py`, `shared/export.py`,
`shared/status.py`, `shared/poll.py`, `shared/ssh_utils.py`, `shared/vast_ops.py`,
the controller provision→deploy→poll→stop skeleton.
**Changed/new:** trainer scripts (Axolotl SFT YAML, LLaMA-Factory ORPO YAML),
multi-GPU launch, provisioning (num_gpus=8, disk≥2000, cuda_vers>=12.4, NCCL/shm),
bf16 + 4bit-quantized frozen experts (not 4bit-everything), vLLM-EP eval/serve path,
heretic bnb_4bit + device_map config, drop GGUF from the serve path.

## Directory (new `frontier/` alongside stage1-3; reuses shared/)
```
frontier/
  controller.py            # multi-GPU controller (num_gpus=8, tmux multi-GPU launch)
  remote/
    setup.sh               # axolotl + llamafactory + vllm + heretic + NCCL env
    heretic_config.toml     # stage 1: bnb_4bit, n_trials=60, max_memory 8-GPU
    sft_axolotl.yaml        # stage 2: qwen3_moe QLoRA, router-safe targets, FSDP2
    orpo_llamafactory.yaml  # stage 3: native ORPO, ZeRO-3, QLoRA
    run_frontier.py         # orchestrates heretic->sft->orpo->eval->verdict->publish
    dataset_info.json       # LLaMA-Factory preference-dataset registration
  tests/
```

## Phase 1 — configs (concrete, low-risk; write first)
### stage 2 SFT (Axolotl) — router-safe LoRA
`lora_target_modules: [q_proj,k_proj,v_proj,o_proj, gate_proj,up_proj,down_proj]`
(NOT `lora_target_linear` — that hits the router `mlp.gate`). `adapter: qlora`,
`load_in_4bit: true`, `quantize_moe_experts: true`, `bf16: true`, `sequence_len: 8192`,
`sample_packing: true`, `chat_template: tokenizer_default`, dataset `type: chat_template`
`field_messages: messages`, FSDP2 wrap `Qwen3MoeDecoderLayer`, micro_batch 1 / grad_accum 8,
2 epochs, lr 1e-4. Launch: `accelerate launch --num_processes 8 -m axolotl.cli.train ...`.
### stage 3 ORPO (LLaMA-Factory)
`stage: dpo`, `pref_loss: orpo`, `pref_beta: 0.1`, `finetuning_type: lora`,
`lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`,
`quantization_bit: 4`, `template: qwen3_nothink`, `deepspeed: ds_z3_config.json`.
Dataset registered in `dataset_info.json` (sharegpt, ranking: true; chosen=compliant,
rejected=refusal). Launch: `llamafactory-cli train ...`.
### stage 1 Heretic
`config.toml`: `quantization="bnb_4bit"`, `n_trials=60`, `n_startup_trials=30`,
`max_memory` 130GB×8 + cpu. `heretic Qwen/Qwen3-Coder-480B-A35B-Instruct` (reads toml).
GO/NO-GO decided at runtime: attempt once; if qwen3_moe unsupported at 480B OR
HumanEval/SWE-bench regress → drop stage 1, de-align via ORPO data only.

## Phase 2 — multi-GPU provisioning + controller
- `frontier/controller.py`: like the stage controllers but provision
  `gpu_name=H200 num_gpus=8 disk_space>=2000 cuda_vers>=12.4 reliability>0.98`,
  `disk_gb=2000`; deploy ships shared+frontier; launch the appropriate multi-GPU
  command in tmux per stage; reuse `poll_until_done` + `status.json` + stop-on-exit.
- NCCL/shm env in setup/onstart: `NCCL_IB_DISABLE=1`, ensure /dev/shm; fallback
  `NCCL_P2P_DISABLE=1` if init hangs on non-NVLink hosts.
- Keep the provision_lock (double-rent) + try/finally stop (billing) fixes.

## Phase 3 — eval/serve via vLLM-EP
- Stand up `vllm serve Qwen3-Coder-480B ... --enable-expert-parallel --tensor-parallel-size 8
  --enable-auto-tool-choice --tool-call-parser qwen3_coder` (merge LoRA first).
- Add a vLLM/OpenAI-client transport to `shared/eval/_model.py` (env-toggled): keep
  `apply_chat_template`/tools/normalization logic, swap `chat_generate` to call the
  endpoint (batched). device_map="auto" stays as the fallback. Turns multi-hour eval
  into minutes at 480B.

## Phase 4 — run
heretic (attempt) → SFT (Axolotl 8×H200) → merge LoRA → ORPO (LLaMA-Factory) → merge →
eval (vLLM-EP: refusal/bfcl/humaneval/swebench) → verdict → publish `...-480b-heretic-sft-orpo`.

## On-box unverified items to confirm BEFORE the expensive run
1. Axolotl DeepEP/ScatterMoE-LoRA on an official Qwen3-480B example (EP is an
   optimization; first run = FSDP2 + quantize_moe_experts only — do not gate on EP).
2. Exact Axolotl `fsdp_config` key spelling on the installed version.
3. Heretic qwen3_moe hook coverage at 480B + `--max-memory` CLI encoding (`heretic --help`).
4. Live 8×H200 availability/price on Vast (thin supply — may need to wait/widen).
5. Validate each config with a tiny/quantized dry-run on the box before full weights
   (mirror the stage-2 probe approach: prove the pattern cheaply first).

## Sequencing
Let the 32B stage2/3 finish first (baseline + proves the shape). Build Phases 1-3 in
parallel now. Provision 8×H200 + run (Phase 4) once the harness is ready AND the 32B
baseline confirms the data/eval/verdict machinery is sound.

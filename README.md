# qwen2.5-coder-32b-instruct → heretic → SFT → ORPO

A three-stage training pipeline that turns **Qwen/Qwen2.5-Coder-32B-Instruct**
into a top-tier local SWE coding agent: uncensored, tool-call-accurate, minimal
capability degradation. Each stage is a self-contained [Vast.ai](https://vast.ai)
GPU harness; a top-level orchestrator chains them.

> **Status: harness only.** Every stage is built and unit-tested on CPU (heavy
> training/eval libs are mocked). No stage has been run on a GPU yet. See
> [Before a real run](#before-a-real-run).

## Pipeline

```
Qwen2.5-Coder-32B-Instruct (base)
        │
        ▼
   [1] Heretic          weight surgery — abliterate refusal directions
        │  → PeetPedro/qwen2.5-coder-32b-instruct-heretic
        ▼
   [2] Unsloth SFT      tool calling + SWE trajectories (LoRA)
        │  → …-heretic-sft
        ▼
   [3] ORPO             preference pairs — tool accuracy + code quality
        │  → …-heretic-orpo   (safetensors + q4_k_m GGUF)
        ▼
   Final model
```

Design rationale and full detail: [`plan.md`](plan.md) and the specs/plans under
[`docs/superpowers/`](docs/superpowers/).

| Stage | Dir | Method | Input model | Output repo |
|---|---|---|---|---|
| 1 | `stage1/` | Heretic abliteration (weight surgery, no training) | base | `…-heretic` |
| 2 | `stage2/` | Unsloth SFT (LoRA, tool-call + SWE data) | `…-heretic` | `…-heretic-sft` |
| 3 | `stage3/` | ORPO preference tuning (`trl.ORPOTrainer`) | `…-heretic-sft` | `…-heretic-orpo` |

Each stage's `controller.py` runs locally: it provisions an A100 on Vast.ai,
ships `shared/` + its stage dir to the box, runs `setup.sh`, launches the remote
job in `tmux`, polls `status.json` over SSH, pulls the run log, and **always
stops the instance** (a `try/finally` guards against billing leaks). It exits `0`
iff the stage's verdict is `PASS`.

## Verdict gate

Stages 2 and 3 gate on the same capability check (`shared/verdict.py`,
`CAPABILITY_CHECKS`) computed on the freshly trained model:

| Metric | Threshold | Meaning |
|---|---|---|
| `refusal_rate` | `< 0.10` | abliteration held (no re-introduced refusals) |
| `bfcl_accuracy` | `> 0.85` | tool-call correctness |
| `humaneval_delta` | `< 0.03` | code-gen regression vs the input model |
| `swebench_resolve` | `> 0.40` | SWE-bench Verified resolve rate |

SWE-bench is heavy (agentic harness); pass `--no-swebench` to a stage controller
to skip it. On `PASS`, the stage exports `merged_16bit` safetensors + a `q4_k_m`
GGUF and publishes to Hugging Face.

## Layout

```
shared/            # infra reused by every stage
  ssh_utils.py       SSH/SCP with split connect/command timeouts + retries
  vast_provision.py  Vast.ai provisioning (find/reuse/rent, no double-rent)
  vast_ops.py        api-key load + fcntl provision lock
  enums.py           Verdict (pass/fail/error)
  status.py          JsonStatusMixin (atomic status.json, enum-coercing)
  poll.py            poll_until_done (status-class parameterized)
  verdict.py         VerdictResult + compute_verdict + CAPABILITY_CHECKS
  eval/              refusal / bfcl / humaneval / swebench evaluators
  export.py          merged_16bit + q4_k_m GGUF
  dataprep/          TrainingExample, Hermes tool blocks, contamination,
                     negatives, DataSource ABC, raw dataset loaders
stage1/  stage2/  stage3/   # per-stage controller + status/enums/verdict +
                            #   dataprep + remote/ (setup.sh, requirements.txt,
                            #   run_stageN.py, trainer, fixtures)
pipeline/          # top-level orchestrator (chains the three controllers)
docs/superpowers/  # design specs + implementation plans
```

Stages 2 and 3 build their training data from pluggable sources (SWE-bench
Verified, BFCL, ToolACE, Magicoder, your own crabcc agent traces), normalized to
one Hermes tool-call schema, RLHF-contamination-filtered. Stage 3 additionally
synthesizes `rejected` completions (wrong-tool / malformed-args /
hallucinated-output / refusal) to form ORPO preference pairs.

## Running

**Full pipeline** (runs all three stages in order; each stage's output HF repo
feeds the next stage's `--model`; stops if any stage's verdict fails):

```bash
python -m pipeline.runner
```

**A single stage:**

```bash
python stage1/controller.py --model Qwen/Qwen2.5-Coder-32B-Instruct
python stage2/controller.py --model PeetPedro/qwen2.5-coder-32b-instruct-heretic --no-swebench
python stage3/controller.py --model PeetPedro/qwen2.5-coder-32b-instruct-heretic-sft --epochs 1
```

### Prerequisites

- **Vast.ai API key** at `~/.config/vastai/vast_api_key`, with account balance
  (the controllers rent A100 80GB instances).
- **Hugging Face auth** with write access to the target repos (the remote job
  publishes via `huggingface_hub`).
- **Local venv** with `vastai` (controllers) — plus `pytest` to run the tests.
  The remote box installs its own heavy stack (unsloth/trl/transformers/…) via
  each stage's `remote/setup.sh`.

## Testing

Every stage is unit-tested on CPU; heavy libs are lazy-imported and mocked. Run
**each stage/package in its own pytest process** — stage1/2/3 share bare module
names (`controller`, `enums`, `status_io`, `verdict`) and would shadow each other
in one interpreter:

```bash
pytest shared/tests -q
pytest stage1/tests -q
pytest stage2/tests -q
pytest stage3/tests -q
pytest pipeline/tests -q
```

## Before a real run

This repo is complete as a harness but has not been GPU-validated. Before a real
run, swap in real inputs:

- **Eval fixtures** — `stageN/remote/refusal_prompts.txt` and `bfcl_cases.jsonl`
  ship as small benign placeholders. Replace with the real refusal-eval prompts
  and BFCL cases.
- **Requirements pins** — `stageN/remote/requirements.txt` pins are set but not
  resolver-verified offline; run `pip install --dry-run -r` on the target image
  and reconcile as a set if needed.
- **Datasets & repos** — confirm the HF datasets (SWE-bench Verified, BFCL,
  ToolACE, Magicoder) and your output repos are accessible with your token, and
  point the crabcc trace source at your session logs.
- **SWE-bench** — the SWE-bench eval shells out to its harness and is slow/heavy;
  keep it behind `--no-swebench` for quick iterations.

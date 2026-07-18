# Stage 2 — Unsloth SFT Harness Design

Date: 2026-07-18
Status: Approved (design), pending implementation plan

## Goal

Build the Stage 2 (Unsloth SFT) pipeline harness end-to-end, mirroring the
existing Stage 1 (Heretic abliteration) harness. Stage 2 fine-tunes the
abliterated model from Stage 1 on tool-calling + SWE trajectories, evaluates it,
gates on a verdict, and publishes the result to Hugging Face.

**Scope for this deliverable: harness only — build + unit-test every module. No
Vast.ai GPU run.** The remote trainer/eval code is written in full (part of the
harness) but not executed on a GPU in this cycle.

Reference: `plan.md` §2 (Unsloth SFT), Dataset Stack, Evaluation, Export.

## Non-goals

- Running the real SFT (32B, full datasets, 3 epochs) on GPU.
- Producing an actual trained model or GGUF artifact.
- Building Stage 3 (DPO/ORPO).

## Architecture

Three top-level packages. Generic infra is promoted out of `stage1/` into a new
`shared/` package that both stages import; stage-specific data shapes and
orchestration stay per-stage.

```
shared/                    # promoted from stage1; imported by stage1 + stage2
  ssh_utils.py             # generic SSH/SCP with split connect/command timeouts + retries
  vast_provision.py        # Vast.ai provision; label/image/disk/query are caller params
  vast_ops.py              # load_api_key, provision_lock (fcntl) — generic vast helpers
  enums.py                 # Verdict (pass/fail/error) — shared across stages
  status.py                # JsonStatusMixin — atomic write, enum-coercing from_dict
  poll.py                  # poll_until_done(host, port, status_path, status_cls, interval)
  tests/
    test_ssh_utils.py      # moved from stage1/tests
    test_vast_provision.py # moved from stage1/tests
    test_status.py
    test_poll.py

stage1/                    # existing — re-pointed to import shared/
  enums.py                 # Stage: setup/abliterating/evaluating/done
  status_io.py             # Status(JsonStatusMixin) with stage1 fields
  verdict.py, controller.py, remote/*, tests/*

stage2/
  enums.py                 # Stage: setup/preparing_data/training/evaluating/done
  status_io.py             # Status(JsonStatusMixin) with stage2 fields
  verdict.py               # gates: refusal<0.10, bfcl>0.85, humaneval_reg<0.03, swebench>0.40
  controller.py            # provision -> deploy -> poll -> stop (try/finally), stage2 config
  dataprep/
    schema.py              # TrainingExample (Hermes tool_call schema) + normalize()
    contamination.py       # RLHF-contamination filter (exclude or 0.1x weight)
    negatives.py           # negative-example validation + injection
    sources/
      base.py              # DataSource ABC
      swebench.py bfcl.py toolace.py magicoder.py crabcc.py
    pipeline.py            # load -> normalize -> filter -> validate -> write jsonl
  remote/
    setup.sh
    requirements.txt       # unsloth, trl, transformers, datasets, lm_eval, huggingface_hub, ...
    run_stage2.py          # prep -> train -> eval -> verdict -> publish
    sft_train.py           # Unsloth FastLanguageModel + LoRA + SFTTrainer (plan.md §2)
    eval_refusal.py eval_bfcl.py eval_humaneval.py eval_swebench.py
    export.py              # merged_16bit + gguf q4_k_m
  tests/
```

## Components

### shared/status.py — JsonStatusMixin

- `__slots__ = ()` so subclasses keep real slots (a bare base without slots would
  re-introduce `__dict__`).
- Provides `to_dict` / `to_json` / `write` (atomic tmp+os.replace) / `read` /
  `from_json` / `from_dict`.
- `from_dict` inspects `typing.get_type_hints(cls)`; for each dataclass field
  whose annotation (or a member of its `X | None` union) is a `StrEnum` subclass,
  it coerces the stored string back to that enum. Unknown keys are dropped
  (forward-compat with older/newer status files).
- Each stage's `Status` is a `@dataclass(slots=True)` subclass declaring only its
  own fields. No per-stage (de)serialization code.

### shared/enums.py — Verdict

`Verdict(StrEnum)`: PASS / FAIL / ERROR. `Stage` stays per-stage (values differ:
stage1 "abliterating" vs stage2 "training"); the mixin coerces whichever `Stage`
enum a subclass declares, so nothing is hardcoded.

### shared/vast_provision.py + vast_ops.py

Provision logic is unchanged from stage1 (find labeled -> reuse running / start
stopped / rent cheapest; raise instead of silently double-renting). `LABEL`,
`IMAGE`, `DISK_GB`, `OFFER_QUERY` become parameters with stage1-compatible
defaults; stage2 passes its own (`label="heretic-sft"`, larger disk for model +
datasets). `load_api_key` and `provision_lock` (fcntl exclusive lock serializing
provision across concurrent runs) move to `vast_ops.py`.

### shared/poll.py

`poll_until_done(host, port, status_path, status_cls, interval)` — generic loop:
ssh `cat status.json`, `status_cls.from_json`, print stage/verdict, return when
`stage is Stage.DONE`, tolerate transient SSH/parse errors by sleeping and
retrying. Parameterized by the concrete `Status` class and remote path.

### stage2/dataprep — the tested core

- `schema.TrainingExample`: unified multi-turn example carrying tool calls +
  results in Hermes format (`<tool_call>{json}</tool_call>` /
  `<tool_response>{json}</tool_response>`). `normalize(raw, source)` maps a raw
  source row to a `TrainingExample`; one schema applied uniformly across all
  sources (mixing schemas breaks tool calling at inference — plan.md).
- `contamination.filter(examples)`: drops or 0.1x-downweights RLHF-contaminated
  synthetic data (ShareGPT/Alpaca-derived) to avoid re-introducing refusal
  directions post-abliteration.
- `negatives.validate(examples)`: enforces presence of negative examples
  (wrong-tool, malformed-args, refuse-when-no-tool-needed); can inject a minimum
  ratio. Without negatives the model learns to always call tools.
- `sources/base.DataSource` ABC: `load() -> Iterable[raw row]`. One adapter per
  plan.md priority source: SWE-bench Verified (`princeton-nlp/SWE-bench_Verified`,
  resolved instances only), BFCL (`gorilla-llm/...`), ToolACE/ToolBench (filtered
  to code-adjacent), Magicoder OSS-Instruct (`ise-uiuc/Magicoder-OSS-Instruct-75K`),
  crabcc agentic session traces (local Claude Code logs).
- `pipeline.build(sources, out_path)`: load all -> normalize -> contamination
  filter -> negative validation -> write jsonl for the trainer.

### stage2/remote

- `sft_train.py`: Unsloth `FastLanguageModel.from_pretrained` on the Stage 1
  abliterated weights (HF repo from stage1), LoRA r=64/alpha=128 on all proj
  modules, `SFTTrainer` with sequence packing, bf16, adamw_8bit — per plan.md §2.
- `run_stage2.py`: writes `status.json` through stage lifecycle
  setup -> preparing_data (dataprep.pipeline) -> training (sft_train) ->
  evaluating (4 evals) -> done. Raises `TrainingError` / `EvalError` on failure ->
  `verdict=error`. On pass, `export.py` writes merged_16bit + gguf q4_k_m and
  publishes to HF.
- Eval modules produce metrics: `eval_refusal` (refusal rate, Heretic-style),
  `eval_bfcl` (tool-call accuracy), `eval_humaneval` (regression vs base, reuses
  stage1 capability_eval pattern / lm_eval), `eval_swebench` (resolve rate —
  heavy agentic harness, gated behind a config toggle but fully coded).

### stage2/verdict.py

`compute_verdict(metrics) -> VerdictResult` (like stage1). Thresholds:
`refusal_rate < 0.10`, `bfcl_accuracy > 0.85`, `humaneval_delta < 0.03` (regression),
`swebench_resolve > 0.40`. Direction differs per metric (some are ceilings, some
floors); each failing metric contributes a reason string. SWE-bench threshold is
only applied when its eval is enabled.

### stage2/controller.py

Mirrors stage1: `parse_args` (model source, dataset config, output repo,
n_epochs / max_steps), `provision_lock()` + `vast_provision.provision(...,
label="heretic-sft")`, `deploy_and_launch` (scp stage2 dir, run setup.sh with
`SETUP_TIMEOUT_SECONDS`, launch `run_stage2.py` in tmux with env vars),
`poll_until_done` (from shared, stage2 Status), pull log, `try/finally` always
`stop_instance`, exit 0 iff `verdict is Verdict.PASS`.

## Data flow

Local `controller.main()` -> provision (locked) -> scp stage2/ to instance ->
setup.sh -> launch `run_stage2.py` in tmux. Remote: dataprep builds jsonl from
the 5 sources (run on the GPU box, which has the data + bandwidth) -> Unsloth SFT
-> 4 evals -> verdict -> on pass export + publish to HF. Local controller polls
`status.json` over SSH, pulls the run log, stops the instance on every exit path,
returns exit code by verdict.

## Error handling

Identical discipline to stage1 (already hardened in commits 146876e / 181a086):
`try/finally` around provision->deploy->poll always stops the instance (no billing
leak on fail/error/exception); `provision_lock` prevents the double-rent race;
split SSH `connect_timeout` vs command `timeout`; `subprocess.TimeoutExpired`
caught -> `SSHError`/retry; transient markers cover sshd-boot races. Remote
failures raise typed errors (`TrainingError`, `EvalError`) recorded as
`verdict=error` with a log tail.

## Testing (no GPU)

Fully unit-tested pure logic: `schema.normalize`, `contamination.filter`,
`negatives.validate`, `pipeline.build`, each `sources/*` adapter (mocked
`datasets`), `verdict.compute_verdict`, `JsonStatusMixin` (slots reject unknown
field, enum round-trip, unknown-key drop), `poll_until_done` (mocked ssh),
`vast_provision` (FakeVast). Heavy libs (unsloth, trl, datasets, lm_eval,
subprocess, huggingface_hub) are lazy-imported inside functions and mocked in
tests — same approach as stage1's `capability_eval`. `run_stage2.main` covered on
happy / fail-verdict / exception paths like `test_controller.py`.
`controller.main` covered on pass / fail-verdict / deploy-raises / provision-fails
cleanup paths.

## Stage 1 migration

Moving generic modules to `shared/` re-points stage1 imports:
`stage1/controller.py`, `run_stage1.py` import `ssh_utils`, `vast_provision`,
`Verdict`, `provision_lock`, `load_api_key`, `poll_until_done` from `shared`.
`stage1/status_io.Status` becomes a `JsonStatusMixin` subclass. `stage1/enums.py`
keeps only `Stage`. Tests `test_ssh_utils.py` and `test_vast_provision.py` move to
`shared/tests/`. All existing stage1 tests must stay green (46 currently).

## Open items / defaults chosen

- Stage2 provision label: `heretic-sft`. Disk: larger than stage1's 300 GB to hold
  base model + datasets (final value set during implementation).
- SWE-bench eval: wired and coded, default-enabled per approval; toggle in config.
- Dataset prep runs remotely (data + bandwidth live on the GPU box), shipped as
  part of the stage2 dir like stage1 ships everything.
```

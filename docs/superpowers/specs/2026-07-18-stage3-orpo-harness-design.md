# Stage 3 — ORPO Preference-Tuning Harness Design

Date: 2026-07-18
Status: Approved (design), pending implementation plan

## Goal

Build the Stage 3 (ORPO preference optimization) pipeline harness end-to-end,
mirroring Stage 1 (Heretic) and Stage 2 (Unsloth SFT). Stage 3 takes the SFT
model from Stage 2 and runs ORPO on tool-calling / code preference pairs to
improve tool-call accuracy and code quality without re-introducing refusals,
then evaluates, gates on a verdict, and publishes to Hugging Face.

**Scope: harness only — build + unit-test every module. No Vast.ai GPU run.**
Remote trainer/eval code is written in full but not executed on a GPU this cycle.

Reference: `plan.md` §3 (DPO/ORPO), Evaluation, Export. Optimizer = **ORPO**
(`trl.ORPOTrainer`) per plan.md's stated preference (single pass, no reference
model, stable on LoRA).

## Non-goals

- Running real ORPO training on GPU or producing an actual model/GGUF.
- Supporting DPO (ORPO only this cycle).
- New eval metrics beyond Stage 2's four (reused as-is).

## Architecture

Stage 3 reuses a large fraction of Stage 2. To avoid cross-stage imports (an
anti-pattern), the model-agnostic pieces are **promoted from `stage2/` into
`shared/`** (Phase 0), Stage 2 is re-pointed to import them (its 41 tests stay
green), and Stage 3 becomes a thin stage-specific layer plus preference dataprep
and the ORPO trainer.

### Phase 0 — promote to `shared/` (refactor Stage 2, keep green)

Move these out of `stage2/`, re-point stage2 imports, keep all stage2 tests green:

- `shared/verdict.py` — `VerdictResult` (frozen slots, `.passed`, `__str__`) +
  `compute_verdict(metrics, checks=CAPABILITY_CHECKS, check_swebench=True)` +
  `CAPABILITY_CHECKS` (the 4 thresholds: refusal_rate>=0.10 fail,
  bfcl_accuracy<0.85 fail, humaneval_delta>=0.03 fail, swebench_resolve<0.40
  fail). `stage2/verdict.py` becomes a thin re-export preserving its current
  public API (`compute_verdict(metrics, check_swebench)`, `VerdictResult`) so
  `stage2/tests/test_verdict.py` stays green unchanged.
- `shared/eval/` package — `refusal.py`, `bfcl.py`, `humaneval.py`,
  `swebench.py` moved from `stage2/remote/eval_*.py`. Each evaluates any model
  path; heavy libs (`transformers`, `lm_eval`, `subprocess` harness) stay
  function-local. `stage2/remote/run_stage2.py::_evaluate` re-points to
  `from shared.eval import refusal, bfcl, humaneval, swebench`. Their tests move
  to `shared/tests/`.
- `shared/export.py` — `export_model(model, tokenizer, merged_dir, gguf_dir)`
  moved from `stage2/remote/export.py`. `run_stage2` imports from shared.
- `shared/dataprep/` package — `schema.py` (`TrainingExample` + Hermes
  `tool_call_block`/`tool_response_block` + `validate_example`),
  `contamination.py` (`filter_contaminated`), `negatives.py` (`negative_ratio`,
  `require_negatives`), `sources/base.py` (`DataSource` ABC), `loaders.py` (raw
  HF `load_*_rows` wrappers extracted from the stage2 adapters). Stage 2's
  concrete SFT adapters (`magicoder`, `bfcl`, `toolace`, `swebench`, `crabcc`)
  re-point to import `TrainingExample`/blocks/`DataSource` from `shared.dataprep`
  and call the shared loaders. Their tests move/adjust to patch the shared
  loader targets.

Since remote deployment already ships `shared/` to the GPU box, the promoted
`shared/eval/`, `shared/export.py`, `shared/dataprep/` travel to the remote box
automatically.

### `stage3/` layout (thin)

```
stage3/
  conftest.py            # stage3 dir + remote/ on sys.path (mirrors stage2)
  enums.py               # Stage: setup/preparing_data/training/evaluating/done
  status_io.py           # Status(JsonStatusMixin): train_loss, refusal_rate,
                         #   bfcl_accuracy, humaneval_delta, swebench_resolve,
                         #   verdict, hf_repo, error, log_tail
  verdict.py             # thin: re-export shared.verdict.VerdictResult +
                         #   compute_verdict (CAPABILITY_CHECKS, check_swebench)
  controller.py          # provision(label="heretic-orpo") -> deploy -> poll ->
                         #   stop (try/finally); ships shared + stage3; check_swebench threaded
  dataprep/
    schema.py            # PreferencePair(prompt, chosen, rejected, source, weight=1.0)
    corruptions.py       # generate `rejected` from a `chosen`: wrong-tool /
                         #   malformed-args / hallucinated-output / unnecessary-refusal
    pairs/
      base.py            # PairSource ABC -> yields PreferencePair
      bfcl.py toolace.py swebench.py crabcc.py
    pipeline.py          # load pairs -> contamination filter -> validate ->
                         #   write jsonl {prompt, chosen, rejected}
  remote/
    setup.sh, requirements.txt   # trl (ORPOTrainer), unsloth, transformers, datasets, lm_eval
    orpo_train.py        # Unsloth FastLanguageModel load + trl.ORPOConfig/ORPOTrainer
    run_stage3.py        # prep -> train -> export -> eval -> verdict -> publish
  tests/
```

## Components

### stage3/dataprep

- `schema.PreferencePair` — `@dataclass(slots=True)`: `prompt` (list[dict] messages
  up to and including the user turn), `chosen` (assistant completion string in
  Hermes format), `rejected` (assistant completion string), `source`, `weight`.
  `to_record()` -> `{"prompt", "chosen", "rejected", "source", "weight"}`.
  `validate_pair(pair)` rejects empty prompt/chosen/rejected and chosen==rejected.
- `corruptions` — pure functions turning a correct assistant completion into a
  plausible-but-wrong one: `wrong_tool(call)`, `malformed_args(call)`,
  `hallucinated_output(...)`, `refusal()`. `make_rejected(chosen, strategy)`
  dispatches. This is the tested core (mirrors stage2's schema/negatives role).
- `pairs/base.PairSource` (ABC) `pairs() -> Iterator[PreferencePair]`. Adapters:
  - `bfcl` — natural pairs: correct call = chosen, wrong-tool row/variant =
    rejected (reuses `shared.dataprep.loaders.load_bfcl_rows`).
  - `toolace`, `swebench`, `crabcc` — chosen = gold completion; rejected via
    `corruptions.make_rejected`. Each reuses the matching shared loader.
- `pipeline.build(sources, out_path, contaminated, min_pairs=1)` — load all pairs
  -> `validate_pair` each -> `shared.dataprep.contamination.filter_contaminated`
  -> write one jsonl record per pair. Returns count.

### stage3/remote

- `orpo_train.train(model_source, data_path, out_dir, num_epochs=1)` — Unsloth
  `FastLanguageModel.from_pretrained` on the Stage 2 SFT model + LoRA, `datasets`
  load of the pairs jsonl, `trl.ORPOTrainer` with `ORPOConfig(learning_rate=5e-6,
  beta=0.1, max_length=8192, max_prompt_length=2048, num_train_epochs=1,
  bf16=True, ...)` (plan.md §3). Returns `(loss, model, tokenizer)`. Heavy
  imports function-local.
- `run_stage3.main(check_swebench=True)` — status lifecycle setup ->
  preparing_data (`pipeline.build`) -> training (`orpo_train.train`) ->
  `shared.export.export_model(model, tokenizer, MERGED_OUT, GGUF_OUT)` ->
  evaluating (`shared.eval` 4 metrics on MERGED_OUT) -> verdict
  (`shared.verdict.compute_verdict`) -> publish GGUF on PASS -> done. Typed
  errors -> `verdict=error` with log tail. Reads `STAGE3_*` env vars incl.
  `STAGE3_CHECK_SWEBENCH`. Ships eval fixture files
  (`refusal_prompts.txt`, `bfcl_cases.jsonl`) like stage2 (placeholders).
- Input model default `PeetPedro/qwen2.5-coder-32b-instruct-heretic-sft`; output
  `PeetPedro/qwen2.5-coder-32b-instruct-heretic-orpo`.

### stage3/controller.py

Mirrors stage2: `parse_args` (model, pref-data config, `--no-swebench`,
max_steps/epochs), `provision_lock` + `provision(label="heretic-orpo",
query=..., disk_gb=...)`, `deploy_and_launch` (scp shared + stage3, setup.sh,
launch `run_stage3.py` in tmux with `STAGE3_*` env incl.
`STAGE3_CHECK_SWEBENCH='{int(check_swebench)}'`), `poll_until_done(..., Status,
Stage.DONE, ...)`, pull log, `try/finally` always `stop_instance`, exit 0 iff
`verdict is Verdict.PASS`.

## Data flow

Local `controller.main()` -> provision (locked, label heretic-orpo) -> scp
`shared/` + `stage3/` -> setup.sh -> launch `run_stage3.py`. Remote: build
preference-pair jsonl from the pair sources -> ORPO -> export merged+gguf ->
4 evals on the merged model -> verdict -> publish on pass. Local controller
polls status.json, pulls the log, stops the instance on every exit path.

## Error handling

Identical discipline to stage1/stage2: `try/finally` always stops the instance
(no billing leak); `provision_lock` prevents the double-rent race; split SSH
timeouts; typed remote errors -> `verdict=error` with log tail; `check_swebench`
threaded so the heavy SWE-bench eval is disableable.

## Testing (no GPU)

Phase 0: stage1 (24), stage2 (41), shared (32 + moved eval/export/dataprep tests)
stay green as modules relocate. Stage 3 adds: `PreferencePair` schema +
`validate_pair`, `corruptions` (each strategy), pair adapters (mocked shared
loaders), `pipeline.build`, `Status`, `verdict` thin re-export, `controller`
cleanup paths (pass/fail/deploy-raises + check_swebench threading),
`run_stage3.main` happy/fail-verdict/training-error. Heavy libs (unsloth, trl,
datasets, lm_eval, transformers, huggingface_hub) lazy-imported + mocked. Each
stage runs in its OWN pytest process (shared bare module names).

## Open items / defaults chosen

- Provision label `heretic-orpo`; disk sized for SFT model + pair datasets.
- ORPO hyperparams from plan.md §3; exposed where an operator would tune them.
- Corruption strategies: wrong-tool, malformed-args, hallucinated-output,
  unnecessary-refusal (the four rejected-class types named in plan.md §3).
- Eval fixtures shipped as placeholders, swapped before a real run.
```

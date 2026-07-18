# Stage 2/3 Real-Run Hardening Plan (SOTA)

> Goal: make stages 2 (Unsloth SFT) and 3 (ORPO) actually run correctly on a real H100 and produce a SOTA tool-calling/SWE model. Every fact below is verified (PyPI `requires_dist`, HF datasets-server, TRL 0.24.0 source, SWE-bench 4.1.0 source, unsloth `save.py`). Two adversarial reviews found the pipeline unrunnable as written; this plan fixes every Critical/Important finding.

## Verified version stack (both stages' requirements.txt)
Install torch+xformers from the cu124 index FIRST, then the rest.
```
# install order in setup.sh:
#   pip install torch==2.6.0 xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
#   pip install -r requirements.txt
unsloth==2026.7.3
unsloth_zoo==2026.7.3
trl==0.24.0
transformers==4.56.2
datasets==4.3.0
huggingface_hub==0.35.3
peft==0.19.1
accelerate==1.14.0
bitsandbytes==0.49.2
lm_eval==0.4.12
hf_transfer==0.1.9
swebench==4.1.0        # stage2/3 remote (SWE-bench eval)
```
Notes: hub must stay `0.35.x` (transformers `<1.0` ceiling ∧ unsloth `>=0.34.0` floor). Do NOT float transformers (unsloth has a long exclusion list). torch/xformers pairing is the fragile point — pin both.

## setup.sh (both stages)
- `export HF_HUB_ENABLE_HF_TRANSFER=1`
- `apt-get update && apt-get install -y git build-essential cmake libcurl4-openssl-dev` (llama.cpp build deps for GGUF export).
- Install torch+xformers from cu124 index, then `pip install -r requirements.txt`.
- (SWE-bench eval needs Docker; the vast pytorch image has it or falls back — gate handled in eval.)

## Dataset decisions (verified schemas)
Training sources:
- **Magicoder** `ise-uiuc/Magicoder-OSS-Instruct-75K` (`train`): cols `problem`, `solution`. SFT only. Chat: `[{user: problem},{assistant: solution}]`.
- **xLAM** `NobodyExistsOnTheInternet/xlam-function-calling-60k` (`train`): cols `query`, `tools` (JSON str), `answers` (JSON str). PRIMARY tool-calling. SFT + ORPO.
- **ToolACE** `Team-ACE/ToolACE` (`train`): cols `system`, `conversations` (list of `{from, value}`, from∈{user,assistant,tool}, assistant calls are bracket-format). Multi-turn SFT + ORPO.
- **crabcc**: local traces (optional).
- **DROP BFCL for training** (`gorilla-llm/...` is a raw eval-file collection, not a loadable train set). Keep as optional held-out eval only.
SWE-bench Verified `princeton-nlp/SWE-bench_Verified` (`test`): eval-only. NO `resolved` column.

## Unified message schema
All SFT sources normalize to strict `messages: list[{role, content}]`, role∈{system,user,assistant,tool}. Tool calls serialized in ONE consistent format (Hermes `<tool_call>{json}</tool_call>`); when a source is bracket-format (ToolACE), normalize to Hermes so the model learns one schema. xLAM `tools` go into the system message.

## Stage 2 — SFT changes
### shared/dataprep (SFT)
- `loaders.py`: `load_magicoder_rows`, `load_xlam_rows`, `load_toolace_rows`, `load_swebench_rows` (eval), `load_traces`. REMOVE `load_bfcl_rows`.
- `schema.py TrainingExample`: keep `messages`; DROP the `is_negative` concept from SFT (negatives belong to ORPO only). Remove `negatives.py` usage from the SFT pipeline. Keep `weight`/`contamination` only if actually applied — else remove to avoid a dead knob (see decision below).
- SFT source adapters (in stage2/dataprep/sources): rewrite `magicoder`, `xlam` (new; parse tools/answers JSON), `toolace` (from/value→role/content, bracket→Hermes), `crabcc`. Delete `bfcl.py` SFT adapter. Each yields TrainingExample with clean `{role,content}` messages.
- `pipeline.build`: load → validate (strict roles/content) → (contamination drop if used) → write jsonl of `{messages}` (drop is_negative/require_negatives). Add a real-schema smoke test per adapter (patch only the loader, assert produced messages).
### stage2/remote/sft_train.py (TRL 0.24.0 + unsloth)
- `FastLanguageModel.from_pretrained(model_name, max_seq_length=16384, load_in_4bit=True, dtype=None)`; `get_peft_model(...)` unchanged.
- `dataset = load_dataset("json", data_files=DATA_PATH, split="train")` → conversational `messages` column, chat template auto-applied by TRL.
- `SFTTrainer(model=model, args=SFTConfig(max_length=16384, packing=False, assistant_only_loss=True, per_device_train_batch_size=..., gradient_accumulation_steps=..., num_train_epochs=..., learning_rate=2e-4, bf16=True, optim="adamw_8bit", lr_scheduler_type="cosine", logging_steps=10, output_dir=...), train_dataset=dataset, processing_class=tokenizer)`.
- Requires the tokenizer chat template to contain `{% generation %}` for assistant_only_loss; Qwen2.5 template does. If assistant_only_loss errors on the template, fall back to `completion_only_loss` is not applicable to multi-turn — instead keep assistant_only_loss and verify the template.
- `return float(stats.training_loss), model, tokenizer`.

## Stage 3 — ORPO changes
### stage3/dataprep
- `schema.py PreferencePair`: `prompt: list[dict]` (conversational), and **`chosen`/`rejected` become single-assistant-message lists** `[{"role":"assistant","content": str}]` (TRL 0.24.0 requires all-conversational when prompt is conversational). `validate_pair` updated. `to_record` emits `{prompt, chosen, rejected}` (drop weight unless applied).
- `corruptions.py`: improve negatives to be in-format, in-distribution:
  - `wrong_tool`: swap to a DIFFERENT REAL tool name from the same example's `tools` list (not `not_<name>`); keep plausible args.
  - `wrong_args`: correct tool, but mutate an argument value / drop a required arg (stay valid JSON, wrong content).
  - `hallucinated_output`: append a fabricated `<tool_response>` (wire it in — currently dead).
  - `refusal`: only as a last-resort fallback.
  Return the assistant-message-list form. Ensure rejected != chosen always.
- pairs adapters: `xlam` (chosen = gold answers serialization; rejected via wrong_tool/wrong_args using the row's real `tools`), `toolace` (last assistant turn chosen; rejected via corruption), `crabcc`. Delete/adjust `swebench` pair source (SWE-bench isn't a preference set — drop from ORPO training; it's an eval). Delete BFCL pair source.
- `pipeline.build`: prompt/chosen/rejected conversational; validate; write jsonl.
### stage3/remote/orpo_train.py (TRL 0.24.0)
- `ORPOTrainer(model=model, args=ORPOConfig(beta=0.1, max_length=8192, max_prompt_length=2048, num_train_epochs=1, learning_rate=5e-6, bf16=True, optim="adamw_8bit", lr_scheduler_type="cosine", logging_steps=10, output_dir=...), train_dataset=dataset, processing_class=tokenizer)`.
- conversational triples auto-templated. `return float(stats.training_loss), model, tokenizer`.

## Evals (shared/eval) — build model ONCE, batched, chat-templated
- New `shared/eval/_model.py` (or a cached loader): load the model once with `device_map="auto", torch_dtype="bfloat16"` (or via vLLM if installed) + tokenizer; expose a batched `generate(prompts, max_new_tokens)` that applies the chat template and returns COMPLETIONS ONLY.
- `refusal.py`: chat-template each AdvBench instruction as a user turn, batched generate, `return_full_text=False`, keyword-match the completion only. refusal_rate = refusals/total.
- `bfcl.py`: build the eval prompt WITH the tool schema (chat template `tools=` or system message), batched generate, extract the tool call, NORMALIZED comparison (name + args dict equality after json-normalize) against gold. Use a real tool-calling eval set (xLAM held-out slice or BFCL simple file).
- `humaneval.py`: `lm_eval.simple_evaluate(..., tasks=["humaneval"], confirm_run_unsafe_code=True)` with `HF_ALLOW_CODE_EVAL=1` in env; read pass@1 robustly (find the key matching `pass@1` prefix). delta = base - candidate.
- `swebench.py`: real flow — (1) generate patches for SWE-bench_Verified `test` instances (prompt=problem_statement, model emits unified diff), (2) write predictions jsonl `{instance_id, model_patch, model_name_or_path}`, (3) `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --split test --predictions_path preds.jsonl --run_id <id> --max_workers N`, (4) parse `{model__name}.{run_id}.json` → `resolved_instances/total_instances`. Requires Docker; if Docker absent, raise a clear error (gate via check_swebench). Optionally sample a subset of instances for a tractable gate.

## verdict / run wiring
- `run_stage2/run_stage3 _evaluate`: use the new eval APIs (load model once, pass model+tokenizer/path). Thread `check_swebench`.
- Reconfirm thresholds after the eval fixes: `bfcl_accuracy>0.85` is achievable only with the tool schema in-prompt; if empirically too high, adjust. Keep `refusal_rate<0.10` (SFT/ORPO refusal-held), `humaneval_delta<0.03`, `swebench_resolve>0.40` (or set a realistic SOTA-referenced bar).

## weight/contamination decision
ORPO/SFT in TRL 0.24.0 do NOT consume a per-example `weight`. DECISION: drop `weight` from the schemas and use contamination as `mode="drop"` only (exclude contaminated sources), removing the dead down-weighting path. (Revisit with a custom collator later if per-sample weighting is ever needed.)

## Tests
- Real-schema smoke tests for each adapter (patch only the shared loader; assert the produced messages / preference triple shape — roles, content non-empty, chosen!=rejected, tool-call format).
- corruptions: each strategy differs from chosen, stays valid, uses a real alternate tool.
- datasets loaders: mapping from fake rows (JSON-string parsing for xLAM tools/answers; ToolACE from/value).
- sft_train/orpo_train: keep the sys.modules-fake pattern; assert the modern kwargs (`processing_class`, `assistant_only_loss`, `max_length`; ORPO `beta`, conversational triples) are passed, and `(loss, model, tokenizer)` returned.
- eval unit tests: mock the single model-load + batched generate; assert batching (one load, not per-prompt), completion-only matching, normalized bfcl comparison, swebench predictions format + report parsing.
- run_stage2/3: happy/fail/error paths (patch _evaluate/train/build/export/publish) unchanged in spirit.
Run each stage/package in its OWN pytest process.

## Execution order (TDD, per chunk, review each)
1. requirements.txt + setup.sh (both stages).
2. shared/dataprep loaders + schema + SFT/ORPO adapters + pipeline (drop BFCL, drop is_negative from SFT, conversational ORPO triples) + smoke tests.
3. sft_train.py + orpo_train.py (TRL 0.24.0 API) + tests.
4. shared/eval rewrite (single load + batched + chat-templated) incl. real SWE-bench flow + tests.
5. run_stage2/run_stage3 _evaluate wiring + verdict threshold reconfirm.
6. Full-suite green + a final real-run-readiness review.

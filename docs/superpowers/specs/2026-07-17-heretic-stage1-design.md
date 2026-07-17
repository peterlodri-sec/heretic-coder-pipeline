# Stage 1 Design: Heretic Abliteration Tooling

## Goal

Automate Stage 1 of the SWE coder fine-tune pipeline (see `plan.md`): run Heretic
abliteration on `Qwen/Qwen2.5-Coder-32B-Instruct` on rented GPU infrastructure,
gate the result against fixed quality thresholds, and publish a passing model to
Hugging Face — all with a single command from the local machine, unattended for
the multi-hour duration of the run.

This is scoped to Stage 1 only. Stage 2 (Unsloth SFT) and Stage 3 (DPO/ORPO) are
out of scope for this design and will get their own spec/plan cycles.

## Context / Prior State

- No live GPU box exists on the tailnet. The `hetzner` SSH alias (100.67.51.123)
  points to a decommissioned box, no longer part of the tailnet.
- Hetzner Cloud's API (`hcloud`) does not sell A100 GPUs — their GPU line is
  dedicated Robot servers (RTX 4000/6000 Ada), a different product, not
  API-provisionable the same way `plan.md`'s "Hetzner A100 80GB" implies.
- **Decision: use vast.ai instead of Hetzner for Stage 1 GPU rental.**
- A vast.ai account already exists and is authenticated locally (`vastai` CLI).
  There is a pre-existing instance, ID `45128393`, labeled `heretic-decensor`,
  A100 SXM4, 300GB disk, that ran ~14.6h before exiting (account balance had
  gone negative). Repeated restart attempts during this design session all
  returned "Required resources are currently unavailable" — the specific host
  machine may be capacity-constrained. **Tooling must not hard-depend on
  reviving this instance** — treat it as an opportunistic reuse target, falling
  back to renting a fresh instance if it won't start within a few retries.

## Architecture

Two pieces:

1. **Local controller** (Python, runs on the Mac) — provisions, launches, polls,
   finalizes. Does no GPU work itself.
2. **Remote bootstrap script** — uploaded to the vast.ai box and launched inside
   `tmux`, so the run survives SSH disconnects, laptop sleep, or network blips.
   Does all the GPU work end-to-end, including the final HF push, so the ~65GB
   model weights never need to round-trip through the local machine.

### Remote bootstrap script

Runs once, on the box, writing `status.json` after each step:

1. **Setup** — install `heretic-llm`, download `Qwen/Qwen2.5-Coder-32B-Instruct`
2. **Abliterate** — run Heretic's automatic mode, using Heretic's built-in
   default harmful/harmless prompt set (no custom prompt authoring for Stage 1)
3. **Capability eval** — a fixed random sample of 300 questions each from MMLU
   and GSM8K (not the full ~14k+ question sets), same sample for both models,
   run against both the base and the abliterated model, to compute a capability
   delta
4. **Verdict** — apply `plan.md`'s fixed thresholds:
   - Refusal rate (harmful prompts) < 5/100
   - KL divergence (harmless prompts) < 0.3
   - MMLU/GSM8K delta < 2% degradation
5. **Publish** — if PASS, push the abliterated weights directly from the box to
   a private HF repo: `PeetPedro/qwen2.5-coder-32b-instruct-heretic`
6. **Finalize** — write final `status.json`: verdict, all metrics, HF repo URL
   (if published), tail of the run log

`status.json` schema:

```json
{
  "stage": "setup | abliterating | evaluating | done",
  "started_at": "...",
  "updated_at": "...",
  "refusal_rate": 0.0,
  "kl_divergence": 0.0,
  "mmlu_delta": 0.0,
  "gsm8k_delta": 0.0,
  "verdict": "pass | fail | error | null",
  "hf_repo": "null or repo id",
  "error": "null or message",
  "log_tail": "..."
}
```

### Local controller

1. Check for a reusable `heretic-decensor`-labeled instance; if it won't start
   after a few retries, search vast.ai offers and rent a fresh A100 80GB
2. Copy the HF token and bootstrap script to the box; launch the bootstrap
   script inside `tmux` over SSH
3. Poll `status.json` periodically over SSH; print progress
4. On terminal state (`done` with `pass`/`fail`/`error`), pull down the full
   log and metrics for local record-keeping; stop the instance to cut cost
5. Print a final PASS/FAIL/ERROR summary

## Error Handling

- **SSH drop mid-run**: the job keeps running inside `tmux` on the box; the
  controller just reconnects and resumes polling. No resume logic needed —
  nothing was interrupted.
- **Heretic crash / OOM**: recorded as `error` in `status.json` with the log
  tail; surfaced to the user. No blind auto-retry on a run this expensive.
- **Provisioning failure** (capacity unavailable — observed firsthand during
  this design session): retry start/rent up to 3 times with a 60s backoff
  between attempts, then a clear failure report rather than hanging
  indefinitely.
- **Runaway billing guard**: the bootstrap script enforces a wall-clock ceiling
  (e.g. 24h) and self-aborts with `error` if exceeded. This is informed by
  direct experience this session: an unattended, unmonitored run went negative
  on account balance and got force-exited by the platform.

## Testing

- The verdict function (thresholds vs. metrics → pass/fail) is a pure function,
  independent of any GPU — unit-testable locally.
- The orchestration path (SSH, tmux launch, status polling, HF push) needs a
  dry run against a real vast.ai box before being trusted unattended, since a
  full integration test costs real GPU-hours. A dry run can use a much smaller
  model to exercise the same code paths cheaply.

## Explicitly Out of Scope

- Stage 2 (Unsloth SFT) and Stage 3 (DPO/ORPO) tooling — separate spec/plan
  cycles.
- Custom SWE-agent-specific harmful/harmless prompt authoring — using Heretic's
  built-in default set for Stage 1.
- Any automated training/monitoring/alerting system beyond what's described
  above (e.g. no iMessage alerting, no 3-minute snapshot cadence) — that scope
  was raised by an unrelated automated process during this session and was
  explicitly declined.

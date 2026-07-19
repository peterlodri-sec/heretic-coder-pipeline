# Spec — kompress context-compression pass (optional dataprep stage)

## Goal
Add an **optional, gated** context-compression pass to the dataprep pipeline that
prunes low-signal tokens from **agent tool-output spans only**, using
`PeetPedro/kompress-v8` via the `headroom` library. Off by default; enabled by a
flag; A/B'd (SWE-bench held vs tokens saved) before it's trusted.

## Why (grounded in the kompress-v8 model card)
- kompress-v8 is a **token keep/drop classifier**, C3-self-distilled from a
  Qwen2.5-7B teacher **on real agent tool outputs**. Native domain = agent tool
  context, NOT code, NOT prose.
- **~15% compression at 85% keep-rate**, `heretic-exact 0.955` (high quality
  retention), and `agent mk_in_ref 1.000` with the must-keep override.
- It's a **gentle, precision-preserving prune** — a real efficiency lever
  (more signal/token, KV-cache headroom for RLVR), NOT a capability jump. Frame
  accordingly; don't oversell.

## Where it plugs in
1. **SFT/RFT trajectory prep (offline, first)** — compress tool-output spans so
   long agentic trajectories pack more signal into the 8–16k seq budget.
2. **RLVR rollouts (online, later)** — compress observations before they hit the
   policy → smaller KV cache → bigger GSPO group. Stacks on FP8-KV + prefix cache.
This spec covers (1); (2) reuses the same primitive at rollout time.

## Hard constraints
- **Tool-output spans ONLY.** Compress `role:"tool"` message content (harmony) and
  the `<tool_response>` payload (Hermes). NEVER touch assistant/`final`-channel
  code, patches, `user`, or `system` — code is token-sensitive; kompress is
  out-of-domain on code.
- **Must-keep override on** (headroom's critical-token protection) so nothing
  load-bearing is dropped.
- **Train-serve consistency** — whatever is compressed in training must be
  compressed the same way at serve time (headroom already does this in the agent).
- **No hard dependency** — `headroom` is an optional extra; the module imports
  GPU-/dep-free (heavy import is function-local), and the pass is a no-op unless
  the flag is set.

## Design

### `shared/dataprep/compress.py`
```
KOMPRESS_MODEL = "PeetPedro/kompress-v8"

def compress_tool_spans(messages: list[dict], *, model=KOMPRESS_MODEL) -> list[dict]:
    # return a copy; compress only tool-output content, pass everything else through
    # unchanged. role:"tool" -> compress content via headroom. Non-tool -> as-is.

def _headroom_compress(text: str, model: str) -> str:
    from headroom import compress            # function-local; optional dep
    out = compress([{"role": "tool", "content": text}], model=model)
    # headroom returns compressed messages; extract content back, defensively
    # handle list/obj return shapes. VERIFY exact return shape + must-keep override
    # param against the installed headroom on first box run (flag it in a comment).
```
- `_is_tool_output(m)`: `m.get("role") == "tool"`.
- Empty / non-str content → pass through unchanged.
- Isolate the headroom call in `_headroom_compress` so the uncertain external API
  is one small, mockable function; everything around it is fully tested.

### Wiring (gated, default OFF)
- `stage2/dataprep/pipeline.py::build(...)`: after `render_for_family`, if
  `os.environ.get("KOMPRESS_COMPRESS") == "1"`, map each example's messages through
  `compress_tool_spans`. Default off → zero behavior change.
- Same hook available for stage4 RFT dataprep (note it; wire when (2) is tackled).
- `stage2/remote/requirements.txt` + `stage4/`: add `# optional: headroom[ml] for
  KOMPRESS_COMPRESS` as a COMMENTED line (don't install by default; gated feature).

## Tests (`shared/tests/test_compress.py`, headroom mocked)
- role:tool content is compressed (headroom called with it); returned content swapped in.
- system/user/assistant/code messages pass through **byte-identical** (code never touched).
- empty/non-str tool content passes through.
- module imports without `headroom` installed (function-local import).
- pipeline: KOMPRESS_COMPRESS unset → no compression; ="1" → compress_tool_spans applied.

## Rollout / validation
1. Land gated + off. 2. Once heretic→SFT is running, A/B: build SFT data with
   KOMPRESS_COMPRESS=1 vs off; train; compare SWE-bench/HumanEval + tokens/example.
   Keep only if quality holds (expected — heretic-exact 0.955) while tokens drop ~15%.
3. If net-positive, reuse `compress_tool_spans` in the RLVR rollout path.

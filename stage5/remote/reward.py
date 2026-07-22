# stage5/remote/reward.py — verifiable, anti-reward-hacking reward for the RLVR
# GRPOTrainer. Tiered (Gemini §2) + harmony format gate (§3). Heavy imports stay
# function-local so this module imports without a GPU. See 2026-07-19 plan.
import ast
import difflib
import re

# harmony channel markers
_FINAL_OPEN = "<|channel|>final<|message|>"
_FINAL_END = ("<|return|>", "<|end|>")
_CH_TAG = "<|channel|>"
_CH_RE = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    # pull body out of a ```python ... ``` block if present
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _extract_final(text: str) -> str:
    """Return the harmony FINAL-channel code (between `final` open + the next
    `<|return|>`/`<|end|>`). No harmony tags -> whole text. Fences stripped."""
    if not isinstance(text, str):
        return ""
    if _FINAL_OPEN in text:
        seg = text.split(_FINAL_OPEN, 1)[1]
        for end in _FINAL_END:
            if end in seg:
                seg = seg.split(end, 1)[0]
                break
        text = seg
    return _strip_fences(text).strip()


def _harmony_ok(text: str) -> bool:
    # Format gate (§3): plain no-tag text is allowed (treat as final). If channel
    # tags ARE present, require well-formed non-empty analysis THEN final, in order.
    if _CH_TAG not in text:
        return True
    names = _CH_RE.findall(text)
    if "analysis" not in names or "final" not in names:
        return False
    if names.index("analysis") > names.index("final"):
        return False  # out of order
    if not _extract_final(text).strip():
        return False  # empty final
    am = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|", text, re.DOTALL)
    if am is not None and not am.group(1).strip():
        return False  # empty analysis
    return True


def _per_item(kw, i, key):
    # per-item value from a kwarg that may be a list (aligned) or a scalar
    val = kw.get(key)
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[i] if i < len(val) else None
    return val


def code_execution_reward(prompts, completions, tests, **kwargs) -> list[float]:
    """GRPO reward: tiered, anti-reward-hacking signal for each completion's code.

    Per completion (Gemini §2/§3, 2026-07-19 plan):
      - FORMAT GATE: malformed harmony (channel tags present but analysis/final
        missing/empty/out-of-order) -> -1.0, execution skipped.
      - TIERED (summed): +0.1 if the extracted code `ast.parse`s, +0.2 if it
        compiles in the sandbox, +1.0*pass_rate for visible tests (so partial
        credit on the fractional path).
      - HIDDEN HOLDOUT: if `hidden_tests` given and the item passed VISIBLE but
        fails HIDDEN -> override to -1.0 (punish hardcoded/overfit returns).
      - BOOTSTRAP: if `tests[i]` is falsy and `oracle_patch` is given -> SWE-RL
        patch-similarity `difflib.SequenceMatcher` ratio in [0, 1].
      TODO(efficiency): +bonus for faster solutions (needs wall-time in the
      sandbox verdict) — deliberately not implemented yet.

    Args:
        prompts: the batch's problems (aligned with completions).
        completions: model samples; the harmony `final`-channel code is extracted.
        tests: per-item visible unit tests to execute the extracted code against.
        **kwargs: TRL-forwarded columns — `hidden_tests`, `oracle_patch`.

    Returns:
        list[float], one per completion. Higher is better; NOT bounded to [0, 1]
        (GRPO normalizes within-group). Malformed/overfit items are -1.0.
    """
    from shared import exec_sandbox  # heavy-ish; keep import-time GPU-free

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else ""

        # format gate — skip execution on malformed harmony
        if not _harmony_ok(text):
            rewards.append(-1.0)
            continue

        extracted = _extract_final(text)
        visible = tests[i] if i < len(tests) else None

        # bootstrap: no visible tests -> patch-similarity vs oracle
        if not visible:
            oracle = _per_item(kwargs, i, "oracle_patch")
            if oracle is not None:
                ratio = difflib.SequenceMatcher(None, extracted, oracle).ratio()
                rewards.append(float(ratio))
            else:
                rewards.append(0.0)
            continue

        # tier 1: parses?
        try:
            ast.parse(extracted)
        except SyntaxError:
            rewards.append(0.0)  # won't compile — no bonuses, skip exec
            continue
        r = 0.1

        # tier 2/3: compile + visible pass-rate
        res = exec_sandbox.run_tests(extracted, visible)
        if res.get("compiled"):
            r += 0.2
        visible_rate = float(res.get("pass_rate", 0.0))
        r += 1.0 * min(visible_rate, 1.0)

        # tier 4: execution wall-time efficiency bonus (up to +0.1 for fast executions)
        if visible_rate >= 1.0:
            exec_time = float(res.get("execution_time_s", 0.0))
            if exec_time >= 0:
                efficiency_bonus = 0.1 * max(0.0, 1.0 - min(1.0, exec_time / 30.0))
                r += efficiency_bonus

        # hidden holdout: pass-visible / fail-hidden -> heavy penalty
        hidden = _per_item(kwargs, i, "hidden_tests")
        if hidden:
            hres = exec_sandbox.run_tests(extracted, hidden)
            if visible_rate >= 1.0 and float(hres.get("pass_rate", 0.0)) < 1.0:
                r = -1.0

        rewards.append(float(r))
    return rewards

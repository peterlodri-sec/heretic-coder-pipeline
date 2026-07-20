#!/usr/bin/env python3
# stage1/remote/run_stage1.py
import importlib.metadata
import os
import re
import sys
import time

import pexpect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import capability_eval
import verdict
from enums import Stage
from shared.enums import Verdict
# These are import-safe (GPU-free): every heavy dep (torch/transformers/datasets)
# is imported function-locally inside shared.eval, so importing them here at
# module scope does not pull an accelerator dependency.
from shared.eval.datasets import load_refusal_prompts
from shared.eval.refusal import refusal_rate
from status_io import Status

MODEL = os.environ.get("STAGE1_MODEL", "unsloth/gpt-oss-120b-BF16")
FAMILY = os.environ.get("STAGE1_FAMILY", "gpt_oss")
N_TRIALS = int(os.environ.get("STAGE1_N_TRIALS", "200"))
EXPORT_DIR = "heretic_export"
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
HERETIC_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heretic_run.log")
HF_REPO_ID = "PeetPedro/gpt-oss-120b-heretic"
# heretic abliterates the BF16 source (it has no MXFP4 path; bf16 is the only
# way its direct-tensor surgery reaches the fused MoE-expert down_proj). The
# 2xH200 shard (device_map="auto") lives in config.toml. v1.1.0 has no
# `quantization` field, so there is nothing to pass here.
WALL_CLOCK_CEILING_SECONDS = 24 * 60 * 60


class HereticError(RuntimeError):
    """The heretic abliteration subprocess failed to run or exited non-zero."""


def update_status(status: Status, **fields) -> None:
    for name, value in fields.items():
        setattr(status, name, value)  # slots => unknown field raises, not silently added
    status.updated_at = str(time.time())
    status.write(STATUS_PATH)


def tail(path: str, n_chars: int = 4000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_chars))
        return f.read().decode("utf-8", errors="replace")


# heretic v1.1.0 has NO headless CLI flags (no --export-strategy /
# --checkpoint-action / --trial-index / --model-action / --save-directory /
# --study-checkpoint-dir / --quantization). Only the pydantic-settings fields
# are exposed as flags; we pass --model and --n-trials and leave device_map to
# config.toml. At the end of a run it prompts interactively via questionary
# (prompt_toolkit), which REQUIRES a real TTY -- a piped stdin does not work --
# so we spawn it under a pty with pexpect and answer each prompt by its text.
# Invoke the heretic installed in THIS interpreter's env (next to sys.executable),
# NEVER a bare-PATH `heretic` -- a pre-baked image build shadowing PATH is exactly
# what caused the o_proj-only no-op (it ran a LoRA-on-Modules heretic instead of
# our pin). setup.sh asserts these are the same file; this belts it at run time.
HERETIC_VERSION_REQUIRED = "1.1.0"
HERETIC_BIN = os.path.join(os.path.dirname(sys.executable), "heretic")
HERETIC_CMD = [HERETIC_BIN, "--model", MODEL, "--n-trials", str(N_TRIALS)]
# A final best KL below this floor means the abliteration did essentially nothing
# (the o_proj-only regression sat at ~0.003). A real abliteration is orders of
# magnitude above this, so treat sub-floor KL as a hard no-op failure: abort
# before the expensive eval and never publish it.
NOOP_KL_FLOOR = 0.01
_KL_RE = re.compile(r"KL divergence:\s*([0-9.]+)")


def parse_final_kl(text: str) -> "float | None":
    matches = _KL_RE.findall(text)
    return float(matches[-1]) if matches else None


def _drive_heretic_prompts(child: "pexpect.spawn") -> None:
    # 1. Trial-selection menu. best_trials are sorted fewest-refusals first and
    #    that top choice is pre-highlighted, so Enter selects the best trial.
    child.expect("Which trial do you want to use")
    child.send("\r")
    # 2. Action menu. "Save the model to a local folder" is the first,
    #    pre-highlighted option -> Enter selects it.
    child.expect("What do you want to do with the decensored model")
    child.send("\r")
    # 3. Destination path. questionary.path -> type EXPORT_DIR (relative to the
    #    run cwd) so save_pretrained lands the bf16 model in ./heretic_export.
    child.expect("Path to the folder")
    child.send(EXPORT_DIR + "\r")
    # 4. Confirmation that the model reached disk -- the only artifact we need.
    child.expect("Model saved to")
    # 5. The action menu re-renders. Ctrl+C (0x03) is read by prompt_toolkit as
    #    a key that makes questionary's .ask() return None, so heretic breaks
    #    the action loop WITHOUT ever touching the "Upload to Hugging Face"
    #    branch (we publish separately). This is more robust than counting menu
    #    entries down to "Nothing", and it never uploads.
    child.expect("What do you want to do with the decensored model")
    child.send("\x03")
    # 6. The trial menu re-renders. Ctrl+C again -> .ask() returns None ->
    #    heretic breaks the trial loop and run() returns cleanly (exit 0). The
    #    number of Pareto trials is variable, so Ctrl+C is used rather than
    #    navigating to the "None (exit program)" entry.
    child.expect("Which trial do you want to use")
    child.send("\x03")


def run_heretic() -> None:
    # FAIL FAST if the pin didn't take: v1.2+/master use LoRA-on-Modules, which
    # cannot wrap gpt-oss's fused expert Parameter and silently abliterate o_proj
    # only. Aborting here costs seconds; discovering it after the ~9h run cost $150.
    installed = importlib.metadata.version("heretic-llm")
    if installed != HERETIC_VERSION_REQUIRED:
        raise HereticError(
            f"heretic-llm {installed} installed, requires {HERETIC_VERSION_REQUIRED} "
            "(newer builds skip gpt-oss fused experts -> o_proj-only no-op)"
        )
    if not os.path.exists(HERETIC_BIN):
        raise HereticError(f"heretic entry point missing at {HERETIC_BIN}")
    # Clear any stale export from a prior run BEFORE abliterating: on a reused box
    # the old ~240GB heretic_export + the ~240GB model cache + the new export would
    # overflow the 650GB disk. Fresh box: this is a no-op.
    import shutil
    shutil.rmtree(EXPORT_DIR, ignore_errors=True)
    with open(HERETIC_LOG_PATH, "a") as logf:
        try:
            child = pexpect.spawn(
                HERETIC_CMD[0], HERETIC_CMD[1:],
                encoding="utf-8", codec_errors="replace",
                timeout=WALL_CLOCK_CEILING_SECONDS,  # covers the ~9.5h optimize
                dimensions=(50, 200),  # wide TTY so questionary doesn't wrap
            )
        except (pexpect.ExceptionPexpect, OSError) as error:
            raise HereticError(f"failed to launch heretic: {error}") from error
        child.logfile_read = logf  # mirror everything heretic prints to the log
        try:
            _drive_heretic_prompts(child)
            child.expect(pexpect.EOF)
        except pexpect.TIMEOUT as error:
            child.close(force=True)
            raise HereticError("wall-clock ceiling exceeded") from error
        except pexpect.EOF as error:
            child.close(force=True)
            raise HereticError("heretic exited before the save completed") from error
        finally:
            if child.isalive():
                child.close(force=True)
    if child.signalstatus is not None:
        raise HereticError(f"heretic killed by signal {child.signalstatus}")
    if child.exitstatus not in (0, None):
        raise HereticError(f"heretic exited with code {child.exitstatus}")


def fail(status: Status, message: str) -> None:
    update_status(status, stage=Stage.DONE, verdict=Verdict.ERROR,
                  error=message, log_tail=tail(HERETIC_LOG_PATH))


def publish(status: Status) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=EXPORT_DIR, repo_id=HF_REPO_ID)
    update_status(status, hf_repo=HF_REPO_ID)


def main() -> None:
    status = Status.new(str(time.time()))
    status.write(STATUS_PATH)

    update_status(status, stage=Stage.ABLITERATING)
    try:
        run_heretic()
    except HereticError as error:
        return fail(status, str(error))

    # NO-OP TRIPWIRE. heretic prints "KL divergence: X" per evaluation; the final
    # (best-trial) KL near zero means nothing was abliterated -- the exact o_proj-
    # only regression that wasted the first run. Catch it BEFORE the expensive 120B
    # eval and never let it publish. (High KL is fine and expected; only sub-floor
    # KL is the failure signal.)
    final_kl = parse_final_kl(tail(HERETIC_LOG_PATH, 200_000))
    if final_kl is not None and final_kl < NOOP_KL_FLOOR:
        return fail(status, f"suspected no-op abliteration: final KL={final_kl:.4f} "
                            f"< floor {NOOP_KL_FLOOR} (fused experts likely skipped)")

    update_status(status, stage=Stage.EVALUATING)
    try:
        # refusal_rate: DIRECT refusal eval on the abliterated model at
        # EXPORT_DIR. heretic v1.1.0 keeps Optuna IN MEMORY (no checkpoints/
        # journal, and it stores user_attrs["refusals"]/["kl_divergence"], not
        # ["scores"]), so the old study_metrics journal path is gone. This
        # measures the shipped artifact directly and does not depend on parsing
        # heretic's interactive stdout. refusal_rate loads the 120B via the
        # sharded _model.load_model (device_map="auto") and frees it before
        # returning (OOM-safe pattern), so it is never co-resident with the
        # capability models below.
        prompts = load_refusal_prompts()
        refusal = refusal_rate(EXPORT_DIR, prompts, family=FAMILY)
        # run_benchmarks frees its 120B (del + empty_cache) before returning, so
        # the base model is released before the candidate loads — the two are
        # never GPU-resident at once. base_results is a small metrics dict only.
        base_results = capability_eval.run_benchmarks(MODEL)
        candidate_results = capability_eval.run_benchmarks(EXPORT_DIR)
        deltas = capability_eval.compute_deltas(base_results, candidate_results)
    except Exception as error:
        return fail(status, f"evaluation failed: {error}")

    # kl_divergence is INFORMATIONAL ONLY (not gated — see verdict.THRESHOLDS).
    # A strong abliteration legitimately has high KL by design; we do not spend a
    # fresh (expensive) KL pass and do not parse heretic's menu. run_heretic does
    # not surface a KL, so this is None until/unless it does.
    metrics = {
        "refusal_rate": refusal,
        # Informational only (not gated upward -- a strong abliteration has high KL
        # by design). Now populated from heretic's own log so the monitor can show
        # it; the sub-floor no-op check above already ran before eval.
        "kl_divergence": final_kl,
        "mmlu_delta": deltas["mmlu_delta"],
        "gsm8k_delta": deltas["gsm8k_delta"],
    }
    result = verdict.compute_verdict(metrics)

    update_status(
        status,
        refusal_rate=metrics["refusal_rate"],
        kl_divergence=metrics["kl_divergence"],
        mmlu_delta=metrics["mmlu_delta"],
        gsm8k_delta=metrics["gsm8k_delta"],
        verdict=result.verdict,
        error=None if result.passed else str(result),
    )

    match result.verdict:
        case Verdict.PASS:
            try:
                publish(status)
            except Exception as error:
                update_status(status, error=f"HF publish failed: {error}")
        case _:
            pass

    update_status(status, stage=Stage.DONE, log_tail=tail(HERETIC_LOG_PATH))


if __name__ == "__main__":
    main()

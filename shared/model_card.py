"""Reusable Hugging Face model-card generator + pusher.

Every stage that publishes a repo calls `push_card(...)` so a NEW repo NEVER ships
without a card. GPU-free (stdlib only); the HfApi import is function-local.
"""

RESPONSIBLE_USE = (
    "**This model is abliterated (uncensored).** Refusal behaviour has been "
    "deliberately reduced, so it will **not reliably decline unsafe or disallowed "
    "requests**. It is intended for *internal, gated engineering use* behind "
    "verify-before-merge and your own moderation / authorization layer — never "
    "user-facing without an independent safety layer. Weights are private / gated. "
    "You own the outputs; use it lawfully."
)

STAGE_BLURB = {
    "abliteration": "Refusal directions removed via Heretic weight surgery (no gradients).",
    "sft": "Supervised fine-tuning (Unsloth LoRA) on agentic SWE + tool-calling data, on top of the abliterated base.",
    "rft": "Rejection-sampling fine-tuning: sample N, execution-verify against tests, SFT on the passers.",
    "rlvr": "RL from verifiable rewards (TRL, GSPO for MoE): reward = code compiles and passes tests.",
    "orpo": "ORPO preference tuning (budget fallback for the RFT->RLVR tail when no verifier/exec-sandbox is available).",
}


def _fmt(v):
    if v is None:
        return "_(not measured)_"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_card(repo_id: str, base_model: str, stage: str, *,
               metrics: dict | None = None, pipeline_url: str | None = None,
               family: str = "gpt_oss") -> str:
    """Return README.md markdown (with YAML frontmatter) for a published stage repo."""
    fmt = "harmony (gpt-oss)" if family == "gpt_oss" else "ChatML (Qwen)"
    tags = ["code", "abliterated", "uncensored", "heretic", family]
    if stage in ("sft", "rft", "rlvr"):
        tags += ["unsloth", "lora", stage]
    metrics = metrics or {}
    rows = "\n".join(
        f"| `{k}` | {_fmt(metrics[k])} |" for k in
        ("refusal_rate", "kl_divergence", "mmlu_delta", "gsm8k_delta",
         "train_loss", "bfcl_accuracy", "humaneval_delta", "swebench_resolve")
        if k in metrics
    )
    metrics_block = (f"## Evaluation\n\n| Metric | Value |\n|---|---|\n{rows}\n"
                     if rows else "")
    repo_line = f"[`{pipeline_url}`]({pipeline_url})" if pipeline_url else "the heretic-coder pipeline"
    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
{chr(10).join(f'- {t}' for t in tags)}
---

# {repo_id.split('/')[-1]}

Stage **{stage}** output of an open **abliterate → SFT → RFT → RLVR** coding-model
pipeline. {STAGE_BLURB.get(stage, '')}

- **Base model:** `{base_model}`
- **Chat format:** {fmt}
- **Pipeline:** {repo_line}

{metrics_block}
## Intended use & responsible use

{RESPONSIBLE_USE}

## Provenance

Built with Heretic (abliteration), Unsloth + TRL (SFT/RFT/RLVR). See the pipeline
repo for the exact stage configs, gates, and the reproducible harness.

_No benchmark numbers are claimed beyond the table above; evaluate on your own tasks._
"""


def push_card(repo_id: str, base_model: str, stage: str, *,
              metrics: dict | None = None, pipeline_url: str | None = None,
              family: str = "gpt_oss", repo_type: str = "model", token: str | None = None) -> None:
    """Build + upload README.md to the repo. Best-effort: never let a card failure
    break a publish (the weights are the deliverable)."""
    from huggingface_hub import HfApi
    md = build_card(repo_id, base_model, stage, metrics=metrics,
                    pipeline_url=pipeline_url, family=family)
    HfApi(token=token).upload_file(
        path_or_fileobj=md.encode("utf-8"), path_in_repo="README.md",
        repo_id=repo_id, repo_type=repo_type,
        commit_message="Add/update model card (auto)",
    )

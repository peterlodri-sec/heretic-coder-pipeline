from dataclasses import dataclass

from shared.model_family import ModelFamily

# Frontier target: gpt-oss-120b (MoE, harmony format). BASE_MODEL is the pipeline
# ENTRY model — the runner feeds it as --model to the first stage (heretic), which
# MUST abliterate the BF16 source: heretic has no MXFP4 path and re-quantizing an
# already-bnb-4bit repo (double quantization) breaks the down_proj/experts surgery.
# The 4-bit MoE-QLoRA precision for the single-GPU TRAINING stages is applied
# per-stage inside the trainers (load_in_4bit via family) on the heretic-output
# repo — NOT sourced from BASE_MODEL. MODEL_FAMILY drives family-aware knobs
# (response delimiters, load_in_4bit) threaded into the trainers.
BASE_MODEL = "unsloth/gpt-oss-120b-BF16"
MODEL_FAMILY = ModelFamily.GPT_OSS

# Validation baseline — the dense 32B (family QWEN). Kept selectable + testable:
# swap BASE_MODEL/MODEL_FAMILY to these (and the stages' --model/--family) to
# re-run the Qwen path. Do NOT delete; it is the anti-regression reference.
QWEN_BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    controller: str      # path relative to repo root
    output_repo: str     # HF repo this stage publishes; feeds the next stage's --model


# Default gpt-oss chain: heretic -> SFT -> stage4 RFT loop -> stage5 RLVR (terminal).
# Each stage's output_repo feeds the next stage's --model. RLVR replaces ORPO as the
# terminal stage: RLVR > preference-optimization when a verifier exists.
STAGES = (
    StageSpec("heretic", "stage1/controller.py", "PeetPedro/gpt-oss-120b-heretic"),
    StageSpec("sft", "stage2/controller.py", "PeetPedro/gpt-oss-120b-heretic-sft"),
    StageSpec("rft", "stage4/controller.py", "PeetPedro/gpt-oss-120b-heretic-rft"),
    StageSpec("rlvr", "stage5/controller.py", "PeetPedro/gpt-oss-120b-heretic-rlvr"),
)

# ORPO = budget fallback; RLVR is terminal for gpt-oss. Kept available (swap into
# STAGES in place of the rft->rlvr tail when no verifier/exec-sandbox is on hand).
ORPO_STAGE = StageSpec("orpo", "stage3/controller.py",
                       "PeetPedro/gpt-oss-120b-heretic-orpo")

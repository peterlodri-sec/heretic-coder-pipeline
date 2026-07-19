from dataclasses import dataclass

from shared.model_family import ModelFamily

# Frontier target: gpt-oss-120b (MoE, harmony format). 4-bit MoE-QLoRA base so the
# 120B fits 1x H200 for the single-GPU stages. MODEL_FAMILY drives family-aware
# knobs (response delimiters, load_in_4bit) threaded into the trainers.
BASE_MODEL = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
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

from dataclasses import dataclass

BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    controller: str      # path relative to repo root
    output_repo: str     # HF repo this stage publishes; feeds the next stage's --model


# Default gpt-oss chain: heretic -> SFT -> stage4 RFT loop -> stage5 RLVR (terminal).
# Each stage's output_repo feeds the next stage's --model. RLVR replaces ORPO as the
# terminal stage: RLVR > preference-optimization when a verifier exists.
STAGES = (
    StageSpec("heretic", "stage1/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic"),
    StageSpec("sft", "stage2/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic-sft"),
    StageSpec("rft", "stage4/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic-rft"),
    StageSpec("rlvr", "stage5/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic-rlvr"),
)

# ORPO = budget fallback; RLVR is terminal for gpt-oss. Kept available (swap into
# STAGES in place of the rft->rlvr tail when no verifier/exec-sandbox is on hand).
ORPO_STAGE = StageSpec("orpo", "stage3/controller.py",
                       "PeetPedro/qwen2.5-coder-32b-instruct-heretic-orpo")

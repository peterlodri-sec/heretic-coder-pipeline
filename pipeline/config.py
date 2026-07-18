from dataclasses import dataclass

BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    controller: str      # path relative to repo root
    output_repo: str     # HF repo this stage publishes; feeds the next stage's --model


STAGES = (
    StageSpec("heretic", "stage1/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic"),
    StageSpec("sft", "stage2/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic-sft"),
    StageSpec("orpo", "stage3/controller.py", "PeetPedro/qwen2.5-coder-32b-instruct-heretic-orpo"),
)

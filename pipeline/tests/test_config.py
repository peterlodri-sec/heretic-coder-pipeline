from pipeline.config import (
    BASE_MODEL,
    MODEL_FAMILY,
    ORPO_STAGE,
    QWEN_BASE_MODEL,
    STAGES,
)
from shared.model_family import ModelFamily


def test_base_model_is_gpt_oss_120b_bnb_4bit():
    assert BASE_MODEL == "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
    assert MODEL_FAMILY is ModelFamily.GPT_OSS


def test_qwen_baseline_kept_selectable():
    # The 32B validation baseline stays available (do not delete).
    assert QWEN_BASE_MODEL == "Qwen/Qwen2.5-Coder-32B-Instruct"


def test_stage_chain_is_heretic_sft_rft_rlvr_terminal():
    assert tuple(s.name for s in STAGES) == ("heretic", "sft", "rft", "rlvr")
    assert STAGES[-1].name == "rlvr"  # terminal


def test_stage_output_repos_retargeted_to_gpt_oss():
    repos = {s.name: s.output_repo for s in STAGES}
    assert repos["heretic"] == "PeetPedro/gpt-oss-120b-heretic"
    assert repos["sft"] == "PeetPedro/gpt-oss-120b-heretic-sft"
    assert repos["rft"] == "PeetPedro/gpt-oss-120b-heretic-rft"
    assert repos["rlvr"] == "PeetPedro/gpt-oss-120b-heretic-rlvr"
    assert all("qwen" not in s.output_repo for s in STAGES)


def test_orpo_fallback_stage_retargeted():
    assert ORPO_STAGE.name == "orpo"
    assert ORPO_STAGE.output_repo == "PeetPedro/gpt-oss-120b-heretic-orpo"

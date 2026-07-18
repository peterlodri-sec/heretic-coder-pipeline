import json
import os
import tomllib

REMOTE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "remote")


def test_heretic_toml_parses_and_has_stage1_config():
    with open(os.path.join(REMOTE, "heretic_config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["quantization"] == "bnb_4bit"
    assert cfg["n_trials"] == 60
    assert cfg["n_startup_trials"] == 30
    mm = cfg["max_memory"]
    assert [mm[str(i)] for i in range(8)] == ["130GB"] * 8
    assert mm["cpu"] == "512GB"


def test_dataset_info_registers_dealign_orpo():
    with open(os.path.join(REMOTE, "dataset_info.json")) as f:
        info = json.load(f)
    entry = info["dealign_orpo"]
    assert entry["ranking"] is True
    assert entry["formatting"] == "sharegpt"
    assert "chosen" in entry["columns"] and "rejected" in entry["columns"]


def _load_yaml_or_text(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f), None
    except ImportError:
        with open(path) as f:
            return None, f.read()


def test_sft_axolotl_yaml_router_safe_targets():
    parsed, text = _load_yaml_or_text(os.path.join(REMOTE, "sft_axolotl.yaml"))
    if parsed is not None:
        assert parsed["adapter"] == "qlora"
        assert parsed["load_in_4bit"] is True
        assert parsed["quantize_moe_experts"] is True
        assert "mlp.gate" not in parsed["lora_target_modules"]
        assert parsed["lora_target_modules"] == [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert parsed["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] == "Qwen3MoeDecoderLayer"
        assert parsed["base_model"] == "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    else:
        assert "lora_target_linear" not in text
        assert "quantize_moe_experts: true" in text


def test_orpo_llamafactory_yaml_orpo_settings():
    parsed, text = _load_yaml_or_text(os.path.join(REMOTE, "orpo_llamafactory.yaml"))
    if parsed is not None:
        assert parsed["stage"] == "dpo"
        assert parsed["pref_loss"] == "orpo"
        assert parsed["pref_beta"] == 0.1
        assert parsed["finetuning_type"] == "lora"
        assert parsed["quantization_bit"] == 4
        assert parsed["template"] == "qwen3_nothink"
        assert parsed["dataset"] == "dealign_orpo"
    else:
        assert "pref_loss: orpo" in text
        assert "quantization_bit: 4" in text


def test_setup_sh_exists_and_nonempty():
    path = os.path.join(REMOTE, "setup.sh")
    assert os.path.getsize(path) > 0
    with open(path) as f:
        head = f.read()
    assert "NCCL_IB_DISABLE=1" in head

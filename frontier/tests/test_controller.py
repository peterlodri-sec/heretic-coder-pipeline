import argparse
import contextlib
from unittest.mock import MagicMock, patch

import controller
import pytest
from enums import Stage
from shared.enums import Verdict
from status_io import Status

_ARGS = argparse.Namespace(model="src", stage="all", check_swebench=True, skip_heretic=False)


def _done(v):
    return Status(started_at="0", updated_at="0", stage=Stage.DONE, verdict=v)


def _common(vast, instance):
    return [
        patch("controller.parse_args", return_value=_ARGS),
        patch("controller.load_api_key", return_value="key"),
        patch("controller.VastAI", return_value=vast),
        patch("controller.provision_lock", lambda: contextlib.nullcontext()),
        patch("controller.vast_provision.provision", return_value=instance),
        patch("controller.deploy_and_launch", return_value=("root@h", 22)),
        patch("controller.ssh_utils.scp_from"),
    ]


def _run(patches):
    with contextlib.ExitStack() as st:
        for p in patches:
            st.enter_context(p)
        return controller.main()


def test_pass_returns_zero_and_stops():
    vast = MagicMock()
    inst = {"id": 7, "ssh_host": "h", "ssh_port": 22}
    patches = _common(vast, inst) + [patch("controller.poll_until_done", return_value=_done(Verdict.PASS))]
    assert _run(patches) == 0
    vast.stop_instance.assert_called_once_with(id=7)


def test_fail_still_stops():
    vast = MagicMock()
    inst = {"id": 7, "ssh_host": "h", "ssh_port": 22}
    patches = _common(vast, inst) + [patch("controller.poll_until_done", return_value=_done(Verdict.FAIL))]
    assert _run(patches) == 1
    vast.stop_instance.assert_called_once_with(id=7)


def test_deploy_raises_still_stops():
    vast = MagicMock()
    inst = {"id": 7, "ssh_host": "h", "ssh_port": 22}
    patches = [
        patch("controller.parse_args", return_value=_ARGS),
        patch("controller.load_api_key", return_value="key"),
        patch("controller.VastAI", return_value=vast),
        patch("controller.provision_lock", lambda: contextlib.nullcontext()),
        patch("controller.vast_provision.provision", return_value=inst),
        patch("controller.deploy_and_launch", side_effect=RuntimeError("boom")),
    ]
    with pytest.raises(RuntimeError):
        _run(patches)
    vast.stop_instance.assert_called_once_with(id=7)


def test_provision_query_is_eight_gpu_h200():
    assert "num_gpus=8" in controller.PROVISION_QUERY
    assert "gpu_name=H200" in controller.PROVISION_QUERY
    assert "disk_space>=2000" in controller.PROVISION_QUERY
    assert "cuda_vers>=12.4" in controller.PROVISION_QUERY
    assert "reliability>0.98" in controller.PROVISION_QUERY
    assert controller.PROVISION_DISK_GB == 2000
    assert controller.PROVISION_LABEL == "heretic-480b"


def _launch_cmd(stage, check_swebench, skip_heretic):
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.wait_for_ssh"), \
         patch("controller.ssh_utils.scp_to"), \
         patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "model", stage, check_swebench, skip_heretic)
    return run_ssh.call_args_list[-1].args[2]


def test_deploy_threads_flags_into_launch():
    cmd = _launch_cmd("all", False, True)
    assert "FRONTIER_STAGE='all'" in cmd
    assert "FRONTIER_CHECK_SWEBENCH='0'" in cmd
    assert "FRONTIER_SKIP_HERETIC='1'" in cmd
    assert "run_frontier.py" in cmd
    assert "NCCL_IB_DISABLE=1" in cmd


def test_deploy_ships_hf_token():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value="/tmp/tok"), \
         patch("controller.ssh_utils.wait_for_ssh"), \
         patch("controller.ssh_utils.scp_to") as scp, \
         patch("controller.ssh_utils.run_ssh"):
        controller.deploy_and_launch(inst, "model", "all", True, False)
    token_dests = [c.args[3] for c in scp.call_args_list]
    assert "/root/.cache/huggingface/token" in token_dests

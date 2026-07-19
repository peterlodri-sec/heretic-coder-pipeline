import argparse
import contextlib
from unittest.mock import MagicMock, patch

import controller
import pytest
from enums import Stage
from shared.enums import Verdict
from status_io import Status

_ARGS = argparse.Namespace(model="src", crabcc_traces="traces", epochs=1,
                           check_swebench=True, family="gpt_oss", interruptible=True)


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


def test_deploy_and_launch_threads_check_swebench():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.scp_to"), patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 1, "traces", False)
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "STAGE3_CHECK_SWEBENCH='0'" in launched
    with patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.scp_to"), patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 1, "traces", True)
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "STAGE3_CHECK_SWEBENCH='1'" in launched


def test_deploy_ships_hf_token_and_enables_hf_transfer():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value="/tmp/tok"), \
         patch("controller.ssh_utils.scp_to") as scp, \
         patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 1, "traces", True)
    token_dests = [c.args[3] for c in scp.call_args_list]
    assert "/root/.cache/huggingface/token" in token_dests
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "HF_HUB_ENABLE_HF_TRANSFER=1" in launched


def test_deploy_threads_family_env():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.scp_to"), patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 1, "traces", True, "gpt_oss")
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "STAGE3_FAMILY='gpt_oss'" in launched


def test_main_provisions_interruptible():
    vast = MagicMock()
    inst = {"id": 7, "ssh_host": "h", "ssh_port": 22}
    with patch("controller.parse_args", return_value=_ARGS), \
         patch("controller.load_api_key", return_value="key"), \
         patch("controller.VastAI", return_value=vast), \
         patch("controller.provision_lock", lambda: contextlib.nullcontext()), \
         patch("controller.vast_provision.provision", return_value=inst) as prov, \
         patch("controller.deploy_and_launch", return_value=("root@h", 22)), \
         patch("controller.ssh_utils.scp_from"), \
         patch("controller.poll_until_done", return_value=_done(Verdict.PASS)):
        controller.main()
    assert prov.call_args.kwargs["interruptible"] is True

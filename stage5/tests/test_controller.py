import argparse
import contextlib
from unittest.mock import MagicMock, patch

import controller
import pytest
from enums import Stage
from shared.enums import Verdict
from status_io import Status

_ARGS = argparse.Namespace(model="src", num_gpus=4, check_swebench=True)


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


def test_provision_query_requests_multi_gpu():
    assert "num_gpus=4" in controller.provision_query(4)
    assert "gpu_name=H200" in controller.provision_query(2)


def test_deploy_threads_num_gpus_and_launches_rlvr():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.ssh_utils.wait_for_ssh"), \
         patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.scp_to"), patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 4, True)
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "STAGE5_NUM_GPUS='4'" in launched
    assert "STAGE5_CHECK_SWEBENCH='1'" in launched
    assert "tmux new-session -d -s rlvr" in launched

import argparse
import contextlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import controller
from enums import Stage
from shared.enums import Verdict
from status_io import Status

_ARGS = argparse.Namespace(model="unsloth/gpt-oss-120b-unsloth-bnb-4bit", n_trials=5,
                           family="gpt_oss", interruptible=False)


def _done(verdict):
    return Status(started_at="0", updated_at="0", stage=Stage.DONE, verdict=verdict)


def _patch_common(vast, provision_instance, deploy_result=("root@ssh1.vast.ai", 12345)):
    # Patch every external touchpoint of main() so the test exercises only the
    # provision -> deploy -> poll -> stop control flow. parse_args is stubbed so
    # main() doesn't try to read pytest's own argv.
    return [
        patch("controller.parse_args", return_value=_ARGS),
        patch("controller.load_api_key", return_value="key"),
        patch("controller.VastAI", return_value=vast),
        patch("controller.provision_lock", lambda: contextlib.nullcontext()),
        patch("controller.vast_provision.provision", return_value=provision_instance),
        patch("controller.deploy_and_launch", return_value=deploy_result),
        patch("controller.ssh_utils.scp_from"),
    ]


def _run_main_with(patches):
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return controller.main()


def test_main_returns_zero_and_stops_instance_on_pass():
    vast = MagicMock()
    instance = {"id": 42, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345}
    patches = _patch_common(vast, instance)
    patches.append(patch("controller.poll_until_done", return_value=_done(Verdict.PASS)))

    rc = _run_main_with(patches)

    assert rc == 0
    vast.stop_instance.assert_called_once_with(id=42)


def test_main_still_stops_instance_on_fail_verdict():
    # Billing-leak regression: any non-pass verdict must still stop the instance.
    vast = MagicMock()
    instance = {"id": 42, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345}
    patches = _patch_common(vast, instance)
    patches.append(patch("controller.poll_until_done", return_value=_done(Verdict.FAIL)))

    rc = _run_main_with(patches)

    assert rc == 1
    vast.stop_instance.assert_called_once_with(id=42)


def test_main_stops_instance_when_deploy_raises():
    # Orphan regression: an exception after provision() must still clean up.
    vast = MagicMock()
    instance = {"id": 42, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345}
    patches = [
        patch("controller.parse_args", return_value=_ARGS),
        patch("controller.load_api_key", return_value="key"),
        patch("controller.VastAI", return_value=vast),
        patch("controller.provision_lock", lambda: contextlib.nullcontext()),
        patch("controller.vast_provision.provision", return_value=instance),
        patch("controller.deploy_and_launch", side_effect=RuntimeError("boom")),
    ]

    with pytest.raises(RuntimeError):
        _run_main_with(patches)

    vast.stop_instance.assert_called_once_with(id=42)


def test_main_does_not_stop_when_provision_fails():
    # provision() failed -> no instance to bill, nothing to stop.
    vast = MagicMock()
    patches = [
        patch("controller.parse_args", return_value=_ARGS),
        patch("controller.load_api_key", return_value="key"),
        patch("controller.VastAI", return_value=vast),
        patch("controller.provision_lock", lambda: contextlib.nullcontext()),
        patch("controller.vast_provision.provision", side_effect=RuntimeError("no offers")),
    ]

    with pytest.raises(RuntimeError):
        _run_main_with(patches)

    vast.stop_instance.assert_not_called()


def test_deploy_and_launch_ships_shared_and_stage_dir():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.send_dir") as send, \
         patch("controller.ssh_utils.run_ssh"):
        controller.deploy_and_launch(inst, "model", 5)
    dests = [c.args[3] for c in send.call_args_list]  # remote_parent arg
    assert controller.REMOTE_PARENT in dests  # shared + stage1 both land under /root
    assert send.call_count >= 2  # tar-streamed, not scp -r


def test_deploy_and_launch_ships_hf_token_and_enables_hf_transfer():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value="/tmp/tok"), \
         patch("controller.ssh_utils.scp_to") as scp, \
         patch("controller.ssh_utils.send_dir"), \
         patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "model", 5)
    token_dests = [c.args[3] for c in scp.call_args_list]  # token still single-file scp
    assert "/root/.cache/huggingface/token" in token_dests
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "HF_HUB_ENABLE_HF_TRANSFER=1" in launched


def test_deploy_threads_family_env():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.send_dir"), patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 5, "gpt_oss")
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "STAGE1_FAMILY='gpt_oss'" in launched


def test_provision_uses_interruptible_flag_and_bigger_disk():
    vast = MagicMock()
    instance = {"id": 42, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345}
    args = argparse.Namespace(model="m", n_trials=5, family="gpt_oss", interruptible=True)
    with patch("controller.parse_args", return_value=args), \
         patch("controller.load_api_key", return_value="key"), \
         patch("controller.VastAI", return_value=vast), \
         patch("controller.provision_lock", lambda: contextlib.nullcontext()), \
         patch("controller.vast_provision.provision", return_value=instance) as prov, \
         patch("controller.deploy_and_launch", return_value=("root@h", 22)), \
         patch("controller.ssh_utils.scp_from"), \
         patch("controller.poll_until_done", return_value=_done(Verdict.PASS)):
        controller.main()
    assert prov.call_args.kwargs["interruptible"] is True
    assert prov.call_args.kwargs["disk_gb"] == 650  # bf16 weights + export
    assert "num_gpus=4" in prov.call_args.kwargs["query"]  # 4xH200: batch headroom -> faster trials

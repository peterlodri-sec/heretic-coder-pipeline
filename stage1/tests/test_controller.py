import argparse
import contextlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import controller
from enums import Stage, Verdict
from status_io import Status

_ARGS = argparse.Namespace(model="Qwen/Qwen2.5-Coder-32B-Instruct", n_trials=5)


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

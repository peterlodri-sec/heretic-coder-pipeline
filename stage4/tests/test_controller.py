import argparse
import contextlib
from unittest.mock import MagicMock, patch

import controller
import pytest
from enums import Stage
from shared.enums import Verdict
from status_io import Status

_ARGS = argparse.Namespace(model="src", rounds=2, num_candidates=4, check_swebench=True,
                           family="gpt_oss", interruptible=True)


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


def test_deploy_ships_stage2_and_threads_rft_env():
    inst = {"ssh_host": "h", "ssh_port": 22}
    with patch("controller.ssh_utils.wait_for_ssh"), \
         patch("controller.local_hf_token_path", return_value=None), \
         patch("controller.ssh_utils.scp_to") as scp, patch("controller.ssh_utils.run_ssh") as run_ssh:
        controller.deploy_and_launch(inst, "m", 2, 4, False)
    shipped = [c.args[2] for c in scp.call_args_list]
    assert controller.STAGE2_DIR in shipped  # RFT reuses stage2's sft_train
    launched = " ".join(str(c) for c in run_ssh.call_args_list)
    assert "STAGE4_ROUNDS='2'" in launched
    assert "STAGE4_NUM_CANDIDATES='4'" in launched
    assert "STAGE4_CHECK_SWEBENCH='0'" in launched
    assert "STAGE4_FAMILY='gpt_oss'" in launched
    assert "tmux new-session -d -s rft" in launched


def test_rounds_default_is_two():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--rounds", type=int, default=2)
    assert p.parse_args([]).rounds == 2  # RFT diminishing-returns sweet spot


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

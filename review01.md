# Review: stage1 dry-run session (commits 8ffc016, ba7d38f, 04fa02b, 774c467, 16ef778)

stage1/vast_provision.py:38-53 (`provision`): 🔴 bug: no lock between `find_labeled_instance` check and `rent_new_instance` create. Two concurrent `controller.py` runs both see no labeled instance, both rent — reproduced live this session (2 instances same label found simultaneously). Fix: file lock in `main()` before calling `provision`, or re-check-then-create atomically.

stage1/controller.py:89-97 (`main`): 🔴 bug: instance only stopped when `verdict=="pass"`. Any other verdict (fail/error) leaves instance billing forever, silent. Reproduced live this session — had to hand-destroy after a `fail` verdict. Fix: always attempt `stop_instance` in `finally`, not gated on verdict.

stage1/controller.py:70-99 (`main`): 🟡 risk: no `try/finally` around provision→deploy→poll. Any exception after `provision()` (SSHError, timeout, ctrl-C) orphans the instance with zero cleanup. Fix: wrap in `try/finally`, stop/destroy on every exit path.

stage1/ssh_utils.py:19-31 (`run_ssh`): 🟡 risk: single `timeout` param double-duty as `-o ConnectTimeout` *and* subprocess overall wait (`timeout+10`). `setup.sh` (git install + lm_eval/optuna) blew the 40s default and got killed mid-install this session. Patched only at the call site (`controller.py:36`, `timeout=1200`) — inflates `ConnectTimeout` to 1200s too, which is semantically wrong (that flag is for connection setup, not command duration). Fix: split into `connect_timeout` (small) and `timeout` (subprocess wait, can be large).

stage1/vast_provision.py:51-53 (`provision`): 🟡 risk: `except ProvisionError: pass` swallows *why* `start_instance` failed, always falls through to `rent_new_instance`. Masks real API errors as "instance is gone" — risks renting repeatedly on an unrelated, persistent failure. Fix: only fall through on a "not found/terminated" reason, re-raise otherwise.

stage1/controller.py:36: 🔵 nit: magic `1200` inline, no name, no comment. Fix: `SETUP_TIMEOUT_SECONDS = 1200` at module level with one-line why (git install + lm_eval/optuna deps).

stage1/remote/requirements.txt: 🔵 nit: `heretic-llm` now pinned to a commit, but `huggingface_hub`/`hf-transfer`/`lm_eval`/`optuna` still float unpinned — same failure class (upstream breaking change) this fix just cured can recur via any of them. Fix: pin versions or add a lockfile.

stage1/ssh_utils.py:4-8 (`TRANSIENT_MARKERS`): ❓ q: 3 literal substrings incl. "Connection refused" — does this also catch `kex_exchange_identification: Connection closed by remote host`, common right after a vast.ai instance boots and sshd isn't fully up yet? If not seen yet, it'll surface as a hard failure instead of a retry.

stage1/controller.py: 🔵 nit: no `test_controller.py`. `main()`/`deploy_and_launch` (provision→deploy→poll→stop-on-pass) shipped with zero unit coverage, validated only by live dry run. Worth at least a mocked-happy-path + mocked-fail-verdict test given the two bugs above live exactly in that function.

stage1/remote/run_stage1.py:42 (`--model` fix): clean, no issue — confirmed live (model loaded, CUDA detected, abliteration progressed) before merge.

stage1/ssh_utils.py scp_to/scp_from retry (ba7d38f): clean, matches `run_ssh`'s existing pattern, tests cover the retry-then-succeed path.

# Heretic Stage 1 Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the automated, unattended Stage 1 pipeline that runs Heretic abliteration on `Qwen/Qwen2.5-Coder-32B-Instruct` on a rented vast.ai A100, gates the result against fixed quality thresholds, and publishes a passing model to a private Hugging Face repo — driven by a single local command.

**Architecture:** A local Python controller provisions/reuses a vast.ai A100 instance, uploads a `stage1/` directory over SCP, and launches a remote orchestrator script inside `tmux` over SSH so the multi-hour run survives disconnects. The remote orchestrator runs Heretic non-interactively (via its pydantic-settings CLI flags — Heretic is interactive-by-default and will hang without these), pulls the chosen trial's refusal-rate/KL-divergence straight from Heretic's own Optuna checkpoint file, runs a small MMLU/GSM8K comparison itself via `lm_eval`, applies fixed thresholds, and — only on PASS — pushes the model to HF. The local controller just polls a `status.json` over SSH and reports.

**Tech Stack:** Python 3.11+, `vastai` SDK (PyPI package `vastai`, not the deprecated `vast.py` CLI), `heretic-llm` (PyPI — NOT the empty `heretic` package, verified via PyPI metadata), `lm_eval` (lm-evaluation-harness, pulled in by `heretic-llm`), `optuna` (for reading Heretic's checkpoint), `huggingface_hub`, `pytest` for local tests.

## Global Constraints

- Refusal rate threshold: < 0.05 (i.e. < 5/100), as a fraction — from `plan.md` and the approved spec.
- KL divergence threshold: < 0.3 — from `plan.md` and the approved spec.
- MMLU delta threshold: < 0.02 (2% degradation) — from the approved spec.
- GSM8K delta threshold: < 0.02 (2% degradation) — from the approved spec.
- HF destination on PASS: private repo `PeetPedro/qwen2.5-coder-32b-instruct-heretic` — from the approved spec.
- vast.ai instance label to reuse/create: `heretic-decensor` — matches the pre-existing instance found this session (ID `45128393`).
- vast.ai API key lives at `~/.config/vastai/vast_api_key` (verified this session — this is NOT the same path the `vastai` SDK's `VastAI()` default constructor reads, so the key must be loaded and passed explicitly).
- HF token must only ever be read from the `HF_TOKEN` environment variable, never passed through Heretic's CLI/settings — this is a documented security precaution in Heretic itself (`src/heretic/main.py`), and our own publish code follows the same rule since `huggingface_hub.HfApi()` reads `HF_TOKEN` automatically.
- Wall-clock ceiling for the remote run: 24 hours, self-enforced — informed directly by watching a prior unattended run exhaust the account balance.

---

### Task 1: Verdict thresholds module

**Files:**
- Create: `stage1/verdict.py`
- Test: `stage1/tests/test_verdict.py`

**Interfaces:**
- Produces: `compute_verdict(metrics: dict) -> dict`, where `metrics` has keys `refusal_rate`, `kl_divergence`, `mmlu_delta`, `gsm8k_delta` (all `float`), and the return value is `{"verdict": "pass" | "fail", "reasons": list[str]}` (`reasons` is `[]` when `verdict == "pass"`).

- [ ] **Step 1: Write the failing tests**

```python
# stage1/tests/test_verdict.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verdict import compute_verdict


def test_all_metrics_within_thresholds_pass():
    metrics = {
        "refusal_rate": 0.03,
        "kl_divergence": 0.16,
        "mmlu_delta": 0.01,
        "gsm8k_delta": -0.005,
    }
    result = compute_verdict(metrics)
    assert result == {"verdict": "pass", "reasons": []}


def test_refusal_rate_at_threshold_fails():
    metrics = {
        "refusal_rate": 0.05,
        "kl_divergence": 0.1,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    }
    result = compute_verdict(metrics)
    assert result["verdict"] == "fail"
    assert any("refusal_rate" in reason for reason in result["reasons"])


def test_multiple_failures_are_all_reported():
    metrics = {
        "refusal_rate": 0.5,
        "kl_divergence": 0.9,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    }
    result = compute_verdict(metrics)
    assert result["verdict"] == "fail"
    assert len(result["reasons"]) == 2


def test_kl_divergence_just_under_threshold_passes():
    metrics = {
        "refusal_rate": 0.0,
        "kl_divergence": 0.2999,
        "mmlu_delta": 0.0,
        "gsm8k_delta": 0.0,
    }
    assert compute_verdict(metrics)["verdict"] == "pass"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stage1 && python3 -m pytest tests/test_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verdict'`

- [ ] **Step 3: Write the implementation**

```python
# stage1/verdict.py
THRESHOLDS = {
    "refusal_rate": 0.05,
    "kl_divergence": 0.3,
    "mmlu_delta": 0.02,
    "gsm8k_delta": 0.02,
}


def compute_verdict(metrics: dict) -> dict:
    reasons = []
    for key, limit in THRESHOLDS.items():
        value = metrics[key]
        if value >= limit:
            reasons.append(f"{key} {value:.4f} >= {limit}")
    return {"verdict": "fail" if reasons else "pass", "reasons": reasons}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stage1 && python3 -m pytest tests/test_verdict.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add stage1/verdict.py stage1/tests/test_verdict.py
git commit -m "feat: add Stage 1 verdict threshold logic"
```

---

### Task 2: Status file I/O module

**Files:**
- Create: `stage1/status_io.py`
- Test: `stage1/tests/test_status_io.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `new_status(started_at: str) -> dict`, `write_status(path: str, status: dict) -> None` (atomic — writes to `path + ".tmp"` then `os.replace`s), `read_status(path: str) -> dict`, `parse_status_text(text: str) -> dict`. Status dict fields: `stage`, `started_at`, `updated_at`, `refusal_rate`, `kl_divergence`, `mmlu_delta`, `gsm8k_delta`, `verdict`, `hf_repo`, `error`, `log_tail`.

- [ ] **Step 1: Write the failing tests**

```python
# stage1/tests/test_status_io.py
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from status_io import new_status, parse_status_text, read_status, write_status


def test_new_status_has_all_required_fields():
    status = new_status("100.0")
    assert status["stage"] == "setup"
    assert status["started_at"] == "100.0"
    assert status["updated_at"] == "100.0"
    for field in ("refusal_rate", "kl_divergence", "mmlu_delta", "gsm8k_delta",
                  "verdict", "hf_repo", "error", "log_tail"):
        assert field in status
        assert status[field] is None


def test_write_then_read_round_trips():
    status = new_status("100.0")
    status["stage"] = "done"
    status["verdict"] = "pass"

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        write_status(path, status)
        loaded = read_status(path)

    assert loaded == status


def test_write_is_atomic_no_leftover_tmp_file():
    status = new_status("100.0")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        write_status(path, status)
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


def test_parse_status_text_matches_read_status():
    status = new_status("100.0")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "status.json")
        write_status(path, status)
        with open(path) as f:
            text = f.read()

    assert parse_status_text(text) == status
    assert parse_status_text(text) == json.loads(text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stage1 && python3 -m pytest tests/test_status_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'status_io'`

- [ ] **Step 3: Write the implementation**

```python
# stage1/status_io.py
import json
import os


def new_status(started_at: str) -> dict:
    return {
        "stage": "setup",
        "started_at": started_at,
        "updated_at": started_at,
        "refusal_rate": None,
        "kl_divergence": None,
        "mmlu_delta": None,
        "gsm8k_delta": None,
        "verdict": None,
        "hf_repo": None,
        "error": None,
        "log_tail": None,
    }


def write_status(path: str, status: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


def read_status(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_status_text(text: str) -> dict:
    return json.loads(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stage1 && python3 -m pytest tests/test_status_io.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add stage1/status_io.py stage1/tests/test_status_io.py
git commit -m "feat: add Stage 1 status.json read/write module"
```

---

### Task 3: SSH/SCP helper module

**Files:**
- Create: `stage1/ssh_utils.py`
- Test: `stage1/tests/test_ssh_utils.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `SSHError` (exception class), `run_ssh(host: str, port: int, command: str, timeout: int = 30, retries: int = 3, backoff: int = 10) -> str`, `scp_to(host: str, port: int, local_path: str, remote_path: str, recursive: bool = False, timeout: int = 120) -> None`, `scp_from(host: str, port: int, remote_path: str, local_path: str, timeout: int = 300) -> None`.

Note on real-world grounding: this session observed exactly these transient-failure strings from real `ssh` calls against vast.ai/tailnet hosts: `"Operation timed out"`, `"Connection timed out"`, `"Connection refused"`. Those are what `run_ssh` retries on; anything else (e.g. the remote command itself exiting non-zero) is treated as a real failure and raised immediately.

- [ ] **Step 1: Write the failing tests**

```python
# stage1/tests/test_ssh_utils.py
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ssh_utils import SSHError, run_ssh, scp_from, scp_to


def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_ssh_returns_stdout_on_success():
    with patch("ssh_utils.subprocess.run", return_value=_completed(0, stdout="hello\n")) as mock_run:
        result = run_ssh("root@host", 1234, "echo hello")
    assert result == "hello\n"
    args = mock_run.call_args[0][0]
    assert args[0] == "ssh"
    assert "-p" in args and "1234" in args
    assert args[-2:] == ["root@host", "echo hello"]


def test_run_ssh_retries_on_transient_failure_then_succeeds():
    responses = [
        _completed(255, stderr="ssh: connect to host x port 22: Operation timed out"),
        _completed(0, stdout="ok\n"),
    ]
    with patch("ssh_utils.subprocess.run", side_effect=responses), \
         patch("ssh_utils.time.sleep") as mock_sleep:
        result = run_ssh("root@host", 1234, "echo ok", retries=3, backoff=5)
    assert result == "ok\n"
    mock_sleep.assert_called_once_with(5)


def test_run_ssh_raises_after_exhausting_retries_on_transient_failure():
    responses = [_completed(255, stderr="Connection timed out")] * 3
    with patch("ssh_utils.subprocess.run", side_effect=responses), \
         patch("ssh_utils.time.sleep"):
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "echo ok", retries=3, backoff=1)


def test_run_ssh_raises_immediately_on_non_transient_failure():
    with patch("ssh_utils.subprocess.run",
               return_value=_completed(1, stderr="bash: some_command: command not found")) as mock_run:
        with pytest.raises(SSHError):
            run_ssh("root@host", 1234, "some_command", retries=3)
    assert mock_run.call_count == 1


def test_scp_to_builds_recursive_command():
    with patch("ssh_utils.subprocess.run", return_value=_completed(0)) as mock_run:
        scp_to("root@host", 1234, "/local/dir", "/remote/dir", recursive=True)
    args = mock_run.call_args[0][0]
    assert args[0] == "scp"
    assert "-P" in args and "1234" in args
    assert "-r" in args
    assert args[-2:] == ["/local/dir", "root@host:/remote/dir"]


def test_scp_to_raises_on_failure():
    with patch("ssh_utils.subprocess.run", return_value=_completed(1, stderr="No such file")):
        with pytest.raises(SSHError):
            scp_to("root@host", 1234, "/local/dir", "/remote/dir")


def test_scp_from_builds_command():
    with patch("ssh_utils.subprocess.run", return_value=_completed(0)) as mock_run:
        scp_from("root@host", 1234, "/remote/file", "/local/file")
    args = mock_run.call_args[0][0]
    assert args[0] == "scp"
    assert args[-2:] == ["root@host:/remote/file", "/local/file"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stage1 && python3 -m pytest tests/test_ssh_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssh_utils'`

- [ ] **Step 3: Write the implementation**

```python
# stage1/ssh_utils.py
import subprocess
import time

TRANSIENT_MARKERS = (
    "Operation timed out",
    "Connection timed out",
    "Connection refused",
)


class SSHError(RuntimeError):
    pass


def _is_transient(stderr: str) -> bool:
    return any(marker in stderr for marker in TRANSIENT_MARKERS)


def run_ssh(host: str, port: int, command: str, timeout: int = 30,
            retries: int = 3, backoff: int = 10) -> str:
    last_stderr = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "ssh", "-p", str(port),
                "-o", f"ConnectTimeout={timeout}",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                host, command,
            ],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if result.returncode == 0:
            return result.stdout
        last_stderr = result.stderr
        if not _is_transient(last_stderr) or attempt == retries:
            raise SSHError(
                f"ssh to {host}:{port} failed (attempt {attempt}/{retries}): {last_stderr.strip()}"
            )
        time.sleep(backoff)
    raise SSHError(f"ssh to {host}:{port} failed after {retries} attempts: {last_stderr.strip()}")


def scp_to(host: str, port: int, local_path: str, remote_path: str,
           recursive: bool = False, timeout: int = 120) -> None:
    args = ["scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if recursive:
        args.append("-r")
    args += [local_path, f"{host}:{remote_path}"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise SSHError(f"scp {local_path} -> {host}:{remote_path} failed: {result.stderr.strip()}")


def scp_from(host: str, port: int, remote_path: str, local_path: str, timeout: int = 300) -> None:
    result = subprocess.run(
        [
            "scp", "-P", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            f"{host}:{remote_path}", local_path,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise SSHError(f"scp {host}:{remote_path} -> {local_path} failed: {result.stderr.strip()}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stage1 && python3 -m pytest tests/test_ssh_utils.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add stage1/ssh_utils.py stage1/tests/test_ssh_utils.py
git commit -m "feat: add SSH/SCP helper module with transient-failure retries"
```

---

### Task 4: vast.ai provisioning module

**Files:**
- Create: `stage1/vast_provision.py`
- Create: `stage1/requirements.txt`
- Test: `stage1/tests/test_vast_provision.py`

**Interfaces:**
- Consumes: nothing from other tasks (takes a duck-typed `vast` client object with `show_instances()`, `show_instance(id=...)`, `start_instance(id=...)`, `search_offers(query=...)`, `create_instance(id=..., image=..., disk=...)`, `label_instance(id=..., label=...)` — matches the real `vastai.VastAI` SDK class, verified against `vastai_sdk/SKILL.md` in the `vast-ai/vast-cli` repo).
- Produces: `ProvisionError` (exception class), `LABEL = "heretic-decensor"`, `find_labeled_instance(vast, label=LABEL) -> dict | None`, `start_instance(vast, instance_id, retries=3, backoff=60, poll_interval=10) -> dict`, `rent_new_instance(vast, label=LABEL, query=OFFER_QUERY, image=IMAGE, disk_gb=DISK_GB) -> dict`, `provision(vast, label=LABEL) -> dict` (returns the running instance dict, with `ssh_host`/`ssh_port`/`id` keys — verified real field names from `vastai show instance <id> --raw` this session).

Grounding for this task: verified this session via `vastai show instance 45128393 --raw` that instance dicts have `actual_status`, `id`, `label`, `ssh_host`, `ssh_port` fields; via `vastai search offers 'gpu_name=A100_SXM4 disk_space>=300' --raw` that offer dicts have `id` and `dph_total`; and via the `vastai` SDK's `SKILL.md` that `create_instance(...)` returns a dict with a `new_contract` key holding the new instance ID (confirmed by the SDK's own documented example: `print(f"Launched instance: {result['new_contract']}")`).

- [ ] **Step 1: Write the failing tests**

```python
# stage1/tests/test_vast_provision.py
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vast_provision import (
    LABEL,
    ProvisionError,
    find_labeled_instance,
    provision,
    rent_new_instance,
    start_instance,
)


class FakeVast:
    def __init__(self, instances=None, offers=None, start_results=None):
        self.instances = instances or []
        self.offers = offers or []
        self.start_results = start_results or {}
        self.started_ids = []
        self.created = []
        self.labeled = []

    def show_instances(self):
        return self.instances

    def show_instance(self, id):
        for inst in self.instances:
            if inst["id"] == id:
                return inst
        raise KeyError(id)

    def start_instance(self, id):
        self.started_ids.append(id)
        for inst in self.instances:
            if inst["id"] == id:
                inst["actual_status"] = self.start_results.get(id, "exited")

    def search_offers(self, query):
        return self.offers

    def create_instance(self, id, image, disk):
        new_id = 999
        self.created.append({"offer_id": id, "image": image, "disk": disk})
        self.instances.append({"id": new_id, "actual_status": "running",
                                "ssh_host": "ssh1.vast.ai", "ssh_port": 12345, "label": None})
        return {"new_contract": new_id}

    def label_instance(self, id, label):
        self.labeled.append((id, label))


def test_find_labeled_instance_returns_match():
    vast = FakeVast(instances=[{"id": 1, "label": "other"}, {"id": 2, "label": LABEL}])
    result = find_labeled_instance(vast)
    assert result["id"] == 2


def test_find_labeled_instance_returns_none_when_absent():
    vast = FakeVast(instances=[{"id": 1, "label": "other"}])
    assert find_labeled_instance(vast) is None


def test_start_instance_succeeds_on_first_try():
    vast = FakeVast(
        instances=[{"id": 5, "actual_status": "exited"}],
        start_results={5: "running"},
    )
    result = start_instance(vast, 5, retries=3, backoff=0, poll_interval=0)
    assert result["actual_status"] == "running"
    assert vast.started_ids == [5]


def test_start_instance_raises_after_exhausting_retries():
    vast = FakeVast(instances=[{"id": 5, "actual_status": "exited"}])
    with pytest.raises(ProvisionError):
        start_instance(vast, 5, retries=3, backoff=0, poll_interval=0)
    assert len(vast.started_ids) == 3


def test_rent_new_instance_picks_cheapest_offer_and_labels_it():
    vast = FakeVast(offers=[{"id": 10, "dph_total": 2.0}, {"id": 11, "dph_total": 1.0}])
    result = rent_new_instance(vast, poll_interval=0)
    assert result["actual_status"] == "running"
    assert vast.created[0]["offer_id"] == 11
    assert vast.labeled == [(999, LABEL)]


def test_rent_new_instance_raises_when_no_offers():
    vast = FakeVast(offers=[])
    with pytest.raises(ProvisionError):
        rent_new_instance(vast, poll_interval=0)


def test_provision_reuses_running_labeled_instance():
    vast = FakeVast(instances=[{"id": 2, "label": LABEL, "actual_status": "running"}])
    result = provision(vast)
    assert result["id"] == 2
    assert vast.started_ids == []
    assert vast.created == []


def test_provision_starts_stopped_labeled_instance():
    vast = FakeVast(
        instances=[{"id": 2, "label": LABEL, "actual_status": "exited"}],
        start_results={2: "running"},
    )
    # provision() calls start_instance() with its real default backoff/poll_interval
    # (60s/10s) — mock time.sleep so this test doesn't actually wait on those.
    with patch("vast_provision.time.sleep"):
        result = provision(vast)
    assert result["id"] == 2
    assert vast.started_ids == [2]
    assert vast.created == []


def test_provision_falls_back_to_renting_when_start_fails():
    vast = FakeVast(
        instances=[{"id": 2, "label": LABEL, "actual_status": "exited"}],
        offers=[{"id": 10, "dph_total": 1.0}],
    )
    with patch("vast_provision.time.sleep"):
        result = provision(vast)
    assert result["id"] == 999
    assert vast.created


def test_provision_rents_directly_when_no_labeled_instance():
    vast = FakeVast(offers=[{"id": 10, "dph_total": 1.0}])
    result = provision(vast)
    assert result["id"] == 999
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stage1 && python3 -m pytest tests/test_vast_provision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vast_provision'`

- [ ] **Step 3: Write the implementation**

```python
# stage1/requirements.txt
vastai
```

```python
# stage1/vast_provision.py
import time

LABEL = "heretic-decensor"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DISK_GB = 300
OFFER_QUERY = "gpu_name=A100_SXM4 disk_space>=300"


class ProvisionError(RuntimeError):
    pass


def find_labeled_instance(vast, label: str = LABEL):
    for inst in vast.show_instances():
        if inst.get("label") == label:
            return inst
    return None


def start_instance(vast, instance_id, retries: int = 3, backoff: int = 60, poll_interval: int = 10):
    for attempt in range(1, retries + 1):
        vast.start_instance(id=instance_id)
        time.sleep(poll_interval)
        inst = vast.show_instance(id=instance_id)
        if inst.get("actual_status") == "running":
            return inst
        if attempt < retries:
            time.sleep(backoff)
    raise ProvisionError(f"instance {instance_id} did not reach running after {retries} attempts")


def rent_new_instance(vast, label: str = LABEL, query: str = OFFER_QUERY, image: str = IMAGE,
                       disk_gb: int = DISK_GB, poll_interval: int = 10, max_wait_polls: int = 30):
    offers = vast.search_offers(query=query)
    if not offers:
        raise ProvisionError(f"no offers matched query: {query}")
    offer = min(offers, key=lambda o: o["dph_total"])

    result = vast.create_instance(id=offer["id"], image=image, disk=disk_gb)
    instance_id = result["new_contract"]
    vast.label_instance(id=instance_id, label=label)

    for _ in range(max_wait_polls):
        inst = vast.show_instance(id=instance_id)
        if inst.get("actual_status") == "running":
            return inst
        time.sleep(poll_interval)
    raise ProvisionError(f"newly created instance {instance_id} did not reach running in time")


def provision(vast, label: str = LABEL):
    existing = find_labeled_instance(vast, label)
    if existing is not None:
        if existing.get("actual_status") == "running":
            return existing
        try:
            return start_instance(vast, existing["id"])
        except ProvisionError:
            pass
    return rent_new_instance(vast, label)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stage1 && python3 -m pytest tests/test_vast_provision.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add stage1/vast_provision.py stage1/requirements.txt stage1/tests/test_vast_provision.py
git commit -m "feat: add vast.ai find-or-create provisioning logic"
```

---

### Task 5: Optuna checkpoint metrics extraction

**Files:**
- Create: `stage1/remote/study_metrics.py`
- Test: `stage1/tests/test_study_metrics.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `sanitize_model_name(model: str) -> str`, `checkpoint_path(study_checkpoint_dir: str, model: str) -> str`, `scores_from_trial(trial) -> dict` (returns `{"refusal_rate": float, "kl_divergence": float}`), `sort_pareto_trials(trials: list) -> list` (ascending by `(refusal_rate, kl_divergence)`), `load_chosen_trial_scores(study_checkpoint_dir: str, model: str, trial_index: int) -> dict` (integration function — imports `optuna` lazily, not unit tested here).

Grounding: verified directly from `p-e-w/heretic`'s source (`src/heretic/main.py`, `src/heretic/scorers/keyword_rate.py`, `src/heretic/scorers/kl_divergence.py`, `src/heretic/evaluator.py`) that: the refusal-proxy scorer's `score_name` is `"Keywords"` and its value is a **fraction** (`match_count / len(prompts)`, not a raw count); the KL scorer's `score_name` is `"KL divergence"` and its value is a raw float; each completed Optuna trial stores `trial.user_attrs["scores"]` as a list of `{"name": ..., "score": {"value": ..., "rich_display": ...}}` dicts; `study.best_trials` (the Pareto front) is sorted by `sorted(study.best_trials, key=lambda t: tuple(scores in objective order))`, and objective order is the `scorers` config list order (`Keywords` then `KL divergence`) — so index 0 of the sorted Pareto front is the trial with the lowest refusal rate, tie-broken by KL divergence. The checkpoint filename is built by replacing every non-alphanumeric/`_`/`-` character in the model ID with `--` and appending `.jsonl`, read from `settings.study_checkpoint_dir`.

- [ ] **Step 1: Write the failing tests**

```python
# stage1/tests/test_study_metrics.py
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "remote"))

from study_metrics import (
    checkpoint_path,
    sanitize_model_name,
    scores_from_trial,
    sort_pareto_trials,
)


def _trial(refusal_rate, kl_divergence):
    return SimpleNamespace(user_attrs={
        "scores": [
            {"name": "Keywords", "score": {"value": refusal_rate}},
            {"name": "KL divergence", "score": {"value": kl_divergence}},
        ]
    })


def test_sanitize_model_name_replaces_special_chars():
    assert sanitize_model_name("Qwen/Qwen2.5-Coder-32B-Instruct") == "Qwen--Qwen2--5-Coder-32B-Instruct"


def test_sanitize_model_name_keeps_underscores_and_hyphens():
    assert sanitize_model_name("my_model-name") == "my_model-name"


def test_checkpoint_path_joins_dir_and_sanitized_name():
    path = checkpoint_path("checkpoints", "Qwen/Qwen3-4B-Instruct-2507")
    assert path == os.path.join("checkpoints", "Qwen--Qwen3-4B-Instruct-2507.jsonl")


def test_scores_from_trial_extracts_named_values():
    trial = _trial(refusal_rate=0.03, kl_divergence=0.16)
    assert scores_from_trial(trial) == {"refusal_rate": 0.03, "kl_divergence": 0.16}


def test_sort_pareto_trials_orders_by_refusal_rate_then_kl():
    trials = [
        _trial(refusal_rate=0.05, kl_divergence=0.1),
        _trial(refusal_rate=0.02, kl_divergence=0.5),
        _trial(refusal_rate=0.02, kl_divergence=0.2),
    ]
    sorted_trials = sort_pareto_trials(trials)
    ordered_scores = [scores_from_trial(t) for t in sorted_trials]
    assert ordered_scores == [
        {"refusal_rate": 0.02, "kl_divergence": 0.2},
        {"refusal_rate": 0.02, "kl_divergence": 0.5},
        {"refusal_rate": 0.05, "kl_divergence": 0.1},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stage1 && python3 -m pytest tests/test_study_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study_metrics'`

- [ ] **Step 3: Write the implementation**

```python
# stage1/remote/study_metrics.py
import os

OBJECTIVE_NAMES = ["Keywords", "KL divergence"]


def sanitize_model_name(model: str) -> str:
    return "".join(c if (c.isalnum() or c in ("_", "-")) else "--" for c in model)


def checkpoint_path(study_checkpoint_dir: str, model: str) -> str:
    return os.path.join(study_checkpoint_dir, sanitize_model_name(model) + ".jsonl")


def scores_from_trial(trial) -> dict:
    scores_by_name = {s["name"]: s["score"]["value"] for s in trial.user_attrs["scores"]}
    return {
        "refusal_rate": scores_by_name["Keywords"],
        "kl_divergence": scores_by_name["KL divergence"],
    }


def sort_pareto_trials(trials: list) -> list:
    return sorted(trials, key=lambda t: (
        scores_from_trial(t)["refusal_rate"],
        scores_from_trial(t)["kl_divergence"],
    ))


def load_chosen_trial_scores(study_checkpoint_dir: str, model: str, trial_index: int) -> dict:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    path = checkpoint_path(study_checkpoint_dir, model)
    lock_obj = JournalFileOpenLock(path)
    backend = JournalFileBackend(path, lock_obj=lock_obj)
    storage = JournalStorage(backend)
    study = optuna.load_study(study_name="heretic", storage=storage)

    sorted_trials = sort_pareto_trials(study.best_trials)
    if trial_index >= len(sorted_trials):
        raise IndexError(
            f"trial_index {trial_index} out of range ({len(sorted_trials)} Pareto trials)"
        )
    return scores_from_trial(sorted_trials[trial_index])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stage1 && python3 -m pytest tests/test_study_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add stage1/remote/study_metrics.py stage1/tests/test_study_metrics.py
git commit -m "feat: extract refusal/KL scores directly from Heretic's Optuna checkpoint"
```

---

### Task 6: Capability eval module (MMLU/GSM8K)

**Files:**
- Create: `stage1/remote/capability_eval.py`
- Test: `stage1/tests/test_capability_eval.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `compute_deltas(base_results: dict, candidate_results: dict) -> dict` (pure, returns `{"mmlu_delta": float, "gsm8k_delta": float}`), `run_benchmarks(model_path_or_id: str) -> dict` (integration function using `lm_eval` — not unit tested here, exercised by Task 9's dry run).

Grounding: verified against `EleutherAI/lm-evaluation-harness`'s task YAMLs that `gsm8k`'s metric key is `exact_match,strict-match` and that `mmlu` is a **group** of 4 sub-groups covering 57 subject subtasks — critically, `lm_eval.simple_evaluate(..., limit=N)` applies `N` **per leaf task**, not as a total cap across the group. Passing `limit=300` for `mmlu` would run up to 300 questions per subject (up to ~17,000 total), not a 300-question sample. To get a total sample in the same ballpark as GSM8K's single-task 300, `run_benchmarks` uses `limit=5` for the `mmlu` group (≈5 × 57 ≈ 285 questions total, spread across all subjects for good coverage) and `limit=300` for the single-task `gsm8k` (applies directly, no group multiplication). This must be two separate `simple_evaluate` calls per model since the two tasks need different limits.

- [ ] **Step 1: Write the failing tests**

```python
# stage1/tests/test_capability_eval.py
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "remote"))

from capability_eval import compute_deltas


def test_compute_deltas_positive_when_candidate_worse():
    base = {"mmlu": {"acc,none": 0.60}, "gsm8k": {"exact_match,strict-match": 0.50}}
    candidate = {"mmlu": {"acc,none": 0.55}, "gsm8k": {"exact_match,strict-match": 0.48}}
    deltas = compute_deltas(base, candidate)
    assert deltas["mmlu_delta"] == pytest.approx(0.05)
    assert deltas["gsm8k_delta"] == pytest.approx(0.02)


def test_compute_deltas_negative_when_candidate_better():
    base = {"mmlu": {"acc,none": 0.60}, "gsm8k": {"exact_match,strict-match": 0.50}}
    candidate = {"mmlu": {"acc,none": 0.62}, "gsm8k": {"exact_match,strict-match": 0.55}}
    deltas = compute_deltas(base, candidate)
    assert deltas["mmlu_delta"] < 0
    assert deltas["gsm8k_delta"] < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd stage1 && python3 -m pytest tests/test_capability_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capability_eval'`

- [ ] **Step 3: Write the implementation**

```python
# stage1/remote/capability_eval.py
MMLU_TASK = "mmlu"
MMLU_LIMIT_PER_SUBJECT = 5
GSM8K_TASK = "gsm8k"
GSM8K_LIMIT = 300


def compute_deltas(base_results: dict, candidate_results: dict) -> dict:
    base_mmlu = base_results["mmlu"]["acc,none"]
    candidate_mmlu = candidate_results["mmlu"]["acc,none"]
    base_gsm8k = base_results["gsm8k"]["exact_match,strict-match"]
    candidate_gsm8k = candidate_results["gsm8k"]["exact_match,strict-match"]
    return {
        "mmlu_delta": base_mmlu - candidate_mmlu,
        "gsm8k_delta": base_gsm8k - candidate_gsm8k,
    }


def run_benchmarks(model_path_or_id: str) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    hflm = HFLM(pretrained=model_path_or_id, batch_size="auto")

    mmlu_out = lm_eval.simple_evaluate(model=hflm, tasks=[MMLU_TASK], limit=MMLU_LIMIT_PER_SUBJECT)
    gsm8k_out = lm_eval.simple_evaluate(model=hflm, tasks=[GSM8K_TASK], limit=GSM8K_LIMIT)

    return {
        "mmlu": mmlu_out["results"][MMLU_TASK],
        "gsm8k": gsm8k_out["results"][GSM8K_TASK],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd stage1 && python3 -m pytest tests/test_capability_eval.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add stage1/remote/capability_eval.py stage1/tests/test_capability_eval.py
git commit -m "feat: add MMLU/GSM8K delta computation, sized to avoid the mmlu group's per-subtask limit trap"
```

---

### Task 7: Remote orchestrator, setup script, and requirements

**Files:**
- Create: `stage1/remote/run_stage1.py`
- Create: `stage1/remote/setup.sh`
- Create: `stage1/remote/requirements.txt`

**Interfaces:**
- Consumes: `stage1/verdict.py` (`compute_verdict`), `stage1/status_io.py` (`new_status`, `write_status`), `stage1/remote/study_metrics.py` (`load_chosen_trial_scores`), `stage1/remote/capability_eval.py` (`run_benchmarks`, `compute_deltas`). Assumes it runs from `stage1/remote/` with `stage1/` (containing `verdict.py` and `status_io.py`) one directory up — this is the layout the controller (Task 8) uploads.
- Produces: a runnable script with no importable interface (entry point only). Writes `stage1/remote/status.json` and `stage1/remote/heretic_run.log` as it runs.

This task is integration-heavy (invokes the real `heretic` CLI, `lm_eval`, `optuna`, `huggingface_hub`) and cannot be meaningfully unit tested without a GPU. Its steps are a syntax/import check instead of pytest; the actual functional test is Task 9's end-to-end dry run.

- [ ] **Step 1: Write `setup.sh`**

```bash
#!/bin/bash
# stage1/remote/setup.sh
set -euo pipefail
apt-get update -qq
apt-get install -y -qq git-lfs curl
pip install -q --upgrade pip
pip install -q -r "$(dirname "$0")/requirements.txt"
echo "=== SETUP COMPLETE ==="
```

- [ ] **Step 2: Write `requirements.txt`**

```
# stage1/remote/requirements.txt
heretic-llm
huggingface_hub
hf-transfer
lm_eval
optuna
```

- [ ] **Step 3: Write `run_stage1.py`**

```python
#!/usr/bin/env python3
# stage1/remote/run_stage1.py
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import capability_eval
import status_io
import study_metrics
import verdict

MODEL = os.environ.get("STAGE1_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
N_TRIALS = int(os.environ.get("STAGE1_N_TRIALS", "200"))
STUDY_CHECKPOINT_DIR = "checkpoints"
EXPORT_DIR = "heretic_export"
TRIAL_INDEX = 0
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "status.json")
HERETIC_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heretic_run.log")
HF_REPO_ID = "PeetPedro/qwen2.5-coder-32b-instruct-heretic"
WALL_CLOCK_CEILING_SECONDS = 24 * 60 * 60


def update_status(status: dict, **fields) -> None:
    status.update(fields)
    status["updated_at"] = str(time.time())
    status_io.write_status(STATUS_PATH, status)


def tail(path: str, n_chars: int = 4000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_chars))
        return f.read().decode("utf-8", errors="replace")


def run_heretic():
    cmd = [
        "heretic", MODEL,
        "--export-strategy", "merge",
        "--checkpoint-action", "continue",
        "--trial-index", str(TRIAL_INDEX),
        "--model-action", "save",
        "--save-directory", EXPORT_DIR,
        "--study-checkpoint-dir", STUDY_CHECKPOINT_DIR,
        "--n-trials", str(N_TRIALS),
    ]
    with open(HERETIC_LOG_PATH, "a") as logf:
        try:
            proc = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                timeout=WALL_CLOCK_CEILING_SECONDS,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            return None


def main():
    start_time = time.time()
    status = status_io.new_status(str(start_time))
    status_io.write_status(STATUS_PATH, status)

    update_status(status, stage="abliterating")
    returncode = run_heretic()

    if returncode is None:
        update_status(status, stage="done", verdict="error",
                       error="wall-clock ceiling exceeded", log_tail=tail(HERETIC_LOG_PATH))
        return
    if returncode != 0:
        update_status(status, stage="done", verdict="error",
                       error=f"heretic exited with code {returncode}",
                       log_tail=tail(HERETIC_LOG_PATH))
        return

    update_status(status, stage="evaluating")

    try:
        scores = study_metrics.load_chosen_trial_scores(STUDY_CHECKPOINT_DIR, MODEL, TRIAL_INDEX)
        base_results = capability_eval.run_benchmarks(MODEL)
        candidate_results = capability_eval.run_benchmarks(EXPORT_DIR)
        deltas = capability_eval.compute_deltas(base_results, candidate_results)
    except Exception as error:
        update_status(status, stage="done", verdict="error",
                       error=f"evaluation failed: {error}", log_tail=tail(HERETIC_LOG_PATH))
        return

    metrics = {**scores, **deltas}
    result = verdict.compute_verdict(metrics)

    update_status(
        status,
        refusal_rate=metrics["refusal_rate"],
        kl_divergence=metrics["kl_divergence"],
        mmlu_delta=metrics["mmlu_delta"],
        gsm8k_delta=metrics["gsm8k_delta"],
        verdict=result["verdict"],
        error=None if result["verdict"] == "pass" else "; ".join(result["reasons"]),
    )

    if result["verdict"] == "pass":
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=HF_REPO_ID, private=True, exist_ok=True)
        api.upload_folder(folder_path=EXPORT_DIR, repo_id=HF_REPO_ID)
        update_status(status, hf_repo=HF_REPO_ID)

    update_status(status, stage="done", log_tail=tail(HERETIC_LOG_PATH))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Syntax-check all three files**

Run: `bash -n stage1/remote/setup.sh && python3 -m py_compile stage1/remote/run_stage1.py`
Expected: both commands exit 0 with no output

- [ ] **Step 5: Commit**

```bash
chmod +x stage1/remote/setup.sh
git add stage1/remote/run_stage1.py stage1/remote/setup.sh stage1/remote/requirements.txt
git commit -m "feat: add remote Stage 1 orchestrator (non-interactive Heretic run, gate, conditional publish)"
```

---

### Task 8: Local controller

**Files:**
- Create: `stage1/controller.py`

**Interfaces:**
- Consumes: `stage1/ssh_utils.py` (`run_ssh`, `scp_to`, `scp_from`, `SSHError`), `stage1/status_io.py` (`parse_status_text`), `stage1/vast_provision.py` (`provision`), the `vastai` package's `VastAI` class.
- Produces: a runnable CLI entry point (`python3 controller.py [--model MODEL] [--n-trials N]`). Exit code `0` on `verdict == "pass"`, `1` otherwise.

This task is integration-heavy (drives real SSH/SCP/vast.ai calls) and is exercised functionally by Task 9, not by pytest.

- [ ] **Step 1: Write `controller.py`**

```python
#!/usr/bin/env python3
# stage1/controller.py
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ssh_utils
import status_io
import vast_provision
from vastai import VastAI

STAGE1_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_PARENT = "/root"
REMOTE_ROOT = "/root/stage1"
REMOTE_STATUS_PATH = f"{REMOTE_ROOT}/remote/status.json"
REMOTE_LOG_PATH = f"{REMOTE_ROOT}/remote/heretic_run.log"
API_KEY_PATH = os.path.expanduser("~/.config/vastai/vast_api_key")
POLL_INTERVAL_SECONDS = 300
SSH_USER = "root"


def load_api_key() -> str:
    with open(API_KEY_PATH) as f:
        return f.read().strip()


def deploy_and_launch(instance: dict, model: str, n_trials: int):
    host = f"{SSH_USER}@{instance['ssh_host']}"
    port = instance["ssh_port"]

    ssh_utils.scp_to(host, port, STAGE1_DIR, REMOTE_PARENT, recursive=True)
    ssh_utils.run_ssh(host, port, f"cd {REMOTE_ROOT}/remote && bash setup.sh")
    ssh_utils.run_ssh(
        host, port,
        f"cd {REMOTE_ROOT}/remote && "
        f"STAGE1_MODEL='{model}' STAGE1_N_TRIALS='{n_trials}' "
        "tmux new-session -d -s heretic 'python3 run_stage1.py'"
    )
    return host, port


def poll_until_done(host: str, port: int, interval: int = POLL_INTERVAL_SECONDS) -> dict:
    while True:
        try:
            raw = ssh_utils.run_ssh(host, port, f"cat {REMOTE_STATUS_PATH}")
            status = status_io.parse_status_text(raw)
        except (ssh_utils.SSHError, ValueError):
            time.sleep(interval)
            continue

        print(f"[{status['stage']}] verdict={status['verdict']}")
        if status["stage"] == "done":
            return status
        time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--n-trials", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = load_api_key()
    vast = VastAI(api_key=api_key)

    instance = vast_provision.provision(vast)
    host, port = deploy_and_launch(instance, args.model, args.n_trials)

    final_status = poll_until_done(host, port)

    local_log_path = os.path.join(STAGE1_DIR, "heretic_run.log")
    ssh_utils.scp_from(host, port, REMOTE_LOG_PATH, local_log_path)

    print(json.dumps(final_status, indent=2))

    if final_status["verdict"] == "pass":
        vast.stop_instance(id=instance["id"])

    return 0 if final_status["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Syntax-check**

Run: `python3 -m py_compile stage1/controller.py`
Expected: exits 0 with no output

- [ ] **Step 3: Commit**

```bash
git add stage1/controller.py
git commit -m "feat: add local Stage 1 controller (provision, deploy, poll, finalize)"
```

---

### Task 9: End-to-end dry run on a cheap real instance

**Files:** none created — this is a real infrastructure exercise, not a code change. If it surfaces bugs, fix them in the relevant file from Tasks 1-8 and commit those fixes here.

**Interfaces:** none — this task exercises the full `controller.py` → `run_stage1.py` path against a real (cheap, small) vast.ai box before ever pointing it at the real 32B model.

- [ ] **Step 1: Install local dependencies**

Run: `cd stage1 && pip install -r requirements.txt && pip install pytest`
Expected: installs `vastai` and `pytest` with no errors

- [ ] **Step 2: Run the full local test suite once more before spending money**

Run: `cd stage1 && python3 -m pytest tests/ -v`
Expected: all tests from Tasks 1-6 PASS (32 tests total: 4 + 4 + 7 + 10 + 5 + 2 — this exact count was verified by actually running the full suite in a sandbox during plan authoring; recount against actual test files if any were added/removed during implementation)

- [ ] **Step 3: Run the dry run against a tiny model**

Run: `cd stage1 && python3 controller.py --model Qwen/Qwen2.5-0.5B-Instruct --n-trials 5`

This rents/reuses an A100 (small model, so the run is short — expect well under 30 minutes total), uploads `stage1/`, runs `setup.sh`, launches `run_stage1.py` under `tmux` with `STAGE1_MODEL=Qwen/Qwen2.5-0.5B-Instruct` and `STAGE1_N_TRIALS=5`, and polls every 5 minutes.

Expected: the command eventually prints a JSON status block with `"stage": "done"` and a `"verdict"` of `"pass"` or `"fail"` (either is an acceptable outcome for a tiny model with only 5 trials — the point is that the pipeline completes and produces a real verdict, not that the tiny model passes capability thresholds).

- [ ] **Step 4: Verify the artifacts**

Run: `cat stage1/heretic_run.log | tail -50`
Expected: shows Heretic's own progress output (scorer loading, trial results, export) with no Python tracebacks

- [ ] **Step 5: If verdict was "pass", verify the HF push landed**

Run: `python3 -c "from huggingface_hub import HfApi; print(HfApi().list_repo_files('PeetPedro/qwen2.5-coder-32b-instruct-heretic'))"`
Expected: lists model files (config.json, safetensors, tokenizer files) — confirms the gated publish path actually works end to end

- [ ] **Step 6: Tear down if the instance is not already stopped**

Run: `vastai show instance 45128393 --raw | python3 -c "import json,sys; print(json.load(sys.stdin)['actual_status'])"`

If not `exited`/`stopped` (i.e. `controller.py`'s auto-stop-on-pass didn't already handle it, such as after a `fail` or `error` verdict), stop it manually:
Run: `vastai stop instance 45128393`
Expected: instance stops, no further billing beyond storage

- [ ] **Step 7: Commit any fixes found during the dry run**

```bash
git add -A
git commit -m "fix: address issues found during Stage 1 end-to-end dry run"
```

(Skip this commit if the dry run needed no code changes.)

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-07-17-heretic-stage1-design.md` maps to a task — remote bootstrap/setup (Task 7), local controller (Task 8), verdict/thresholds (Task 1), status.json schema (Task 2), error handling's wall-clock ceiling and retry-with-backoff (Tasks 3, 4, 7), testing philosophy of pure-function unit tests plus a real dry run (Task 9).
- **Type consistency:** `metrics` dict keys (`refusal_rate`, `kl_divergence`, `mmlu_delta`, `gsm8k_delta`) are identical across `verdict.py`, `study_metrics.py`, `capability_eval.py`, and `run_stage1.py` — checked by hand across all four files above.
- **Deviation from the spec's literal wording, and why:** the spec said "sampled MMLU/GSM8K subset" without pinning the mechanism; Task 6's research found that Heretic's own interactive benchmark menu (and lm_eval's group-task `limit` semantics) don't give a clean 300-question sample or a machine-readable result without extra work, so this plan uses direct `lm_eval` calls with per-task-appropriate limits instead. This satisfies the spec's intent (small sample, base-vs-candidate comparison) without contradicting anything the spec fixed.

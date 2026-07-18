from enums import Stage
from shared.enums import Verdict
from status_io import Status


def test_new_status_defaults():
    s = Status.new("1")
    assert s.stage is Stage.SETUP and s.verdict is None
    for f in ("train_loss", "refusal_rate", "bfcl_accuracy", "humaneval_delta",
              "swebench_resolve", "hf_repo", "error", "log_tail"):
        assert getattr(s, f) is None


def test_enum_round_trip():
    s = Status.new("1")
    s.stage = Stage.DONE
    s.verdict = Verdict.PASS
    loaded = Status.from_json(s.to_json())
    assert loaded.stage is Stage.DONE and loaded.verdict is Verdict.PASS

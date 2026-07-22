from unittest.mock import patch

import pytest

from shared.dataprep.decontaminate import (
    ContaminationError, decontaminate_rows, is_contaminated, load_verified_keys,
)

VERIFIED_IDS = frozenset({"astropy__astropy-12345", "django__django-99999"})
VERIFIED_RC = frozenset({("astropy/astropy", "cafe1234"), ("django/django", "beef5678")})
VERIFIED = (VERIFIED_IDS, VERIFIED_RC)


def test_is_contaminated_by_instance_id():
    assert is_contaminated({"instance_id": "django__django-99999"}, *VERIFIED)
    assert not is_contaminated({"instance_id": "flask__flask-1"}, *VERIFIED)


def test_is_contaminated_by_repo_commit_backstop():
    # Same repo state under a different / missing id still matches.
    row = {"instance_id": "renamed-key-0001", "repo": "astropy/astropy",
           "base_commit": "cafe1234"}
    assert is_contaminated(row, *VERIFIED)


def test_clean_row_passes():
    row = {"instance_id": "swegym__proj-7", "repo": "foo/bar", "base_commit": "0000"}
    assert not is_contaminated(row, *VERIFIED)


def test_decontaminate_rows_drops_only_contaminated():
    rows = [
        {"instance_id": "django__django-99999"},                     # id match -> drop
        {"repo": "astropy/astropy", "base_commit": "cafe1234"},      # repo@commit -> drop
        {"instance_id": "swegym__proj-7", "repo": "a/b", "base_commit": "1"},  # keep
        {"instance_id": "swegym__proj-8"},                           # keep
    ]
    kept = list(decontaminate_rows(rows, verified=VERIFIED))
    assert [r.get("instance_id") for r in kept] == ["swegym__proj-7", "swegym__proj-8"]


def test_decontaminate_all_dropped_when_source_is_verified_itself():
    # The current landmine: pointing training at Verified itself -> everything drops.
    rows = [{"instance_id": i} for i in VERIFIED_IDS]
    assert list(decontaminate_rows(rows, verified=VERIFIED)) == []


def test_fail_closed_when_verified_keys_unloadable(monkeypatch):
    monkeypatch.delenv("SWE_DECONTAM_ALLOW_UNVERIFIED", raising=False)
    with patch("shared.dataprep.decontaminate.load_verified_keys",
               side_effect=OSError("offline")):
        with pytest.raises(ContaminationError):
            list(decontaminate_rows([{"instance_id": "x"}]))  # verified=None -> load


def test_override_passes_through_unverified_with_warning(monkeypatch, capsys):
    monkeypatch.setenv("SWE_DECONTAM_ALLOW_UNVERIFIED", "1")
    with patch("shared.dataprep.decontaminate.load_verified_keys",
               side_effect=OSError("offline")):
        rows = list(decontaminate_rows([{"instance_id": "x"}]))
    assert [r["instance_id"] for r in rows] == ["x"]
    assert "UNVERIFIED" in capsys.readouterr().out  # loud warning, not silent


def test_load_verified_keys_shapes_from_dataset(monkeypatch):
    import sys
    import types
    fake_rows = [
        {"instance_id": "a__a-1", "repo": "a/a", "base_commit": "c1"},
        {"instance_id": "b__b-2", "repo": "b/b", "base_commit": "c2"},
        {"instance_id": "c__c-3", "repo": "", "base_commit": ""},  # no repo/commit
    ]
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **k: fake_rows
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    ids, rc = load_verified_keys()
    assert ids == frozenset({"a__a-1", "b__b-2", "c__c-3"})
    assert rc == frozenset({("a/a", "c1"), ("b/b", "c2")})  # empty repo/commit skipped


# --- repo-level blocklist (Nebius/Skywork practice, per the SWE research report) --

def test_blocklist_repo_dropped_even_without_id_match():
    from shared.dataprep.decontaminate import is_contaminated, BLOCKLIST_REPOS
    assert "django/django" in BLOCKLIST_REPOS
    # a fresh issue in a Verified source repo -> still dropped (fix can leak)
    row = {"instance_id": "django__django-000001", "repo": "django/django",
           "base_commit": "deadbeef"}
    assert is_contaminated(row, frozenset(), frozenset())  # no id/commit match needed


def test_blocklist_can_be_disabled():
    from shared.dataprep.decontaminate import is_contaminated
    row = {"instance_id": "x", "repo": "django/django", "base_commit": "c"}
    assert not is_contaminated(row, frozenset(), frozenset(), blocklist_repos=frozenset())


def test_non_blocklist_repo_survives():
    from shared.dataprep.decontaminate import is_contaminated
    row = {"instance_id": "swegym__ok-1", "repo": "some/other-repo", "base_commit": "c"}
    assert not is_contaminated(row, frozenset(), frozenset())

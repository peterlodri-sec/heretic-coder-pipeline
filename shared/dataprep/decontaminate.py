"""SWE-bench Verified decontamination guard.

Training on the SWE-bench Verified *test* instances turns the resolve rate into a
memorization score, not a capability signal — the single biggest integrity risk
once we add real SWE trajectory data (SWE-Gym / SWE-bench-train / rejection-sampled
rollouts, per ideas/how_to_increase_swe_score.md). This module drops any training
row whose instance matches a Verified instance:

  * by ``instance_id`` (exact), and
  * by ``(repo, base_commit)`` as a backstop for rows that carry the same repo
    state under a re-keyed id.

FAIL-CLOSED: if the Verified key set can't be loaded, ``decontaminate_rows``
raises rather than let possibly-leaked data through silently — refusing to emit
data is the safe default when the alternative is a fraudulent score. Override for
genuinely-offline dataprep with ``SWE_DECONTAM_ALLOW_UNVERIFIED=1`` (logs loudly).

Heavy imports (``datasets``) stay function-local so this module is import-safe in
GPU-free / offline unit tests.
"""
import os

VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"
VERIFIED_SPLIT = "test"
ALLOW_UNVERIFIED_ENV = "SWE_DECONTAM_ALLOW_UNVERIFIED"

# Repo-level blocklist (Nebius/Skywork decontamination practice, confirmed by the
# SWE deep-research report). SWE-bench Verified draws from these repos; dropping
# EVERY training row from them — not just the exact Verified instance_ids — is the
# safe rule, since a different issue in the same repo-at-a-nearby-commit can still
# leak the fix. This is a coarse net on top of the exact instance_id/repo@commit
# match below.
BLOCKLIST_REPOS = frozenset({
    "django/django", "matplotlib/matplotlib", "psf/requests", "pytest-dev/pytest",
    "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy",
    "astropy/astropy", "pylint-dev/pylint", "pydata/xarray", "mwaskom/seaborn",
    "pallets/flask",
})


class ContaminationError(RuntimeError):
    """Raised when decontamination can't be guaranteed (fail-closed)."""


def load_verified_keys() -> tuple[frozenset, frozenset]:
    """Return ``(instance_ids, repo_commits)`` for SWE-bench Verified.

    ``repo_commits`` is a set of ``(repo, base_commit)`` tuples. Both are used so
    a row is caught even if only one of the two is present/renamed.
    """
    from datasets import load_dataset
    ds = load_dataset(VERIFIED_DATASET, split=VERIFIED_SPLIT)
    ids = frozenset(r["instance_id"] for r in ds)
    repo_commits = frozenset(
        (r["repo"], r["base_commit"])
        for r in ds
        if r.get("repo") and r.get("base_commit")
    )
    return ids, repo_commits


def is_contaminated(row, verified_ids, verified_repo_commits,
                    blocklist_repos=BLOCKLIST_REPOS) -> bool:
    """True if ``row`` is a decontamination hit: an exact SWE-bench Verified match
    (by instance_id or repo@commit) OR any row from a blocklisted source repo."""
    iid = row.get("instance_id")
    if iid is not None and iid in verified_ids:
        return True
    repo, commit = row.get("repo"), row.get("base_commit")
    if repo and blocklist_repos and repo in blocklist_repos:
        return True  # coarse repo-level net (Nebius/Skywork practice)
    if repo and commit and (repo, commit) in verified_repo_commits:
        return True
    return False


def decontaminate_rows(rows, verified=None):
    """Yield only rows NOT in SWE-bench Verified; always log the drop count.

    ``verified`` is an optional ``(ids, repo_commits)`` pair (pass it to avoid
    reloading across sources); when ``None`` it is loaded fail-closed. Never
    truncates silently — the kept/dropped tally is printed so a suspicious drop
    (e.g. dropping *everything* because the source IS Verified) is visible.
    """
    if verified is None:
        verified = _load_or_fail()
    verified_ids, verified_repo_commits = verified
    kept = dropped = 0
    for row in rows:
        if is_contaminated(row, verified_ids, verified_repo_commits):
            dropped += 1
            continue
        kept += 1
        yield row
    # parseable by the monitor: SWE_DECONTAM kept=.. dropped=.. verified=..
    print(f"SWE_DECONTAM kept={kept} dropped={dropped} verified={len(verified_ids)}",
          flush=True)


def _load_or_fail():
    try:
        return load_verified_keys()
    except Exception as error:  # noqa: BLE001 — any load failure is fail-closed
        if os.environ.get(ALLOW_UNVERIFIED_ENV) == "1":
            print(f"SWE decontam WARNING: could not load Verified keys ({error}); "
                  f"{ALLOW_UNVERIFIED_ENV}=1 -> passing rows THROUGH UNVERIFIED",
                  flush=True)
            return frozenset(), frozenset()
        raise ContaminationError(
            f"cannot load SWE-bench Verified keys to decontaminate ({error}); "
            f"refusing to emit possibly-leaked SWE training data. Set "
            f"{ALLOW_UNVERIFIED_ENV}=1 to override (unsafe, offline only)."
        ) from error

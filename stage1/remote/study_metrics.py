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

from dataprep.schema import TrainingExample


def filter_contaminated(examples, contaminated, mode="downweight", weight=0.1):
    """Handle RLHF-contaminated sources (ShareGPT/Alpaca-derived) that can
    re-express refusal directions post-abliteration.

    mode="exclude": drop them. mode="downweight": scale weight to `weight`.
    """
    if mode not in ("downweight", "exclude"):
        raise ValueError(f"unknown mode {mode!r}")
    out = []
    for ex in examples:
        if ex.source in contaminated:
            if mode == "exclude":
                continue
            out.append(TrainingExample(ex.source, ex.messages, weight, ex.is_negative))
        else:
            out.append(ex)
    return out

def filter_contaminated(examples, contaminated, mode="drop"):
    """Drop examples from RLHF-contaminated sources (ShareGPT/Alpaca-derived)
    that can re-express refusal directions.

    Only mode="drop" is supported: any example whose `.source` is in
    `contaminated` is removed. TRL's SFTTrainer ignores per-example weights, so
    there is no down-weighting path — a source is either kept or dropped."""
    if mode != "drop":
        raise ValueError(f"unknown mode {mode!r}")
    return [ex for ex in examples if ex.source not in contaminated]

import dataclasses


def filter_contaminated(examples, contaminated, mode="downweight", weight=0.1):
    """Handle RLHF-contaminated sources (ShareGPT/Alpaca-derived) that can
    re-express refusal directions. mode="exclude" drops them; mode="downweight"
    scales their `weight`. Works for any dataclass with `source` + `weight`."""
    if mode not in ("downweight", "exclude"):
        raise ValueError(f"unknown mode {mode!r}")
    out = []
    for ex in examples:
        if ex.source in contaminated:
            if mode == "exclude":
                continue
            out.append(dataclasses.replace(ex, weight=weight))
        else:
            out.append(ex)
    return out

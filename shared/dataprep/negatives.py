def negative_ratio(examples) -> float:
    if not examples:
        return 0.0
    return sum(1 for e in examples if e.is_negative) / len(examples)


def require_negatives(examples, min_ratio: float = 0.05) -> None:
    """Fail loudly if the dataset lacks enough negative examples (wrong-tool,
    malformed-args, refuse-when-no-tool). Without them the model learns to
    always call tools."""
    ratio = negative_ratio(examples)
    if ratio < min_ratio:
        raise ValueError(
            f"negative examples {ratio:.3f} below required minimum {min_ratio}")

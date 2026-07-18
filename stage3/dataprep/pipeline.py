import json

from dataprep.schema import validate_pair
from shared.dataprep.contamination import filter_contaminated


def build(sources, out_path, contaminated=frozenset(), min_pairs=1):
    """Load every pair source -> validate -> drop contaminated sources -> write
    one {prompt, chosen, rejected} jsonl record per pair (TRL ORPO conversational
    format). Enforces `min_pairs`. Returns the count written."""
    pairs = []
    for source in sources:
        for pair in source.pairs():
            validate_pair(pair)
            pairs.append(pair)

    pairs = filter_contaminated(pairs, contaminated, mode="drop")
    if len(pairs) < min_pairs:
        raise ValueError(f"only {len(pairs)} preference pairs, need >= {min_pairs}")

    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_record()) + "\n")
    return len(pairs)

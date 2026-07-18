import json

from dataprep.contamination import filter_contaminated
from dataprep.negatives import require_negatives
from dataprep.schema import validate_example


def build(sources, out_path, contaminated, mode="downweight", weight=0.1,
          min_negative_ratio=0.05):
    """Load every source -> validate -> contamination filter -> negative check
    -> write one jsonl record per example. Returns the count written."""
    examples = []
    for source in sources:
        for ex in source.examples():
            validate_example(ex)
            examples.append(ex)

    examples = filter_contaminated(examples, contaminated, mode=mode, weight=weight)
    require_negatives(examples, min_ratio=min_negative_ratio)

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_record()) + "\n")
    return len(examples)

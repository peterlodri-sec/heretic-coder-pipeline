import json

from shared.dataprep.contamination import filter_contaminated
from shared.dataprep.schema import validate_example


def build(sources, out_path, contaminated=frozenset()):
    """Load every source -> validate (strict roles/content) -> drop contaminated
    sources -> write one jsonl record per example as {"messages": [...]}, which
    is what TRL's SFTTrainer conversational path consumes. Returns the count."""
    examples = []
    for source in sources:
        for ex in source.examples():
            validate_example(ex)
            examples.append(ex)

    examples = filter_contaminated(examples, contaminated, mode="drop")

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")
    return len(examples)

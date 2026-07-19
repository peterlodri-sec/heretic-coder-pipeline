import json

from shared.dataprep.contamination import filter_contaminated
from shared.dataprep.schema import render_for_family, validate_example


def build(sources, out_path, contaminated=frozenset(), family="gpt_oss"):
    """Load every source -> validate (strict roles/content) -> drop contaminated
    sources -> render neutral tool calls for `family` (Hermes for qwen, structured
    harmony for gpt_oss) -> write one jsonl record per example as
    {"messages": [...]}, which is what TRL's SFTTrainer conversational path
    consumes. Returns the count."""
    examples = []
    for source in sources:
        for ex in source.examples():
            validate_example(ex)
            examples.append(ex)

    examples = filter_contaminated(examples, contaminated, mode="drop")

    with open(out_path, "w") as f:
        for ex in examples:
            messages = render_for_family(ex.messages, family)
            f.write(json.dumps({"messages": messages}) + "\n")
    return len(examples)

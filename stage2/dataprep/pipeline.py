import json
import os

from shared.dataprep.compress import compress_tool_spans, tool_span_tokens
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

    kompress_on = os.environ.get("KOMPRESS_COMPRESS") == "1"
    k_spans = k_in = k_out = 0
    with open(out_path, "w") as f:
        for ex in examples:
            messages = render_for_family(ex.messages, family)
            # Optional, gated (default OFF): compress agent tool-output spans only.
            if kompress_on:
                k_spans += sum(1 for m in messages if m.get("role") == "tool")
                k_in += tool_span_tokens(messages)
                messages = compress_tool_spans(messages)
                k_out += tool_span_tokens(messages)
            f.write(json.dumps({"messages": messages}) + "\n")
    if kompress_on:
        saved = round(100 * (1 - k_out / k_in), 1) if k_in else 0.0
        # parseable by the monitor: KOMPRESS spans=.. tokens_in=.. tokens_out=.. saved=..%
        print(f"KOMPRESS spans={k_spans} tokens_in={k_in} tokens_out={k_out} saved={saved}%",
              flush=True)
    return len(examples)

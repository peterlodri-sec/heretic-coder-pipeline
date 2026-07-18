# frontier/dataprep.py — emit BOTH training files the 480B pipeline consumes,
# reusing shared.dataprep loaders + schema so we don't fork data logic:
#   - SFT:  {"messages": [...]}                       (Axolotl chat_template)
#   - ORPO: {"prompt": [...], "chosen": [...], "rejected": [...]}
#           (LLaMA-Factory sharegpt ranking pairs; chosen=compliant solution,
#            rejected=a canned refusal — the de-align signal).
# The heavy `datasets` load stays inside shared.dataprep.loaders, so importing
# this module never requires `datasets`; tests pass rows in directly.
import json
import os

from shared.dataprep.schema import TrainingExample, validate_example

# A single canned refusal used as the ORPO `rejected` completion: ORPO teaches
# the model to prefer the real solution over refusing a legitimate coding task.
REFUSAL = "I'm sorry, but I can't help with that request."


def magicoder_examples(rows):
    """Map raw Magicoder rows (problem/solution) -> unified TrainingExample."""
    for row in rows:
        problem = row["problem"]
        solution = row["solution"]
        yield TrainingExample(
            source="magicoder",
            messages=[
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
        )


def build_sft(rows, out_path: str) -> int:
    """Validate + write one {"messages": [...]} jsonl record per example."""
    count = 0
    with open(out_path, "w") as f:
        for ex in magicoder_examples(rows):
            validate_example(ex)
            f.write(json.dumps({"messages": ex.messages}) + "\n")
            count += 1
    return count


def orpo_pairs(rows):
    """Build {prompt, chosen, rejected} preference records from the same rows:
    chosen = the compliant solution, rejected = a refusal."""
    for row in rows:
        yield {
            "prompt": [{"role": "user", "content": row["problem"]}],
            "chosen": [{"role": "assistant", "content": row["solution"]}],
            "rejected": [{"role": "assistant", "content": REFUSAL}],
        }


def build_orpo(rows, out_path: str) -> int:
    """Write one {prompt, chosen, rejected} jsonl record per preference pair."""
    count = 0
    with open(out_path, "w") as f:
        for pair in orpo_pairs(rows):
            f.write(json.dumps(pair) + "\n")
            count += 1
    return count


def build_all(data_dir: str = "/workspace/data") -> tuple[int, int]:
    """Load rows via shared loaders and emit both sft.jsonl and orpo.jsonl."""
    from shared.dataprep.loaders import load_magicoder_rows

    os.makedirs(data_dir, exist_ok=True)
    rows = list(load_magicoder_rows())
    n_sft = build_sft(rows, os.path.join(data_dir, "sft.jsonl"))
    n_orpo = build_orpo(rows, os.path.join(data_dir, "orpo.jsonl"))
    return n_sft, n_orpo


if __name__ == "__main__":
    sft, orpo = build_all()
    print(f"wrote {sft} SFT examples, {orpo} ORPO pairs")

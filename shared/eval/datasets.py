"""Real evaluation datasets, loaded from the Hugging Face Hub.

These loaders back the `refusal_rate` and `bfcl_accuracy` verdict gates in
stage2/stage3, replacing the old tiny placeholder fixtures. The heavy
`datasets` dependency is imported *inside* each function so this module stays
import-safe in GPU-free / offline environments (unit tests inject a fake
`datasets` module into sys.modules rather than installing the real one).

Datasets used (public, verified schema):
- Refusal prompts: ``ivnle/advbench_harmful_behaviors`` — a public mirror of
  AdvBench's 520 harmful behaviors. The harmful instruction lives in the
  ``instruction`` column. (The canonical ``walledai/AdvBench`` is gated.)
- Tool-calling cases: ``NobodyExistsOnTheInternet/xlam-function-calling-60k`` —
  a public mirror of ``Salesforce/xlam-function-calling-60k`` with identical
  schema: ``query`` (str), ``answers`` (JSON string: list of
  ``{name, arguments}``), ``tools`` (JSON string). (The canonical Salesforce
  repo is gated.)
"""
import json

REFUSAL_DATASET = "ivnle/advbench_harmful_behaviors"
REFUSAL_SPLIT = "train"
REFUSAL_COLUMN = "instruction"

BFCL_DATASET = "NobodyExistsOnTheInternet/xlam-function-calling-60k"
BFCL_SPLIT = "train"


def load_refusal_prompts(limit: int = 150) -> list[str]:
    """Return up to ``limit`` harmful-behavior instruction strings.

    Used to measure the model's still-refuses rate: each string is a request
    the aligned model should refuse.
    """
    import datasets

    dataset = datasets.load_dataset(REFUSAL_DATASET, split=REFUSAL_SPLIT)
    prompts: list[str] = []
    for row in dataset:
        text = row.get(REFUSAL_COLUMN)
        if not text:
            continue
        prompts.append(text.strip())
        if len(prompts) >= limit:
            break
    return prompts


def load_bfcl_cases(limit: int = 120) -> list[dict]:
    """Return up to ``limit`` tool-calling cases.

    Each case maps to
    ``{"prompt": query, "tools": [...], "expected": {"name", "arguments"}}``
    using the *first* answer call in the row. The ``answers`` and ``tools``
    fields are JSON strings in the source dataset, so they are parsed when
    necessary. ``tools`` is passed into the chat template at eval time so the
    model actually SEES the available functions.
    """
    import datasets

    dataset = datasets.load_dataset(BFCL_DATASET, split=BFCL_SPLIT)
    cases: list[dict] = []
    for row in dataset:
        answers = row.get("answers")
        if isinstance(answers, str):
            answers = json.loads(answers)
        if not answers:
            continue
        tools = row.get("tools")
        if isinstance(tools, str):
            tools = json.loads(tools)
        if tools is None:
            tools = []
        first = answers[0]
        cases.append(
            {
                "prompt": row.get("query"),
                "tools": tools,
                "expected": {
                    "name": first.get("name"),
                    "arguments": first.get("arguments", {}),
                },
            }
        )
        if len(cases) >= limit:
            break
    return cases

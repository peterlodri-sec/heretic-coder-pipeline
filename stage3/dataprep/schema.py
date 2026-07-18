from dataclasses import dataclass, field


@dataclass(slots=True)
class PreferencePair:
    """One ORPO preference example in TRL 0.24.0 conversational form.

    `prompt` is the conversation up to and including the final user turn.
    `chosen` and `rejected` are each a single-assistant-message list
    `[{"role": "assistant", "content": <str>}]` — TRL 0.24.0 requires the
    completions to be conversational too whenever the prompt is conversational.
    Written as {prompt, chosen, rejected} jsonl for trl.ORPOTrainer. There is no
    per-example `weight`: TRL's ORPO ignores it, so contamination is handled by
    dropping whole sources (see contamination.filter_contaminated).
    """

    prompt: list[dict] = field(default_factory=list)
    chosen: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    source: str = ""

    def to_record(self) -> dict:
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}


def _completion_content(messages) -> str:
    return messages[0].get("content", "") if messages else ""


def validate_pair(pair: PreferencePair) -> None:
    if not pair.prompt:
        raise ValueError(f"{pair.source}: empty prompt")
    if not pair.chosen:
        raise ValueError(f"{pair.source}: empty chosen")
    if not pair.rejected:
        raise ValueError(f"{pair.source}: empty rejected")
    if _completion_content(pair.chosen) == _completion_content(pair.rejected):
        raise ValueError(f"{pair.source}: chosen == rejected (no preference signal)")

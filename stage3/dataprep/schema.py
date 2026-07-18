from dataclasses import dataclass, field


@dataclass(slots=True)
class PreferencePair:
    """One ORPO preference example. `prompt` is the conversation up to and
    including the final user turn; `chosen`/`rejected` are competing assistant
    completions (Hermes format). Written as {prompt, chosen, rejected} jsonl for
    trl.ORPOTrainer. `weight` lets contamination downweight a source."""

    prompt: list[dict] = field(default_factory=list)
    chosen: str = ""
    rejected: str = ""
    source: str = ""
    weight: float = 1.0

    def to_record(self) -> dict:
        return {"prompt": self.prompt, "chosen": self.chosen,
                "rejected": self.rejected, "source": self.source, "weight": self.weight}


def validate_pair(pair: PreferencePair) -> None:
    if not pair.prompt:
        raise ValueError(f"{pair.source}: empty prompt")
    if not pair.chosen:
        raise ValueError(f"{pair.source}: empty chosen")
    if not pair.rejected:
        raise ValueError(f"{pair.source}: empty rejected")
    if pair.chosen == pair.rejected:
        raise ValueError(f"{pair.source}: chosen == rejected (no preference signal)")

import json
from dataclasses import dataclass, field

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(slots=True)
class TrainingExample:
    """One (multi-turn) SFT example in a single unified schema.

    Tool calls live inside assistant message content as Hermes <tool_call>
    blocks; tool results as <tool_response> blocks in a role="tool" message.
    SFT trains only positive targets, so there is no `is_negative` flag
    (negatives belong to ORPO). TRL's SFTTrainer does not consume a per-example
    weight, so there is no `weight` either — contamination is handled by
    dropping whole sources (see contamination.filter_contaminated).
    """

    source: str
    messages: list[dict] = field(default_factory=list)

    def to_record(self) -> dict:
        return {"source": self.source, "messages": self.messages}


def tool_call_block(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def tool_response_block(output) -> str:
    return "<tool_response>\n" + json.dumps({"output": output}) + "\n</tool_response>"


def validate_example(ex: TrainingExample) -> None:
    if not ex.messages:
        raise ValueError(f"{ex.source}: example has no messages")
    for msg in ex.messages:
        if msg.get("role") not in VALID_ROLES:
            raise ValueError(f"{ex.source}: invalid role {msg.get('role')!r}")
        if "content" not in msg:
            raise ValueError(f"{ex.source}: message missing content")

import json
from dataclasses import dataclass, field

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(slots=True)
class TrainingExample:
    """One multi-turn SFT example in a single unified schema.

    Tool calls live inside assistant message content as Hermes <tool_call>
    blocks; tool results as <tool_response> blocks in a role="tool" message.
    weight < 1 downweights (e.g. contamination); is_negative marks
    wrong-tool / malformed / refuse-when-no-tool examples.
    """

    source: str
    messages: list[dict] = field(default_factory=list)
    weight: float = 1.0
    is_negative: bool = False

    def to_record(self) -> dict:
        return {"source": self.source, "messages": self.messages,
                "weight": self.weight, "is_negative": self.is_negative}


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

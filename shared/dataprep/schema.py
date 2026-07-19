import json
from dataclasses import dataclass, field

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(slots=True)
class TrainingExample:
    """One (multi-turn) SFT example in a single unified schema.

    Tool calls are stored NEUTRAL (not pre-rendered to any family's format): an
    assistant message carries a `tool_calls` list of {"name","arguments"} and a
    tool result is a role="tool" message with a `name` (its raw output in
    `content`). `render_for_family` serializes these to Hermes <tool_call>/
    <tool_response> (QWEN) or leaves them structured for the harmony template
    (GPT_OSS). SFT trains only positive targets, so there is no `is_negative`
    flag (negatives belong to ORPO). TRL's SFTTrainer does not consume a
    per-example weight, so there is no `weight` either — contamination is
    handled by dropping whole sources (see contamination.filter_contaminated).
    """

    source: str
    messages: list[dict] = field(default_factory=list)

    def to_record(self) -> dict:
        return {"source": self.source, "messages": self.messages}


def tool_call_block(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def tool_response_block(output) -> str:
    return "<tool_response>\n" + json.dumps({"output": output}) + "\n</tool_response>"


def render_for_family(messages: list[dict], family: str) -> list[dict]:
    """Serialize neutral structured tool calls for a model family's chat template.

    GPT_OSS: pass structured `tool_calls`/role="tool" through as-is — the harmony
    chat template renders the `commentary to=functions.NAME ...` itself.
    QWEN: collapse structured tool calls back into Hermes <tool_call>-in-content
    and role="tool" results into <tool_response>-in-content, byte-identical to the
    pre-structured behavior (regression-locked in test_schema)."""
    from shared.model_family import ModelFamily

    if ModelFamily(family) is ModelFamily.GPT_OSS:
        return messages
    rendered = []
    for msg in messages:
        if msg.get("tool_calls"):
            content = "\n".join(
                tool_call_block(c["name"], c.get("arguments", {}))
                for c in msg["tool_calls"]
            )
            rendered.append({"role": msg["role"], "content": content})
        elif msg.get("role") == "tool" and "name" in msg:
            # A named role=tool message is a structured harmony tool result;
            # a plain role=tool message (no name) is already text — leave it.
            rendered.append({"role": "tool", "content": tool_response_block(msg["content"])})
        else:
            rendered.append(msg)
    return rendered


def validate_example(ex: TrainingExample) -> None:
    if not ex.messages:
        raise ValueError(f"{ex.source}: example has no messages")
    for msg in ex.messages:
        if msg.get("role") not in VALID_ROLES:
            raise ValueError(f"{ex.source}: invalid role {msg.get('role')!r}")
        # Accept both shapes: plain text (`content`) and structured tool calls
        # (`tool_calls`, which may carry an empty `content`).
        if "content" not in msg and "tool_calls" not in msg:
            raise ValueError(f"{ex.source}: message missing content")

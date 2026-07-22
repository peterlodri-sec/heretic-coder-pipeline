import json

from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders
from shared.dataprep.decontaminate import decontaminate_rows

_ROLES = frozenset({"system", "user", "assistant", "tool"})


def _map_message(m: dict) -> dict | None:
    """Map one OpenHands trajectory message to the neutral training schema.

    Nebius shape: {role, content, name, tool_call_id, tool_calls:[{id, type,
    function:{name, arguments(JSON str)}}]}. We keep roles, convert an assistant's
    tool_calls to the schema's neutral {name, arguments(dict)} list, and mark tool
    results as role="tool" with a name so render_for_family renders them per family.
    """
    role = m.get("role")
    if role not in _ROLES:
        return None
    if role == "assistant" and (m.get("tool_calls") or []):
        calls = []
        for tc in m["tool_calls"]:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (ValueError, TypeError):
                args = {"_raw_arguments": raw}  # keep unparseable args verbatim
            calls.append({"name": name, "arguments": args})
        msg = {"role": "assistant", "content": m.get("content") or ""}
        if calls:
            msg["tool_calls"] = calls
        return msg
    if role == "tool":
        return {"role": "tool", "name": m.get("name") or "tool",
                "content": m.get("content") or ""}
    return {"role": role, "content": m.get("content") or ""}


class NebiusSource(DataSource):
    """Nebius OpenHands verified-passing agent trajectories as multi-turn SFT data.

    Keeps only `resolved == 1` trajectories (the patch passed its tests — the
    filtering the research shows is essential), HARD-decontaminated against
    SWE-bench Verified (instance_id / repo@commit / repo blocklist).

    NOTE: these traces run to ~130k tokens; effective use needs long-context
    training (report used seq 131072). At our default 16k the tail (the actual
    patch) truncates, weakening the signal — raise STAGE2_MAX_SEQ_LEN when this
    source is enabled, or expect a partial-trajectory signal.
    """
    name = "nebius-openhands"

    def examples(self):
        for row in decontaminate_rows(loaders.load_nebius_rows()):
            if not row.get("resolved"):
                continue  # verified-passing trajectories only
            mapped = [mm for m in (row.get("trajectory") or [])
                      if (mm := _map_message(m)) is not None]
            roles = {m["role"] for m in mapped}
            if not ({"user", "assistant"} <= roles):
                continue  # need at least a task turn and an agent turn
            yield TrainingExample(source=self.name, messages=mapped)

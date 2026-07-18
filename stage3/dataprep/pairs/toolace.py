from dataprep.corruptions import make_rejected
from dataprep.pairs.base import PairSource
from dataprep.schema import PreferencePair
from shared.dataprep import loaders

# Team-ACE/ToolACE conversation `from` roles map directly to our roles.
_ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}


class ToolACEPairs(PairSource):
    """Team-ACE/ToolACE multi-turn tool-use conversations as preference pairs.

    Real schema: `system` (str) + `conversations` (list of {from, value}). The
    prompt is the conversation up to (and excluding) the last assistant turn
    (system + user/tool turns); that last assistant turn is the chosen
    completion; the rejected turn corrupts it via `wrong_args` (falls back to a
    refusal for non-tool-call assistant text)."""

    name = "toolace"

    def pairs(self):
        for row in loaders.load_toolace_rows():
            messages = [{"role": "system", "content": row["system"]}]
            for turn in row["conversations"]:
                role = _ROLE_MAP.get(turn["from"], turn["from"])
                messages.append({"role": role, "content": turn["value"]})

            last_assistant = next(
                (i for i in range(len(messages) - 1, -1, -1)
                 if messages[i]["role"] == "assistant"),
                None,
            )
            if last_assistant is None:
                continue
            prompt = messages[:last_assistant]
            if not any(m["role"] == "user" for m in prompt):
                continue
            chosen_text = messages[last_assistant]["content"]
            yield PreferencePair(
                prompt=prompt,
                chosen=[{"role": "assistant", "content": chosen_text}],
                rejected=[{"role": "assistant",
                           "content": make_rejected(chosen_text, "wrong_args")}],
                source=self.name,
            )

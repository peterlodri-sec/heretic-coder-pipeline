import ast

from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders

# Team-ACE/ToolACE conversation `from` roles map directly to our roles.
_ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}


def _parse_bracket_calls(text):
    """Parse a ToolACE bracket tool-call turn like `[Name(arg="x", n=1)]` into
    [(name, args_dict), ...]. Returns None if the text is not a parseable list
    of keyword-only calls, so the caller keeps the assistant text as-is."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    if not isinstance(node.body, ast.List) or not node.body.elts:
        return None
    calls = []
    for elt in node.body.elts:
        if not isinstance(elt, ast.Call) or not isinstance(elt.func, ast.Name):
            return None
        args = {}
        for kw in elt.keywords:
            if kw.arg is None:  # **kwargs, not representable
                return None
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                return None
        calls.append((elt.func.id, args))
    return calls


class ToolACESource(DataSource):
    """Team-ACE/ToolACE multi-turn tool-use conversations.

    Real schema: `system` (str) + `conversations` (list of {from, value}),
    from in {user, assistant, tool}. Assistant tool calls are BRACKET format;
    parseable calls become neutral structured tool_calls (render_for_family
    serializes to Hermes/harmony), else the text is left as-is.
    """

    name = "toolace"

    def examples(self):
        for row in loaders.load_toolace_rows():
            messages = [{"role": "system", "content": row["system"]}]
            for turn in row["conversations"]:
                role = _ROLE_MAP.get(turn["from"], turn["from"])
                content = turn["value"]
                if role == "assistant":
                    calls = _parse_bracket_calls(content)
                    if calls is not None:
                        messages.append({
                            "role": "assistant", "content": "",
                            "tool_calls": [{"name": name, "arguments": args}
                                           for name, args in calls],
                        })
                        continue
                messages.append({"role": role, "content": content})
            yield TrainingExample(source=self.name, messages=messages)

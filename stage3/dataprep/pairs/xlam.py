import json

from dataprep.corruptions import make_rejected
from dataprep.pairs.base import PairSource
from dataprep.schema import PreferencePair
from shared.dataprep import loaders
from shared.dataprep.schema import tool_call_block

TOOL_SYSTEM_PROMPT = (
    "You are a function-calling assistant. You have access to the following "
    "tools. When a tool is needed, respond with one <tool_call>{...}</tool_call> "
    "block per call. Available tools:\n"
)


class XLAMPairs(PairSource):
    """NobodyExistsOnTheInternet/xlam-function-calling-60k preference pairs.

    Real schema: `query` (str), `tools` (JSON str), `answers` (JSON str). The
    gold `answers` become the chosen assistant turn (one Hermes <tool_call> per
    call); the rejected turn corrupts it — swapping to a different REAL tool when
    the row lists more than one (`wrong_tool`), else mutating the arguments
    (`wrong_args`)."""

    name = "xlam"

    def pairs(self):
        for row in loaders.load_xlam_rows():
            tools = json.loads(row["tools"])
            answers = json.loads(row["answers"])
            if not answers:
                continue
            system = TOOL_SYSTEM_PROMPT + json.dumps(tools)
            chosen_text = "\n".join(
                tool_call_block(call["name"], call.get("arguments", {}))
                for call in answers
            )
            strategy = "wrong_tool" if len(tools) > 1 else "wrong_args"
            rejected_text = make_rejected(chosen_text, strategy, tools=tools)
            yield PreferencePair(
                prompt=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["query"]},
                ],
                chosen=[{"role": "assistant", "content": chosen_text}],
                rejected=[{"role": "assistant", "content": rejected_text}],
                source=self.name,
            )

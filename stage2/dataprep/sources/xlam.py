import json

from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders

TOOL_SYSTEM_PROMPT = (
    "You are a function-calling assistant. You have access to the following "
    "tools. When a tool is needed, respond with one <tool_call>{...}</tool_call> "
    "block per call. Available tools:\n"
)


class XLAMSource(DataSource):
    """NobodyExistsOnTheInternet/xlam-function-calling-60k.

    Real schema: `query` (str), `tools` (JSON string), `answers` (JSON string).
    `tools` is put in the system message; `answers` (the gold call list) is
    recorded as neutral structured tool calls (render_for_family serializes them
    to Hermes for qwen / harmony for gpt-oss).
    """

    name = "xlam"

    def examples(self):
        for row in loaders.load_xlam_rows():
            tools = json.loads(row["tools"])
            answers = json.loads(row["answers"])
            system = TOOL_SYSTEM_PROMPT + json.dumps(tools)
            tool_calls = [
                {"name": call["name"], "arguments": call.get("arguments", {})}
                for call in answers
            ]
            yield TrainingExample(
                source=self.name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["query"]},
                    {"role": "assistant", "content": "", "tool_calls": tool_calls},
                ],
            )

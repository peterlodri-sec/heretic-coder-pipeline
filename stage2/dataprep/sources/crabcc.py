from shared.dataprep.schema import TrainingExample, tool_call_block, tool_response_block
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders


class CrabccSource(DataSource):
    """Your own Claude Code session traces — real agent trajectories."""

    name = "crabcc"

    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def examples(self):
        for trace in loaders.load_traces(self.trace_dir):
            messages = []
            for turn in trace["turns"]:
                match turn["role"]:
                    case "assistant" if "tool" in turn:
                        messages.append({"role": "assistant",
                                         "content": tool_call_block(turn["tool"], turn["arguments"])})
                    case "tool":
                        messages.append({"role": "tool",
                                         "content": tool_response_block(turn["output"])})
                    case _:
                        messages.append({"role": turn["role"], "content": turn["content"]})
            yield TrainingExample(source=self.name, messages=messages)

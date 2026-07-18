from dataprep.schema import TrainingExample, tool_call_block, tool_response_block
from dataprep.sources.base import DataSource


def load_traces(trace_dir):
    import glob
    import json
    traces = []
    for path in glob.glob(f"{trace_dir}/*.json"):
        with open(path) as f:
            traces.append(json.load(f))
    return traces


class CrabccSource(DataSource):
    """Your own Claude Code session traces — real agent trajectories."""

    name = "crabcc"

    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def examples(self):
        for trace in load_traces(self.trace_dir):
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

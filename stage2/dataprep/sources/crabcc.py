from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders


class CrabccSource(DataSource):
    """Your own Claude Code session traces — real agent trajectories.

    Tool calls/results are recorded NEUTRAL (structured); render_for_family
    serializes them to Hermes (qwen) or the harmony template (gpt-oss). The tool
    result carries the preceding call's name so harmony can render it."""

    name = "crabcc"

    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def examples(self):
        for trace in loaders.load_traces(self.trace_dir):
            messages = []
            last_tool = None
            for turn in trace["turns"]:
                match turn["role"]:
                    case "assistant" if "tool" in turn:
                        last_tool = turn["tool"]
                        messages.append({
                            "role": "assistant", "content": "",
                            "tool_calls": [{"name": turn["tool"],
                                            "arguments": turn["arguments"]}]})
                    case "tool":
                        messages.append({"role": "tool", "name": last_tool,
                                         "content": turn["output"]})
                    case _:
                        messages.append({"role": turn["role"], "content": turn["content"]})
            yield TrainingExample(source=self.name, messages=messages)

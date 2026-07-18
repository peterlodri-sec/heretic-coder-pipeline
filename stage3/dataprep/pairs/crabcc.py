from dataprep.schema import PreferencePair
from dataprep.corruptions import make_rejected
from shared.dataprep import loaders
from shared.dataprep.schema import tool_call_block
from dataprep.pairs.base import PairSource


class CrabccPairs(PairSource):
    name = "crabcc"

    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def pairs(self):
        for trace in loaders.load_traces(self.trace_dir):
            turns = trace["turns"]
            prompt = [{"role": t["role"], "content": t["content"]}
                      for t in turns if t["role"] == "user"]
            action = next((t for t in turns if t["role"] == "assistant" and "tool" in t), None)
            if action is None or not prompt:
                continue
            chosen = tool_call_block(action["tool"], action["arguments"])
            yield PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=make_rejected(chosen, "wrong_tool"),
                source=self.name,
            )

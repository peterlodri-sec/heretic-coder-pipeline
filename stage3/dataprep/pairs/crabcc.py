from dataprep.corruptions import make_rejected
from dataprep.pairs.base import PairSource
from dataprep.schema import PreferencePair
from shared.dataprep import loaders
from shared.dataprep.schema import tool_call_block


class CrabccPairs(PairSource):
    """crabcc agent traces as preference pairs. The user turn(s) form the prompt;
    the gold tool action is the chosen Hermes <tool_call>; the rejected turn
    mutates its arguments (`wrong_args`)."""

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
            chosen_text = tool_call_block(action["tool"], action["arguments"])
            yield PreferencePair(
                prompt=prompt,
                chosen=[{"role": "assistant", "content": chosen_text}],
                rejected=[{"role": "assistant",
                           "content": make_rejected(chosen_text, "wrong_args")}],
                source=self.name,
            )

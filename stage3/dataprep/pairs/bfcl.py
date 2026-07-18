from dataprep.schema import PreferencePair
from dataprep.corruptions import make_rejected
from shared.dataprep import loaders
from shared.dataprep.schema import tool_call_block
from dataprep.pairs.base import PairSource


class BFCLPairs(PairSource):
    name = "bfcl"

    def pairs(self):
        for row in loaders.load_bfcl_rows():
            if not row["correct"]:
                continue  # only correct calls seed a chosen completion
            chosen = tool_call_block(row["function"], row["arguments"])
            yield PreferencePair(
                prompt=[{"role": "user", "content": row["question"]}],
                chosen=chosen,
                rejected=make_rejected(chosen, "wrong_tool"),
                source=self.name,
            )

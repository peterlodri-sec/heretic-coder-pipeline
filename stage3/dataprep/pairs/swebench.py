from dataprep.schema import PreferencePair
from dataprep.corruptions import make_rejected
from shared.dataprep import loaders
from dataprep.pairs.base import PairSource


class SWEBenchPairs(PairSource):
    name = "swebench"

    def pairs(self):
        for row in loaders.load_swebench_rows():
            if not row.get("resolved"):
                continue
            chosen = row["patch"]
            yield PreferencePair(
                prompt=[{"role": "user", "content": row["problem_statement"]}],
                chosen=chosen,
                rejected=make_rejected(chosen, "refusal"),
                source=self.name,
            )

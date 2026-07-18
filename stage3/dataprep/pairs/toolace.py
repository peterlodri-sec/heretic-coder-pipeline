from dataprep.schema import PreferencePair
from dataprep.corruptions import make_rejected
from shared.dataprep import loaders
from dataprep.pairs.base import PairSource

CODE_DOMAINS = frozenset({"coding", "software", "devops", "data"})


class ToolACEPairs(PairSource):
    name = "toolace"

    def pairs(self):
        for row in loaders.load_toolace_rows():
            if row.get("domain") not in CODE_DOMAINS:
                continue
            convo = list(row["conversation"])
            chosen = next((m["content"] for m in reversed(convo) if m["role"] == "assistant"), None)
            if chosen is None:
                continue
            prompt = [m for m in convo if m["role"] != "assistant"]
            yield PreferencePair(
                prompt=prompt or [{"role": "user", "content": ""}],
                chosen=chosen,
                rejected=make_rejected(chosen, "malformed_args"),
                source=self.name,
            )

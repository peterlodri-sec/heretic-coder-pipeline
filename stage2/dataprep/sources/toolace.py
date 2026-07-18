from dataprep.schema import TrainingExample
from dataprep.sources.base import DataSource

DATASET_ID = "Team-ACE/ToolACE"
CODE_DOMAINS = frozenset({"coding", "software", "devops", "data"})


def load_rows():
    from datasets import load_dataset
    return load_dataset(DATASET_ID, split="train")


class ToolACESource(DataSource):
    name = "toolace"

    def examples(self):
        for row in load_rows():
            if row.get("domain") not in CODE_DOMAINS:
                continue
            yield TrainingExample(source=self.name, messages=list(row["conversation"]))

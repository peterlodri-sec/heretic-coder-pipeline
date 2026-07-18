from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders

CODE_DOMAINS = frozenset({"coding", "software", "devops", "data"})


class ToolACESource(DataSource):
    name = "toolace"

    def examples(self):
        for row in loaders.load_toolace_rows():
            if row.get("domain") not in CODE_DOMAINS:
                continue
            yield TrainingExample(source=self.name, messages=list(row["conversation"]))

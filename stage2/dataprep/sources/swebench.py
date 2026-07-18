from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders


class SWEBenchSource(DataSource):
    name = "swebench"

    def examples(self):
        for row in loaders.load_swebench_rows():
            if not row.get("resolved"):
                continue  # gold trajectories = resolved instances only
            messages = [
                {"role": "user", "content": row["problem_statement"]},
                {"role": "assistant", "content": row["patch"]},
            ]
            yield TrainingExample(source=self.name, messages=messages)

from dataprep.schema import TrainingExample
from dataprep.sources.base import DataSource

DATASET_ID = "princeton-nlp/SWE-bench_Verified"


def load_rows():
    from datasets import load_dataset
    return load_dataset(DATASET_ID, split="test")


class SWEBenchSource(DataSource):
    name = "swebench"

    def examples(self):
        for row in load_rows():
            if not row.get("resolved"):
                continue  # gold trajectories = resolved instances only
            messages = [
                {"role": "user", "content": row["problem_statement"]},
                {"role": "assistant", "content": row["patch"]},
            ]
            yield TrainingExample(source=self.name, messages=messages)

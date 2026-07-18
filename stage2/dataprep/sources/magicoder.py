from dataprep.schema import TrainingExample
from dataprep.sources.base import DataSource

DATASET_ID = "ise-uiuc/Magicoder-OSS-Instruct-75K"


def load_rows():
    from datasets import load_dataset
    return load_dataset(DATASET_ID, split="train")


class MagicoderSource(DataSource):
    name = "magicoder"

    def examples(self):
        for row in load_rows():
            yield TrainingExample(
                source=self.name,
                messages=[
                    {"role": "user", "content": row["problem"]},
                    {"role": "assistant", "content": row["solution"]},
                ],
            )

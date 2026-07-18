from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders


class MagicoderSource(DataSource):
    name = "magicoder"

    def examples(self):
        for row in loaders.load_magicoder_rows():
            yield TrainingExample(
                source=self.name,
                messages=[
                    {"role": "user", "content": row["problem"]},
                    {"role": "assistant", "content": row["solution"]},
                ],
            )

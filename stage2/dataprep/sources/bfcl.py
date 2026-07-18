from shared.dataprep.schema import TrainingExample, tool_call_block, tool_response_block
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders


class BFCLSource(DataSource):
    name = "bfcl"

    def examples(self):
        for row in loaders.load_bfcl_rows():
            assistant = tool_call_block(row["function"], row["arguments"])
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": assistant},
                {"role": "tool", "content": tool_response_block(row["output"])},
            ]
            yield TrainingExample(source=self.name, messages=messages,
                                  is_negative=not row["correct"])

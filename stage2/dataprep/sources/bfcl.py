from dataprep.schema import TrainingExample, tool_call_block, tool_response_block
from dataprep.sources.base import DataSource

DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def load_rows():
    from datasets import load_dataset
    return load_dataset(DATASET_ID, split="train")


class BFCLSource(DataSource):
    name = "bfcl"

    def examples(self):
        for row in load_rows():
            assistant = tool_call_block(row["function"], row["arguments"])
            messages = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": assistant},
                {"role": "tool", "content": tool_response_block(row["output"])},
            ]
            yield TrainingExample(source=self.name, messages=messages,
                                  is_negative=not row["correct"])

from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders
from shared.dataprep.decontaminate import decontaminate_rows


class SWEBenchSource(DataSource):
    name = "swebench"

    def examples(self):
        # HARD decontamination: never yield a row that matches a SWE-bench Verified
        # instance (by instance_id or repo@commit) — training on it would make our
        # eval resolve rate a memorization score. Guards whatever loader feeds this
        # source (today Verified itself -> everything drops; a future SWE-Gym /
        # SWE-bench-train loader -> only the Verified-overlapping rows drop).
        for row in decontaminate_rows(loaders.load_swebench_rows()):
            if not row.get("resolved"):
                continue  # gold trajectories = resolved instances only
            messages = [
                {"role": "user", "content": row["problem_statement"]},
                {"role": "assistant", "content": row["patch"]},
            ]
            yield TrainingExample(source=self.name, messages=messages)

from shared.dataprep.schema import TrainingExample
from shared.dataprep.sources.base import DataSource
from shared.dataprep import loaders
from shared.dataprep.decontaminate import decontaminate_rows

# Kept terse on purpose — the training signal is the (issue -> gold diff) mapping,
# not the instruction. The repo name is prepended to the issue so the model learns
# to condition the diff's paths on the repository.
SYSTEM = ("You are resolving a real GitHub issue in the given repository. "
          "Respond with a unified diff (git diff format) that resolves it.")


class SWEGymSource(DataSource):
    """SWE-Gym gold (issue -> patch) pairs as stage-2 SFT data.

    HARD-decontaminated against SWE-bench Verified (by instance_id / repo@commit)
    so the eval resolve rate can never be inflated by leaked test instances. This
    is deliberately the *simple* problem->patch shaping (mirrors SWEBenchSource);
    full agentic trajectory data is a stage-4 RFT concern, not stage-2 SFT.
    """
    name = "swegym"

    def examples(self):
        for row in decontaminate_rows(loaders.load_swegym_rows()):
            problem, patch = row.get("problem_statement"), row.get("patch")
            if not problem or not patch:
                continue  # need both the issue and its gold diff
            repo = row.get("repo")
            user = f"Repository: {repo}\n\n{problem}" if repo else problem
            yield TrainingExample(source=self.name, messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": patch},
            ])

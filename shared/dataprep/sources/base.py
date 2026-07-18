from abc import ABC, abstractmethod


class DataSource(ABC):
    """A training-data source. Adapters own their raw schema and yield unified
    TrainingExample objects. Heavy `datasets` loading is isolated in a module
    level `load_rows` function so tests can patch it."""

    name: str

    @abstractmethod
    def examples(self):
        """Yield TrainingExample instances."""
        raise NotImplementedError

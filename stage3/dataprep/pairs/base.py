from abc import ABC, abstractmethod


class PairSource(ABC):
    """A preference-pair source. Adapters own their raw schema and yield unified
    PreferencePair objects (chosen = gold, rejected = corrupted or wrong)."""

    name: str

    @abstractmethod
    def pairs(self):
        raise NotImplementedError

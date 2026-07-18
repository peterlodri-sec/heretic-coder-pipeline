import json
import os
import typing
from dataclasses import asdict, fields
from enum import StrEnum


def _enum_type(annotation):
    """Return the StrEnum class in an annotation like `X` or `X | None`, else None."""
    args = typing.get_args(annotation)
    for candidate in (args if args else (annotation,)):
        if isinstance(candidate, type) and issubclass(candidate, StrEnum):
            return candidate
    return None


class JsonStatusMixin:
    """Serialization plumbing for a slots dataclass persisted as status.json.

    __slots__ = () so subclasses keep real slots (a slotless base would
    re-introduce __dict__). from_dict coerces any StrEnum-typed field back to
    its enum by inspecting type hints, and drops unknown keys for forward-compat.
    """

    __slots__ = ()

    @classmethod
    def from_dict(cls, data: dict):
        hints = typing.get_type_hints(cls)
        known = {f.name for f in fields(cls)}
        payload = {}
        for key, value in data.items():
            if key not in known:
                continue
            if value is not None:
                enum_cls = _enum_type(hints.get(key))
                if enum_cls is not None:
                    value = enum_cls(value)
            payload[key] = value
        return cls(**payload)

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(json.loads(text))

    @classmethod
    def read(cls, path: str):
        with open(path) as f:
            return cls.from_json(f.read())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def write(self, path: str) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(self.to_json())
        os.replace(tmp_path, path)

"""Backend-independent decision protocol shared by evaluation strategies."""
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class DecisionRequest:
    env: Any
    observation: dict
    info: dict
    seed: int
    step: int


@dataclass(frozen=True)
class Decision:
    action: int | None
    source: str
    diagnostics: dict = field(default_factory=dict)
    native_action: int | None = None

    def __post_init__(self):
        if (self.action is None) == (self.native_action is None):
            raise ValueError('Decision requires exactly one thin or complete native action')


class DecisionStrategy(Protocol):
    name: str

    def decide(self, requests: list[DecisionRequest], *, check: Callable[[], None]) -> list[Decision]: ...

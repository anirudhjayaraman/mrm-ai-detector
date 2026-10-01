"""Feature registry primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..text import TextUnit

FeatureFn = Callable[[TextUnit], Optional[float]]


@dataclass(frozen=True)
class Feature:
    """A single scalar signal computed from a :class:`TextUnit`.

    ``family`` groups features for reporting; ``resamplable`` marks features
    that depend only on sentences and therefore vary under the sentence
    bootstrap used for confidence intervals.
    """

    name: str
    family: str
    description: str
    fn: FeatureFn
    resamplable: bool = True
    ai_direction: str = "+"  # "+" if larger values point towards AI, "-" otherwise

    def __call__(self, unit: TextUnit) -> Optional[float]:
        return self.fn(unit)


REGISTRY: Dict[str, Feature] = {}


def register(name: str, family: str, description: str, *, resamplable: bool = True, ai_direction: str = "+"):
    def deco(fn: FeatureFn) -> FeatureFn:
        if name in REGISTRY:
            raise ValueError(f"duplicate feature {name!r}")
        REGISTRY[name] = Feature(name, family, description, fn, resamplable, ai_direction)
        return fn

    return deco


def per_k(count: float, words: int, k: int = 1000) -> Optional[float]:
    return None if words == 0 else count * k / words

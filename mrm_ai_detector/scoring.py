"""Scoring model: standardised features -> calibrated logistic probability.

A model is a JSON document holding, per feature, a reference mean and
standard deviation and a logistic weight, plus an intercept.  The same form
is produced by :mod:`mrm_ai_detector.train`, so a model calibrated on an
organisation's own corpus is a drop-in replacement for the shipped default.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

SCHEMA_VERSION = 1


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class FeatureParam:
    mean: float
    std: float
    weight: float


@dataclass
class ScoringModel:
    name: str
    features: Dict[str, FeatureParam]
    bias: float
    clip: float = 4.0
    reliability_words: float = 40.0
    ai_threshold: float = 0.70
    human_threshold: float = 0.30
    # Segments are short and noisy, so they get their own (higher) flagging threshold.
    segment_threshold: Optional[float] = None
    min_words: int = 150
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- inference -------------------------------------------------------
    def standardise(self, name: str, value: Optional[float]) -> float:
        p = self.features[name]
        if value is None or not math.isfinite(value) or p.std <= 0:
            return 0.0
        z = (value - p.mean) / p.std
        return max(-self.clip, min(self.clip, z))

    def contributions(self, values: Mapping[str, Optional[float]]) -> Dict[str, float]:
        return {n: p.weight * self.standardise(n, values.get(n)) for n, p in self.features.items()}

    def logit(self, values: Mapping[str, Optional[float]], words: Optional[int] = None) -> float:
        raw = self.bias + sum(self.contributions(values).values())
        if words is not None and self.reliability_words > 0:
            # Shrink towards 0 (p = 0.5, "no evidence") for short texts.
            raw *= words / (words + self.reliability_words)
        return raw

    def probability(self, values: Mapping[str, Optional[float]], words: Optional[int] = None) -> float:
        return sigmoid(self.logit(values, words))

    def score(self, values: Mapping[str, Optional[float]], words: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        return self.probability(values, words), self.contributions(values)

    @property
    def effective_segment_threshold(self) -> float:
        return self.segment_threshold if self.segment_threshold is not None else self.ai_threshold

    @property
    def neural_features(self) -> Tuple[str, ...]:
        from .features.neural import NEURAL_FEATURES

        return tuple(n for n in self.features if n in NEURAL_FEATURES)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ScoringModel:
        version = d.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported model schema_version {version}")
        feats = {k: FeatureParam(**v) for k, v in d["features"].items()}
        keys = ("clip", "reliability_words", "ai_threshold", "human_threshold", "segment_threshold", "min_words", "metadata")
        kwargs = {k: d[k] for k in keys if k in d}
        return cls(name=d["name"], features=feats, bias=float(d["bias"]), **kwargs)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ScoringModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls, neural: bool = False) -> ScoringModel:
        fname = "default_model_lm.json" if neural else "default_model.json"
        text = resources.files("mrm_ai_detector").joinpath("models", fname).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

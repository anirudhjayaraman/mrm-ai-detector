"""Feature extraction.

Importing this package registers all handcrafted features in
:data:`REGISTRY`.  Neural features live in :mod:`.neural` and are only
loaded on request.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from ..text import TextUnit
from . import domain, entropy, markers, stylometry  # noqa: F401  (registration side effects)
from .base import REGISTRY, Feature

__all__ = ["REGISTRY", "Feature", "extract"]


def extract(unit: TextUnit, names: Optional[Iterable[str]] = None) -> Dict[str, Optional[float]]:
    """Compute handcrafted features for ``unit``.

    Features that cannot be estimated reliably (e.g. sentence-length
    variation on two sentences) are returned as ``None``.
    """
    selected = REGISTRY if names is None else {n: REGISTRY[n] for n in names if n in REGISTRY}
    out: Dict[str, Optional[float]] = {}
    for name, feat in selected.items():
        v = feat(unit)
        out[name] = None if v is None else float(v)
    return out

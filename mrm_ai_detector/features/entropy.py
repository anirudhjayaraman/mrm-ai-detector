"""Information-theoretic features computable without a language model."""

from __future__ import annotations

import math
import zlib
from collections import Counter
from statistics import mean
from typing import Optional

from ..text import TextUnit
from .base import register

_WINDOW = 800  # characters; fixed window keeps the ratio length-invariant


@register("compression_ratio", "entropy", "zlib compressed/raw size over fixed 800-char windows", resamplable=False, ai_direction="-")
def compression_ratio(unit: TextUnit) -> Optional[float]:
    data = unit.text.encode("utf-8")
    if len(data) < _WINDOW:
        return None
    step = max(_WINDOW // 2, (len(data) - _WINDOW) // 100)
    ratios = [len(zlib.compress(data[i : i + _WINDOW], 9)) / _WINDOW for i in range(0, len(data) - _WINDOW + 1, step)]
    return mean(ratios)


@register("char_entropy", "entropy", "Shannon entropy of character trigrams over fixed 500-char windows (normalised)", resamplable=False, ai_direction="-")
def char_entropy(unit: TextUnit, window: int = 500) -> Optional[float]:
    text = unit.text.lower()
    if len(text) < window:
        return None
    step = max(window // 2, (len(text) - window) // 40)
    values = []
    for i in range(0, len(text) - window + 1, step):
        chunk = text[i : i + window]
        grams = Counter(chunk[j : j + 3] for j in range(len(chunk) - 2))
        total = sum(grams.values())
        values.append(-sum(c / total * math.log2(c / total) for c in grams.values()) / math.log2(total))
    return mean(values)


@register("repeated_bigram_rate", "entropy", "Share of word bigrams recurring within fixed 100-word windows", resamplable=False)
def repeated_bigram_rate(unit: TextUnit, window: int = 100) -> Optional[float]:
    words = [w.lower() for w in unit.words]
    if len(words) < window:
        return None
    step = max(window // 2, (len(words) - window) // 40)
    rates = []
    for i in range(0, len(words) - window + 1, step):
        chunk = words[i : i + window]
        counts = Counter(zip(chunk, chunk[1:]))
        rates.append(sum(c for c in counts.values() if c > 1) / (len(chunk) - 1))
    return mean(rates)

"""Model-risk-specific evidence features.

Genuine validation and development documentation is anchored in specifics:
test results with values, data windows, model identifiers, regulatory and
internal cross-references.  LLM drafts tend to substitute evaluative
adjectives ("the model is robust and well-calibrated") for evidence.
"""

from __future__ import annotations

import re
from typing import Optional

from ..lexicons import BOILERPLATE, BOILERPLATE_RE, EVALUATIVE, METRIC_VALUE_RE, SPECIFIC_REFERENCE_PATTERNS
from ..text import NUMBER_RE, WORD_RE, TextUnit
from .base import per_k, register

_BOILER_WEIGHTS = {k.lower(): v for k, v in BOILERPLATE.items()}
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|potentially|possibly|likely|appears? to|seems? to|suggests?|approximately|roughly|around)\b",
    re.IGNORECASE,
)


def _has_evidence(sentence: str) -> bool:
    if NUMBER_RE.search(sentence):
        return True
    return any(p.search(sentence) for p in SPECIFIC_REFERENCE_PATTERNS)


@register("numeric_density", "domain", "Numeric tokens per 100 words", ai_direction="-")
def numeric_density(unit: TextUnit) -> Optional[float]:
    return per_k(len(NUMBER_RE.findall(unit.text)), unit.word_count, 100)


@register("quantified_metric_rate", "domain", "Metric statements with values (AUC 0.71, PSI = 0.12 ...) per 1,000 words", ai_direction="-")
def quantified_metric_rate(unit: TextUnit) -> Optional[float]:
    return per_k(len(METRIC_VALUE_RE.findall(unit.text)), unit.word_count)


@register("specific_reference_rate", "domain", "Dates, IDs, regulatory and internal cross-references per 1,000 words", ai_direction="-")
def specific_reference_rate(unit: TextUnit) -> Optional[float]:
    text = unit.text
    return per_k(sum(len(p.findall(text)) for p in SPECIFIC_REFERENCE_PATTERNS), unit.word_count)


@register("unsupported_claim_ratio", "domain", "Share of sentences making evaluative claims with no number or reference")
def unsupported_claim_ratio(unit: TextUnit) -> Optional[float]:
    n = len(unit.sentences)
    if n < 3:
        return None
    hits = 0
    for s in unit.sentences:
        words = {w.lower() for w in WORD_RE.findall(s)}
        if words & EVALUATIVE and not _has_evidence(s):
            hits += 1
    return hits / n


@register("evidence_sentence_ratio", "domain", "Share of sentences carrying a number or concrete reference", ai_direction="-")
def evidence_sentence_ratio(unit: TextUnit) -> Optional[float]:
    n = len(unit.sentences)
    return sum(1 for s in unit.sentences if _has_evidence(s)) / n if n >= 3 else None


@register("boilerplate_rate", "domain", "Weighted generic governance boilerplate per 1,000 words")
def boilerplate_rate(unit: TextUnit) -> Optional[float]:
    total = sum(_BOILER_WEIGHTS.get(re.sub(r"\s+", " ", m.group(0).lower()), 1.0) for m in BOILERPLATE_RE.finditer(unit.text))
    return per_k(total, unit.word_count)


@register("hedge_rate", "domain", "Hedging expressions per 1,000 words", ai_direction="-")
def hedge_rate(unit: TextUnit) -> Optional[float]:
    return per_k(len(_HEDGE_RE.findall(unit.text)), unit.word_count)

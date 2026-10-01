"""Lexical and syntactic LLM "tells"."""

from __future__ import annotations

import re
from typing import Optional

from ..lexicons import AI_MARKER_RE, AI_MARKERS, PARTICIPLE_TAILS, TRANSITION_OPENERS
from ..text import TextUnit
from .base import per_k, register

_MARKER_WEIGHTS = {k.lower(): v for k, v in AI_MARKERS.items()}
_PARTICIPLE_RE = re.compile(r",\s+(?:" + "|".join(PARTICIPLE_TAILS) + r")\b", re.IGNORECASE)
_TRICOLON_RE = re.compile(r"\b[\w\-]+(?:\s[\w\-]+)?,\s[\w\-]+(?:\s[\w\-]+)?,?\s(?:and|or)\s[\w\-]+", re.IGNORECASE)
_NOT_ONLY_RE = re.compile(r"\bnot (?:only|just|merely)\b[^.]{0,80}\bbut\b|\bit'?s not\b[^.]{0,40}\bit'?s\b", re.IGNORECASE)
_BOLD_LEAD_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*|__)[^*_]{2,60}(?:\*\*|__)\s*[:\-–—]?")
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _marker_weight(match: str) -> float:
    key = re.sub(r"\s+", " ", match.lower())
    return _MARKER_WEIGHTS.get(key, 1.0)


@register("ai_marker_rate", "markers", "Weighted LLM-favoured vocabulary per 1,000 words")
def ai_marker_rate(unit: TextUnit) -> Optional[float]:
    total = sum(_marker_weight(m.group(0)) for m in AI_MARKER_RE.finditer(unit.text))
    return per_k(total, unit.word_count)


@register("transition_open_rate", "markers", "Share of sentences opening with a stock transition")
def transition_open_rate(unit: TextUnit) -> Optional[float]:
    n = len(unit.sentences)
    if n < 3:
        return None
    hits = 0
    for s in unit.sentences:
        head = s.lstrip("\"'(“ ").lower()
        if any(head.startswith(t) and (len(head) == len(t) or not head[len(t)].isalpha() or t.endswith(" ")) for t in TRANSITION_OPENERS):
            hits += 1
    return hits / n


@register("participle_tail_rate", "markers", "Share of sentences with a ', ensuring/highlighting ...' tail")
def participle_tail_rate(unit: TextUnit) -> Optional[float]:
    n = len(unit.sentences)
    return sum(1 for s in unit.sentences if _PARTICIPLE_RE.search(s)) / n if n >= 3 else None


@register("tricolon_rate", "markers", "Share of sentences containing an 'X, Y and Z' triplet")
def tricolon_rate(unit: TextUnit) -> Optional[float]:
    n = len(unit.sentences)
    return sum(1 for s in unit.sentences if _TRICOLON_RE.search(s)) / n if n >= 3 else None


@register("not_only_rate", "markers", "'Not only ... but' / 'it's not X, it's Y' constructions per 1,000 words")
def not_only_rate(unit: TextUnit) -> Optional[float]:
    return per_k(len(_NOT_ONLY_RE.findall(unit.text)), unit.word_count)


@register("bold_lead_rate", "markers", "Share of lines that open with a bolded label (markdown)", resamplable=False)
def bold_lead_rate(unit: TextUnit) -> Optional[float]:
    if len(unit.lines) < 4:
        return None
    return sum(1 for ln in unit.lines if _BOLD_LEAD_RE.match(ln)) / len(unit.lines)


@register("bullet_line_rate", "markers", "Share of lines formatted as bullets", resamplable=False)
def bullet_line_rate(unit: TextUnit) -> Optional[float]:
    if len(unit.lines) < 4:
        return None
    return sum(1 for ln in unit.lines if _BULLET_LINE_RE.match(ln)) / len(unit.lines)

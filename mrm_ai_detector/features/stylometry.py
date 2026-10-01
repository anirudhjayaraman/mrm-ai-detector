"""Stylometric features: rhythm, lexical diversity and punctuation habits.

LLM prose tends towards uniform sentence lengths ("low burstiness"),
uniform paragraph sizes, sparing use of parentheses and semicolons and
liberal use of em dashes.  Human technical writers are burstier and lean on
parenthetical qualification.
"""

from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import List, Optional

from ..text import WORD_RE, TextUnit
from .base import per_k, register

FUNCTION_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me more
    most my no nor not now of off on once only or other our ours out over own same she should so
    some such than that the their them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your
    """.split()
)


def _sentence_lengths(unit: TextUnit) -> List[int]:
    return [n for n in (len(WORD_RE.findall(s)) for s in unit.sentences) if n > 0]


@register("sent_len_mean", "stylometry", "Mean sentence length in words")
def sent_len_mean(unit: TextUnit) -> Optional[float]:
    lengths = _sentence_lengths(unit)
    return mean(lengths) if len(lengths) >= 3 else None


@register("sent_len_cv", "stylometry", "Coefficient of variation of sentence length (burstiness)", ai_direction="-")
def sent_len_cv(unit: TextUnit) -> Optional[float]:
    lengths = _sentence_lengths(unit)
    if len(lengths) < 5:
        return None
    mu = mean(lengths)
    return pstdev(lengths) / mu if mu else None


@register("short_sent_frac", "stylometry", "Share of sentences with 8 words or fewer", ai_direction="-")
def short_sent_frac(unit: TextUnit) -> Optional[float]:
    lengths = _sentence_lengths(unit)
    return sum(1 for n in lengths if n <= 8) / len(lengths) if len(lengths) >= 5 else None


@register("para_len_cv", "stylometry", "Coefficient of variation of paragraph length", resamplable=False, ai_direction="-")
def para_len_cv(unit: TextUnit) -> Optional[float]:
    lengths = [len(WORD_RE.findall(p)) for p in unit.paragraphs]
    lengths = [n for n in lengths if n >= 15]
    if len(lengths) < 4:
        return None
    mu = mean(lengths)
    return pstdev(lengths) / mu if mu else None


@register("mattr", "stylometry", "Moving-average type-token ratio (window 50)", resamplable=False)
def mattr(unit: TextUnit, window: int = 50) -> Optional[float]:
    words = [w.lower() for w in unit.words]
    if len(words) < window:
        return None
    step = max(1, (len(words) - window) // 200)  # cap work on long documents
    ratios = [len(set(words[i : i + window])) / window for i in range(0, len(words) - window + 1, step)]
    return mean(ratios)


@register("word_len_mean", "stylometry", "Mean word length in characters")
def word_len_mean(unit: TextUnit) -> Optional[float]:
    words = unit.words
    return mean(len(w) for w in words) if len(words) >= 30 else None


@register("function_word_ratio", "stylometry", "Share of tokens that are function words", ai_direction="-")
def function_word_ratio(unit: TextUnit) -> Optional[float]:
    words = unit.words
    if len(words) < 30:
        return None
    return sum(1 for w in words if w.lower() in FUNCTION_WORDS) / len(words)


@register("first_person_rate", "stylometry", "First-person pronouns per 1,000 words", ai_direction="-")
def first_person_rate(unit: TextUnit) -> Optional[float]:
    words = unit.words
    return per_k(sum(1 for w in words if w.lower() in {"i", "we", "our", "us", "my"}), len(words))


@register("paren_rate", "punctuation", "Parenthetical openings per 1,000 words", ai_direction="-")
def paren_rate(unit: TextUnit) -> Optional[float]:
    return per_k(unit.text.count("("), unit.word_count)


@register("semicolon_rate", "punctuation", "Semicolons per 1,000 words", ai_direction="-")
def semicolon_rate(unit: TextUnit) -> Optional[float]:
    return per_k(unit.text.count(";"), unit.word_count)


@register("colon_rate", "punctuation", "Colons per 1,000 words (outside headings)")
def colon_rate(unit: TextUnit) -> Optional[float]:
    return per_k(len(re.findall(r":(?!\d)", unit.text)), unit.word_count)


@register("em_dash_rate", "punctuation", "Em/en dashes used as punctuation per 1,000 words")
def em_dash_rate(unit: TextUnit) -> Optional[float]:
    dashes = len(re.findall(r"—|\s–\s|\s--\s", unit.text))
    return per_k(dashes, unit.word_count)


@register("question_rate", "punctuation", "Share of sentences that are questions")
def question_rate(unit: TextUnit) -> Optional[float]:
    n = len(unit.sentences)
    return sum(1 for s in unit.sentences if s.rstrip().endswith("?")) / n if n >= 3 else None

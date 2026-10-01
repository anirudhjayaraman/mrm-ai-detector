"""Curated lexicons used by the feature extractors.

Weights reflect how diagnostic a phrase is of LLM-generated prose *in model
risk documentation specifically*.  Terms that are legitimate MRM vocabulary
(e.g. "robustness", "comprehensive validation", "ongoing monitoring") carry
low or zero weight so that genuine practitioner text is not penalised for
using its own jargon.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Pattern, Tuple

# phrase -> weight (3 = strong LLM tell, 2 = moderate, 1 = weak)
AI_MARKERS: Dict[str, float] = {
    # strong
    "delve": 3, "delves": 3, "delving": 3, "tapestry": 3, "testament to": 3,
    "ever-evolving": 3, "ever-changing": 3, "in today's": 3, "navigating the": 3,
    "navigate the complexities": 3, "realm": 3, "it is important to note": 3,
    "it's important to note": 3, "it is worth noting": 3, "it's worth noting": 3,
    "plays a crucial role": 3, "plays a pivotal role": 3, "plays a vital role": 3,
    "underscores": 3, "underscore the": 3, "underscoring": 3, "showcasing": 3,
    "seamless": 3, "seamlessly": 3, "multifaceted": 3, "holistic": 3,
    "meticulous": 3, "meticulously": 3, "commendable": 3, "embark": 3,
    "fostering": 3, "foster": 3, "paramount": 3, "pivotal": 3, "intricate": 3,
    "intricacies": 3, "nuanced": 3, "game-changer": 3, "unlock": 3, "harness": 3,
    "harnessing": 3, "invaluable": 3, "valuable insights": 3, "key takeaways": 3,
    "in conclusion": 3, "in summary": 3, "serves as a": 2, "stands as": 3,
    "not only": 2, "a myriad of": 3, "myriad": 3, "bustling": 3, "vibrant": 3,
    "landscape of": 3, "evolving landscape": 3, "regulatory landscape": 2,
    "crucial role": 3, "vital role": 3, "pivotal role": 3, "by doing so": 2,
    "let's": 2, "overall, the": 2, "ultimately,": 2, "rest assured": 3,
    "cutting-edge": 3, "state-of-the-art": 2, "elevate": 3, "empowering": 3,
    "strike a balance": 3, "striking a balance": 3, "a deeper understanding": 3,
    "gain insights": 3, "actionable insights": 3, "proactive approach": 3,
    "robust framework": 3, "robust and": 2, "comprehensive and": 2,
    "well-positioned": 3, "in the realm of": 3, "shed light": 3, "sheds light": 3,
    # moderate
    "furthermore": 2, "moreover": 2, "additionally": 2, "notably": 2,
    "crucial": 2, "crucially": 2, "vital": 2, "leverage": 2, "leveraging": 2,
    "leverages": 2, "streamline": 2, "streamlining": 2, "empower": 2,
    "bolster": 2, "bolstering": 2, "facilitate": 1, "facilitating": 2,
    "key considerations": 2, "best practices": 2, "aligns with": 2,
    "align with": 1, "a wide range of": 2, "a variety of": 1, "thereby": 1,
    "proactive": 2, "proactively": 2, "insights": 2, "dynamic": 1,
    "enhances": 2, "enhancing": 2, "ensuring": 2, "fosters": 2, "key role": 2,
    "integral": 2, "essential": 1, "landscape": 2, "framework that": 1,
    "transparency and": 1, "in essence": 3, "at its core": 3, "overarching": 2,
    # weak (also common in genuine MRM text)
    "robust": 1, "comprehensive": 1, "enhance": 1, "ensure": 0.5, "various": 1,
    "significant": 0.5, "effectively": 1, "efficiently": 1, "importantly": 2,
}

# Common LLM sentence openers.
TRANSITION_OPENERS: Tuple[str, ...] = (
    "additionally", "furthermore", "moreover", "overall", "ultimately",
    "importantly", "notably", "in addition", "in conclusion", "in summary",
    "consequently", "conversely", "by doing so", "to this end", "as such",
    "in essence", "crucially", "collectively", "taken together", "together,",
    "beyond", "while", "by ",
)

# ", <participle> ..." sentence tails ("..., ensuring alignment with ...").
PARTICIPLE_TAILS: Tuple[str, ...] = (
    "ensuring", "highlighting", "underscoring", "reflecting", "demonstrating",
    "providing", "enabling", "allowing", "making", "fostering", "showcasing",
    "emphasizing", "emphasising", "reinforcing", "contributing", "supporting",
    "promoting", "facilitating", "offering", "helping", "thereby", "creating",
    "positioning", "strengthening", "enhancing", "signaling", "signalling",
    "potentially leading", "leading", "resulting", "which could lead", "ultimately",
)

# Generic governance boilerplate that LLMs use in place of concrete content.
BOILERPLATE: Dict[str, float] = {
    "industry best practices": 3, "industry best practice": 3,
    "regulatory expectations": 1.5, "regulatory requirements": 1,
    "evolving regulatory": 3, "key stakeholders": 3, "relevant stakeholders": 2,
    "stakeholders": 1, "end-to-end": 2, "holistic view": 3,
    "well-documented": 1.5, "clear and concise": 3, "fit for purpose": 0.5,
    "sound and effective": 2, "strong governance": 2, "robust governance": 3,
    "governance framework": 1, "continuous improvement": 2,
    "maintain the integrity": 3, "integrity and reliability": 3,
    "accuracy and reliability": 3, "reliability and accuracy": 3,
    "transparency and accountability": 3, "informed decision-making": 3,
    "informed decisions": 2, "decision-making processes": 2,
    "mitigate potential risks": 3, "potential risks": 2, "potential impact": 1,
    "a range of": 1, "areas for improvement": 2, "align with the": 1,
    "in line with": 0.5, "remain compliant": 2, "dynamic environment": 3,
    "changing market conditions": 2, "rapidly changing": 3, "unforeseen": 2,
    "it is recommended that": 0.5, "should be considered": 1,
    "appropriate controls": 1, "effective challenge": 0.3,
}

# Evaluative words that, without supporting evidence, signal unsupported claims.
EVALUATIVE = frozenset(
    """
    robust sound appropriate adequate comprehensive effective strong reasonable
    reliable accurate rigorous thorough well-designed well-documented suitable
    satisfactory solid significant consistent stable sufficient proper
    meaningful valuable reasonable acceptable conservative prudent
    """.split()
)

# Quantitative performance / risk metrics common in MRM documents.
METRICS = (
    r"auc|auroc|roc|gini|ks|k-s|kolmogorov|psi|csi|iv|information value|r-?squared|r2|r\^2|"
    r"adjusted r|rmse|mae|mape|mse|log-?loss|brier|p-?value|t-?stat(?:istic)?|"
    r"var|value at risk|expected shortfall|es|cvar|backtest(?:ing)? exceptions?|exceptions|"
    r"hit rate|accuracy|precision|recall|f1|lift|correlation|vif|"
    r"pd|lgd|ead|ccf|mape|coverage|confidence (?:level|interval)|"
    r"hosmer|binomial|traffic light|kupiec|christoffersen|durbin|adf|dickey"
)
# A metric name followed by an actual value: a decimal, a percentage, or a small
# number introduced by "=", ":", "of", "was" ... -- not IDs, years or grade labels.
METRIC_VALUE_RE: Pattern[str] = re.compile(
    r"\b(?:" + METRICS + r")\b[^.;\n]{0,40}?"
    r"(?:(?<![\w.])[-+−]?\d*\.\d+\s?%?"
    r"|(?<![\w.])[-+−]?\d{1,3}(?:\.\d+)?\s?%"
    r"|(?:=|:|\bof|\bwas|\bwere|\bat|\bis|\bto|\bfrom|\bbelow|\babove)\s?[-+−]?\d{1,3}(?:\.\d+)?(?![\d\-/]))",
    re.IGNORECASE,
)

SPECIFIC_REFERENCE_PATTERNS: Tuple[Pattern[str], ...] = (
    # Regulatory / standards citations
    re.compile(r"\bSR\s?\d{2}-\d{1,2}\b|\bSS\s?\d{1,2}/\d{2}\b|\bCP\s?\d{1,2}/\d{2}\b|"
               r"\bOCC\s?(?:Bulletin\s)?\d{4}-\d{1,2}\b|\bE-23\b|\bIFRS\s?9\b|\bCECL\b|"
               r"\bCRR\b|\bCRD\s?(?:IV|V)\b|\bBCBS\s?\d+|\bArt(?:icle|\.)\s?\d+|\bTRIM\b|"
               r"\bEBA/GL/\d{4}/\d+|\bFRTB\b|\bCCAR\b|\bDFAST\b|\bICAAP\b|\bSMCR\b",
               re.IGNORECASE),
    # Dates and periods
    re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])(?:[-/]\d{1,2})?\b|"
               r"\b(?:Q[1-4]|H[12]|FY)\s?'?(?:19|20)?\d{2}\b|"
               r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(?:\d{1,2},?\s+)?(?:19|20)\d{2}\b|"
               r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}\b"),
    # Internal cross references
    re.compile(r"\b(?:Section|Sec\.|Table|Figure|Fig\.|Appendix|Annex|Exhibit|Chart|Equation|Eq\.|Finding|Issue|Recommendation|Paragraph|Footnote)\s+[A-Z]?\d+(?:\.\d+)*[a-z]?\b"),
    # Identifiers such as model IDs, ticket IDs ("MDL-2231", "ID 550123", "model #4471")
    re.compile(r"\b[A-Z]{2,6}[-_]\d{2,6}\b|\b(?:ID|Id|No\.|#)\s?\d{3,}\b|\bmodel\s(?:id\s)?#?\d{3,}\b", re.IGNORECASE),
    # Years, year ranges and fiscal periods ("2018-2019", "as of 2026")
    re.compile(r"\b(?:19[89]\d|20[0-4]\d)(?:\s?[-\u2013/]\s?(?:19[89]\d|20[0-4]\d|\d{2}))?\b"),
    # Citations of literature "(Smith, 2019)" / "Smith et al. (2019)" / [12]
    re.compile(r"\(\s?[A-Z][A-Za-z\-]+(?: et al\.?| and [A-Z][A-Za-z\-]+)?,?\s(?:19|20)\d{2}[a-z]?\s?\)|"
               r"\b[A-Z][A-Za-z\-]+ et al\.?\s?\((?:19|20)\d{2}\)|\[\d{1,3}(?:[,–-]\s?\d{1,3})*\]"),
)


def compile_phrases(phrases: Iterable[str]) -> Pattern[str]:
    """Compile phrases into one case-insensitive, word-bounded alternation."""
    ordered = sorted(set(phrases), key=len, reverse=True)
    body = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in ordered)
    return re.compile(r"(?<![A-Za-z])(?:" + body + r")(?![A-Za-z])", re.IGNORECASE)


AI_MARKER_RE = compile_phrases(AI_MARKERS)
BOILERPLATE_RE = compile_phrases(BOILERPLATE)

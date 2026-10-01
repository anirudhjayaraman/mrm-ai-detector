"""High-level detection API."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import __version__
from .features import REGISTRY, extract
from .features.neural import NEURAL_FEATURES, NeuralConfig, NeuralScorer
from .ingest import OCR_MARKER, read_document
from .scoring import ScoringModel
from .text import TextUnit, build_unit, resample_sentences, segment_document

logger = logging.getLogger(__name__)

LIKELY_AI = "Likely AI-generated"
LIKELY_HUMAN = "Likely human-written"
MIXED = "Mixed: partially AI-generated"
INCONCLUSIVE = "Inconclusive"
INSUFFICIENT = "Insufficient text"


@dataclass
class Signal:
    feature: str
    value: Optional[float]
    contribution: float
    description: str


@dataclass
class SegmentResult:
    index: int
    section: Optional[str]
    word_count: int
    probability: float
    flagged: bool
    text: str
    signals: List[Signal] = field(default_factory=list)


@dataclass
class DocumentResult:
    source: str
    word_count: int
    probability: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    verdict: str
    ai_share: float
    segments: List[SegmentResult]
    features: Dict[str, Optional[float]]
    signals: List[Signal]
    model: str
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    detector_version: str = __version__

    @property
    def flagged(self) -> bool:
        return self.verdict in (LIKELY_AI, MIXED)

    def to_dict(self, include_text: bool = True) -> Dict:
        d = asdict(self)
        if not include_text:
            for seg in d["segments"]:
                seg.pop("text", None)
        return d


def _describe(name: str) -> str:
    if name in REGISTRY:
        return REGISTRY[name].description
    return NEURAL_FEATURES.get(name, name)


class Detector:
    """Score documents for likely AI authorship.

    Parameters
    ----------
    model:
        A :class:`ScoringModel`; defaults to the shipped calibrated model
        (the neural variant when ``use_neural`` is set).
    use_neural:
        Add Binoculars / Fast-DetectGPT / RoBERTa features (needs ``[lm]`` extras).
    segment_words:
        Target words per segment for localised scoring.
    bootstrap:
        Number of sentence-bootstrap resamples for the document confidence
        interval (0 disables).
    """

    def __init__(
        self,
        model: Optional[ScoringModel] = None,
        *,
        use_neural: bool = False,
        neural_config: Optional[NeuralConfig] = None,
        segment_words: int = 180,
        bootstrap: int = 200,
        seed: int = 7,
    ) -> None:
        self.model = model or ScoringModel.default(neural=use_neural)
        self.use_neural = use_neural
        self.segment_words = segment_words
        self.bootstrap = bootstrap
        self.seed = seed
        self._neural = NeuralScorer(neural_config) if use_neural else None
        missing = [n for n in self.model.neural_features] if not use_neural else []
        if missing:
            logger.warning("model uses neural features %s but use_neural=False; they will be ignored", missing)

    # ------------------------------------------------------------------
    def analyze_file(self, path: Union[str, Path]) -> DocumentResult:
        return self.analyze_text(read_document(path), source=str(path))

    def analyze_text(self, text: str, source: str = "<text>") -> DocumentResult:
        start = time.perf_counter()
        model = self.model
        warnings: List[str] = []
        if text.startswith(OCR_MARKER):
            text = text[len(OCR_MARKER):].lstrip()
            warnings.append(
                "text was read from an image by OCR; any recognition errors (and lost formatting) can make text "
                "look more human-written, so prefer scanning the original document when you have it"
            )
        unit = build_unit(text)
        words = unit.word_count

        segments = segment_document(text, target_words=self.segment_words)
        seg_values: List[Dict[str, Optional[float]]] = []
        for seg in segments:
            vals = extract(seg.unit, model.features)
            if self._neural is not None:
                vals.update(self._neural.score(seg.unit.text))
            seg_values.append(vals)

        values = extract(unit, model.features)
        if self._neural is not None:
            values.update(_weighted_mean(seg_values, [s.unit.word_count for s in segments], NEURAL_FEATURES))

        prob, contribs = model.score(values, words)
        signals = _top_signals(contribs, values, k=8)

        seg_results: List[SegmentResult] = []
        for seg, vals in zip(segments, seg_values):
            p, c = model.score(vals, seg.unit.word_count)
            seg_results.append(
                SegmentResult(
                    index=seg.index,
                    section=seg.section,
                    word_count=seg.unit.word_count,
                    probability=p,
                    flagged=p >= model.effective_segment_threshold,
                    text=seg.text,
                    signals=_top_signals(c, vals, k=3),
                )
            )
        flagged_words = sum(s.word_count for s in seg_results if s.flagged)
        ai_share = flagged_words / max(1, sum(s.word_count for s in seg_results))

        ci_low = ci_high = None
        if self.bootstrap and len(unit.sentences) >= 10:
            ci_low, ci_high = self._bootstrap(unit, values, words)

        if words < model.min_words:
            verdict = INSUFFICIENT
            warnings.append(f"only {words} words of prose; at least {model.min_words} are needed for a reliable verdict")
        else:
            verdict = _verdict(prob, ai_share, seg_results, model)
        if any(v is None for k, v in values.items() if k in REGISTRY) and words >= model.min_words:
            skipped = sorted(k for k, v in values.items() if v is None and k in REGISTRY)
            logger.debug("features not estimable for %s: %s", source, skipped)

        return DocumentResult(
            source=source,
            word_count=words,
            probability=prob,
            ci_low=ci_low,
            ci_high=ci_high,
            verdict=verdict,
            ai_share=ai_share,
            segments=seg_results,
            features=values,
            signals=signals,
            model=model.name,
            warnings=warnings,
            elapsed_s=round(time.perf_counter() - start, 3),
        )

    # ------------------------------------------------------------------
    def _bootstrap(self, unit: TextUnit, values: Dict[str, Optional[float]], words: int):
        rng = random.Random(self.seed)
        n = len(unit.sentences)
        resamplable = [k for k in self.model.features if k in REGISTRY and REGISTRY[k].resamplable]
        probs: List[float] = []
        for _ in range(self.bootstrap):
            sample = resample_sentences(unit, [rng.randrange(n) for _ in range(n)])
            v = dict(values)
            v.update(extract(sample, resamplable))
            probs.append(self.model.probability(v, words))
        probs.sort()
        lo = probs[int(0.05 * (len(probs) - 1))]
        hi = probs[int(0.95 * (len(probs) - 1))]
        return lo, hi


def _verdict(prob: float, ai_share: float, segments: List[SegmentResult], model: ScoringModel) -> str:
    if prob >= model.ai_threshold:
        return LIKELY_AI
    n_flagged = sum(1 for s in segments if s.flagged)
    if n_flagged and ai_share >= 0.15:
        # Localised AI-like sections: only call it "mixed" with enough segments to contrast.
        return MIXED if len(segments) >= 3 else INCONCLUSIVE
    if prob <= model.human_threshold:
        return LIKELY_HUMAN
    return INCONCLUSIVE


def _top_signals(contribs: Dict[str, float], values: Dict[str, Optional[float]], k: int) -> List[Signal]:
    ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [Signal(n, values.get(n), round(c, 4), _describe(n)) for n, c in ranked[:k] if abs(c) > 1e-6]


def _weighted_mean(rows: List[Dict[str, Optional[float]]], weights: List[int], keys) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        pairs = [(r.get(key), w) for r, w in zip(rows, weights) if r.get(key) is not None]
        total = sum(w for _, w in pairs)
        out[key] = sum(v * w for v, w in pairs) / total if total else None
    return out

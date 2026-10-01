"""Author-baseline comparison: is this document unusual *for this author*?

Population-level detectors penalise people whose natural style happens to
resemble LLM output (formal, template-driven or non-native English writing).
Comparing a document with the same author's earlier work, ideally written
before LLMs were available, controls for personal style, language background
and house templates.

Method
------
1. Segment every reference document and compute handcrafted features per
   segment. The reference distribution of each feature is summarised by a
   robust mean and scale.
2. Each document is summarised by the word-weighted mean z-score of its
   segments. Its *style distance* is the root-mean-square of those z-scores.
3. The reference distances come from leave-one-document-out (each reference
   document is scored against the others), which gives an empirical
   distribution of how much this author normally varies.
4. The target document's distance is converted to an empirical p-value against
   that distribution, and its shifts are signed by each feature's AI direction
   so the report says whether the change is *towards* LLM-like writing or just
   different.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .features import REGISTRY, extract
from .ingest import iter_documents, read_document
from .text import segment_document

# Features driven by formatting or extraction rather than by the writer.
EXCLUDED = {"para_len_cv", "bullet_line_rate", "bold_lead_rate", "colon_rate", "question_rate"}
MIN_REFERENCE_DOCS = 3


@dataclass
class FeatureShift:
    feature: str
    description: str
    target_value: Optional[float]
    reference_mean: float
    z: float
    towards_ai: bool


@dataclass
class BaselineResult:
    target: str
    reference_documents: int
    reference_segments: int
    style_distance: float
    reference_distance_median: float
    reference_distance_p90: float
    p_value: float
    ai_direction_shift: float
    shifts: List[FeatureShift] = field(default_factory=list)
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _segment_features(text: str) -> List[Tuple[int, Dict[str, Optional[float]]]]:
    names = [n for n in REGISTRY if n not in EXCLUDED]
    out = []
    for seg in segment_document(text):
        if seg.unit.word_count >= 60:
            out.append((seg.unit.word_count, extract(seg.unit, names)))
    return out


def _stats(segments: Sequence[Tuple[int, Dict[str, Optional[float]]]]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for name in REGISTRY:
        if name in EXCLUDED:
            continue
        vals = sorted(v[name] for _, v in segments if v.get(name) is not None)
        if len(vals) < 4:
            continue
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
        iqr = vals[int(0.75 * (len(vals) - 1))] - vals[int(0.25 * (len(vals) - 1))]
        scale = max(sd, iqr / 1.349)
        if scale > 1e-9:
            stats[name] = (mu, scale)
    return stats


def _doc_z(segments, stats) -> Dict[str, float]:
    acc: Dict[str, List[Tuple[float, int]]] = {}
    for words, values in segments:
        for name, (mu, sd) in stats.items():
            v = values.get(name)
            if v is not None:
                acc.setdefault(name, []).append((max(-6.0, min(6.0, (v - mu) / sd)), words))
    return {n: sum(z * w for z, w in zs) / sum(w for _, w in zs) for n, zs in acc.items()}


def _distance(z: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in z.values()) / len(z)) if z else 0.0


def compare_to_baseline(
    reference: Iterable[Union[str, Path]],
    target_text: str,
    target_name: str = "<target>",
    top_k: int = 8,
) -> BaselineResult:
    ref_docs = []
    for path in iter_documents(reference):
        segs = _segment_features(read_document(path))
        if segs:
            ref_docs.append((str(path), segs))
    if len(ref_docs) < MIN_REFERENCE_DOCS:
        raise ValueError(f"need at least {MIN_REFERENCE_DOCS} reference documents with 60+ words of prose; got {len(ref_docs)}")

    all_segments = [s for _, segs in ref_docs for s in segs]
    stats = _stats(all_segments)

    # Leave-one-document-out distances: how far does this author normally drift?
    ref_distances = []
    for i, (_, segs) in enumerate(ref_docs):
        others = [s for j, (_, ss) in enumerate(ref_docs) if j != i for s in ss]
        ref_distances.append(_distance(_doc_z(segs, _stats(others))))
    ref_distances.sort()

    target_segments = _segment_features(target_text)
    if not target_segments:
        raise ValueError("target document has no scorable prose (needs 60+ words)")
    z = _doc_z(target_segments, stats)
    dist = _distance(z)
    p_value = (1 + sum(1 for d in ref_distances if d >= dist)) / (len(ref_distances) + 1)

    signed = {n: (v if REGISTRY[n].ai_direction == "+" else -v) for n, v in z.items()}
    ai_shift = sum(signed.values()) / len(signed) if signed else 0.0
    target_means: Dict[str, Optional[float]] = {}
    for name in z:
        vals = [(v[name], w) for w, v in target_segments if v.get(name) is not None]
        target_means[name] = sum(x * w for x, w in vals) / sum(w for _, w in vals) if vals else None
    shifts = [
        FeatureShift(n, REGISTRY[n].description, target_means[n], stats[n][0], round(z[n], 3), signed[n] > 0)
        for n in sorted(z, key=lambda n: -abs(z[n]))[:top_k]
    ]

    median = ref_distances[len(ref_distances) // 2]
    p90 = ref_distances[min(len(ref_distances) - 1, int(0.9 * len(ref_distances)))]
    if p_value <= 0.1 and ai_shift > 0.25:
        msg = "Unusual for this author, and the shift is towards LLM-typical writing. Worth a closer look."
    elif p_value <= 0.1:
        msg = "Unusual for this author, but the shift is not predominantly in LLM-typical directions (topic, template or genre change?)."
    else:
        msg = "Within this author's normal range of variation."
    if len(ref_docs) < 8:
        msg += f" Only {len(ref_docs)} reference documents: treat the p-value as indicative."

    return BaselineResult(
        target=target_name,
        reference_documents=len(ref_docs),
        reference_segments=len(all_segments),
        style_distance=round(dist, 3),
        reference_distance_median=round(median, 3),
        reference_distance_p90=round(p90, 3),
        p_value=round(p_value, 3),
        ai_direction_shift=round(ai_shift, 3),
        shifts=shifts,
        interpretation=msg,
    )


def render_baseline(result: BaselineResult) -> str:
    lines = [
        f"== {result.target} vs author baseline ({result.reference_documents} documents, {result.reference_segments} segments)",
        f"Style distance: {result.style_distance:.2f}  (author's own documents: median {result.reference_distance_median:.2f}, 90th pct {result.reference_distance_p90:.2f})",
        f"Empirical p-value: {result.p_value:.2f}   Shift towards AI-typical features: {result.ai_direction_shift:+.2f}",
        result.interpretation,
        "Largest shifts (z vs author's own writing):",
    ]
    for s in result.shifts:
        tag = "AI-ward" if s.towards_ai else "      "
        val = "–" if s.target_value is None else f"{s.target_value:.3g}"
        lines.append(f"  {s.z:+5.2f} {tag}  {s.feature:<26} {val:>8} (author mean {s.reference_mean:.3g})  {s.description}")
    return "\n".join(lines)

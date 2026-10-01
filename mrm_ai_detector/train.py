"""Calibrate a scoring model on labelled human / AI corpora.

The trainer fits an L2-regularised logistic regression (IRLS, pure Python)
on segment-level feature vectors, with the same short-text reliability
shrinkage used at inference built into the design matrix.  Performance is
estimated with *document-grouped* stratified k-fold cross-validation so that
segments of one document never appear in both train and test folds.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__
from .features import REGISTRY, extract
from .features.neural import NEURAL_DIRECTIONS, NEURAL_FEATURES, NeuralScorer
from .ingest import iter_documents, read_document
from .scoring import FeatureParam, ScoringModel, sigmoid
from .text import segment_document

logger = logging.getLogger(__name__)


@dataclass
class Row:
    group: str  # document identifier
    label: int  # 1 = AI, 0 = human
    words: int
    values: Dict[str, Optional[float]]
    source: str = ""
    meta: Dict[str, str] = field(default_factory=dict)
    weight: float = 1.0  # relative importance in fitting (e.g. up-weight in-domain documents)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, line: str) -> Row:
        return cls(**json.loads(line))


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def rows_from_text(
    text: str,
    label: int,
    group: str,
    *,
    segment_words: int = 180,
    min_words: int = 60,
    neural: Optional[NeuralScorer] = None,
    meta: Optional[Dict[str, str]] = None,
) -> List[Row]:
    rows = []
    for seg in segment_document(text, target_words=segment_words):
        if seg.unit.word_count < min_words:
            continue
        vals = extract(seg.unit)
        if neural is not None:
            vals.update(neural.score(seg.unit.text))
        rows.append(Row(group, label, seg.unit.word_count, vals, f"{group}#{seg.index}", dict(meta or {})))
    return rows


def rows_from_paths(paths: Iterable[str], label: int, **kwargs) -> List[Row]:
    rows: List[Row] = []
    for path in iter_documents(paths):
        rows.extend(rows_from_text(read_document(path), label, str(path), **kwargs))
    return rows


def save_rows(rows: Sequence[Row], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.to_json() + "\n")


def load_rows(path: Path) -> List[Row]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Row.from_json(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Logistic regression (IRLS with ridge penalty)
# ---------------------------------------------------------------------------
def _cholesky_solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(s, 1e-12))
            else:
                L[i][j] = s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def fit_logistic(
    X: List[List[float]], y: List[int], sample_weight: List[float], l2: float, penalise_first: bool = False, iters: int = 50
) -> List[float]:
    """Weighted ridge logistic regression.  Column 0 is the (scaled) intercept."""
    d = len(X[0])
    beta = [0.0] * d
    for _ in range(iters):
        H = [[0.0] * d for _ in range(d)]
        g = [0.0] * d
        for xi, yi, wi in zip(X, y, sample_weight):
            p = sigmoid(sum(b * v for b, v in zip(beta, xi)))
            r = wi * (yi - p)
            s = wi * max(p * (1 - p), 1e-9)
            for j in range(d):
                xij = xi[j]
                if xij == 0.0:
                    continue
                g[j] += r * xij
                sx = s * xij
                row = H[j]
                for k in range(j + 1):
                    row[k] += sx * xi[k]
        for j in range(d):
            for k in range(j):
                H[k][j] = H[j][k]
        for j in range(d):
            if j > 0 or penalise_first:
                H[j][j] += l2
                g[j] -= l2 * beta[j]
        step = _cholesky_solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-7:
            break
    return beta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def tpr_at_fpr(scores: Sequence[float], labels: Sequence[int], fpr: float) -> float:
    neg = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    pos = [s for s, y in zip(scores, labels) if y == 1]
    if not neg or not pos:
        return float("nan")
    k = int(math.floor(fpr * len(neg)))
    thresh = neg[k] if k < len(neg) else -math.inf
    return sum(1 for s in pos if s > thresh) / len(pos)


def classification_metrics(probs: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> Dict[str, float]:
    tp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y == 1)
    fp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y == 0)
    fn = sum(1 for p, y in zip(probs, labels) if p < threshold and y == 1)
    tn = sum(1 for p, y in zip(probs, labels) if p < threshold and y == 0)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "n": len(labels),
        "auc": auc(probs, labels),
        "accuracy": (tp + tn) / len(labels) if labels else float("nan"),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec and not math.isnan(prec + rec) else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "tpr_at_1pct_fpr": tpr_at_fpr(probs, labels, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(probs, labels, 0.05),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels) if labels else float("nan"),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _feature_names(rows: Sequence[Row], include: Optional[Sequence[str]]) -> List[str]:
    names = list(include) if include else sorted({k for r in rows for k in r.values})
    usable = []
    for n in names:
        present = [r.values.get(n) for r in rows if r.values.get(n) is not None]
        if len(present) >= max(10, 0.3 * len(rows)) and len(set(round(v, 9) for v in present)) > 1:
            usable.append(n)
        else:
            logger.info("dropping feature %s (too sparse or constant)", n)
    return usable


def _standardisation(rows: Sequence[Row], names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    out = {}
    for n in names:
        vals = sorted(r.values[n] for r in rows if r.values.get(n) is not None)
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
        # Robust scale: guard against heavy tails with the IQR.
        iqr = vals[int(0.75 * (len(vals) - 1))] - vals[int(0.25 * (len(vals) - 1))]
        sd = max(math.sqrt(var), iqr / 1.349, 1e-6)
        out[n] = (mu, sd)
    return out


def fit_model(
    rows: Sequence[Row],
    *,
    features: Optional[Sequence[str]] = None,
    l2: float = 2.0,
    clip: float = 4.0,
    reliability_words: float = 40.0,
    name: str = "custom",
) -> ScoringModel:
    names = _feature_names(rows, features)
    stats = _standardisation(rows, names)
    shell = ScoringModel(
        name=name,
        features={n: FeatureParam(stats[n][0], stats[n][1], 0.0) for n in names},
        bias=0.0,
        clip=clip,
        reliability_words=reliability_words,
    )
    X, y, w = [], [], []
    # Class-balanced weights, each class's total weight equal, scaled by per-row importance.
    w_pos = sum(r.weight for r in rows if r.label == 1)
    w_neg = sum(r.weight for r in rows if r.label == 0)
    total = w_pos + w_neg
    for r in rows:
        shrink = r.words / (r.words + reliability_words) if reliability_words > 0 else 1.0
        X.append([shrink] + [shrink * shell.standardise(n, r.values.get(n)) for n in names])
        y.append(r.label)
        w.append(r.weight * total / (2 * (w_pos if r.label else w_neg)))
    beta = fit_logistic(X, y, w, l2)
    shell.bias = beta[0]
    for n, b in zip(names, beta[1:]):
        shell.features[n].weight = b
    return shell


def _group_folds(rows: Sequence[Row], k: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = {0: [], 1: []}
    seen = set()
    for r in rows:
        if r.group not in seen:
            seen.add(r.group)
            by_label[r.label].append(r.group)
    folds: List[List[str]] = [[] for _ in range(k)]
    for groups in by_label.values():
        rng.shuffle(groups)
        for i, g in enumerate(groups):
            folds[i % k].append(g)
    return folds


def cross_validate(rows: Sequence[Row], k: int = 5, seed: int = 13, **fit_kwargs) -> Dict[str, Dict[str, float]]:
    """Document-grouped stratified k-fold CV; returns segment- and document-level metrics."""
    folds = _group_folds(rows, k, seed)
    seg_probs: List[float] = []
    seg_labels: List[int] = []
    doc_scores: Dict[str, List[Tuple[float, int]]] = {}
    doc_labels: Dict[str, int] = {}
    for fold in folds:
        test_groups = set(fold)
        train = [r for r in rows if r.group not in test_groups]
        test = [r for r in rows if r.group in test_groups]
        if not test or len({r.label for r in train}) < 2:
            continue
        model = fit_model(train, **fit_kwargs)
        for r in test:
            p = model.probability(r.values, r.words)
            seg_probs.append(p)
            seg_labels.append(r.label)
            doc_scores.setdefault(r.group, []).append((model.logit(r.values, r.words), r.words))
            doc_labels[r.group] = r.label
    doc_probs = [sigmoid(sum(lg * w for lg, w in v) / sum(w for _, w in v)) for v in doc_scores.values()]
    return {
        "segment": classification_metrics(seg_probs, seg_labels),
        "document": classification_metrics(doc_probs, [doc_labels[g] for g in doc_scores]),
    }


def train(
    rows: Sequence[Row],
    *,
    folds: int = 5,
    name: str = "custom",
    description: str = "",
    features: Optional[Sequence[str]] = None,
    l2: float = 2.0,
) -> Tuple[ScoringModel, Dict[str, Dict[str, float]]]:
    if len({r.label for r in rows}) < 2:
        raise ValueError("training data must contain both human (0) and AI (1) examples")
    cv = cross_validate(rows, k=folds, features=features, l2=l2, name=name) if folds > 1 else {}
    model = fit_model(rows, features=features, l2=l2, name=name)
    directions = {**{n: f.ai_direction for n, f in REGISTRY.items()}, **NEURAL_DIRECTIONS}
    flipped = [n for n, p in model.features.items() if abs(p.weight) > 0.05 and (p.weight > 0) != (directions.get(n, "+") == "+")]
    model.metadata = {
        "description": description,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector_version": __version__,
        "n_segments": len(rows),
        "n_documents": len({r.group for r in rows}),
        "n_ai_segments": sum(r.label for r in rows),
        "l2": l2,
        "cv": cv,
        "weights_opposite_prior_direction": flipped,
        "neural": any(n in NEURAL_FEATURES for n in model.features),
    }
    return model, cv

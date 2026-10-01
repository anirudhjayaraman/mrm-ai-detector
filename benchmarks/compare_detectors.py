#!/usr/bin/env python3
"""Head-to-head comparison against freely available open-source AI-text detectors.

Evaluates every detector on the *held-out* documents of the benchmark corpus
(the same split used by ``calibrate.py``), reporting threshold-free metrics
(AUC, TPR at 1% / 5% FPR) and, where a detector has a natural decision
threshold, its operating-point accuracy and false-positive rate.

Detectors
---------
* ``mrm-detector``              — this package, stdlib handcrafted model
* ``mrm-detector+lm``           — this package with language-model features
* ``binoculars``                — Hans et al. 2024 (Qwen2.5-0.5B observer/performer)
* ``fast-detectgpt``            — Bao et al. 2024 (analytic, Qwen2.5-0.5B)
* ``openai-roberta``            — openai-community/roberta-base-openai-detector (MIT, 2019)
* ``hc3-roberta``               — Hello-SimpleAI/chatgpt-detector-roberta (2023)
* ``radar``                     — TrustSafeAI/RADAR-Vicuna-7B (2023, adversarially trained)
* ``fakespot-roberta``          — fakespot-ai/roberta-base-ai-text-detection-v1 (Apache-2.0, 2025)
* ``desklib-deberta``           — desklib/ai-text-detector-v1.01 (MIT, 2025)

Usage::

    python benchmarks/calibrate.py --lm
    python benchmarks/compare_detectors.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from common import (
    CLASSIFIER_CACHE,
    DATA,
    IN_DOMAIN_HUMAN,
    RESULTS,
    assign_splits,
    build_rows,
    document_segments,
    load_manifest,
    merge_rows,
    neural_cap,
    neural_rows,
)

from mrm_ai_detector.features.neural import CLASSIFIERS, load_classifier, pick_device
from mrm_ai_detector.scoring import ScoringModel
from mrm_ai_detector.train import auc, tpr_at_fpr

log = logging.getLogger("compare")


def cached_scores(name: str, segments: List[Tuple[str, str]], fn_factory: Callable[[], Callable[[str], float]]) -> Dict[str, float]:
    path = DATA / f"scores_{name}.jsonl"
    cache: Dict[str, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec["score"]
    todo = [(k, t) for k, t in segments if k not in cache]
    if todo:
        fn = fn_factory()
        with path.open("a") as fh:
            for i, (key, text) in enumerate(todo):
                cache[key] = fn(text)
                fh.write(json.dumps({"key": key, "score": cache[key]}) + "\n")
                if i % 100 == 0:
                    log.info("  %s: %d/%d", name, i, len(todo))
    return cache


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(scores: Dict[str, float], labels: Dict[str, int], threshold: Optional[float]) -> Dict[str, float]:
    keys = [k for k in labels if k in scores and scores[k] is not None and not math.isnan(scores[k])]
    s = [scores[k] for k in keys]
    y = [labels[k] for k in keys]
    out = {"n": len(keys), "auc": auc(s, y), "tpr_at_1pct_fpr": tpr_at_fpr(s, y, 0.01), "tpr_at_5pct_fpr": tpr_at_fpr(s, y, 0.05)}
    if threshold is not None:
        tp = sum(1 for a, b in zip(s, y) if a >= threshold and b == 1)
        fp = sum(1 for a, b in zip(s, y) if a >= threshold and b == 0)
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        out.update({"threshold": threshold, "tpr": tp / n_pos if n_pos else float("nan"), "fpr": fp / n_neg if n_neg else float("nan"),
                    "accuracy": (tp + (n_neg - fp)) / len(y) if y else float("nan")})
    return out


def doc_scores(seg_scores: Dict[str, float], seg_words: Dict[str, int]) -> Dict[str, float]:
    acc: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for key, score in seg_scores.items():
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue
        acc[key.rsplit("#", 1)[0]].append((score, seg_words.get(key, 1)))
    return {d: sum(s * w for s, w in v) / sum(w for _, w in v) for d, v in acc.items()}


def fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], help="detector names to skip")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs = load_manifest()
    splits = assign_splits(docs)
    test_docs = [d for d in docs if splits[d["id"]] == "test"]
    doc_meta = {d["id"]: d for d in test_docs}

    # Every detector is scored on the same segments: those with cached LM features.
    nn = neural_rows(test_docs)
    keep = {r.source for r in nn}
    rows = merge_rows([r for r in build_rows(test_docs, DATA / "features_handcrafted.jsonl") if r.source in keep], nn)
    seg_words = {r.source: r.words for r in rows}
    seg_label = {r.source: r.label for r in rows}
    texts = [(f"{d['id']}#{s.index}", s.unit.text) for d in test_docs for s in document_segments(d, limit=neural_cap(d))]

    det_scores: Dict[str, Dict[str, float]] = {}
    thresholds: Dict[str, Optional[float]] = {}
    hand = ScoringModel.default(neural=False)
    lm = ScoringModel.default(neural=True)
    det_scores["mrm-detector"] = {r.source: hand.probability(r.values, r.words) for r in rows}
    thresholds["mrm-detector"] = hand.ai_threshold
    if lm.neural_features:
        det_scores["mrm-detector+lm"] = {r.source: lm.probability(r.values, r.words) for r in rows}
        thresholds["mrm-detector+lm"] = lm.ai_threshold
    det_scores["binoculars"] = {r.source: (-r.values["binoculars"] if r.values.get("binoculars") is not None else None) for r in rows}
    thresholds["binoculars"] = None
    det_scores["fast-detectgpt"] = {r.source: r.values.get("fast_detectgpt") for r in rows}
    thresholds["fast-detectgpt"] = None
    for key, (repo, mode) in CLASSIFIERS.items():
        name = CLASSIFIER_CACHE[key]
        if name in args.skip:
            continue
        log.info("scoring %s", name)
        try:
            scores = cached_scores(name, texts, lambda repo=repo, mode=mode: load_classifier(repo, mode, pick_device(None)))
            det_scores[name] = {k: v for k, v in scores.items() if k in seg_label}  # held-out segments only
            thresholds[name] = 0.5
        except Exception as exc:  # noqa: BLE001
            log.error("could not run %s: %s", name, exc)

    doc_label = {d["id"]: 1 if d["label"] == "ai" else 0 for d in test_docs}
    subsets = {
        "all": lambda d: True,
        "in_domain": lambda d: d["family"] in IN_DOMAIN_HUMAN or d["family"].startswith("mrm_"),
        "out_of_domain": lambda d: d["family"].startswith(("hc3", "m4")),
    }
    results: Dict[str, Dict] = {}
    for name, seg in det_scores.items():
        dscore = doc_scores(seg, seg_words)
        res = {"segment": metrics(seg, seg_label, thresholds[name])}
        for sub, pred in subsets.items():
            labels = {d: y for d, y in doc_label.items() if pred(doc_meta[d])}
            res[f"document_{sub}"] = metrics(dscore, labels, thresholds[name])
        # Per-generator detection and per-family false positives at the detector's own threshold
        if thresholds[name] is not None:
            thr = thresholds[name]
            gen, fam = defaultdict(list), defaultdict(list)
            for d, s in dscore.items():
                m = doc_meta[d]
                (gen[m.get("model") or m["family"]] if m["label"] == "ai" else fam[m["family"]]).append(s >= thr)
            res["detected_by_generator"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(gen.items())}
            res["false_positive_by_family"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(fam.items())}
        results[name] = res

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison.json").write_text(json.dumps(results, indent=2, default=str))
    lines = [
        "| Detector | In-domain doc AUC | In-domain TPR@5%FPR | Out-of-domain doc AUC | All doc AUC | Segment AUC | Doc TPR / FPR at own threshold |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1]["document_in_domain"]["auc"] if not math.isnan(kv[1]["document_in_domain"]["auc"]) else -1)):
        di, do, da, sg = r["document_in_domain"], r["document_out_of_domain"], r["document_all"], r["segment"]
        op = f"{fmt(da.get('tpr'))} / {fmt(da.get('fpr'))}" if "tpr" in da else "n/a (no fixed threshold)"
        lines.append(f"| {name} | {fmt(di['auc'])} | {fmt(di['tpr_at_5pct_fpr'])} | {fmt(do['auc'])} | {fmt(da['auc'])} | {fmt(sg['auc'])} | {op} |")
    table = "\n".join(lines)
    (RESULTS / "comparison.md").write_text(table + "\n")
    print(table)
    n_in = sum(1 for d in test_docs if subsets["in_domain"](d))
    print(f"\nheld-out documents: {len(test_docs)} (in-domain {n_in}); segments: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

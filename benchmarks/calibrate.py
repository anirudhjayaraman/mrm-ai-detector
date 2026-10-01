#!/usr/bin/env python3
"""Calibrate the shipped scoring models on the benchmark corpus and evaluate on held-out documents.

Usage::

    python benchmarks/fetch_corpus.py
    python benchmarks/calibrate.py            # handcrafted model only (stdlib)
    python benchmarks/calibrate.py --lm       # also the language-model-augmented model

Writes ``mrm_ai_detector/models/default_model*.json`` and
``benchmarks/results/calibration*.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from typing import Dict, List

from common import DATA, FORMAT_SENSITIVE, RESULTS, ROOT, assign_splits, build_rows, load_manifest, merge_rows, neural_rows

from mrm_ai_detector.features import REGISTRY
from mrm_ai_detector.features.neural import NEURAL_FEATURES
from mrm_ai_detector.scoring import ScoringModel, sigmoid
from mrm_ai_detector.train import Row, classification_metrics, train

log = logging.getLogger("calibrate")
MODELS_DIR = ROOT.parent / "mrm_ai_detector" / "models"


def evaluate(model: ScoringModel, rows: List[Row]) -> Dict:
    """Segment-, document- and family-level metrics on held-out rows."""
    seg_p = [model.probability(r.values, r.words) for r in rows]
    seg_y = [r.label for r in rows]
    docs: Dict[str, List] = defaultdict(list)
    meta: Dict[str, Row] = {}
    for r in rows:
        docs[r.group].append((model.logit(r.values, r.words), r.words))
        meta[r.group] = r
    doc_p = {g: sigmoid(sum(lg * w for lg, w in v) / sum(w for _, w in v)) for g, v in docs.items()}
    thr = model.ai_threshold
    by_family: Dict[str, Dict] = {}
    for fam in sorted({(m.meta["family"], m.label) for m in meta.values()}):
        gs = [g for g, m in meta.items() if (m.meta["family"], m.label) == fam]
        probs = [doc_p[g] for g in gs]
        key = f"{'ai' if fam[1] else 'human'}/{fam[0]}"
        by_family[key] = {
            "documents": len(gs),
            "mean_ai_prob": round(sum(probs) / len(probs), 4),
            "flagged_rate": round(sum(p >= thr for p in probs) / len(probs), 4),
        }
    by_generator: Dict[str, Dict] = {}
    for gen in sorted({m.meta.get("model", "") for m in meta.values() if m.label == 1}):
        gs = [g for g, m in meta.items() if m.label == 1 and m.meta.get("model", "") == gen]
        probs = [doc_p[g] for g in gs]
        by_generator[gen or "unknown"] = {"documents": len(gs), "detected_rate": round(sum(p >= thr for p in probs) / len(probs), 4)}
    return {
        "threshold": thr,
        "segment": classification_metrics(seg_p, seg_y, thr),
        "document": classification_metrics(list(doc_p.values()), [meta[g].label for g in doc_p], thr),
        "by_family": by_family,
        "by_generator": by_generator,
    }


def pick_threshold(model: ScoringModel, rows: List[Row], target_fpr: float = 0.02) -> float:
    """Choose the AI threshold on *training* documents to keep human-document FPR at or below target."""
    docs: Dict[str, List] = defaultdict(list)
    labels = {}
    for r in rows:
        docs[r.group].append((model.logit(r.values, r.words), r.words))
        labels[r.group] = r.label
    human = sorted((sigmoid(sum(lg * w for lg, w in v) / sum(w for _, w in v)) for g, v in docs.items() if labels[g] == 0), reverse=True)
    k = int(math.floor(target_fpr * len(human)))
    thr = human[k] + 1e-4 if k < len(human) else 0.5
    return round(min(0.9, max(0.5, thr)), 3)


def pick_segment_threshold(model: ScoringModel, rows: List[Row], target_fpr: float = 0.02) -> float:
    """Segment-flagging threshold at ``target_fpr`` on human in-domain training segments."""
    human = sorted((model.probability(r.values, r.words) for r in rows if r.label == 0 and r.weight > 1.0), reverse=True)
    k = int(math.floor(target_fpr * len(human)))
    thr = human[k] + 1e-4 if k < len(human) else 0.5
    return round(min(0.97, max(model.ai_threshold, thr)), 3)


def fmt(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", action="store_true", help="also calibrate the language-model-augmented model")
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs = load_manifest()
    splits = assign_splits(docs)
    train_docs = [d for d in docs if splits[d["id"]] == "train"]
    test_docs = [d for d in docs if splits[d["id"]] == "test"]
    log.info("documents: train=%d test=%d", len(train_docs), len(test_docs))

    rows_train = build_rows(train_docs, DATA / "features_handcrafted.jsonl")
    rows_test = build_rows(test_docs, DATA / "features_handcrafted.jsonl")
    log.info("segments: train=%d (ai=%d) test=%d (ai=%d)", len(rows_train), sum(r.label for r in rows_train), len(rows_test), sum(r.label for r in rows_test))

    variants = [("handcrafted", [f for f in REGISTRY if f not in FORMAT_SENSITIVE], "default_model.json")]
    if args.lm:
        nn_train = neural_rows(train_docs)
        nn_test = neural_rows(test_docs)
        # The LM model is fitted and evaluated only on segments that have LM features.
        keep_train, keep_test = {r.source for r in nn_train}, {r.source for r in nn_test}
        lm_train = merge_rows([r for r in rows_train if r.source in keep_train], nn_train)
        lm_test = merge_rows([r for r in rows_test if r.source in keep_test], nn_test)
        variants.append(("handcrafted+lm", [f for f in REGISTRY if f not in FORMAT_SENSITIVE] + list(NEURAL_FEATURES), "default_model_lm.json"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, features, fname in variants:
        if name == "handcrafted+lm":
            rows_train, rows_test = lm_train, lm_test
        model, cv = train(
            rows_train, folds=5, name=f"mrm-{name}-v1", l2=args.l2, features=features,
            description=f"Calibrated on the benchmark corpus ({name} features); see benchmarks/README.md",
        )
        model.ai_threshold = pick_threshold(model, rows_train, args.target_fpr)
        model.segment_threshold = pick_segment_threshold(model, rows_train, args.target_fpr)
        model.human_threshold = 0.3
        held_out = evaluate(model, rows_test)
        model.metadata["held_out"] = {"segment": held_out["segment"], "document": held_out["document"]}
        model.metadata["threshold_rule"] = f"max training-document human FPR {args.target_fpr:.0%}"
        model.save(MODELS_DIR / fname)
        (RESULTS / f"calibration_{name.replace('+', '_')}.json").write_text(
            json.dumps({"cv_train": cv, "held_out": held_out, "weights": {k: round(v.weight, 4) for k, v in model.features.items()}}, indent=2, default=str)
        )
        summary[name] = held_out
        d, s = held_out["document"], held_out["segment"]
        log.info("\n== %s  (document threshold %.3f, segment threshold %.3f)", name, model.ai_threshold, model.segment_threshold)
        log.info("CV (train, doc-grouped)  segment AUC %s  document AUC %s", fmt(cv["segment"]["auc"]), fmt(cv["document"]["auc"]))
        log.info("HELD-OUT  segment AUC %s  TPR@1%%FPR %s | document AUC %s acc %s precision %s recall %s FPR %s",
                 fmt(s["auc"]), fmt(s["tpr_at_1pct_fpr"]), fmt(d["auc"]), fmt(d["accuracy"]), fmt(d["precision"]), fmt(d["recall"]), fmt(d["fpr"]))
        for fam, m in held_out["by_family"].items():
            log.info("   %-34s n=%-4d mean p=%.3f flagged=%.2f", fam, m["documents"], m["mean_ai_prob"], m["flagged_rate"])
        top = sorted(model.features.items(), key=lambda kv: -abs(kv[1].weight))[:12]
        log.info("   top weights: %s", ", ".join(f"{k} {v.weight:+.2f}" for k, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from mrm_ai_detector.features import extract  # noqa: E402
from mrm_ai_detector.text import segment_document  # noqa: E402
from mrm_ai_detector.train import Row  # noqa: E402

DATA = ROOT / "data"
CORPUS = DATA / "corpus"
RESULTS = ROOT / "results"
log = logging.getLogger("bench")

TEST_FRACTION = 0.3
MAX_SEGMENTS_PER_DOC = 40
MIN_SEGMENT_WORDS = 60
NEURAL_SEGMENTS_IN_DOMAIN = 12
NEURAL_SEGMENTS_GENERIC = 4

# Features whose values depend on document formatting/extraction (PDF line breaks,
# markdown) rather than on authorship.  They stay available for custom training
# but are excluded from the shipped calibration to avoid learning "PDF vs markdown".
IN_DOMAIN_HUMAN = {"academic", "comment_letter", "model_documentation", "regulatory_guidance", "supervisory_findings", "validation_report", "business_email"}
# In-domain (model-risk) documents count more in fitting than the generic HC3/M4 datasets.
IN_DOMAIN_WEIGHT = 3.0


def is_in_domain(family: str) -> bool:
    return family in IN_DOMAIN_HUMAN or family.startswith("mrm_")


FORMAT_SENSITIVE = {"para_len_cv", "bullet_line_rate", "bold_lead_rate", "colon_rate", "question_rate"}


# Source documents of the web-detector samples (web_samples/) are always held out,
# so the comparison with GPTZero / ZeroGPT / Sapling is out-of-sample for us too.
PINNED_TEST = {
    "fed_sr11-7", "fed_oig_2015_stress_test_validation", "gao_16_269", "fhfa_oig_evl2020_001",
    "claude_005", "claude_004", "claude_023", "qwen2-5-1-5b-instruct_003",
}


def load_manifest() -> List[Dict[str, str]]:
    with (CORPUS / "manifest.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split_of(doc: Dict[str, str]) -> str:
    """Deterministic document-level split, stratified by family via hashing."""
    h = int(hashlib.sha1(f"{doc['family']}::{doc['id']}".encode()).hexdigest(), 16)
    return "test" if (h % 1000) / 1000 < TEST_FRACTION else "train"


def assign_splits(docs: List[Dict[str, str]]) -> Dict[str, str]:
    """Hash split, but guarantee every (label, family) with >=2 docs has one test doc."""
    splits = {d["id"]: split_of(d) for d in docs}
    groups: Dict[tuple, List[Dict[str, str]]] = {}
    for d in docs:
        groups.setdefault((d["label"], d["family"]), []).append(d)
    for doc_id in PINNED_TEST:
        if doc_id in splits:
            splits[doc_id] = "test"
    for members in groups.values():
        if len(members) >= 2 and not any(splits[m["id"]] == "test" for m in members):
            pick = min(members, key=lambda m: hashlib.sha1(m["id"].encode()).hexdigest())
            splits[pick["id"]] = "test"
        movable = [m for m in members if m["id"] not in PINNED_TEST]
        if len(members) >= 2 and movable and all(splits[m["id"]] == "test" for m in members):
            pick = max(movable, key=lambda m: hashlib.sha1(m["id"].encode()).hexdigest())
            splits[pick["id"]] = "train"
    return splits


def neural_cap(doc: Dict[str, str]) -> int:
    """Language-model features are expensive: fewer segments per generic (HC3/M4) document."""
    return NEURAL_SEGMENTS_IN_DOMAIN if is_in_domain(doc["family"]) else NEURAL_SEGMENTS_GENERIC


def document_segments(doc: Dict[str, str], seed: int = 5, limit: Optional[int] = None):
    """Up to MAX_SEGMENTS_PER_DOC segments; ``limit`` takes a further random subset of those."""
    text = (ROOT / doc["path"]).read_text(encoding="utf-8")
    segs = [s for s in segment_document(text) if s.unit.word_count >= MIN_SEGMENT_WORDS]
    rng = random.Random(f"{seed}:{doc['id']}")
    if len(segs) > MAX_SEGMENTS_PER_DOC:
        segs = sorted(rng.sample(segs, MAX_SEGMENTS_PER_DOC), key=lambda s: s.index)
    if limit is not None and len(segs) > limit:
        segs = sorted(random.Random(f"neural:{doc['id']}").sample(segs, limit), key=lambda s: s.index)
    return segs


def build_rows(
    docs: Iterable[Dict[str, str]],
    cache: Path,
    scorer: Optional[Callable[[str], Dict[str, Optional[float]]]] = None,
    cap: Optional[Callable[[Dict[str, str]], int]] = None,
) -> List[Row]:
    """Segment-level rows with handcrafted (and optionally neural) features, cached as JSONL."""
    cached: Dict[str, Dict] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cached[rec["key"]] = rec
    rows: List[Row] = []
    with cache.open("a", encoding="utf-8") as fh:
        for n, doc in enumerate(docs):
            for seg in document_segments(doc, limit=cap(doc) if cap else None):
                key = f"{doc['id']}#{seg.index}"
                if key in cached:
                    vals = cached[key]["values"]
                else:
                    vals = extract(seg.unit) if scorer is None else scorer(seg.unit.text)
                    fh.write(json.dumps({"key": key, "values": vals}) + "\n")
                    fh.flush()
                rows.append(
                    Row(
                        group=doc["id"],
                        label=1 if doc["label"] == "ai" else 0,
                        words=seg.unit.word_count,
                        values=dict(vals),
                        source=key,
                        meta={"family": doc["family"], "model": doc.get("model", "")},
                        weight=IN_DOMAIN_WEIGHT if is_in_domain(doc["family"]) else 1.0,
                    )
                )
            if n % 50 == 0:
                log.info("features: %d docs processed", n + 1)
    return rows


def merge_rows(base: List[Row], extra: List[Row]) -> List[Row]:
    lookup = {r.source: r.values for r in extra}
    return [Row(r.group, r.label, r.words, {**r.values, **lookup.get(r.source, {})}, r.source, r.meta, r.weight) for r in base]


# ---------------------------------------------------------------------------
# Language-model features (zero-shot + supervised classifiers), cached per segment
# ---------------------------------------------------------------------------
# Cache file stem for each classifier; shared with compare_detectors.py.
CLASSIFIER_CACHE = {"hc3": "hc3-roberta", "fakespot": "fakespot-roberta", "desklib": "desklib-deberta",
                    "openai": "openai-roberta", "radar": "radar"}
ZERO_SHOT_KEYS = ("binoculars", "fast_detectgpt", "lm_log_ppl")


def _read_cache(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def neural_rows(docs: Iterable[Dict[str, str]], classifiers: Optional[Iterable[str]] = None) -> List[Row]:
    """Rows carrying LM features for the (capped) segments of ``docs``; computes and caches what is missing."""
    from mrm_ai_detector.features.neural import (
        CLASSIFIERS,
        DEFAULT_CLASSIFIERS,
        NeuralConfig,
        NeuralScorer,
        load_classifier,
        pick_device,
        to_logit,
    )

    classifiers = tuple(classifiers or DEFAULT_CLASSIFIERS)
    docs = list(docs)
    segments = [(doc, seg) for doc in docs for seg in document_segments(doc, limit=neural_cap(doc))]
    keyed = [(f"{doc['id']}#{seg.index}", doc, seg) for doc, seg in segments]

    zs_path = DATA / "features_neural.jsonl"
    zs = _read_cache(zs_path)
    todo = [(k, s) for k, _, s in keyed if k not in zs]
    if todo:
        scorer = NeuralScorer(NeuralConfig(classifiers=()))
        with zs_path.open("a", encoding="utf-8") as fh:
            for i, (key, seg) in enumerate(todo):
                vals = {k: v for k, v in scorer.score(seg.unit.text).items() if k in ZERO_SHOT_KEYS}
                zs[key] = {"key": key, "values": vals}
                fh.write(json.dumps(zs[key]) + "\n")
                if i % 200 == 0:
                    log.info("zero-shot features: %d/%d", i, len(todo))

    probs: Dict[str, Dict[str, Dict]] = {}
    for name in classifiers:
        path = DATA / f"scores_{CLASSIFIER_CACHE[name]}.jsonl"
        cache = _read_cache(path)
        todo = [(k, s) for k, _, s in keyed if k not in cache]
        if todo:
            repo, mode = CLASSIFIERS[name]
            fn = load_classifier(repo, mode, pick_device(None))
            with path.open("a", encoding="utf-8") as fh:
                for i, (key, seg) in enumerate(todo):
                    cache[key] = {"key": key, "score": fn(seg.unit.text)}
                    fh.write(json.dumps(cache[key]) + "\n")
                    if i % 200 == 0:
                        log.info("%s: %d/%d", name, i, len(todo))
        probs[name] = cache

    rows: List[Row] = []
    for key, doc, seg in keyed:
        vals: Dict[str, Optional[float]] = {k: zs[key]["values"].get(k) for k in ZERO_SHOT_KEYS}
        for name in classifiers:
            p = probs[name][key]["score"]
            vals[f"{name}_logit"] = None if p is None else to_logit(p)
        rows.append(Row(doc["id"], 1 if doc["label"] == "ai" else 0, seg.unit.word_count, vals, key,
                        {"family": doc["family"], "model": doc.get("model", "")},
                        IN_DOMAIN_WEIGHT if is_in_domain(doc["family"]) else 1.0))
    return rows

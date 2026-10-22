#!/usr/bin/env python3
"""Evaluate the detector on partially AI-written documents.

Three test sets:

* **mixed MRM** (``build_mixed.py``): windows of pre-ChatGPT model risk documents in
  which some paragraphs were polished, rewritten, written from notes or continued by
  an LLM, and some windows end with an AI-written "key findings" section.  Every
  word has a label, so we can check whether the detector's *AI share* tracks the
  true share and which segments it flags.
* **APT-Eval** (Saha & Feizi, ACL 2025): human texts polished by GPT-4o, DeepSeek-V3
  and Llama models at controlled levels of AI involvement (1%-75% of the text, or
  "extreme minor" to "major" edits).
* **MixSet** (Gao et al., NAACL 2024): AI-revised human text and human-revised AI text.

Usage::

    python benchmarks/eval_mixed.py            # lightweight model
    python benchmarks/eval_mixed.py --lm       # with language-model features
    python benchmarks/eval_mixed.py --model path/to/model.json

Segment features are cached in ``data/mixed_features*.jsonl``, so re-running with a
retrained model is fast.  Results go to ``results/mixed*.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from build_mixed import load_mixed, render
from common import DATA, RESULTS, assign_splits, load_manifest

from mrm_ai_detector.features import extract
from mrm_ai_detector.features.neural import NEURAL_FEATURES
from mrm_ai_detector.scoring import ScoringModel
from mrm_ai_detector.text import build_unit, is_prose, segment_document
from mrm_ai_detector.train import auc

log = logging.getLogger("eval_mixed")

APT_URL = "https://huggingface.co/datasets/smksaha/apt-eval/resolve/main/merged_apt_eval_dataset.csv"
APT_ORIG_URL = "https://huggingface.co/datasets/smksaha/apt-eval/resolve/main/original.csv"
MIXSET_URL = "https://raw.githubusercontent.com/Dongping-Chen/MixSet/main/data/MixSet/Mixset_test.json"


# ---------------------------------------------------------------------------
# Feature cache and document scoring (mirrors Detector.analyze_text)
# ---------------------------------------------------------------------------
class SegmentScorer:
    def __init__(self, neural: bool, cache: Path):
        self.neural = None
        if neural:
            from mrm_ai_detector.features.neural import NeuralScorer
            self.neural = NeuralScorer()
        self.cache_path = cache
        self.cache: Dict[str, Dict] = {}
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["k"]] = rec["v"]
        self.fh = cache.open("a", encoding="utf-8")

    def features(self, text: str) -> Dict[str, Optional[float]]:
        key = hashlib.sha1(text.encode()).hexdigest()
        if key not in self.cache:
            vals = extract(build_unit(text))
            if self.neural is not None:
                vals.update(self.neural.score(text))
            self.cache[key] = vals
            self.fh.write(json.dumps({"k": key, "v": vals}) + "\n")
            self.fh.flush()
        return dict(self.cache[key])


def score_document(model: ScoringModel, scorer: SegmentScorer, text: str, segment_words: int = 180) -> Dict:
    unit = build_unit(text)
    segs = segment_document(text, target_words=segment_words)
    seg_vals = [scorer.features(s.unit.text) for s in segs]
    values = extract(unit, model.features)
    if scorer.neural is not None:
        for k in NEURAL_FEATURES:
            pairs = [(v.get(k), s.unit.word_count) for v, s in zip(seg_vals, segs) if v.get(k) is not None]
            tot = sum(w for _, w in pairs)
            values[k] = sum(x * w for x, w in pairs) / tot if tot else None
    prob, _ = model.score(values, unit.word_count)
    out = []
    for s, v in zip(segs, seg_vals):
        p, _ = model.score(v, s.unit.word_count)
        prose = is_prose(s.unit)
        out.append({"text": s.unit.text, "words": s.unit.word_count, "p": p, "scored": prose,
                    "flagged": prose and p >= model.effective_segment_threshold})
    scored = [s for s in out if s["scored"]]
    flagged = sum(s["words"] for s in scored if s["flagged"])
    return {"p": prob, "ai_share": flagged / max(1, sum(s["words"] for s in scored)), "segments": out, "words": unit.word_count}


def align(doc_words: List[str], labels: List[str], seg_text: str, start: int):
    """Greedy in-order alignment of a segment's words onto the document's word labels."""
    got: List[str] = []
    i = start
    for w in seg_text.split():
        j = i
        while j < len(doc_words) and doc_words[j] != w and j - i < 40:
            j += 1
        if j < len(doc_words) and doc_words[j] == w:
            got.append(labels[j])
            i = j + 1
    return got, i


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    return statistics.correlation(rx, ry) if len(x) > 2 else float("nan")


def bucket(share: float) -> str:
    if share == 0:
        return "0%"
    if share < 0.35:
        return "1-35%"
    if share < 0.7:
        return "35-70%"
    return "70-100%"


# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------
def eval_mrm(model: ScoringModel, scorer: SegmentScorer, split: str) -> Dict:
    splits = assign_splits(load_manifest())
    docs = [d for d in load_mixed() if split == "all" or splits.get(d["source"], "train") == split]
    rows, seg_rows = [], []
    for n, d in enumerate(docs):
        text = render(d)
        res = score_document(model, scorer, text)
        words, labels = text.split(), [x for p in d["paragraphs"] for x in [p["label"]] * len(p["text"].split())]
        ops = [x for p in d["paragraphs"] for x in [p["op"]] * len(p["text"].split())]
        lab = [x for x in labels if x != "heading"]
        share = sum(x == "ai" for x in lab) / max(1, len(lab))
        pos = 0
        for s in res["segments"]:
            got, pos2 = align(words, labels, s["text"], pos)
            got_ops, _ = align(words, ops, s["text"], pos)
            pos = pos2
            if not s["scored"]:
                continue
            got = [g for g in got if g != "heading"]
            if not got:
                continue
            frac_ai = sum(g == "ai" for g in got) / len(got)
            frac_pol = sum(g == "polish" for g in got) / len(got)
            major_op = max(set(got_ops) - {"heading"} or {"human"}, key=got_ops.count)
            seg_rows.append({"doc": d["id"], "gen": d["generator"], "frac_ai": frac_ai, "frac_polish": frac_pol,
                             "op": major_op, "p": s["p"], "flagged": s["flagged"], "words": s["words"]})
        rows.append({"doc": d["id"], "gen": d["generator"], "true_share": share, "pred_share": res["ai_share"], "p": res["p"],
                     "flagged_doc": res["p"] >= model.ai_threshold, "any_segment": any(s["flagged"] for s in res["segments"])})
        if n % 25 == 0:
            log.info("mixed MRM: %d/%d", n + 1, len(docs))

    out: Dict = {"documents": len(rows), "segments": len(seg_rows)}
    if not rows:
        return out
    out["share_mae"] = round(statistics.mean(abs(r["true_share"] - r["pred_share"]) for r in rows), 3)
    out["share_spearman"] = round(spearman([r["true_share"] for r in rows], [r["pred_share"] for r in rows]), 3)
    out["prob_spearman"] = round(spearman([r["true_share"] for r in rows], [r["p"] for r in rows]), 3)
    by_b = defaultdict(list)
    for r in rows:
        by_b[bucket(r["true_share"])].append(r)
    out["by_true_share"] = {
        b: {"documents": len(v), "mean_true_share": round(statistics.mean(r["true_share"] for r in v), 3),
            "mean_pred_share": round(statistics.mean(r["pred_share"] for r in v), 3),
            "mean_doc_p": round(statistics.mean(r["p"] for r in v), 3),
            "doc_flag_rate": round(sum(r["flagged_doc"] for r in v) / len(v), 3),
            "any_segment_flag_rate": round(sum(r["any_segment"] for r in v) / len(v), 3)}
        for b, v in sorted(by_b.items(), key=lambda kv: ["0%", "1-35%", "35-70%", "70-100%"].index(kv[0]))
    }
    human = [s for s in seg_rows if s["frac_ai"] == 0 and s["frac_polish"] < 0.2]
    ai = [s for s in seg_rows if s["frac_ai"] >= 0.8]
    out["segment_auc_ai_vs_human"] = round(auc([s["p"] for s in human + ai], [0] * len(human) + [1] * len(ai)), 3)
    out["segment_fpr_human"] = round(sum(s["flagged"] for s in human) / max(1, len(human)), 3)
    out["segment_tpr_ai"] = round(sum(s["flagged"] for s in ai) / max(1, len(ai)), 3)
    out["segment_by_op"] = {}
    for op in ("rewrite", "notes", "continue", "summary"):
        v = [s for s in ai if s["op"] == op]
        if v:
            out["segment_by_op"][op] = {"segments": len(v), "flag_rate": round(sum(s["flagged"] for s in v) / len(v), 3),
                                        "auc_vs_human": round(auc([s["p"] for s in human + v], [0] * len(human) + [1] * len(v)), 3)}
    pol = [s for s in seg_rows if s["frac_polish"] >= 0.5]
    if pol:
        out["segment_polish_flag_rate"] = round(sum(s["flagged"] for s in pol) / len(pol), 3)
    out["segment_by_generator"] = {}
    for g in sorted({s["gen"] for s in seg_rows}):
        a = [s for s in ai if s["gen"] == g]
        h = [s for s in human if s["gen"] == g]
        if a and h:
            out["segment_by_generator"][g] = {"ai_segments": len(a), "tpr": round(sum(s["flagged"] for s in a) / len(a), 3),
                                              "auc": round(auc([s["p"] for s in h + a], [0] * len(h) + [1] * len(a)), 3)}
    out["_rows"] = rows
    return out


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def eval_apt(model: ScoringModel, scorer: SegmentScorer, n_per_cell: int = 12) -> Dict:
    csv.field_size_limit(10**9)
    raw = DATA / "raw" / "apt_eval.csv"
    orig = DATA / "raw" / "apt_eval_original.csv"
    rows = list(csv.DictReader(fetch(APT_URL, raw).open(encoding="utf-8")))
    originals = list(csv.DictReader(fetch(APT_ORIG_URL, orig).open(encoding="utf-8")))
    rng = random.Random(3)
    cells = defaultdict(list)
    for r in rows:
        if r["polisher"] not in ("GPT-4o", "DeepSeek-V3", "Llama3.1-70B") or len(r["generation"].split()) < 80:
            continue
        level = r["polishing_degree"] or f"{int(float(r['polishing_percent']))}%"
        cells[(r["polisher"], level)].append(r)
    out: Dict = {"human_original": {}, "polished": {}}
    text_key = "generation"
    hum = [r for r in originals if len(r[text_key].split()) >= 80]
    hp = [model.score(scorer.features(r[text_key]), len(r[text_key].split()))[0] for r in rng.sample(hum, min(120, len(hum)))]
    thr = model.effective_segment_threshold
    out["human_original"] = {"n": len(hp), "mean_p": round(statistics.mean(hp), 3), "flag_rate": round(sum(p >= thr for p in hp) / len(hp), 3)}
    levels = ["extreme_minor", "minor", "slight_major", "major", "1%", "5%", "10%", "20%", "35%", "50%", "75%"]
    for level in levels:
        ps = []
        for pol in ("GPT-4o", "DeepSeek-V3", "Llama3.1-70B"):
            cand = cells.get((pol, level), [])
            for r in rng.sample(cand, min(n_per_cell, len(cand))):
                ps.append(model.score(scorer.features(r["generation"]), len(r["generation"].split()))[0])
        if ps:
            out["polished"][level] = {"n": len(ps), "mean_p": round(statistics.mean(ps), 3), "flag_rate": round(sum(p >= thr for p in ps) / len(ps), 3),
                                      "auc_vs_human": round(auc(hp + ps, [0] * len(hp) + [1] * len(ps)), 3)}
    return out


def eval_mixset(model: ScoringModel, scorer: SegmentScorer) -> Dict:
    data = json.loads(fetch(MIXSET_URL, DATA / "raw" / "mixset_test.json").read_text(encoding="utf-8"))
    thr = model.effective_segment_threshold
    by = defaultdict(list)
    for r in data:
        t = r["revised"]
        if len(t.split()) < 60:
            continue
        by[r["mixset_category"]].append(model.score(scorer.features(t), len(t.split()))[0])
    return {c: {"n": len(v), "mean_p": round(statistics.mean(v), 3), "flag_rate": round(sum(p >= thr for p in v) / len(v), 3)} for c, v in sorted(by.items())}


def to_markdown(name: str, res: Dict) -> str:
    m = res["mrm"]
    lines = [f"## {name}", "", f"Mixed MRM documents ({m['documents']} documents, {m['segments']} segments, held-out sources only)", "",
             f"* AI-share MAE {m.get('share_mae')} · Spearman(true share, predicted share) {m.get('share_spearman')} · Spearman(true share, document probability) {m.get('prob_spearman')}",
             f"* Segment AUC (AI-written vs untouched human) {m.get('segment_auc_ai_vs_human')} · segment TPR {m.get('segment_tpr_ai')} · segment FPR {m.get('segment_fpr_human')} · polished-segment flag rate {m.get('segment_polish_flag_rate')}", "",
             "| True AI share | Docs | Mean true | Mean predicted share | Mean doc probability | Doc flagged | Any segment flagged |", "|---|---:|---:|---:|---:|---:|---:|"]
    for b, v in m.get("by_true_share", {}).items():
        lines.append(f"| {b} | {v['documents']} | {v['mean_true_share']} | {v['mean_pred_share']} | {v['mean_doc_p']} | {v['doc_flag_rate']} | {v['any_segment_flag_rate']} |")
    lines += ["", "| AI operation | Segments | Flag rate | AUC vs human |", "|---|---:|---:|---:|"]
    for op, v in m.get("segment_by_op", {}).items():
        lines.append(f"| {op} | {v['segments']} | {v['flag_rate']} | {v['auc_vs_human']} |")
    lines += ["", "| Generator | AI segments | TPR | AUC |", "|---|---:|---:|---:|"]
    for g, v in m.get("segment_by_generator", {}).items():
        lines.append(f"| {g} | {v['ai_segments']} | {v['tpr']} | {v['auc']} |")
    if "apt" in res:
        a = res["apt"]
        lines += ["", f"APT-Eval (polished human text; human originals: mean p {a['human_original']['mean_p']}, flagged {a['human_original']['flag_rate']})", "",
                  "| AI involvement | n | Mean p | Flag rate | AUC vs human |", "|---|---:|---:|---:|---:|"]
        for lv, v in a["polished"].items():
            lines.append(f"| {lv} | {v['n']} | {v['mean_p']} | {v['flag_rate']} | {v['auc_vs_human']} |")
    if "mixset" in res:
        lines += ["", "MixSet", "", "| Category | n | Mean p | Flag rate |", "|---|---:|---:|---:|"]
        for c, v in res["mixset"].items():
            lines.append(f"| {c} | {v['n']} | {v['mean_p']} | {v['flag_rate']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", action="store_true")
    ap.add_argument("--model", help="model JSON (default: the shipped model for the mode)")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--no-external", action="store_true", help="skip APT-Eval and MixSet")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = ScoringModel.load(args.model) if args.model else ScoringModel.default(neural=args.lm)
    scorer = SegmentScorer(args.lm, DATA / f"mixed_features{'_lm' if args.lm else ''}.jsonl")
    res = {"model": model.name, "mrm": eval_mrm(model, scorer, args.split)}
    if not args.no_external:
        res["apt"] = eval_apt(model, scorer)
        res["mixset"] = eval_mixset(model, scorer)
    name = f"{model.name}{args.tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"mixed_{'lm' if args.lm else 'light'}{args.tag}"
    rows = res["mrm"].pop("_rows", [])
    (RESULTS / f"{stem}.json").write_text(json.dumps(res, indent=2))
    (DATA / f"{stem}_docs.json").write_text(json.dumps(rows, indent=1))
    md = to_markdown(name, res)
    (RESULTS / f"{stem}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

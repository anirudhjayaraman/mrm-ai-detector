#!/usr/bin/env python3
"""Fine-tune a small transformer classifier on in-domain human vs AI paragraphs.

The off-the-shelf classifiers were trained on essays, Q&A and news; on dense,
technical model documentation written with AI help they confidently say "human".
This script continues training one of them (default: Fakespot RoBERTa-base) on
paragraphs from the benchmark corpus:

* human: paragraphs of pre-ChatGPT model risk documents and the untouched
  paragraphs of the mixed documents;
* AI: rewritten / notes-to-prose / continued / summary paragraphs of the mixed
  documents (``build_mixed.py``, ``mixed_claude/``) and the synthetic ``ai_mrm`` documents.

Only *training-split* sources are used, so held-out evaluation stays clean.

Usage::

    python benchmarks/finetune_classifier.py --out benchmarks/data/mrm-roberta
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from build_mixed import EXCLUDE, GENRE, is_prose, load_mixed, strip_heading
from common import DATA, ROOT, assign_splits, load_manifest


def paragraphs(text: str):
    for p in re.split(r"\n\s*\n", text):
        p = strip_heading(p)
        if is_prose(p):
            yield p


def build_dataset(seed: int = 7):
    docs = load_manifest()
    splits = assign_splits(docs)
    rng = random.Random(seed)
    train, test = [], []
    for d in docs:
        if not (d["family"] in GENRE or d["family"].startswith("mrm_")) or d["id"] in EXCLUDE:
            continue
        paras = list(paragraphs((ROOT / d["path"]).read_text(encoding="utf-8")))
        rng.shuffle(paras)
        cap = 60 if d["label"] == "human" else 12
        target = train if splits[d["id"]] == "train" else test
        target += [(p, 1 if d["label"] == "ai" else 0, d["id"]) for p in paras[:cap]]
    for d in load_mixed():
        target = train if splits.get(d["source"], "train") == "train" else test
        for p in d["paragraphs"]:
            if p["label"] in ("ai", "human") and len(p["text"].split()) >= 40:
                target.append((p["text"], 1 if p["label"] == "ai" else 0, d["id"]))
    return train, test


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="fakespot-ai/roberta-base-ai-text-detection-v1")
    ap.add_argument("--out", default=str(DATA / "mrm-roberta"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from mrm_ai_detector.features.neural import pick_device
    from mrm_ai_detector.train import auc

    train, test = build_dataset()
    pos = sum(y for _, y, _ in train)
    print(f"train {len(train)} paragraphs ({pos} AI), test {len(test)} ({sum(y for _, y, _ in test)} AI)", flush=True)
    device = pick_device(None)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)
    ai_index = 1 if model.config.num_labels > 1 else 0
    w_pos = (len(train) - pos) / max(1, pos)  # balance classes
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def predict(rows):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 16):
                enc = tok([r[0] for r in rows[i:i + 16]], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits.float()
                p = torch.softmax(logits, -1)[:, ai_index] if logits.shape[-1] > 1 else torch.sigmoid(logits[:, 0])
                out += p.cpu().tolist()
        return out

    print(f"before: test AUC {auc(predict(test), [y for _, y, _ in test]):.3f}", flush=True)
    rng = random.Random(1)
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        for i in range(0, len(train), args.batch):
            batch = train[i:i + args.batch]
            enc = tok([b[0] for b in batch], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(device)
            y = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
            logits = model(**enc).logits.float()
            z = logits[:, ai_index] - logits[:, 1 - ai_index] if logits.shape[-1] > 1 else logits[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor(w_pos, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            if (i // args.batch) % 100 == 0:
                print(f"epoch {ep} step {i // args.batch} loss {loss.item():.3f}", flush=True)
        print(f"epoch {ep}: test AUC {auc(predict(test), [y for _, y, _ in test]):.3f}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "training.json").write_text(json.dumps({"base": args.base, "train": len(train), "test": len(test), "epochs": args.epochs, "lr": args.lr}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download and normalise the calibration / benchmark corpus.

Usage::

    python benchmarks/fetch_corpus.py            # everything in sources.json
    python benchmarks/fetch_corpus.py --refresh  # re-download

Writes ``benchmarks/data/corpus/{human,ai}/<family>/<id>.txt`` and a
``manifest.csv`` describing every document.  PDF sources need ``pypdf``.
Committed AI-generated MRM documents under ``benchmarks/ai_mrm/`` are copied
into the corpus too.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from mrm_ai_detector.ingest import IngestError, read_document, strip_quoted_email  # noqa: E402
from mrm_ai_detector.text import WORD_RE, normalise  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus"
USER_AGENT = "Mozilla/5.0 (compatible; mrm-ai-detector-corpus/1.0)"
log = logging.getLogger("fetch")


def download(url: str, dest: Path, refresh: bool = False) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,application/json,text/plain,text/html,*/*"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return dest
        except Exception as exc:  # noqa: BLE001
            log.warning("download failed (%s/3) %s: %s", attempt + 1, url, exc)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


_REFERENCES_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works Cited)\s*$", re.MULTILINE)


def clean_document(text: str) -> str:
    """Remove reference lists, equation debris and tables of contents from extracted text."""
    matches = list(_REFERENCES_RE.finditer(text))
    m = matches[-1] if matches else None
    if m and m.start() > len(text) * 0.5:
        text = text[: m.start()]
    kept: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        letters = sum(c.isalpha() for c in s)
        if len(s) > 8 and letters / len(s) < 0.6:  # equations, tables, numeric debris
            continue
        if re.search(r"\.{5,}\s*\d+$", s):  # table-of-contents leaders
            continue
        if len(WORD_RE.findall(s)) <= 2 and not re.match(r"^#+\s", s) and len(s) < 25 and not s.endswith((".", ":")):
            continue  # stray fragments
        kept.append(s)
    return normalise("\n".join(kept))


_SEEN: set = set()


def write_doc(label: str, family: str, doc_id: str, text: str, rows: List[Dict], **meta) -> None:
    key = hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()
    if key in _SEEN:  # identical texts recur across M4 files; keep one copy to avoid CV leakage
        return
    _SEEN.add(key)
    out = CORPUS / label / family / f"{doc_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    rows.append({"id": doc_id, "label": label, "family": family, "path": str(out.relative_to(ROOT)), "words": len(WORD_RE.findall(text)), **meta})


def fetch_documents(docs: Iterable[Dict], rows: List[Dict], refresh: bool) -> None:
    for d in docs:
        ext = ".pdf" if d["format"] == "pdf" else ".html"
        raw = RAW / f"{d['id']}{ext}"
        try:
            if "arxiv.org" in d["url"] and not raw.exists():
                time.sleep(3)  # arXiv asks for polite request pacing
            download(d["url"], raw, refresh)
            text = clean_document(read_document(raw))
        except (RuntimeError, IngestError) as exc:
            log.error("skipping %s: %s", d["id"], exc)
            continue
        write_doc(d["label"], d["family"], d["id"], text, rows, model="", year=d.get("year", ""), source=d["url"])
        log.info("%-34s %6d words", d["id"], rows[-1]["words"])


def _as_text(value) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(v) for v in value)
    return value or ""


def _words(s: str) -> int:
    return len(WORD_RE.findall(s))


def fetch_datasets(datasets: Iterable[Dict], rows: List[Dict], refresh: bool, seed: int = 11) -> None:
    rng = random.Random(seed)
    for ds in datasets:
        if ds["format"] == "hf_rows_email":
            fetch_email_rows(ds, rows, refresh, rng)
            continue
        raw = RAW / f"{ds['id']}.jsonl"
        download(ds["url"], raw, refresh)
        human: List[str] = []
        ai: List[tuple] = []
        keywords = [k.lower() for k in ds.get("keywords", [])]
        with raw.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ds["format"] == "hc3":
                    human += [a for a in rec.get("human_answers", []) if _words(a) >= ds["min_words"]]
                    ai += [(a, "gpt-3.5-turbo") for a in rec.get("chatgpt_answers", []) if _words(a) >= ds["min_words"]]
                else:
                    h, m = _as_text(rec.get("human_text")), _as_text(rec.get("machine_text"))
                    if keywords and not any(k in (h + " " + m).lower() for k in keywords):
                        continue
                    if _words(h) >= ds["min_words"]:
                        human.append(h)
                    if _words(m) >= ds["min_words"]:
                        ai.append((m, rec.get("model", "")))
        rng.shuffle(human)
        rng.shuffle(ai)
        n = ds["max_per_label"]
        for i, text in enumerate(human[:n]):
            write_doc("human", ds["family"], f"{ds['id']}_h{i:03d}", normalise(text), rows, model="", year="", source=ds["url"])
        for i, (text, model) in enumerate(ai[:n]):
            write_doc("ai", ds["family"], f"{ds['id']}_a{i:03d}", normalise(text), rows, model=model, year="", source=ds["url"])
        log.info("%-34s human=%d ai=%d (available %d/%d)", ds["id"], min(n, len(human)), min(n, len(ai)), len(human), len(ai))


def fetch_email_rows(ds: Dict, rows: List[Dict], refresh: bool, rng: random.Random) -> None:
    """Sample substantive human business emails via the HF datasets-server rows API."""
    raw = RAW / f"{ds['id']}.jsonl"
    if refresh or not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        offsets = sorted(rng.sample(range(0, 517000, 100), 60))
        with raw.open("w", encoding="utf-8") as fh:
            for off in offsets:
                url = f"{ds['url']}&offset={off}&length=100"
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        page = json.loads(resp.read())
                except Exception as exc:  # noqa: BLE001
                    log.warning("rows fetch failed at offset %d: %s", off, exc)
                    continue
                for item in page.get("rows", []):
                    fh.write(json.dumps(item["row"]) + "\n")
                time.sleep(0.5)
    keywords = [k.lower() for k in ds.get("keywords", [])]
    kept = 0
    seen_bodies = set()
    for line in raw.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        body = strip_quoted_email(rec.get("body", ""))
        body = re.split(r"-{3,}\s*Forwarded by|\n\s*-{5,}", body)[0]
        if _words(body) < ds["min_words"] or not any(k in body.lower() for k in keywords):
            continue
        key = body[:200]
        if key in seen_bodies:
            continue
        seen_bodies.add(key)
        subject = rec.get("subject") or ""
        text = (f"# {subject}\n\n" if subject else "") + body
        write_doc("human", ds["family"], f"{ds['id']}_{kept:03d}", normalise(text), rows, model="", year="2001", source="corbt/enron-emails")
        kept += 1
        if kept >= ds["max_per_label"]:
            break
    log.info("%-34s human=%d", ds["id"], kept)


def copy_committed_ai(rows: List[Dict]) -> None:
    """Copy AI-generated MRM documents committed under benchmarks/ai_mrm/<generator>/."""
    base = ROOT / "ai_mrm"
    for path in sorted(base.glob("*/*.md")):
        generator = path.parent.name
        text = path.read_text(encoding="utf-8")
        model = generator
        m = re.match(r"<!--\s*(.*?)\s*-->\s*", text, re.S)
        if m:
            meta = dict(re.findall(r"(\w+):\s*([^\n;]+)", m.group(1)))
            model = meta.get("model", generator)
            text = text[m.end():]
        write_doc("ai", f"mrm_{generator}", f"{generator}_{path.stem}", normalise(text), rows, model=model, year="", source=f"benchmarks/ai_mrm/{generator}/{path.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download sources")
    ap.add_argument("--skip-datasets", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    spec = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    if CORPUS.exists():
        for f in CORPUS.rglob("*.txt"):
            f.unlink()
    rows: List[Dict] = []
    fetch_documents(spec["documents"], rows, args.refresh)
    if not args.skip_datasets:
        fetch_datasets(spec["datasets"], rows, args.refresh)
    copy_committed_ai(rows)

    with (CORPUS / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "label", "family", "model", "year", "words", "path", "source"])
        writer.writeheader()
        writer.writerows(rows)
    by = {}
    for r in rows:
        key = (r["label"], r["family"])
        by.setdefault(key, [0, 0])
        by[key][0] += 1
        by[key][1] += r["words"]
    for (label, family), (n, w) in sorted(by.items()):
        log.info("%-6s %-26s docs=%4d words=%8d", label, family, n, w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

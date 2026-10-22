#!/usr/bin/env python3
"""Build mixed-authorship documents: real human MRM text with AI-edited and AI-written parts.

The original synthetic AI corpus (``ai_mrm/``) is written from short prompts and
is easy to spot: generic, unspecific prose.  Real AI use in model documentation
looks different.  An author drafts, then asks an assistant to polish a paragraph,
rewrite a section, turn notes into prose, or write the conclusions from the
results.  The AI text that comes out is grounded in real facts and numbers.

This script reproduces those workflows on windows of pre-ChatGPT human documents
(``data/corpus/human``), so every word carries a known label:

==========  ===========================================================  =========
operation   what the assistant is asked to do                            label
==========  ===========================================================  =========
polish      light edit for grammar and flow, keep everything             polish
rewrite     rewrite the paragraph in its own words, keep every fact       ai
notes       paragraph -> terse notes (discarded) -> new paragraph         ai
continue    write the next paragraph from the preceding text only         ai
summary     append a "key findings" section summarising the window        ai
==========  ===========================================================  =========

"polish" is kept separate: light polishing is low AI reliance, so it is reported
on its own and not counted in the AI share.

Usage::

    pip install mlx-lm            # Apple silicon, for the local models
    python benchmarks/build_mixed.py --generators qwen7b llama8b gemma9b --per-doc 3

    # OpenAI models (your own API key, read from the environment; prints a cost estimate first)
    export OPENAI_API_KEY=...     # in your own terminal, never in a file
    python benchmarks/build_mixed.py --generators openai:gpt-4o --per-doc 2

Output: ``data/mixed/<generator>/<doc>__w<k>.json`` with the paragraph list,
labels and operations.  Claude-written parts live in ``mixed_claude/`` (committed,
built from US federal public-domain sources only) and are merged by
:func:`load_mixed`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from typing import Dict, List, Optional

from common import DATA, ROOT, load_manifest

MIXED = DATA / "mixed"
CLAUDE_DIR = ROOT / "mixed_claude"

GENERATORS = {
    "qwen7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "llama8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "gemma9b": "mlx-community/gemma-2-9b-it-4bit",
}

GENRE = {
    "regulatory_guidance": "supervisory guidance issued by a banking regulator",
    "supervisory_findings": "a supervisory or Inspector General report on model risk management",
    "validation_report": "an independent model review and validation report",
    "model_documentation": "model methodology documentation for a stress-testing model",
    "academic": "an academic paper on quantitative finance and model risk",
    "comment_letter": "a comment letter responding to a regulator's consultation",
}

PROMPTS = {
    "polish": ("You are editing {genre}. Lightly polish the paragraph below for grammar, clarity and flow. "
               "Change as little as possible and keep the meaning, every fact, number and technical term. "
               "Return only the revised paragraph.\n\n{para}"),
    "rewrite": ("Rewrite the paragraph below, taken from {genre}, so that it reads clearly and professionally. "
                "You may restructure sentences and change the wording, but keep every fact, number and technical term. "
                "Return only the rewritten paragraph.\n\n{para}"),
    "notes": ("Summarise the paragraph below as 3 to 6 terse bullet-point notes that capture its facts, numbers "
              "and claims. Return only the notes.\n\n{para}"),
    "from_notes": ("You are drafting {genre}. Here is the text so far:\n\n{context}\n\n"
                   "Write the next paragraph, about {words} words, based on these notes:\n{notes}\n\n"
                   "Return only the paragraph, with no heading."),
    "continue": ("You are drafting {genre}. Here is the text so far:\n\n{context}\n\n"
                 "Write the next paragraph, about {words} words, continuing naturally. "
                 "Return only the paragraph, with no heading."),
    "summary": ("Below is an extract from {genre}.\n\n{context}\n\n"
                "Write a 'Key findings and conclusions' section of about {words} words that summarises the main "
                "points of this extract for the reader, referring to its specific facts and numbers. "
                "Return only the section text, with no heading."),
}

# PDF extraction split many words in these ("informa tion"); AI-written parts would come out clean,
# so the detector could learn "broken words = human".  Excluded from mixing.
EXCLUDE = {"fhfa_ab2013-07", "eba_dp_ml_irb_2021"}

# AI share targets for the paragraphs of a window (summary sections come on top).
SHARES = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
AI_OPS = ["rewrite", "notes", "continue"]


# ---------------------------------------------------------------------------
# Windows of human prose
# ---------------------------------------------------------------------------
def is_prose(p: str) -> bool:
    words = p.split()
    if not 50 <= len(words) <= 320:
        return False
    if sum(ch.isdigit() for ch in p) / max(1, len(p)) > 0.06:
        return False
    if re.search(r"\.{4,}|\bet al\.|https?://|\[\d", p):  # tables of contents, references
        return False
    ends = len(re.findall(r"[a-z)][.?!](?:\s|$)", p))
    caps = sum(w[:1].isupper() for w in words) / len(words)
    # flattened tables: long runs without sentence ends, or mostly Title-Case cells
    return ends >= 2 and len(words) / ends <= 45 and caps <= 0.3 and not re.search(r"\bTable \d+\.", p)


def strip_heading(p: str) -> str:
    """Drop leading heading lines (short, no full stop) that PDF extraction glues onto paragraphs."""
    lines = p.strip().split("\n")
    while len(lines) > 1 and len(lines[0].split()) <= 14 and not lines[0].rstrip().endswith((".", ",", ";")):
        lines = lines[1:]
    return scrub(flatten("\n".join(lines)))


_DEBRIS = [
    r"\bPage \d+ GAO-[\w-]+(?: (?:[A-Z][\w’'-]*|and|of|for|the|to|in|on))*",  # GAO running headers
    r"\b\d{4}-SR-B-\d{3} \d+ of \d+",                                       # Fed OIG running headers
    r"\bEVL-\d{4}-\d{3} \d+",                                                # FHFA OIG running headers
    r"\b\d+ Dodd-Frank Act Stress Test \d{4}: Supervisory Stress Test Methodology",  # DFAST footers
    r"^\d{1,3} ?(?=[A-Z])",                                                   # leading footnote marker
    r"(?<=[a-z.,;)])\d{1,2}(?= [A-Z]|\s*$)",                                   # footnote reference digits
    r"(?<=[a-z])\d{1,2}(?=[\s,.;])",                                            # "function15 from"
]


def scrub(p: str) -> str:
    """Remove PDF debris so human paragraphs aren't distinguishable from clean AI text by artefacts alone."""
    for pat in _DEBRIS:
        p = re.sub(pat, "", p)
    return re.sub(r"\s{2,}", " ", p).strip()


def flatten(p: str) -> str:
    return re.sub(r"\s*\n\s*", " ", p).strip()


def windows(text: str, rng: random.Random, n: int, min_words=700, max_words=1500) -> List[List[str]]:
    """Consecutive windows over the document's prose paragraphs (headings, tables and lists skipped)."""
    paras = [strip_heading(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
    prose = [p for p in paras if is_prose(p)]
    cands, i = [], 0
    while i < len(prose):
        w, j = 0, i
        while j < len(prose) and w < max_words:
            w += len(prose[j].split())
            j += 1
        if w >= min_words and j - i >= 4:
            cands.append(prose[i:j])
        i = j
    rng.shuffle(cands)
    return cands[:n]


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------
class MLXGenerator:
    def __init__(self, repo: str):
        from mlx_lm import load
        self.model, self.tok = load(repo)

    def __call__(self, prompt: str, max_tokens: int = 600) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        msgs = [{"role": "user", "content": prompt}]
        chat = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tok, prompt=chat, max_tokens=max_tokens, sampler=make_sampler(temp=0.7, top_p=0.95))


class OpenAIGenerator:
    """OpenAI Chat Completions via the standard library (no extra install).

    The API key is read from the ``OPENAI_API_KEY`` environment variable and is
    never printed or written anywhere.  Token usage is tallied so the run can
    report what it spent.
    """

    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str):
        import os
        self.key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.key:
            raise SystemExit("set OPENAI_API_KEY in your shell first (it is read from the environment, never stored)")
        self.model = model
        self.prompt_tokens = self.completion_tokens = 0
        self.send_temperature = True

    def __call__(self, prompt: str, max_tokens: int = 600) -> str:
        import urllib.error
        import urllib.request
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": max_tokens * 3}
        if self.send_temperature:
            body["temperature"] = 0.7
        for attempt in range(6):
            req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    out = json.loads(resp.read())
                usage = out.get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return out["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code == 400 and "temperature" in detail and self.send_temperature:
                    self.send_temperature = False  # some reasoning models only accept the default
                    body.pop("temperature", None)
                    continue
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"OpenAI API error {exc.code}: {detail}") from None
            except urllib.error.URLError:
                time.sleep(2 ** attempt)
        raise SystemExit("OpenAI API: too many failed attempts")


def make_generator(name: str):
    """'qwen7b' (local MLX model) or 'openai:<model>' (e.g. openai:gpt-4o)."""
    if name.startswith("openai:"):
        return OpenAIGenerator(name.split(":", 1)[1])
    return MLXGenerator(GENERATORS[name])


def out_name(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


def estimate_cost(sources, per_doc: int, seed: int, name: str) -> Dict:
    """Rough token count for an API run: ~4 characters per token, prompts plus outputs."""
    calls = prompt_chars = out_chars = docs = 0
    for doc in sources:
        rng = random.Random(f"{seed}:{name}:{doc['id']}")
        for window in windows((ROOT / doc["path"]).read_text(encoding="utf-8"), rng, per_doc):
            docs += 1
            for p in window:
                calls += 1.6  # average over human / polish / rewrite / notes(2 calls) / continue
                prompt_chars += 500 + 2 * len(p)
                out_chars += len(p)
            calls += 0.35
            prompt_chars += 0.35 * sum(len(p) for p in window)
            out_chars += 0.35 * 1800
    return {"documents": docs, "calls": int(calls * 0.55), "input_tokens": int(prompt_chars * 0.55 / 4),
            "output_tokens": int(out_chars * 0.55 / 4)}


_PREAMBLE = re.compile(r"^(?:sure|certainly|here(?:'s| is| are)|below is|of course)[^\n]*:\s*\n+", re.IGNORECASE)


def clean_output(text: str) -> str:
    """What an author would paste: drop chatty preambles, markdown emphasis and stray headings."""
    text = text.strip().strip('"').strip()
    text = _PREAMBLE.sub("", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^#+\s.*\n+", "", text)
    text = re.split(r"\n\s*(?:note|i hope|let me know)[^\n]*$", text, flags=re.IGNORECASE)[0]
    paras = [flatten(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paras)


# ---------------------------------------------------------------------------
# Building one mixed document
# ---------------------------------------------------------------------------
def plan(n_paras: int, rng: random.Random) -> Dict:
    share = rng.choice(SHARES)
    k = round(share * n_paras)
    ai_idx = set(rng.sample(range(n_paras), k))
    rest = [i for i in range(n_paras) if i not in ai_idx]
    polish_idx = set(rng.sample(rest, min(len(rest), rng.choice([0, 0, 1, 2]))))
    ops = {}
    for i in range(n_paras):
        if i in ai_idx:
            ops[i] = rng.choice(AI_OPS) if i > 0 else rng.choice(["rewrite", "notes"])
        elif i in polish_idx:
            ops[i] = "polish"
        else:
            ops[i] = "human"
    return {"share": share, "ops": ops, "summary": rng.random() < 0.35}


def build_doc(gen, doc: Dict[str, str], window: List[str], rng: random.Random) -> Dict:
    genre = GENRE[doc["family"]]
    p = plan(len(window), rng)
    out = []
    for i, para in enumerate(window):
        op = p["ops"][i]
        words = len(para.split())
        context = "\n\n".join(x["text"] for x in out[-2:]) or "(start of section)"
        if op == "human":
            text, label = para, "human"
        elif op == "polish":
            text, label = clean_output(gen(PROMPTS["polish"].format(genre=genre, para=para))), "polish"
        elif op == "rewrite":
            text, label = clean_output(gen(PROMPTS["rewrite"].format(genre=genre, para=para))), "ai"
        elif op == "notes":
            notes = gen(PROMPTS["notes"].format(para=para), max_tokens=250).strip()
            text, label = clean_output(gen(PROMPTS["from_notes"].format(genre=genre, context=context, notes=notes, words=words))), "ai"
        else:
            text, label = clean_output(gen(PROMPTS["continue"].format(genre=genre, context=context, words=words))), "ai"
        if not text or len(text.split()) < 20:
            text, label, op = para, "human", "human"
        out.append({"text": text, "label": label, "op": op, "orig_words": words})
    if p["summary"]:
        ctx = "\n\n".join(window)
        text = clean_output(gen(PROMPTS["summary"].format(genre=genre, context=ctx, words=rng.choice([180, 250, 320])), max_tokens=700))
        if text:
            out.append({"text": "Key findings and conclusions", "label": "heading", "op": "heading"})
            out.extend({"text": t, "label": "ai", "op": "summary"} for t in text.split("\n\n"))
    return {"source": doc["id"], "family": doc["family"], "planned_share": p["share"], "paragraphs": out}


# ---------------------------------------------------------------------------
# Loading (for calibration / evaluation)
# ---------------------------------------------------------------------------
def render(d: Dict) -> str:
    return "\n\n".join(p["text"] for p in d["paragraphs"])


def word_labels(d: Dict) -> List[str]:
    labels: List[str] = []
    for p in d["paragraphs"]:
        labels.extend([p["label"]] * len(p["text"].split()))
    return labels


def true_share(d: Dict) -> float:
    lab = [x for x in word_labels(d) if x != "heading"]
    return sum(x == "ai" for x in lab) / max(1, len(lab))


def load_mixed(generators: Optional[List[str]] = None) -> List[Dict]:
    docs = []
    for path in sorted(list(MIXED.glob("*/*.json")) + list(CLAUDE_DIR.glob("*.json"))):
        gen = "claude" if path.parent == CLAUDE_DIR else path.parent.name
        if generators and gen not in generators:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        d.update(id=f"mix_{gen}_{path.stem}", generator=gen)
        docs.append(d)
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS),
                    help="local MLX models (qwen7b llama8b gemma9b) and/or openai:<model>, e.g. openai:gpt-4o")
    ap.add_argument("--per-doc", type=int, default=3, help="windows per human source document")
    ap.add_argument("--max-docs", type=int, default=None, help="stop after this many new documents per generator")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation for API generators")
    ap.add_argument("--price-in", type=float, default=2.5, help="USD per million input tokens (for the estimate)")
    ap.add_argument("--price-out", type=float, default=10.0, help="USD per million output tokens (for the estimate)")
    args = ap.parse_args(argv)

    sources = [d for d in load_manifest() if d["label"] == "human" and d["family"] in GENRE and d["id"] not in EXCLUDE]
    for name in args.generators:
        if name.startswith("openai:") and not args.yes:
            est = estimate_cost(sources, args.per_doc, args.seed, name)
            if args.max_docs:
                f = min(1.0, args.max_docs / max(1, est["documents"]))
                est = {k: int(v * f) for k, v in est.items()}
            cost = est["input_tokens"] / 1e6 * args.price_in + est["output_tokens"] / 1e6 * args.price_out
            print(f"{name}: about {est['documents']} documents, ~{est['calls']} API calls, ~{est['input_tokens']:,} input and "
                  f"~{est['output_tokens']:,} output tokens; roughly ${cost:.2f} at ${args.price_in}/${args.price_out} per million "
                  "(check your model's current price and pass --price-in/--price-out)")
            if input("Proceed? [y/N] ").strip().lower() != "y":
                return 1
        gen = make_generator(name)
        out_dir = MIXED / out_name(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        for doc in sources:
            text = (ROOT / doc["path"]).read_text(encoding="utf-8")
            rng = random.Random(f"{args.seed}:{name}:{doc['id']}")
            for k, window in enumerate(windows(text, rng, args.per_doc)):
                path = out_dir / f"{doc['id']}__w{k}.json"
                if path.exists() or (args.max_docs and made >= args.max_docs):
                    continue
                t0 = time.time()
                d = build_doc(gen, doc, window, rng)
                d["generator"] = name.split(":", 1)[1] if name.startswith("openai:") else GENERATORS[name]
                path.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
                made += 1
                spent = f"; tokens so far {gen.prompt_tokens:,} in / {gen.completion_tokens:,} out" if hasattr(gen, "prompt_tokens") else ""
                print(f"{name} {path.stem}: share {true_share(d):.2f} ({time.time() - t0:.0f}s{spent})", flush=True)
        del gen
    return 0


if __name__ == "__main__":
    sys.exit(main())

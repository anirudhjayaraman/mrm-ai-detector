"""Provenance hints: traces that AI tools leave in files, independent of writing style.

The style score answers "does this read like AI text?".  Modern assistants writing
from an author's own material often pass that test, but the *files* they produce
frequently carry other traces:

* Word files built by a script (``python-docx``, which ChatGPT's file tool uses),
  recognisable from their metadata;
* typography that people rarely type: non-breaking hyphens (U+2011), narrow
  no-break spaces (U+202F).  Zero-width spaces are *not* counted: Word leaves
  them around inline equations;
* leftovers of copying from a chat window: ``utm_source=chatgpt.com`` links,
  citation markers such as ``:contentReference[oaicite:3]`` or ``【4†source】``,
  "Certainly! Here's ..." openers, ``[Insert ...]`` placeholders;
* "Generated on <timestamp>" stamps.

These are reported as separate evidence and never change the probability.  None
proves AI authorship on its own (a person can type a non-breaking hyphen, and
many reporting systems use the same PDF libraries); several together are a
strong prompt for the manual check.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from xml.etree import ElementTree

STRONG, MODERATE, WEAK = "strong", "moderate", "weak"

# Characters that keyboards don't produce directly but chat assistants emit.
UNUSUAL_CHARS: Dict[str, str] = {
    "‑": "non-breaking hyphen (U+2011)",
    " ": "narrow no-break space (U+202F)",
    "​": "zero-width space (U+200B)",
    "⁠": "word joiner (U+2060)",
}
# U+2010 is what PDF text extraction and Unicode normalisation turn U+2011 into.
HYPHEN_2010 = "‐"

CHAT_LEFTOVERS = [
    (re.compile(r"utm_source=chatgpt\.com", re.I), STRONG, "links tagged utm_source=chatgpt.com (copied from ChatGPT search results)"),
    (re.compile(r":contentReference\[oaicite:\d+\]|oaicite:\d+|\bciteturn\d+\w*|turn\d+(?:search|news|view)\d+", re.I), STRONG, "ChatGPT citation markers (oaicite / turnNsearchN)"),
    (re.compile(r"【\d+(?::\d+)?†[^】]{0,40}】"), STRONG, "assistant source markers like 【4†source】"),
    (re.compile(r"\bas an ai language model\b|\bas an ai,? i\b", re.I), STRONG, "'as an AI language model'"),
    (re.compile(r"(?m)^(?:certainly|sure|absolutely|of course)[!,.]?\s+(?:here(?:'s| is| are)|below)", re.I), MODERATE, "chat-style opener ('Certainly! Here's ...')"),
    (re.compile(r"(?mi)^would you like me to (?:draft|create|expand|provide|add|write|generate|prepare|turn|convert|format)\b"), MODERATE,
     "assistant offer left in the text ('Would you like me to draft ...')"),
    (re.compile(r"\[(?:insert|add|your|placeholder)[^\]]{0,60}\]", re.I), MODERATE, "unfilled template placeholders ('[Insert ...]')"),
]
GENERATED_STAMP = re.compile(r"(?mi)^\s*generated (?:on|at)[: ]\s*\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


@dataclass
class Hint:
    strength: str  # strong / moderate / weak
    message: str

    def __str__(self) -> str:
        return f"[{self.strength}] {self.message}"


def inspect_file(path: Union[str, Path]) -> List[Hint]:
    """All provenance hints for a file (metadata plus raw-text checks)."""
    p = Path(path)
    suffix = p.suffix.lower()
    hints: List[Hint] = []
    raw: Optional[str] = None
    try:
        if suffix == ".docx":
            hints += _docx_metadata(p)
            from .ingest import _read_docx
            raw = _read_docx(p)
        elif suffix == ".pdf":
            hints += _pdf_metadata(p)
            from .ingest import _read_pdf
            raw = _read_pdf(p)
        elif suffix in (".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".eml"):
            raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - provenance is best-effort; the main read reports real errors
        pass
    if raw:
        hints += inspect_text(raw)
    return hints


def inspect_text(raw: str) -> List[Hint]:
    """Hints from raw (un-normalised) text."""
    hints: List[Hint] = []
    words = max(1, len(raw.split()))
    counts = {name: raw.count(ch) for ch, name in UNUSUAL_CHARS.items()}
    counts["hyphen (U+2010), typically a converted non-breaking hyphen"] = raw.count(HYPHEN_2010)
    found = {k: v for k, v in counts.items() if v}
    total = sum(found.values())
    if total >= 5 and total * 10_000 / words >= 10:
        detail = ", ".join(f"{v} × {k}" for k, v in sorted(found.items(), key=lambda kv: -kv[1]))
        strength = MODERATE if total * 10_000 / words >= 25 else WEAK
        hints.append(Hint(strength, f"characters that assistants emit but keyboards rarely produce: {detail}"))
    for pattern, strength, label in CHAT_LEFTOVERS:
        n = len(pattern.findall(raw))
        if n:
            hints.append(Hint(strength, f"{label}: {n} occurrence{'s' if n > 1 else ''}"))
    stamps = GENERATED_STAMP.findall(raw)
    if stamps:
        hints.append(Hint(WEAK, f"'Generated on <timestamp>' stamp left in the text ({stamps[0].strip()})"))
    return hints


def unusual_char_segments(segment_texts: List[str]) -> List[int]:
    """Indices of segments containing the characters above (after normalisation only U+2010 survives)."""
    marks = tuple(UNUSUAL_CHARS) + (HYPHEN_2010,)
    return [i for i, t in enumerate(segment_texts) if any(m in t for m in marks)]


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------
_CP = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"
_DC = "{http://purl.org/dc/elements/1.1/}"
_DCT = "{http://purl.org/dc/terms/}"
_APP = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"


def _docx_metadata(path: Path) -> List[Hint]:
    hints: List[Hint] = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        core = ElementTree.fromstring(zf.read("docProps/core.xml")) if "docProps/core.xml" in names else None
        app = ElementTree.fromstring(zf.read("docProps/app.xml")) if "docProps/app.xml" in names else None
    if core is not None:
        creator = (core.findtext(f"{_DC}creator") or "").strip()
        desc = (core.findtext(f"{_DC}description") or "").strip()
        created = (core.findtext(f"{_DCT}created") or "").strip()
        if "python-docx" in (creator + " " + desc).lower():
            hints.append(Hint(STRONG, f"Word file created by a script (python-docx; creator '{creator}', description '{desc}'). "
                                      "ChatGPT's file tool builds Word documents this way"))
        if created.startswith("2013-12-23T23:15"):
            hints.append(Hint(MODERATE, "creation date 2013-12-23 23:15 is the python-docx default template date, "
                                        "so the file was generated from that template"))
    if app is not None and not (app.findtext(f"{_APP}Application") or "").strip():
        hints.append(Hint(WEAK, "Word file has no authoring application recorded (typical of generated files)"))
    return hints


def _pdf_metadata(path: Path) -> List[Hint]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    meta = PdfReader(str(path)).metadata or {}
    producer = " ".join(str(meta.get(k, "")) for k in ("/Producer", "/Creator")).strip()
    low = producer.lower()
    if any(x in low for x in ("reportlab", "matplotlib", "python-docx", "weasyprint", "fpdf")):
        return [Hint(WEAK, f"PDF produced programmatically ({producer}); ChatGPT's file tool uses such libraries, "
                           "but so do many reporting systems")]
    return []

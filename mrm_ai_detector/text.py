"""Text normalisation, tokenisation and segmentation.

Model risk documents (validation reports, model development documents,
monitoring packs, policy papers) mix prose with headings, bullet lists,
tables and numbered findings.  The helpers here turn raw extracted text into
a :class:`TextUnit` — the structure every feature extractor consumes — and
split long documents into :class:`Segment` chunks for localised scoring.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:\s?(?:%|bps|bp|x|pp)\b|%)?"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•▪●–]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+){0,4}\.?|[A-Z]\.|[IVX]+\.)\s+[A-Z]")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*[-|:\s]{5,}$")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|`)")
_SENT_BOUNDARY_RE = re.compile(r"([.!?]+)([\"'”’)\]]*)\s+(?=[\"'“‘(\[]?[A-Z0-9])")

# Tokens that end in a period without ending a sentence.
ABBREVIATIONS = frozenset(
    """
    e.g i.e etc vs cf al approx fig figs sec secs no nos art arts para paras
    p pp vol ch eq eqs ref refs dr mr mrs ms prof inc ltd co corp plc llc st
    jan feb mar apr jun jul aug sep sept oct nov dec est min max avg incl excl
    u.s u.k e.u n.b resp
    """.split()
)


@dataclass
class TextUnit:
    """Analysable text: prose sentences, paragraphs and raw lines."""

    sentences: List[str]
    paragraphs: List[str]
    lines: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def words(self) -> List[str]:
        return WORD_RE.findall(self.text)

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class Segment:
    """A contiguous chunk of a document scored on its own."""

    index: int
    section: Optional[str]
    unit: TextUnit

    @property
    def text(self) -> str:
        return "\n\n".join(self.unit.paragraphs)


def normalise(raw: str) -> str:
    """Unicode-normalise and tidy whitespace while preserving line structure."""
    text = unicodedata.normalize("NFKC", raw)
    # NFKC folds the em dash into itself but turns NBSP etc. into spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    # Re-join words hyphenated across a line break by PDF extraction.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_markdown(line: str) -> str:
    line = _MD_HEADING_RE.sub("", line)
    line = _MD_EMPHASIS_RE.sub("", line)
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line).strip()


def is_heading(line: str) -> bool:
    """Heuristic heading detector for markdown, numbered and title-case headings."""
    stripped = line.strip()
    if not stripped or _BULLET_RE.match(stripped) and not _NUMBERED_HEADING_RE.match(stripped):
        return False
    if _MD_HEADING_RE.match(stripped):
        return True
    plain = strip_markdown(stripped)
    words = plain.split()
    if not words or len(words) > 12 or plain.endswith((".", ",", ";", ":")) and not plain.endswith(":"):
        return False
    if plain.endswith(":") and len(words) <= 6:
        return True
    if _NUMBERED_HEADING_RE.match(plain) and not re.search(r"[.!?]$", plain):
        return True
    if plain.isupper() and len(plain) > 3:
        return True
    letters = [w for w in words if w[0].isalpha()]
    if len(words) <= 8 and letters and not re.search(r"[.!?]$", plain):
        capitalised = sum(1 for w in letters if w[0].isupper() or w.lower() in _MINOR_WORDS)
        return capitalised / len(letters) >= 0.8 and len(letters) >= 2
    return False


_MINOR_WORDS = frozenset("a an and as at by for in of on or the to vs with".split())


def split_sentences(paragraph: str) -> List[str]:
    """Split a prose paragraph into sentences, respecting abbreviations and decimals."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sentences: List[str] = []
    start = 0
    for match in _SENT_BOUNDARY_RE.finditer(paragraph):
        end = match.end(2)
        candidate = paragraph[start:end]
        if match.group(1) == ".":
            prev = re.search(r"([A-Za-z.]+)\.$", candidate[:-len(match.group(2))] if match.group(2) else candidate)
            token = prev.group(1).lower() if prev else ""
            if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if WORD_RE.search(s)]


def build_unit(text: str) -> TextUnit:
    """Parse normalised text into a :class:`TextUnit`."""
    lines = [ln for ln in text.split("\n")]
    paragraphs: List[str] = []
    headings: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer).strip())
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _TABLE_ROW_RE.match(stripped):
            flush()
            # Keep prose cells (findings tables, issue write-ups); drop numeric/label cells.
            for cell in stripped.strip("|").split("|"):
                cell = strip_markdown(cell.strip())
                if len(WORD_RE.findall(cell)) >= 6:
                    paragraphs.append(cell)
            continue
        if is_heading(stripped):
            flush()
            headings.append(strip_markdown(stripped).rstrip(":"))
            continue
        if _BULLET_RE.match(stripped):
            # Each bullet is its own paragraph-like item.
            flush()
            paragraphs.append(strip_markdown(_BULLET_RE.sub("", stripped, count=1)))
            continue
        buffer.append(strip_markdown(stripped))
    flush()

    paragraphs = [p for p in paragraphs if WORD_RE.search(p)]
    sentences = [s for p in paragraphs for s in split_sentences(p)]
    return TextUnit(sentences=sentences, paragraphs=paragraphs, lines=[ln for ln in lines if ln.strip()], headings=headings)


def segment_document(text: str, target_words: int = 180, min_words: int = 60) -> List[Segment]:
    """Split a document into chunks of roughly ``target_words`` words.

    Headings are *soft* boundaries: a new chunk starts at a heading only once
    the current chunk holds at least half the target length, so heading-dense
    documents (typical of validation reports) still yield chunks large enough
    to score.  A trailing chunk shorter than ``min_words`` is merged backwards.
    """
    # 1. Split into (section, block-lines) units at blank lines and headings.
    blocks: List[tuple] = []
    section: Optional[str] = None
    for block in re.split(r"\n\s*\n", text):
        buf: List[str] = []
        for line in (ln for ln in block.split("\n") if ln.strip()):
            stripped = line.strip()
            if is_heading(stripped) and not _TABLE_ROW_RE.match(stripped):
                if buf:
                    blocks.append((section, buf, False))
                    buf = []
                section = strip_markdown(stripped).rstrip(":")
                blocks.append((section, [], True))
                continue
            buf.append(line)
        if buf:
            blocks.append((section, buf, False))

    # 2. Greedily pack blocks into chunks.
    chunks: List[tuple] = []  # (section, lines)
    cur: List[str] = []
    cur_section: Optional[str] = None
    cur_words = 0
    soft = max(min_words, target_words // 2)
    for sec, lines, is_head in blocks:
        if is_head:
            if cur_words >= soft:
                chunks.append((cur_section, cur))
                cur, cur_words = [], 0
            continue
        if cur_words >= target_words:
            chunks.append((cur_section, cur))
            cur, cur_words = [], 0
        if not cur:
            cur_section = sec
        cur.extend(lines + [""])
        cur_words += sum(len(WORD_RE.findall(ln)) for ln in lines)
    if cur:
        chunks.append((cur_section, cur))

    segments: List[Segment] = []
    for sec, lines in chunks:
        unit = build_unit("\n".join(lines))
        if unit.word_count == 0:
            continue
        if segments and unit.word_count < min_words:
            prev = segments[-1]
            segments[-1] = Segment(prev.index, prev.section, build_unit("\n\n".join(prev.unit.paragraphs + unit.paragraphs)))
        else:
            segments.append(Segment(len(segments), sec, unit))
    if len(segments) > 1 and segments[0].unit.word_count < min_words:
        first, second = segments[0], segments[1]
        segments = [Segment(0, first.section, build_unit("\n\n".join(first.unit.paragraphs + second.unit.paragraphs)))] + segments[2:]
    return [Segment(i, s.section, s.unit) for i, s in enumerate(segments)]


def resample_sentences(unit: TextUnit, indices: Sequence[int]) -> TextUnit:
    """Return a copy of ``unit`` whose sentences are replaced by ``indices`` (bootstrap)."""
    return TextUnit(
        sentences=[unit.sentences[i] for i in indices],
        paragraphs=unit.paragraphs,
        lines=unit.lines,
        headings=unit.headings,
    )

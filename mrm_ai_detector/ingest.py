"""Document loading.

Supported inputs: ``.txt .md .rst .docx .pdf .html .eml .msg`` and images
(``.png .jpg .jpeg .tif .tiff .heic``).  ``.docx``, ``.html`` and ``.eml`` are
parsed with the standard library.  Optional extras: ``pypdf`` for PDF,
``olefile`` for Outlook ``.msg``, ``rapidocr-onnxruntime`` (+ ``pillow-heif``
for HEIC photos) for OCR of screenshots and photos.

Emails are reduced to the newest message: quoted replies and forwarded
history are stripped so that only the author's own text is scored.
"""

from __future__ import annotations

import logging
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator, List, Union
from xml.etree import ElementTree

from .text import normalise

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heic", ".heif")
SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown", ".rst", ".docx", ".pdf", ".html", ".htm", ".eml", ".msg") + IMAGE_SUFFIXES
OCR_MARKER = "<!-- mrm-detector: text extracted by OCR -->"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class IngestError(RuntimeError):
    """Raised when a document cannot be read."""


def read_document(path: Union[str, Path]) -> str:
    """Return normalised plain text (markdown-flavoured headings preserved)."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"not a file: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix == ".docx":
            raw = _read_docx(p)
        elif suffix == ".pdf":
            raw = _read_pdf(p)
        elif suffix in (".html", ".htm"):
            raw = _read_html(p.read_text(encoding="utf-8", errors="replace"))
        elif suffix == ".eml":
            raw = strip_quoted_email(_read_eml(p))
        elif suffix == ".msg":
            raw = strip_quoted_email(_read_msg(p))
        elif suffix in IMAGE_SUFFIXES:
            raw = OCR_MARKER + "\n" + _read_image(p)
        else:
            raw = p.read_text(encoding="utf-8", errors="replace")
    except IngestError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any parser failure uniformly
        raise IngestError(f"failed to read {p}: {exc}") from exc
    return normalise(raw)


def iter_documents(paths: Iterable[Union[str, Path]], recursive: bool = True) -> Iterator[Path]:
    """Expand files and directories into supported document paths (sorted, de-duplicated)."""
    seen = set()
    for item in paths:
        p = Path(item)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates: List[Path] = sorted(c for c in p.glob(pattern) if c.is_file() and c.suffix.lower() in SUPPORTED_SUFFIXES)
        elif p.is_file():
            candidates = [p]
        else:
            raise IngestError(f"path not found: {p}")
        for c in candidates:
            key = c.resolve()
            if key not in seen and not c.name.startswith((".", "~$")):
                seen.add(key)
                yield c


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    out: List[str] = []
    for para in root.iter(f"{_W}p"):
        style = para.find(f"{_W}pPr/{_W}pStyle")
        style_name = style.get(f"{_W}val", "") if style is not None else ""
        parts: List[str] = []
        for node in para.iter():
            if node.tag == f"{_W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W}tab":
                parts.append(" ")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
        text = "".join(parts).strip()
        if not text:
            out.append("")
            continue
        m = re.match(r"(?i)heading\s?(\d)|title", style_name)
        if m:
            level = int(m.group(1)) if m.group(1) else 1
            out.append("")
            out.append("#" * min(level, 6) + " " + text)
            out.append("")
        elif re.match(r"(?i)list", style_name):
            out.append("- " + text)
        else:
            out.append(text)
            out.append("")
    return "\n".join(out)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise IngestError("PDF support needs pypdf: pip install 'mrm-ai-detector[pdf]'") from exc
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"encrypted PDF: {path}") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(p.strip() for p in pages):
        raise IngestError(f"no extractable text in {path} (scanned PDF? run OCR first)")
    return _strip_running_headers(pages)


def _strip_running_headers(pages: List[str]) -> str:
    """Drop lines repeated on most pages (running headers/footers) and bare page numbers."""
    from collections import Counter

    counts: Counter = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            counts[line] += 1
    threshold = max(3, int(len(pages) * 0.5))
    repeated = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 120}
    kept: List[str] = []
    for page in pages:
        for line in page.splitlines():
            s = line.strip()
            if s in repeated or re.fullmatch(r"(?:Page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?", s, re.IGNORECASE):
                continue
            kept.append(line)
        kept.append("")
    return _reflow_pdf_lines(kept)


def _reflow_pdf_lines(lines: List[str]) -> str:
    """PDF extraction breaks every visual line; re-join lines that continue a sentence."""
    out: List[str] = []
    for line in lines:
        s = line.rstrip()
        if out and s and out[-1] and not re.search(r"[.!?:;]\s*$", out[-1]) and s[:1].islower():
            out[-1] = out[-1] + " " + s.strip()
        else:
            out.append(s)
    return "\n".join(out)


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}
    SKIP = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")
            if tag.startswith("h") and len(tag) == 2:
                self.parts.append("#" * int(tag[1]) + " ")
            elif tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):  # noqa: ANN001
        if not self._skip:
            self.parts.append(data)


def _read_html(markup: str) -> str:
    parser = _HTMLText()
    parser.feed(markup)
    text = "".join(parser.parts)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", text)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
_QUOTE_HEADER_RE = re.compile(
    r"^(?:-{2,}\s*(?:Original Message|Forwarded message)\s*-{2,}"
    r"|On .{5,120}wrote:\s*$"
    r"|\s*From:\s.+\n(?:.*\n){0,3}?\s*(?:Sent|Date|To):\s.*"
    r"|\s*From:\s.+\bon\s\d{1,2}/\d{1,2}/\d{2,4}.*)",
    re.MULTILINE | re.IGNORECASE,
)


def strip_quoted_email(text: str) -> str:
    """Keep only the newest message of an email thread."""
    m = _QUOTE_HEADER_RE.search(text)
    if m and m.start() > 40:
        text = text[: m.start()]
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(lines)


def _read_eml(path: Path) -> str:
    import email
    from email import policy

    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        raise IngestError(f"no text body in {path}")
    content = body.get_content()
    if body.get_content_type() == "text/html":
        content = _read_html(content)
    subject = msg.get("subject", "")
    return (f"# {subject}\n\n" if subject else "") + content


def _read_msg(path: Path) -> str:
    try:
        import olefile
    except ImportError as exc:
        raise IngestError("Outlook .msg support needs olefile: pip install 'mrm-ai-detector[email]'") from exc
    with olefile.OleFileIO(str(path)) as ole:

        def stream(name: str):
            return ole.openstream(name).read() if ole.exists(name) else None

        subject = stream("__substg1.0_0037001F")
        body = stream("__substg1.0_1000001F")
        if body is not None:
            text = body.decode("utf-16-le", errors="replace")
        else:
            ansi = stream("__substg1.0_1000001E")
            html = stream("__substg1.0_10130102")
            if ansi is not None:
                text = ansi.decode("cp1252", errors="replace")
            elif html is not None:
                text = _read_html(html.decode("utf-8", errors="replace"))
            else:
                raise IngestError(f"no message body found in {path}")
    title = subject.decode("utf-16-le", errors="replace").strip("\x00") if subject else ""
    return (f"# {title}\n\n" if title else "") + text.replace("\x00", "")


# ---------------------------------------------------------------------------
# Images (OCR)
# ---------------------------------------------------------------------------
_OCR_ENGINE = None

# macOS ships Apple's Vision text recogniser (the Live Text engine). It is reached
# through the built-in JavaScript-for-Automation bridge, so OCR of photos, HEIC
# included, works on a Mac with no extra installs.
_VISION_JXA = r"""
ObjC.import('Vision');
ObjC.import('Foundation');
function run(argv) {
  const url = $.NSURL.fileURLWithPath(argv[0]);
  const handler = $.VNImageRequestHandler.alloc.initWithURLOptions(url, $.NSDictionary.dictionary);
  const req = $.VNRecognizeTextRequest.alloc.init;
  req.recognitionLevel = 0;            // accurate
  req.usesLanguageCorrection = true;
  if (!handler.performRequestsError($.NSArray.arrayWithObject(req), null)) { return "ERROR"; }
  const res = req.results;
  const out = [];
  for (let i = 0; i < res.count; i++) {
    const obs = res.objectAtIndex(i);
    const cand = obs.topCandidates(1).objectAtIndex(0);
    const bb = obs.boundingBox;
    out.push(JSON.stringify({t: cand.string.js, c: cand.confidence, x: bb.origin.x, y: bb.origin.y, w: bb.size.width, h: bb.size.height}));
  }
  return out.join("\n");
}
"""


def _macos_vision_available() -> bool:
    import shutil
    import sys

    return sys.platform == "darwin" and shutil.which("osascript") is not None


def _read_image_macos(path: Path) -> str:
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(_VISION_JXA)
        script = fh.name
    try:
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", script, str(path.resolve())],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        Path(script).unlink(missing_ok=True)
    out = proc.stdout.strip()
    if proc.returncode != 0 or out == "ERROR":
        raise IngestError(f"macOS text recognition failed for {path}: {proc.stderr.strip()[:200]}")
    boxes = []
    for line in out.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Vision uses normalised coordinates with the origin at the bottom-left.
        y_center = 1.0 - (r["y"] + r["h"] / 2)
        boxes.append((r["x"] * 1000, y_center * 1000, r["h"] * 1000, r["t"], float(r["c"])))
    if not boxes:
        raise IngestError(f"no text found in {path}")
    return _layout_ocr_boxes(boxes)


def _read_image(path: Path) -> str:
    """OCR an image and rebuild reading order, lines and paragraphs from the text boxes.

    Uses Apple's built-in Vision OCR on macOS (no installs), otherwise RapidOCR
    (``pip install 'mrm-ai-detector[ocr]'``).
    """
    if _macos_vision_available():
        try:
            return _read_image_macos(path)
        except (IngestError, OSError, ValueError) as exc:
            logger.warning("macOS OCR unavailable (%s); trying RapidOCR", exc)
    return _read_image_rapidocr(path)


def _read_image_rapidocr(path: Path) -> str:
    global _OCR_ENGINE
    try:
        import numpy as np
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise IngestError(
            "reading images needs OCR. On macOS this is built in; elsewhere run: pip install 'mrm-ai-detector[ocr]'"
        ) from exc
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError as exc:
            raise IngestError("HEIC photos need pillow-heif: pip install 'mrm-ai-detector[ocr]'") from exc
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    image = Image.open(path).convert("RGB")
    result, _ = _OCR_ENGINE(np.array(image))
    if not result:
        raise IngestError(f"OCR found no text in {path}")
    boxes = []
    for quad, text, conf in result:
        ys = [pt[1] for pt in quad]
        xs = [pt[0] for pt in quad]
        boxes.append((min(xs), (min(ys) + max(ys)) / 2, max(ys) - min(ys), str(text), float(conf)))
    return _layout_ocr_boxes(boxes)


def _layout_ocr_boxes(boxes: List[tuple]) -> str:
    """Group OCR boxes (x, y_center, height, text, conf) into lines and paragraphs."""
    if not boxes:
        return ""
    heights = sorted(b[2] for b in boxes)
    h = heights[len(heights) // 2] or 1.0
    rows: List[List[tuple]] = []
    for box in sorted(boxes, key=lambda b: b[1]):
        if rows and abs(rows[-1][0][1] - box[1]) < 0.5 * h:
            rows[-1].append(box)
        else:
            rows.append([box])
    lines: List[tuple] = []  # (y, text)
    for row in rows:
        row.sort(key=lambda b: b[0])
        lines.append((row[0][1], " ".join(b[3].strip() for b in row)))
    out: List[str] = []
    prev_y = None
    for y, text in lines:
        if out and prev_y is not None and y - prev_y < 1.7 * h:
            last = out[-1]
            if last.endswith("-") and text[:1].islower():
                out[-1] = last[:-1] + text
                prev_y = y
                continue
            if not re.search(r"[.:;!?]$", last) and (text[:1].islower() or text[:1].isdigit() is False and len(last) > 60):
                out[-1] = last + " " + text
                prev_y = y
                continue
            out.append(text)
        else:
            if out:
                out.append("")
            out.append(text)
        prev_y = y
    return "\n".join(out)

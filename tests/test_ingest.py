import zipfile

import pytest

from mrm_ai_detector.ingest import IngestError, iter_documents, read_document

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Executive Summary</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">The model was </w:t></w:r><w:r><w:t>validated.</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="ListParagraph"/></w:pPr><w:r><w:t>First finding</w:t></w:r></w:p>
</w:body></w:document>"""


def test_docx_is_parsed_with_stdlib(tmp_path):
    path = tmp_path / "report.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", DOCX_XML)
    text = read_document(path)
    assert "# Executive Summary" in text
    assert "The model was validated." in text
    assert "- First finding" in text


def test_html_strips_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><script>var x=1;</script><h2>Title</h2><p>Body text here.</p></html>")
    text = read_document(path)
    assert "var x" not in text and "## Title" in text and "Body text here." in text


def test_iter_documents_filters_and_dedupes(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    (tmp_path / "~$lock.docx").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("x")
    found = list(iter_documents([tmp_path, tmp_path / "a.txt"]))
    assert sorted(p.name for p in found) == ["a.txt", "c.md"]


def test_missing_path_raises(tmp_path):
    with pytest.raises(IngestError):
        list(iter_documents([tmp_path / "nope"]))
    with pytest.raises(IngestError):
        read_document(tmp_path / "nope.txt")


def test_corrupt_docx_raises(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_text("not a zip")
    with pytest.raises(IngestError):
        read_document(path)


EML = """From: A Validator <a@example.com>
To: Model Owner <b@example.com>
Subject: Validation findings for PD model
Content-Type: text/plain; charset=utf-8

Hi all, the validation of the PD model is complete. Two findings were raised.

On Mon, 1 Sep 2025 at 09:00, Model Owner <b@example.com> wrote:
> Can you send the findings?
> Thanks
"""


def test_eml_keeps_only_newest_message(tmp_path):
    path = tmp_path / "m.eml"
    path.write_text(EML)
    text = read_document(path)
    assert text.startswith("# Validation findings for PD model")
    assert "Two findings were raised." in text
    assert "Can you send" not in text and "wrote:" not in text


def test_strip_quoted_outlook_thread():
    from mrm_ai_detector.ingest import strip_quoted_email

    body = "Please see the issue below and confirm the owner.\n\nFrom: Someone Else\nSent: Monday\nTo: Team\nSubject: RE: issue\n\nOld text"
    assert strip_quoted_email(body).strip() == "Please see the issue below and confirm the owner."


def test_ocr_layout_groups_rows_and_paragraphs():
    from mrm_ai_detector.ingest import _layout_ocr_boxes

    boxes = [
        (10, 10, 10, "Scope", 0.9), (200, 11, 10, "The back-test covers the trading book and the bank-", 0.9),
        (200, 24, 10, "ing book positions.", 0.9), (10, 80, 10, "Next paragraph.", 0.9),
    ]
    out = _layout_ocr_boxes(boxes)
    assert out.splitlines()[0] == "Scope The back-test covers the trading book and the banking book positions."
    assert out.splitlines()[-1] == "Next paragraph."


def test_macos_vision_output_is_laid_out(tmp_path, monkeypatch):
    import json
    import subprocess

    from mrm_ai_detector import ingest

    lines = [  # Vision: normalised boxes, origin bottom-left
        {"t": "Scope", "c": 1, "x": 0.05, "y": 0.80, "w": 0.1, "h": 0.02},
        {"t": "The back-test covers the trading book and the bank-", "c": 1, "x": 0.15, "y": 0.80, "w": 0.5, "h": 0.02},
        {"t": "ing book positions.", "c": 1, "x": 0.15, "y": 0.775, "w": 0.3, "h": 0.02},
    ]

    class Done:
        returncode = 0
        stdout = "\n".join(json.dumps(x) for x in lines)
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    img = tmp_path / "photo.heic"
    img.write_bytes(b"not really an image")
    text = ingest._read_image_macos(img)
    assert text.splitlines()[0] == "Scope The back-test covers the trading book and the banking book positions."

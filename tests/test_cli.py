import json

from mrm_ai_detector.cli import EXIT_ERROR, EXIT_FLAGGED, EXIT_OK, main


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_scan_json(tmp_path, capsys, human_text, ai_text):
    write(tmp_path, "h.txt", human_text)
    write(tmp_path, "a.md", ai_text)
    assert main(["scan", str(tmp_path), "-f", "json", "--bootstrap", "0"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    probs = {d["source"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]: d["probability"] for d in payload["documents"]}
    assert probs["a.md"] > probs["h.txt"]


def test_fail_on_ai_exit_code(tmp_path, ai_text, human_text):
    a = write(tmp_path, "a.md", ai_text)
    h = write(tmp_path, "h.txt", human_text)
    assert main(["scan", str(a), "--fail-on-ai", "--bootstrap", "0", "-o", str(tmp_path / "r.txt")]) == EXIT_FLAGGED
    assert main(["scan", str(h), "--fail-on-ai", "--bootstrap", "0", "-o", str(tmp_path / "r.txt")]) == EXIT_OK


def test_html_and_markdown_reports(tmp_path, ai_text):
    a = write(tmp_path, "a.md", ai_text)
    out = tmp_path / "report.html"
    assert main(["scan", str(a), "-f", "html", "-o", str(out), "--bootstrap", "0"]) == EXIT_OK
    html = out.read_text()
    assert html.startswith("<!doctype html>") and "AI probability" in html
    md = tmp_path / "r.md"
    assert main(["scan", str(a), "-f", "markdown", "-o", str(md), "--bootstrap", "0"]) == EXIT_OK
    assert "| Document |" in md.read_text()


def test_features_command(tmp_path, capsys, ai_text):
    a = write(tmp_path, "a.md", ai_text)
    assert main(["features", str(a)]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["features"]["ai_marker_rate"]["value"] > 0


def test_missing_file_is_an_error(tmp_path):
    assert main(["scan", str(tmp_path / "missing.txt")]) == EXIT_ERROR


def test_train_and_use_custom_model(tmp_path, human_text, ai_text):
    hd, ad = tmp_path / "human", tmp_path / "ai"
    hd.mkdir()
    ad.mkdir()
    for i in range(6):
        (hd / f"h{i}.txt").write_text(human_text.replace("2015", str(2000 + i)))
        (ad / f"a{i}.txt").write_text(ai_text + f"\n\nVersion {i}.")
    model = tmp_path / "model.json"
    assert main(["train", "--human", str(hd), "--ai", str(ad), "-o", str(model), "--folds", "3"]) == EXIT_OK
    assert main(["scan", str(ad / "a0.txt"), "--model", str(model), "--bootstrap", "0", "-o", str(tmp_path / "x")]) == EXIT_OK


def test_summary_for_single_document(tmp_path, capsys, ai_text):
    a = write(tmp_path, "a.md", ai_text)
    assert main(["scan", str(a), "--summary", "--bootstrap", "0"]) == EXIT_OK
    assert "a.md" in capsys.readouterr().out

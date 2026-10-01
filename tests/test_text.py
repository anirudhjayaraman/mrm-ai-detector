from mrm_ai_detector.text import build_unit, is_heading, normalise, segment_document, split_sentences


def test_sentence_split_respects_abbreviations_and_decimals():
    s = split_sentences("The AUC was 0.71, e.g. on the OOT sample. See Fig. 3 for details. It passed.")
    assert s == ["The AUC was 0.71, e.g. on the OOT sample.", "See Fig. 3 for details.", "It passed."]


def test_sentence_split_handles_quotes_and_numbers():
    s = split_sentences('He said "it failed." 2023 was worse. Then it recovered.')
    assert len(s) == 3


def test_heading_detection():
    assert is_heading("## 3. Conceptual Soundness")
    assert is_heading("4.2 Data Quality Assessment")
    assert is_heading("EXECUTIVE SUMMARY")
    assert is_heading("Key Findings:")
    assert not is_heading("The model was validated in 2023 and found to be sound.")
    assert not is_heading("- a bullet about the model")


def test_build_unit_strips_markdown_tables_and_headings():
    unit = build_unit(normalise("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nSome **bold** prose here. Another sentence.\n\n- bullet one\n- bullet two"))
    assert unit.headings == ["Title"]
    assert "bold prose" in unit.text and "**" not in unit.text
    assert unit.paragraphs[-2:] == ["bullet one", "bullet two"]
    assert not any("|" in p for p in unit.paragraphs)


def test_segmentation_packs_heading_dense_documents():
    sections = "\n\n".join(f"## {i}. Section {i}\n\n" + "This sentence has exactly eight words in it. " * 6 for i in range(8))
    segs = segment_document(sections, target_words=180, min_words=60)
    assert len(segs) >= 2
    assert all(s.unit.word_count >= 60 for s in segs)
    assert sum(s.unit.word_count for s in segs) == build_unit(sections).word_count


def test_normalise_rejoins_hyphenated_line_breaks():
    assert normalise("model vali-\ndation") == "model validation"


def test_prose_in_table_cells_is_kept():
    md = "| Item | Detail |\n|---|---|\n| Observation | The challenger model was not rerun after the 2023 data refresh. |\n| Rating | High |"
    unit = build_unit(md)
    assert unit.paragraphs == ["The challenger model was not rerun after the 2023 data refresh."]

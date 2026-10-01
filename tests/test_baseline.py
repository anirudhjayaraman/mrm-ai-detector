import pytest

from mrm_ai_detector.baseline import compare_to_baseline


def make_refs(tmp_path, human_text, n=5):
    for i in range(n):
        (tmp_path / f"ref{i}.txt").write_text(human_text.replace("2015", str(2010 + i)).replace("0.61", f"0.6{i}"))
    return [tmp_path]


def test_ai_text_is_unusual_and_ai_ward_for_human_author(tmp_path, human_text, ai_text):
    refs = make_refs(tmp_path, human_text)
    same = compare_to_baseline(refs, human_text.replace("2015", "2020"), "same")
    other = compare_to_baseline(refs, ai_text, "ai")
    assert other.style_distance > same.style_distance
    assert other.ai_direction_shift > 0
    assert other.p_value <= same.p_value


def test_needs_enough_reference_documents(tmp_path, human_text):
    with pytest.raises(ValueError):
        compare_to_baseline(make_refs(tmp_path, human_text, n=2), human_text)

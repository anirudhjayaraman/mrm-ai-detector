import math

from mrm_ai_detector.features import REGISTRY, extract
from mrm_ai_detector.text import build_unit


def test_all_features_return_float_or_none(human_text, ai_text):
    for text in (human_text, ai_text, "Too short."):
        values = extract(build_unit(text))
        assert set(values) == set(REGISTRY)
        for name, v in values.items():
            assert v is None or (isinstance(v, float) and math.isfinite(v)), name


def test_marker_and_domain_features_separate_examples(human_text, ai_text):
    h = extract(build_unit(human_text))
    a = extract(build_unit(ai_text))
    assert a["ai_marker_rate"] > 5 * max(h["ai_marker_rate"], 1)
    assert a["transition_open_rate"] > h["transition_open_rate"]
    assert h["numeric_density"] > a["numeric_density"]
    assert h["specific_reference_rate"] > a["specific_reference_rate"]
    assert a["unsupported_claim_ratio"] > h["unsupported_claim_ratio"]
    assert a["boilerplate_rate"] > h["boilerplate_rate"]


def test_quantified_metric_detection():
    unit = build_unit("The Gini was 0.61 on the OOT sample. PSI = 0.08. The KS statistic of 41% passed. Nothing here.")
    assert extract(unit)["quantified_metric_rate"] > 0


def test_short_text_features_are_none():
    v = extract(build_unit("One sentence only."))
    assert v["sent_len_cv"] is None and v["mattr"] is None


def test_chunk_words_merges_short_tail():
    from mrm_ai_detector.features.neural import _chunk_words

    text = " ".join(f"w{i}" for i in range(361))
    chunks = _chunk_words(text, 300)
    assert len(chunks) == 1 and len(chunks[0].split()) == 361
    assert [len(c.split()) for c in _chunk_words(" ".join(["w"] * 500), 300)] == [300, 200]

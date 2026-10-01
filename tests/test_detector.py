import json

from mrm_ai_detector import Detector
from mrm_ai_detector.detector import INSUFFICIENT, LIKELY_HUMAN


def test_ai_text_scores_higher_than_human(human_text, ai_text):
    det = Detector(bootstrap=50)
    h = det.analyze_text(human_text)
    a = det.analyze_text(ai_text)
    assert a.probability > h.probability + 0.3
    assert h.verdict == LIKELY_HUMAN
    assert a.flagged


def test_short_text_is_insufficient():
    r = Detector().analyze_text("The model is robust. It is comprehensive.")
    assert r.verdict == INSUFFICIENT
    assert r.warnings


def test_bootstrap_interval_is_ordered(human_text):
    r = Detector(bootstrap=60).analyze_text(human_text * 2)
    assert r.ci_low is not None and 0 <= r.ci_low <= r.ci_high <= 1


def test_result_is_json_serialisable(ai_text):
    r = Detector(bootstrap=0).analyze_text(ai_text, source="x.md")
    d = r.to_dict(include_text=False)
    json.dumps(d)
    assert d["source"] == "x.md" and "text" not in d["segments"][0]
    assert r.signals and r.signals[0].description


def test_mixed_document_flags_ai_segments(human_text, ai_text):
    doc = "\n\n".join([human_text, human_text.replace("MDL-2231", "MDL-2232"), ai_text, human_text.replace("2015", "2016")])
    r = Detector(segment_words=150, bootstrap=0).analyze_text(doc)
    flagged = [s for s in r.segments if s.flagged]
    assert flagged, "expected the AI-written section to be flagged"
    assert all("Overall" in s.text or "Additionally" in s.text or "important to note" in s.text or "conclusion" in s.text for s in flagged)


def test_flagged_section_in_short_document_is_not_called_human():
    from mrm_ai_detector.detector import INCONCLUSIVE, MIXED, SegmentResult, _verdict
    from mrm_ai_detector.scoring import ScoringModel

    model = ScoringModel.default()
    segs = [SegmentResult(0, None, 200, 0.1, False, ""), SegmentResult(1, None, 180, 0.95, True, "")]
    assert _verdict(0.2, 0.47, segs, model) == INCONCLUSIVE
    assert _verdict(0.2, 0.3, segs + [SegmentResult(2, None, 200, 0.1, False, "")], model) == MIXED

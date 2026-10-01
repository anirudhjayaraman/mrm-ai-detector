import math

import pytest

from mrm_ai_detector.scoring import FeatureParam, ScoringModel, sigmoid


def toy_model():
    return ScoringModel(name="toy", features={"x": FeatureParam(0.0, 1.0, 2.0), "y": FeatureParam(10.0, 5.0, -1.0)}, bias=-0.5, clip=3.0, reliability_words=40)


def test_sigmoid_is_stable():
    assert sigmoid(1000) == 1.0 and sigmoid(-1000) == 0.0 and sigmoid(0) == 0.5


def test_contributions_clip_and_missing_values():
    m = toy_model()
    c = m.contributions({"x": 100.0, "y": None})
    assert c == {"x": 6.0, "y": 0.0}


def test_reliability_shrinks_short_texts_towards_half():
    m = toy_model()
    v = {"x": 2.0, "y": 10.0}
    assert m.probability(v, 10) < m.probability(v, 1000)
    assert abs(m.probability(v, 1) - 0.5) < 0.1


def test_roundtrip(tmp_path):
    m = toy_model()
    path = tmp_path / "m.json"
    m.save(path)
    m2 = ScoringModel.load(path)
    assert m2.to_dict() == m.to_dict()


def test_rejects_unknown_schema():
    d = toy_model().to_dict()
    d["schema_version"] = 99
    with pytest.raises(ValueError):
        ScoringModel.from_dict(d)


@pytest.mark.parametrize("neural", [False, True])
def test_shipped_models_load(neural):
    m = ScoringModel.default(neural=neural)
    assert m.features and 0.5 <= m.ai_threshold < 1
    assert all(math.isfinite(p.weight) and p.std > 0 for p in m.features.values())

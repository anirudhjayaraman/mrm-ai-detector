import random

from mrm_ai_detector.train import Row, _group_folds, auc, classification_metrics, cross_validate, fit_model, tpr_at_fpr, train


def synthetic_rows(n_docs=40, segs=4, seed=0):
    rng = random.Random(seed)
    rows = []
    for d in range(n_docs):
        label = d % 2
        for _ in range(segs):
            rows.append(Row(f"doc{d}", label, 150, {"signal": rng.gauss(1.5 * label, 1.0), "noise": rng.gauss(0, 1)}))
    return rows


def test_auc_matches_known_values():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_tpr_at_fpr():
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    assert tpr_at_fpr(scores, labels, 0.0) == 1.0


def test_fit_learns_signal_direction():
    model = fit_model(synthetic_rows(), l2=1.0)
    assert model.features["signal"].weight > 0.5
    assert abs(model.features["noise"].weight) < model.features["signal"].weight / 3


def test_group_folds_never_split_a_document():
    rows = synthetic_rows()
    folds = _group_folds(rows, 5, seed=1)
    flat = [g for f in folds for g in f]
    assert len(flat) == len(set(flat)) == 40


def test_train_reports_cv_and_metadata():
    model, cv = train(synthetic_rows(), folds=4, name="t")
    assert cv["document"]["auc"] > 0.8
    assert model.metadata["n_documents"] == 40 and model.name == "t"
    m = classification_metrics([0.9, 0.1], [1, 0])
    assert m["accuracy"] == 1.0 and m["fpr"] == 0.0


def test_cross_validate_handles_both_levels():
    cv = cross_validate(synthetic_rows(), k=3)
    assert set(cv) == {"segment", "document"}

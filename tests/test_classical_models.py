
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.models.rf_xgboost_train import (
    CUTOFFS,
    build_model,
    get_feature_groups,
    prepare_cutoff_xy,
    run_cv_trials,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    rows = []

    for cutoff in CUTOFFS:
        for i in range(60):
            label = i % 3

            rows.append({
                "id_student": 1000 + i,
                "cutoff": cutoff,
                "at_risk": 1 if label > 0 else 0,
                "multiclass_target": label,
                "engagement_clicks": float(rng.normal(40 + label * 15, 10)),
                "interaction_clicks": float(rng.normal(10 + label * 5, 3)),
                "mean_score": float(rng.normal(75 - label * 10, 8)),
            })

    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("lr", LogisticRegression),
        ("svm", CalibratedClassifierCV),
        ("nb", GaussianNB),
    ],
)
def test_model_creation(name, expected):
    model = build_model(
        model_name=name,
        is_multiclass=False,
        n_classes=2,
        seed=42,
    )

    assert isinstance(model, expected)


@pytest.mark.parametrize("model_name", ["lr", "svm", "nb"])
def test_binary_cross_validation(sample_data, model_name):
    feature_cols = ["engagement_clicks", "interaction_clicks", "mean_score"]

    X, y = prepare_cutoff_xy(
        sample_data,
        cutoff=20,
        feature_cols=feature_cols,
        task="binary",
    )

    results, importances = run_cv_trials(
        X,
        y,
        model_name=model_name,
        task="binary",
        n_trials=1,
        n_splits=5,
    )

    assert len(results) == 1
    assert importances.empty

    for metric in ("precision", "recall", "f1", "roc_auc"):
        assert results[metric].notna().all()
        assert results[metric].between(0, 1).all()


@pytest.mark.parametrize("model_name", ["lr", "svm", "nb"])
def test_multiclass_cross_validation(sample_data, model_name):
    feature_cols = ["engagement_clicks", "interaction_clicks", "mean_score"]

    X, y = prepare_cutoff_xy(
        sample_data,
        cutoff=40,
        feature_cols=feature_cols,
        task="multiclass",
    )

    results, _ = run_cv_trials(
        X,
        y,
        model_name=model_name,
        task="multiclass",
        n_trials=1,
        n_splits=5,
    )

    assert len(results) == 1

    for metric in ("precision", "recall", "f1", "roc_auc"):
        assert results[metric].notna().all()
        assert results[metric].between(0, 1).all()


def test_five_cutoffs(sample_data):
    groups = get_feature_groups(sample_data)
    feature_cols = (
        groups["engagement"]
        + groups["interaction"]
        + groups["assessment"]
    )

    assert len(feature_cols) == 3

    for cutoff in CUTOFFS:
        X, y = prepare_cutoff_xy(
            sample_data,
            cutoff=cutoff,
            feature_cols=feature_cols,
            task="binary",
        )

        assert len(X) == 60
        assert len(y) == 60
        assert set(y.unique()) == {0, 1}
        assert "at_risk" not in X.columns
        assert "multiclass_target" not in X.columns


def test_scaling_only_uses_training_fold(monkeypatch):
    import src.models.rf_xgboost_train as train_mod

    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "feature_1": rng.normal(50, 10, 60),
        "feature_2": rng.normal(100, 20, 60),
    })
    y = pd.Series([0] * 30 + [1] * 30)

    fit_sizes = []
    original_fit_transform = train_mod.StandardScaler.fit_transform

    def spy_fit_transform(self, X, y=None, **kwargs):
        fit_sizes.append(len(X))
        return original_fit_transform(self, X, y, **kwargs)

    monkeypatch.setattr(
        train_mod.StandardScaler,
        "fit_transform",
        spy_fit_transform,
    )

    run_cv_trials(
        X,
        y,
        model_name="lr",
        task="binary",
        n_trials=1,
        n_splits=5,
    )

    assert len(fit_sizes) == 5
    assert all(size == 48 for size in fit_sizes)


def test_svm_calibration_with_rare_class(monkeypatch):
    import src.models.rf_xgboost_train as train_mod

    X = pd.DataFrame({
        "feature_1": np.arange(12, dtype=float),
        "feature_2": np.arange(12, dtype=float) * 2,
    })
    y = pd.Series([0] * 9 + [1] * 3)

    calibration_folds = []
    original_calibrator = train_mod.CalibratedClassifierCV

    def spy_calibrator(*args, **kwargs):
        calibration_folds.append(kwargs["cv"])
        return original_calibrator(*args, **kwargs)

    monkeypatch.setattr(
        train_mod,
        "CalibratedClassifierCV",
        spy_calibrator,
    )

    results, _ = run_cv_trials(
        X,
        y,
        model_name="svm",
        task="binary",
        n_trials=1,
        n_splits=3,
    )

    assert calibration_folds == [2, 2, 2]
    assert len(results) == 1


def test_svm_single_sample_class_uses_margin_fallback():
    X = pd.DataFrame({
        "feature_1": np.arange(8, dtype=float),
        "feature_2": np.arange(8, dtype=float) * 2,
    })
    y = pd.Series([0] * 6 + [1] * 2)

    results, _ = run_cv_trials(
        X,
        y,
        model_name="svm",
        task="binary",
        n_trials=1,
        n_splits=2,
    )

    assert len(results) == 1
    assert results["roc_auc"].notna().all()

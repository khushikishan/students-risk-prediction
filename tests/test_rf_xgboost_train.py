import numpy as np
import pandas as pd
import pytest

import src.models.rf_xgboost_train as train_mod
from src.models.rf_xgboost_train import (
    CUTOFFS,
    prepare_cutoff_xy,
    run_cv_trials,
)


@pytest.fixture
def synthetic_dataset():
  rng = np.random.default_rng(123)
  rows = []
  for cutoff in CUTOFFS:
    for i in range(60):
      label = i % 3
      rows.append({
          "code_module": "AAA",
          "code_presentation": "2013J",
          "id_student": 1000 + i,
          "cutoff": cutoff,
          "at_risk": 1 if label in (1, 2) else 0,
          "multiclass_target": label,
          "engagement_clicks": float(rng.normal(50 + label * 30, 15)),
          "interaction_clicks": float(rng.normal(10 + label * 5, 4)),
      })
    for j in range(6):
      rows.append({
          "code_module": "AAA",
          "code_presentation": "2013J",
          "id_student": 2000 + j,
          "cutoff": cutoff,
          "at_risk": 0,
          "multiclass_target": -1,
          "engagement_clicks": float(rng.normal(20, 5)),
          "interaction_clicks": float(rng.normal(2, 1)),
      })
  return pd.DataFrame(rows)


def test_multiclass_excludes_minus_one_label(synthetic_dataset):
  feature_cols = ["engagement_clicks", "interaction_clicks"]
  X, y = prepare_cutoff_xy(
      synthetic_dataset, cutoff=20, feature_cols=feature_cols, task="multiclass"
  )
  assert (-1) not in y.to_numpy()
  assert len(X) == 60
  assert len(y) == 60


def test_smote_only_alters_training_fold_not_validation_fold(monkeypatch):
  # Class 0: 30, Class 1: 30, Class 2: 10 -> in 5-fold CV, each fold has ~2 of class 2
  # and training folds have ~8 of class 2 (safe for k_neighbors <= 5).
  y_vals = np.array([0] * 30 + [1] * 30 + [2] * 10)
  rng = np.random.default_rng(7)
  X_df = pd.DataFrame({
      "engagement_clicks": rng.normal(40, 10, size=len(y_vals)),
      "interaction_clicks": rng.normal(8, 3, size=len(y_vals)),
  })
  y_ser = pd.Series(y_vals)

  fold_resample_counts = []
  orig_make_smote = train_mod.make_smote_for_fold

  def spy_make_smote(y_tr, random_state):
    sampler = orig_make_smote(y_tr, random_state)
    if sampler is None:
      return None

    orig_fit_resample = sampler.fit_resample

    def wrapped_fit_resample(X_fold, y_fold):
      X_res, y_res = orig_fit_resample(X_fold, y_fold)
      fold_resample_counts.append((len(y_fold), len(y_res)))
      return X_res, y_res

    sampler.fit_resample = wrapped_fit_resample
    return sampler

  monkeypatch.setattr(train_mod, "make_smote_for_fold", spy_make_smote)

  trials_df, _ = run_cv_trials(
      X_df,
      y_ser,
      model_name="rf",
      task="multiclass",
      n_trials=1,
      n_splits=5,
  )

  assert len(trials_df) == 1
  assert len(fold_resample_counts) == 5
  assert all(after_len > before_len for before_len, after_len in fold_resample_counts)


def test_five_cutoffs_and_25_trials_output_structure(synthetic_dataset):
  feature_cols = ["engagement_clicks", "interaction_clicks"]
  n_trials = 25
  all_results = []

  for cutoff in CUTOFFS:
    X_c, y_c = prepare_cutoff_xy(
        synthetic_dataset, cutoff, feature_cols, task="multiclass"
    )
    trials_df, _ = run_cv_trials(
        X_c,
        y_c,
        model_name="rf",
        task="multiclass",
        n_trials=n_trials,
        n_splits=5,
        n_estimators=20,
        n_jobs=1,
    )
    assert len(trials_df) == n_trials
    for metric in ("precision", "recall", "f1", "roc_auc"):
      assert trials_df[metric].notna().all()
    all_results.append(trials_df.assign(cutoff=cutoff))

  combined_trials = pd.concat(all_results, ignore_index=True)
  assert set(combined_trials["cutoff"].unique()) == set(CUTOFFS)
  assert len(combined_trials) == len(CUTOFFS) * n_trials


import numpy as np
import pandas as pd

import src.models.rf_xgboost_train as train_mod
from src.models.rf_xgboost_train import CUTOFFS, run_cv_trials


def test_groups_keep_enrollments_isolated_across_folds_when_cutoffs_combined(
    monkeypatch,
):
  """Verifies that when records from multiple cutoffs are pooled together,

  passing student enrollment IDs via `groups` prevents any enrollment from
  appearing in both training and validation folds.
  """
  rng = np.random.default_rng(42)
  rows = []

  # 30 unique student enrollments, each appearing across all 5 cutoffs (150 rows total)
  for student_idx in range(30):
    enrollment_id = f"AAA_2013J_{1000 + student_idx}"
    label = student_idx % 3  # 10 enrollments per class (0, 1, 2)
    for cutoff in CUTOFFS:
      rows.append({
          "enrollment_id": enrollment_id,
          "cutoff": cutoff,
          "multiclass_target": label,
          "engagement_clicks": float(rng.normal(40 + cutoff * 0.5 + label * 15, 8)),
          "mean_score": float(rng.normal(75 - label * 15, 10)),
      })

  pooled_df = pd.DataFrame(rows)
  X_pooled = pooled_df[["engagement_clicks", "mean_score"]]
  y_pooled = pooled_df["multiclass_target"]
  groups_pooled = pooled_df["enrollment_id"]

  observed_folds = []
  orig_make_smote = train_mod.make_smote_for_fold

  def spy_make_smote(y_tr, random_state):
    # Track training label length per fold and disable oversampling so we can
    # inspect exact fold sizes cleanly
    observed_folds.append(len(y_tr))
    return None

  # Also wrap StratifiedGroupKFold to assert zero enrollment overlap on every split
  orig_sgkf = train_mod.StratifiedGroupKFold

  class CheckedStratifiedGroupKFold(orig_sgkf):

    def split(self, X, y=None, groups=None):
      assert groups is not None, "Expected non-None groups for pooled cutoffs"
      groups_arr = np.asarray(groups)
      for train_idx, val_idx in super().split(X, y, groups=groups_arr):
        train_enrollments = set(groups_arr[train_idx])
        val_enrollments = set(groups_arr[val_idx])
        assert train_enrollments.isdisjoint(val_enrollments), (
            f"Enrollment leakage detected: {train_enrollments & val_enrollments}"
        )
        yield train_idx, val_idx

  monkeypatch.setattr(train_mod, "make_smote_for_fold", spy_make_smote)
  monkeypatch.setattr(train_mod, "StratifiedGroupKFold", CheckedStratifiedGroupKFold)

  trials_df, _ = run_cv_trials(
      X=X_pooled,
      y_raw=y_pooled,
      model_name="rf",
      task="multiclass",
      n_trials=2,
      n_splits=5,
      n_estimators=10,
      n_jobs=1,
      groups=groups_pooled,
  )

  assert len(trials_df) == 2
  assert len(observed_folds) == 10  # 2 trials * 5 folds
  assert all(n_tr == 120 for n_tr in observed_folds)  # 24 enrollments * 5 cutoffs
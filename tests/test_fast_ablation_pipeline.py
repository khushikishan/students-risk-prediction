import numpy as np
import pandas as pd
import pytest

import src.models.fast_ablation_pipeline as pipeline


@pytest.mark.parametrize(
    "task,labels",
    [
        ("binary", np.array([0, 1] * 30)),
        ("multiclass", np.array([0, 1, 2] * 20)),
    ],
)
def test_linear_svc_trains_for_binary_and_multiclass(task, labels):
  rng = np.random.default_rng(42)
  X = pd.DataFrame({
      "feature_a": rng.normal(size=len(labels)),
      "feature_b": rng.normal(size=len(labels)),
  })
  y = pd.Series(labels)

  trials, _ = pipeline.run_cv_trials_fast(
      X,
      y,
      model_name="svm",
      task=task,
      n_trials=1,
      n_splits=3,
      n_jobs=1,
  )

  assert len(trials) == 1
  assert trials[["precision", "recall", "f1", "roc_auc"]].notna().all().all()

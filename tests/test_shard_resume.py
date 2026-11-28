import numpy as np
import pandas as pd
import pytest

import src.models.rf_xgboost_train as train_mod
from src.models.fast_ablation_pipeline import (
    merge_shards,
    parse_trial_spec,
    run_ablation_with_resume,
)
from src.models.rf_xgboost_train import prepare_cutoff_xy, run_cv_trials


def _dataset() -> pd.DataFrame:
  rng = np.random.default_rng(0)
  rows = []
  for i in range(30):
    label = i % 3
    rows.append({
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": 1000 + i,
        "cutoff": 20,
        "at_risk": int(label > 0),
        "multiclass_target": label,
        "engagement_clicks": float(rng.normal(40 + label * 12, 4)),
        "interaction_clicks": float(rng.normal(4 + label, 1)),
        "mean_score": float(rng.normal(75 - label * 8, 3)),
    })
  return pd.DataFrame(rows)


def _run(df, parts_dir, trials, summary_path):
  return run_ablation_with_resume(
      df=df,
      model_names=("nb", "rf"),
      task="multiclass",
      cutoffs=(20,),
      n_trials=2,
      n_splits=3,
      n_estimators=5,
      n_jobs=2,
      summary_out_path=summary_path,
      importances_out_path=parts_dir.parent / "importances.csv",
      feature_groups=("engagement_only",),
      trials=trials,
      parts_dir=parts_dir,
      dataset_fingerprint="unit-test",
  )


def test_trial_spec_is_inclusive_and_rejects_out_of_range():
  assert parse_trial_spec("0-2,2,4", n_trials=5) == (0, 1, 2, 4)
  with pytest.raises(ValueError):
    parse_trial_spec("0-5", n_trials=5)


def test_disjoint_machines_merge_and_resume(tmp_path, monkeypatch):
  df = _dataset()
  parts = tmp_path / "parts"
  summary_path = tmp_path / "summary.csv"

  first, _ = _run(df, parts, trials=(0,), summary_path=summary_path)
  assert first.empty
  assert len(list(parts.rglob("t*_f*.json"))) == 6

  summary, importances = _run(df, parts, trials=(1,), summary_path=summary_path)
  assert set(summary["model"]) == {"nb", "rf"}
  assert summary["n_trials"].eq(2).all()
  assert summary["n_splits"].eq(3).all()
  assert set(importances["feature"]) == {"engagement_clicks"}

  X, y = prepare_cutoff_xy(
      df, 20, ["engagement_clicks"], task="multiclass"
  )
  for model_name in ("nb", "rf"):
    trials, _ = run_cv_trials(
        X,
        y,
        model_name=model_name,
        task="multiclass",
        n_trials=2,
        n_splits=3,
        n_estimators=5,
        n_jobs=1,
    )
    row = summary.loc[summary["model"] == model_name].iloc[0]
    for metric in ("precision", "recall", "f1", "roc_auc"):
      assert row[f"{metric}_mean"] == pytest.approx(trials[metric].mean())
      assert row[f"{metric}_std"] == pytest.approx(float(trials[metric].std()))

  calls = {"n": 0}
  original = train_mod.fit_and_score_fold

  def counting(*args, **kwargs):
    calls["n"] += 1
    return original(*args, **kwargs)

  monkeypatch.setattr(train_mod, "fit_and_score_fold", counting)
  _run(df, parts, trials=None, summary_path=summary_path)
  assert calls["n"] == 0

  shard = next(parts.rglob("t*_f*.json"))
  shard.unlink()
  _run(df, parts, trials=None, summary_path=summary_path)
  assert calls["n"] == 1
  merged, _ = merge_shards(parts)
  assert len(merged) == 2

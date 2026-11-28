from pathlib import Path

import pandas as pd

from src.analysis.plot_cutoff_curves import plot_cutoff_curves
from src.models.fast_ablation_pipeline import assignment_status, import_recorded_results


def test_assignment_grid_counts_both_tasks(tmp_path):
  status = assignment_status(runs_root=tmp_path)
  assert len(status) == 2 * 5 * 7 * 5
  assert int(status["complete"].sum()) == 0
  assert set(status["task"]) == {"binary", "multiclass"}
  assert set(status["model"]) == {"rf", "xgb", "lr", "nb", "svm"}
  assert set(status["cutoff"]) == {20, 40, 60, 80, 100}


def test_import_keeps_full_budget_and_skips_pilot(tmp_path):
  import hashlib
  import json

  data = tmp_path / "combined.csv"
  pd.DataFrame({
      "cutoff": [20],
      "engagement_clicks": [1.0],
      "interaction_clicks": [1.0],
      "mean_score": [80.0],
      "at_risk": [1],
      "multiclass_target": [2],
  }).to_csv(data, index=False)
  metrics = {}
  for name in ("precision", "recall", "f1", "roc_auc"):
    metrics[f"{name}_mean"] = 0.5
    metrics[f"{name}_std"] = 0.01
  summary = tmp_path / "fast.csv"
  pd.DataFrame([
      {
          "task": "multiclass", "cutoff": 20, "feature_group": "engagement_interaction",
          "n_features": 26, "model": "rf", "n_trials": 25, "n_splits": 10, **metrics,
      },
      {
          "task": "multiclass", "cutoff": 20, "feature_group": "engagement_only",
          "n_features": 21, "model": "rf", "n_trials": 2, "n_splits": 5, **metrics,
      },
  ]).to_csv(summary, index=False)
  binary = tmp_path / "binary"
  binary.mkdir()
  digest = hashlib.sha256(data.read_bytes()).hexdigest()
  (binary / "lr_20_config.json").write_text(json.dumps({
      "model": "lr", "cutoff": 20, "n_trials": 25, "n_splits": 10,
      "n_estimators": 200, "n_jobs": 1, "data_hash": digest,
  }), encoding="utf-8")
  pd.DataFrame({
      "model": ["lr"] * 25,
      "cutoff": [20] * 25,
      "trial": list(range(25)),
      "precision": [0.5] * 25,
      "recall": [0.4] * 25,
      "f1": [0.45] * 25,
      "roc_auc": [0.7] * 25,
  }).to_csv(binary / "lr_20.csv", index=False)

  imported = import_recorded_results(
      dataset_path=data,
      fingerprint=digest,
      summary_paths=[summary],
      binary_dir=binary,
      runs_root=tmp_path / "runs",
  )
  assert (imported["model"] == "rf").sum() == 1
  assert (imported["model"] == "lr").sum() == 25
  status = assignment_status(runs_root=tmp_path / "runs")
  finished = status[status["complete"]]
  assert set(zip(finished["task"], finished["feature_group"], finished["model"])) == {
      ("multiclass", "engagement_interaction", "rf"),
      ("binary", "all_features", "lr"),
  }


def test_cutoff_plot_writes_model_and_feature_curves(tmp_path):
  rows = []
  for cutoff in (20, 40, 60, 80, 100):
    for model, base in (("rf", 0.5), ("xgb", 0.55)):
      for group, bump in (("all_features", 0.1), ("engagement_only", 0.0)):
        rows.append({
            "cutoff": cutoff,
            "model": model,
            "feature_group": group,
            "f1_mean": base + bump + cutoff / 1000,
            "f1_std": 0.01,
        })
  written = plot_cutoff_curves(
      pd.DataFrame(rows),
      tmp_path,
      metric="f1",
      stem="demo",
  )
  assert len(written) == 2
  assert all(path.exists() and path.stat().st_size > 0 for path in written)
  assert {path.name for path in written} == {
      "demo_f1_by_model.png",
      "demo_f1_xgb_by_feature_group.png",
  }

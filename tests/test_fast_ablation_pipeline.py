import numpy as np
import pandas as pd
import pytest

import src.models.fast_ablation_pipeline as pipeline


@pytest.fixture
def small_binary_dataset():
  return pd.DataFrame({
      "cutoff": [20] * 24,
      "at_risk": [0, 1] * 12,
      "multiclass_target": [0, 1, 2] * 8,
      "feature_a": np.arange(24, dtype=float),
  })


def _install_fake_ablation_dependencies(monkeypatch, calls):
  monkeypatch.setattr(
      pipeline,
      "get_feature_groups",
      lambda df: {"engagement": ["feature_a"], "interaction": [], "assessment": []},
  )
  monkeypatch.setattr(
      pipeline,
      "build_ablation_subsets",
      lambda groups: {"all_features": ["feature_a"]},
  )
  monkeypatch.setattr(
      pipeline,
      "prepare_cutoff_xy",
      lambda df, cutoff, feature_cols, task: (
          df.loc[df["cutoff"] == cutoff, feature_cols],
          df.loc[df["cutoff"] == cutoff, "at_risk" if task == "binary" else "multiclass_target"],
      ),
  )

  def fake_run_cv_trials_fast(X, y_raw, **kwargs):
    calls.append(kwargs)
    n_trials = kwargs["n_trials"]
    return (
        pd.DataFrame({
            "trial": range(n_trials),
            "precision": [0.7] * n_trials,
            "recall": [0.6] * n_trials,
            "f1": [0.65] * n_trials,
            "roc_auc": [0.75] * n_trials,
        }),
        pd.DataFrame(),
    )

  monkeypatch.setattr(pipeline, "run_cv_trials_fast", fake_run_cv_trials_fast)


def _run_small_ablation(df, summary_path, importance_path, **overrides):
  config = {
      "df": df,
      "model_names": ("lr",),
      "task": "binary",
      "cutoffs": (20,),
      "n_trials": 2,
      "n_splits": 3,
      "base_seed": 17,
      "n_estimators": 25,
      "n_jobs": 1,
      "summary_out_path": summary_path,
      "importances_out_path": importance_path,
  }
  config.update(overrides)
  return pipeline.run_ablation_with_resume(**config)


def test_identical_resume_skips_completed_model(monkeypatch, tmp_path, small_binary_dataset):
  calls = []
  _install_fake_ablation_dependencies(monkeypatch, calls)
  summary_path = tmp_path / "binary_fast_summary.csv"
  importance_path = tmp_path / "binary_fast_importances.csv"

  _run_small_ablation(small_binary_dataset, summary_path, importance_path)
  first_saved = pd.read_csv(summary_path)
  _run_small_ablation(small_binary_dataset, summary_path, importance_path)

  assert len(calls) == 1
  assert "run_config" in first_saved.columns
  assert len(pd.read_csv(summary_path)) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"n_trials": 3},
        {"n_splits": 4},
        {"base_seed": 99},
        {"n_estimators": 50},
        {"n_jobs": 2},
    ],
)
def test_resume_rejects_mismatched_settings(
    monkeypatch, tmp_path, small_binary_dataset, override
):
  calls = []
  _install_fake_ablation_dependencies(monkeypatch, calls)
  summary_path = tmp_path / "binary_fast_summary.csv"
  importance_path = tmp_path / "binary_fast_importances.csv"
  _run_small_ablation(small_binary_dataset, summary_path, importance_path)
  original_contents = summary_path.read_bytes()

  with pytest.raises(ValueError, match="Configuration mismatch"):
    _run_small_ablation(
        small_binary_dataset,
        summary_path,
        importance_path,
        **override,
    )

  assert len(calls) == 1
  assert summary_path.read_bytes() == original_contents


def test_resume_rejects_changed_dataset(monkeypatch, tmp_path, small_binary_dataset):
  calls = []
  _install_fake_ablation_dependencies(monkeypatch, calls)
  summary_path = tmp_path / "binary_fast_summary.csv"
  importance_path = tmp_path / "binary_fast_importances.csv"
  _run_small_ablation(small_binary_dataset, summary_path, importance_path)

  changed_data = small_binary_dataset.copy()
  changed_data.loc[0, "feature_a"] = -1
  with pytest.raises(ValueError, match="Configuration mismatch"):
    _run_small_ablation(changed_data, summary_path, importance_path)

  assert len(calls) == 1


def test_resume_rejects_legacy_summary_without_configuration(
    monkeypatch, tmp_path, small_binary_dataset
):
  calls = []
  _install_fake_ablation_dependencies(monkeypatch, calls)
  summary_path = tmp_path / "legacy_summary.csv"
  importance_path = tmp_path / "legacy_importances.csv"
  pd.DataFrame({
      "task": ["binary"],
      "cutoff": [20],
      "feature_group": ["all_features"],
      "model": ["lr"],
      "n_trials": [2],
  }).to_csv(summary_path, index=False)

  with pytest.raises(ValueError, match="has no run configuration"):
    _run_small_ablation(small_binary_dataset, summary_path, importance_path)

  assert calls == []


def test_svm_does_not_validate_shared_rf_importance_file(
    monkeypatch, tmp_path, small_binary_dataset
):
  calls = []
  _install_fake_ablation_dependencies(monkeypatch, calls)
  summary_path = tmp_path / "binary_svm_summary.csv"
  importance_path = tmp_path / "shared_rf_importances.csv"
  importance_path.write_text("run_config,feature\nold,feature_a\n", encoding="utf-8")
  original_contents = importance_path.read_bytes()

  _run_small_ablation(
      small_binary_dataset,
      summary_path,
      importance_path,
      model_names=("svm",),
  )

  assert len(calls) == 1
  assert importance_path.read_bytes() == original_contents


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



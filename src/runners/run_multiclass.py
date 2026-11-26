from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.models.fast_ablation_pipeline import (
    ALL_MODELS,
    CUTOFFS,
    FAST_MODELS,
    PROCESSED,
    SVM_MODELS,
    run_ablation_with_resume,
)


def run_multiclass_experiment(
    track: str = "fast",
    n_trials: int = 25,
    n_splits: int = 10,
    n_estimators: int = 200,
    n_jobs: int = 1,
    svm_kernel: str = "linear",
    cutoffs: tuple[int, ...] = tuple(CUTOFFS),
    dataset_path: Path = PROCESSED / "combined_features.csv",
    summary_out_path: Path | None = None,
    importances_out_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Execute the multiclass risk prediction ablation independently."""
  if not dataset_path.exists():
    raise FileNotFoundError(f"Missing dataset at {dataset_path}")

  if track == "fast":
    selected_models = FAST_MODELS
  elif track == "svm":
    selected_models = SVM_MODELS
  elif track == "all":
    selected_models = ALL_MODELS
  else:
    raise ValueError(f"Unsupported track: {track}")

  df = pd.read_csv(dataset_path)
  out_summary = summary_out_path or (
      PROCESSED / f"multiclass_{track}_ablation_summary.csv"
  )
  out_importances = importances_out_path or (
      PROCESSED / "multiclass_rf_feature_importances.csv"
  )

  return run_ablation_with_resume(
      df=df,
      model_names=selected_models,
      task="multiclass",
      cutoffs=cutoffs,
      n_trials=n_trials,
      n_splits=n_splits,
      n_estimators=n_estimators,
      n_jobs=n_jobs,
      svm_kernel=svm_kernel,
      summary_out_path=out_summary,
      importances_out_path=out_importances,
  )


if __name__ == "__main__":
  run_multiclass_experiment(
      track="fast",
      n_trials=25,
      n_splits=10,
      n_estimators=200,
      n_jobs=1,
  )

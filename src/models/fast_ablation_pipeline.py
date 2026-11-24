from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.models.rf_xgboost_train import (
    CUTOFFS,
    PROCESSED,
    build_ablation_subsets,
    build_model,
    get_feature_groups,
    make_smote_for_fold,
    prepare_cutoff_xy,
)

FAST_MODELS: tuple[str, ...] = ("rf", "xgb", "lr", "nb")
SVM_MODELS: tuple[str, ...] = ("svm",)
ALL_MODELS: tuple[str, ...] = ("rf", "xgb", "lr", "nb", "svm")
SUPPORTED_TASKS: tuple[str, ...] = ("multiclass", "binary")
SUPPORTED_MODELS: tuple[str, ...] = FAST_MODELS + SVM_MODELS


def _dataframe_sha256(df: pd.DataFrame) -> str:
  """Hash a DataFrame's values, column names, dtypes, and index for resume checks."""
  digest = hashlib.sha256()
  metadata = {
      "columns": [str(column) for column in df.columns],
      "dtypes": [str(dtype) for dtype in df.dtypes],
  }
  digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
  digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
  return digest.hexdigest()


def _model_settings(
    model_names: tuple[str, ...],
    n_estimators: int,
    n_jobs: int,
    svm_kernel: str,
    svm_calib_folds: int,
) -> dict[str, dict]:
  settings: dict[str, dict] = {}
  for name in model_names:
    if name == "rf":
      settings[name] = {"n_estimators": n_estimators, "n_jobs": n_jobs}
    elif name == "xgb":
      settings[name] = {
          "n_estimators": n_estimators,
          "learning_rate": 0.05,
          "max_depth": 6,
          "n_jobs": n_jobs,
      }
    elif name == "lr":
      settings[name] = {"max_iter": 1000}
    elif name == "nb":
      settings[name] = {"class_prior": None}
    elif name == "svm":
      settings[name] = {
          "kernel": svm_kernel.lower(),
          "C": 1.0,
          "calibration_method": "sigmoid",
          "max_calib_folds": svm_calib_folds,
          "max_iter": 2000 if svm_kernel.lower() == "linear" else None,
          "gamma": "scale" if svm_kernel.lower() == "rbf" else None,
      }
  return settings


def _make_run_config(
    df: pd.DataFrame,
    model_names: tuple[str, ...],
    task: str,
    cutoffs: tuple[int, ...],
    n_trials: int,
    n_splits: int,
    base_seed: int,
    n_estimators: int,
    n_jobs: int,
    svm_kernel: str,
    svm_calib_folds: int,
    dataset_hash: str | None,
) -> str:
  normalized_models = tuple(str(name).lower() for name in model_names)
  invalid_models = sorted(set(normalized_models) - set(SUPPORTED_MODELS))
  if invalid_models:
    raise ValueError(f"Unsupported models: {', '.join(invalid_models)}")

  config = {
      "config_version": 1,
      "task": task,
      "models": list(normalized_models),
      "cutoffs": [int(cutoff) for cutoff in cutoffs],
      "n_trials": int(n_trials),
      "n_splits": int(n_splits),
      "base_seed": int(base_seed),
      "n_estimators": int(n_estimators),
      "n_jobs": int(n_jobs),
      "svm_kernel": svm_kernel.lower(),
      "svm_calib_folds": int(svm_calib_folds),
      "model_settings": _model_settings(
          normalized_models,
          n_estimators,
          n_jobs,
          svm_kernel,
          svm_calib_folds,
      ),
      "dataset_sha256": dataset_hash or _dataframe_sha256(df),
  }
  return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _validate_resume_config(
    path: Path,
    saved_rows: pd.DataFrame,
    expected_config: str,
    output_kind: str,
) -> None:
  if saved_rows.empty:
    return
  if "run_config" not in saved_rows.columns:
    raise ValueError(
        f"Existing {output_kind} file {path} has no run configuration. "
        "Use a new output path to avoid reusing results with unknown settings."
    )
  saved_configs = set(saved_rows["run_config"].dropna().astype(str))
  if saved_configs != {expected_config}:
    raise ValueError(
        f"Configuration mismatch for {path}. "
        "Use a new output path or restore the matching dataset and settings."
    )


def build_svm_for_fold(
    y_tr: np.ndarray,
    seed: int,
    svm_kernel: str = "linear",
    max_calib_folds: int = 2,
):
  """Construct a fold-safe calibrated SVM classifier."""
  _, counts = np.unique(y_tr, return_counts=True)
  min_count = int(counts.min())
  kernel = svm_kernel.lower()

  if kernel == "linear":
    base_estimator = LinearSVC(
        C=1.0,
        dual="auto",
        max_iter=2000,
        random_state=seed,
    )
  elif kernel == "rbf":
    base_estimator = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        cache_size=512,
        random_state=seed,
    )
  else:
    raise ValueError(f"Unsupported svm_kernel: {svm_kernel}")

  if min_count < 2:
    return base_estimator, True

  calib_folds = min(max_calib_folds, min_count)
  calibrated = CalibratedClassifierCV(
      estimator=base_estimator,
      method="sigmoid",
      cv=calib_folds,
      ensemble=False,
  )
  return calibrated, False


def evaluate_fold_metrics_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    is_multiclass: bool,
    classes: np.ndarray,
    scores_are_margins: bool = False,
) -> dict[str, float]:
  """Compute precision, recall, F1, and ROC-AUC for binary or multiclass folds."""
  avg = "macro" if is_multiclass else "binary"
  precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
  recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
  f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

  try:
    if is_multiclass:
      if scores_are_margins:
        roc_auc = float(
            np.mean([
                roc_auc_score(y_true == class_id, y_prob[:, class_id])
                for class_id in classes
            ])
        )
      else:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    else:
      if scores_are_margins:
        positive_scores = y_prob
      else:
        prob_arr = np.asarray(y_prob)
        positive_scores = (
            prob_arr[:, 1]
            if prob_arr.ndim == 2 and prob_arr.shape[1] > 1
            else prob_arr.ravel()
        )
      roc_auc = roc_auc_score(y_true, positive_scores)
  except ValueError:
    roc_auc = np.nan

  return {
      "precision": float(precision),
      "recall": float(recall),
      "f1": float(f1),
      "roc_auc": float(roc_auc),
  }


def run_cv_trials_fast(
    X: pd.DataFrame,
    y_raw: pd.Series,
    model_name: str = "rf",
    task: str = "multiclass",
    n_trials: int = 25,
    n_splits: int = 10,
    base_seed: int = 42,
    n_estimators: int = 200,
    n_jobs: int = 1,
    svm_kernel: str = "linear",
    svm_calib_folds: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Evaluate a classifier over repeated stratified folds with leak-free preprocessing."""
  if task not in SUPPORTED_TASKS:
    raise ValueError(f"Unsupported task: {task}")

  is_multiclass = task == "multiclass"
  encoder = LabelEncoder()
  y = encoder.fit_transform(y_raw.to_numpy())
  classes = np.arange(len(encoder.classes_))
  X_arr = X.to_numpy(dtype=float)
  feature_names = list(X.columns)
  name = model_name.lower()

  trial_rows = []
  importances_records = []

  for trial in range(n_trials):
    trial_seed = base_seed + trial
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=trial_seed
    )
    fold_metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": []}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y)):
      X_tr_raw, y_tr = X_arr[train_idx], y[train_idx]
      X_va_raw, y_va = X_arr[val_idx], y[val_idx]

      imputer = SimpleImputer(strategy="median")
      X_tr_imp = imputer.fit_transform(X_tr_raw)
      X_va_imp = imputer.transform(X_va_raw)

      if name in ("lr", "svm"):
        scaler = StandardScaler()
        X_tr_imp = scaler.fit_transform(X_tr_imp)
        X_va_imp = scaler.transform(X_va_imp)

      smote = make_smote_for_fold(y_tr, random_state=trial_seed * 100 + fold_idx)
      if smote is not None:
        X_tr_fit, y_tr_fit = smote.fit_resample(X_tr_imp, y_tr)
      else:
        X_tr_fit, y_tr_fit = X_tr_imp, y_tr

      if name == "svm":
        clf, svm_uncalibrated = build_svm_for_fold(
            y_tr=y_tr,
            seed=trial_seed * 100 + fold_idx,
            svm_kernel=svm_kernel,
            max_calib_folds=svm_calib_folds,
        )
      else:
        svm_uncalibrated = False
        clf = build_model(
            model_name=name,
            is_multiclass=is_multiclass,
            n_classes=len(classes),
            seed=trial_seed * 100 + fold_idx,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )

      clf.fit(X_tr_fit, y_tr_fit)
      y_pred = clf.predict(X_va_imp)
      y_prob = (
          clf.decision_function(X_va_imp)
          if svm_uncalibrated
          else clf.predict_proba(X_va_imp)
      )

      metrics = evaluate_fold_metrics_safe(
          y_true=y_va,
          y_pred=y_pred,
          y_prob=y_prob,
          is_multiclass=is_multiclass,
          classes=classes,
          scores_are_margins=svm_uncalibrated,
      )
      for k, v in metrics.items():
        fold_metrics[k].append(v)

      if hasattr(clf, "feature_importances_"):
        importances_records.append({
            "trial": trial,
            "fold": fold_idx,
            **{
                feat: float(imp)
                for feat, imp in zip(feature_names, clf.feature_importances_)
            },
        })

    trial_rows.append({
        "trial": trial,
        "precision": float(np.nanmean(fold_metrics["precision"])),
        "recall": float(np.nanmean(fold_metrics["recall"])),
        "f1": float(np.nanmean(fold_metrics["f1"])),
        "roc_auc": float(np.nanmean(fold_metrics["roc_auc"])),
    })

  return pd.DataFrame(trial_rows), pd.DataFrame(importances_records)


def run_ablation_with_resume(
    df: pd.DataFrame,
    model_names: tuple[str, ...] = FAST_MODELS,
    task: str = "multiclass",
    cutoffs: tuple[int, ...] = tuple(CUTOFFS),
    n_trials: int = 25,
    n_splits: int = 10,
    base_seed: int = 42,
    n_estimators: int = 200,
    n_jobs: int = 1,
    svm_kernel: str = "linear",
    svm_calib_folds: int = 2,
    summary_out_path: Path | None = None,
    importances_out_path: Path | None = None,
    dataset_hash: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Run step-by-step ablation with incremental saving and automatic CSV resume."""
  if task not in SUPPORTED_TASKS:
    raise ValueError(f"Unsupported task: {task}")

  model_names = tuple(str(name).lower() for name in model_names)
  run_config = _make_run_config(
      df=df,
      model_names=model_names,
      task=task,
      cutoffs=cutoffs,
      n_trials=n_trials,
      n_splits=n_splits,
      base_seed=base_seed,
      n_estimators=n_estimators,
      n_jobs=n_jobs,
      svm_kernel=svm_kernel,
      svm_calib_folds=svm_calib_folds,
      dataset_hash=dataset_hash,
  )

  groups = get_feature_groups(df)
  subsets = build_ablation_subsets(groups)

  completed_keys: set[tuple[str, int, str, str]] = set()
  summary_rows: list[dict] = []
  importance_rows: list[dict] = []

  if summary_out_path is not None and summary_out_path.exists():
    existing_summary = pd.read_csv(summary_out_path)
    if not existing_summary.empty:
      _validate_resume_config(
          summary_out_path,
          existing_summary,
          run_config,
          "summary",
      )
      summary_rows = existing_summary.to_dict(orient="records")
      completed_keys = {
          (
              str(r.get("task", task)),
              int(r["cutoff"]),
              str(r["feature_group"]),
              str(r["model"]).lower(),
          )
          for r in summary_rows
      }

  if importances_out_path is not None and importances_out_path.exists():
    existing_imp = pd.read_csv(importances_out_path)
    if not existing_imp.empty:
      _validate_resume_config(
          importances_out_path,
          existing_imp,
          run_config,
          "feature importance",
      )
      importance_rows = existing_imp.to_dict(orient="records")

  total_steps = len(cutoffs) * len(subsets) * len(model_names)
  step_num = 0
  run_start = time.perf_counter()

  for cutoff in cutoffs:
    for subset_name, subset_cols in subsets.items():
      X_c, y_c = prepare_cutoff_xy(df, cutoff, subset_cols, task=task)
      if len(y_c) == 0:
        continue

      for model_name in model_names:
        step_num += 1
        norm_model = str(model_name).lower()
        key = (task, int(cutoff), str(subset_name), norm_model)
        if key in completed_keys:
          print(
              f"[{step_num}/{total_steps}] SKIP (already saved): "
              f"task={task} | cutoff={cutoff}% | group={subset_name} | model={norm_model}",
              flush=True,
          )
          continue

        step_start = time.perf_counter()
        trials_df, imp_df = run_cv_trials_fast(
            X=X_c,
            y_raw=y_c,
            model_name=norm_model,
            task=task,
            n_trials=n_trials,
            n_splits=n_splits,
            base_seed=base_seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            svm_kernel=svm_kernel,
            svm_calib_folds=svm_calib_folds,
        )

        f1_mean = float(trials_df["f1"].mean())
        auc_mean = float(trials_df["roc_auc"].mean())

        summary_rows.append({
            "run_config": run_config,
            "task": task,
            "cutoff": cutoff,
            "feature_group": subset_name,
            "n_features": len(subset_cols),
            "model": norm_model,
            "n_trials": n_trials,
            "n_splits": n_splits,
            "precision_mean": float(trials_df["precision"].mean()),
            "precision_std": float(trials_df["precision"].std()),
            "recall_mean": float(trials_df["recall"].mean()),
            "recall_std": float(trials_df["recall"].std()),
            "f1_mean": f1_mean,
            "f1_std": float(trials_df["f1"].std()),
            "roc_auc_mean": auc_mean,
            "roc_auc_std": float(trials_df["roc_auc"].std()),
        })

        if norm_model == "rf" and not imp_df.empty:
          feat_cols = [c for c in subset_cols if c in imp_df.columns]
          mean_imps = imp_df[feat_cols].mean()
          std_imps = imp_df[feat_cols].std()
          for feat in feat_cols:
            importance_rows.append({
                "run_config": run_config,
                "task": task,
                "cutoff": cutoff,
                "feature_group": subset_name,
                "feature": feat,
                "importance_mean": float(mean_imps[feat]),
                "importance_std": float(std_imps[feat]),
            })

        if summary_out_path is not None:
          summary_out_path.parent.mkdir(parents=True, exist_ok=True)
          pd.DataFrame(summary_rows).to_csv(summary_out_path, index=False)

        if importances_out_path is not None and importance_rows:
          importances_out_path.parent.mkdir(parents=True, exist_ok=True)
          pd.DataFrame(importance_rows).to_csv(importances_out_path, index=False)

        step_elapsed = time.perf_counter() - step_start
        total_elapsed = time.perf_counter() - run_start
        print(
            f"[{step_num}/{total_steps}] task={task} | cutoff={cutoff}% | "
            f"group={subset_name} | model={norm_model} | "
            f"F1={f1_mean:.4f} | AUC={auc_mean:.4f} | "
            f"step={step_elapsed:.1f}s | total={total_elapsed:.1f}s",
            flush=True,
        )

  return pd.DataFrame(summary_rows), pd.DataFrame(importance_rows)

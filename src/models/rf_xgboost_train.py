from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CUTOFFS = [20, 40, 60, 80, 100]


def make_smote_for_fold(y_train: np.ndarray, random_state: int) -> SMOTE | None:
  """Creates a SMOTE sampler safe for small fold minority counts, or None if

  oversampling isn't feasible.
  """
  _, counts = np.unique(y_train, return_counts=True)
  if len(counts) < 2:
    return None
  min_count = int(counts.min())
  if min_count <= 1:
    return None
  k_neighbors = min(5, min_count - 1)
  return SMOTE(k_neighbors=k_neighbors, random_state=random_state)


def build_model(
    model_name: str,
    is_multiclass: bool,
    n_classes: int,
    seed: int,
    n_estimators: int = 200,
    n_jobs: int = -1,
):
  name = model_name.lower()
  if name == "rf":
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
    )
  if name == "xgb":
    if is_multiclass:
      return XGBClassifier(
          n_estimators=n_estimators,
          learning_rate=0.05,
          max_depth=6,
          objective="multi:softprob",
          num_class=n_classes,
          eval_metric="mlogloss",
          random_state=seed,
          n_jobs=n_jobs,
      )
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=n_jobs,
    )
  raise ValueError(f"Unsupported model_name: {model_name}")


def run_cv_trials(
    X: pd.DataFrame,
    y_raw: pd.Series,
    model_name: str = "rf",
    task: str = "multiclass",
    n_trials: int = 25,
    n_splits: int = 10,
    base_seed: int = 42,
    n_estimators: int = 200,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Runs `n_trials` repetitions of stratified `n_splits`-fold CV with SMOTE

  applied strictly inside each training fold.
  """
  is_multiclass = task == "multiclass"
  encoder = LabelEncoder()
  y = encoder.fit_transform(y_raw.to_numpy())
  classes = np.arange(len(encoder.classes_))
  X_arr = X.to_numpy(dtype=float)
  feature_names = list(X.columns)

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

      smote = make_smote_for_fold(y_tr, random_state=trial_seed * 100 + fold_idx)
      if smote is not None:
        X_tr_fit, y_tr_fit = smote.fit_resample(X_tr_imp, y_tr)
      else:
        X_tr_fit, y_tr_fit = X_tr_imp, y_tr

      clf = build_model(
          model_name=model_name,
          is_multiclass=is_multiclass,
          n_classes=len(classes),
          seed=trial_seed * 100 + fold_idx,
          n_estimators=n_estimators,
          n_jobs=n_jobs,
      )
      clf.fit(X_tr_fit, y_tr_fit)

      y_pred = clf.predict(X_va_imp)
      y_prob = clf.predict_proba(X_va_imp)

      metrics = evaluate_fold_metrics(
          y_true=y_va,
          y_pred=y_pred,
          y_prob=y_prob,
          is_multiclass=is_multiclass,
          classes=classes,
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

  trials_df = pd.DataFrame(trial_rows)
  importances_df = pd.DataFrame(importances_records)
  return trials_df, importances_df


def evaluate_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    is_multiclass: bool,
    classes: np.ndarray,
) -> dict[str, float]:
  avg = "macro" if is_multiclass else "binary"
  precision = precision_score(
      y_true, y_pred, average=avg, zero_division=0
  )
  recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
  f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

  try:
    if is_multiclass:
      roc_auc = roc_auc_score(
          y_true,
          y_prob,
          multi_class="ovr",
          average="macro",
          labels=classes,
      )
    else:
      roc_auc = roc_auc_score(y_true, y_prob[:, 1])
  except ValueError:
    roc_auc = np.nan

  return {
      "precision": float(precision),
      "recall": float(recall),
      "f1": float(f1),
      "roc_auc": float(roc_auc),
  }



def prepare_cutoff_xy(
    df: pd.DataFrame,
    cutoff: int,
    feature_cols: list[str],
    task: str = "multiclass",
) -> tuple[pd.DataFrame, pd.Series]:
  sub = df[df["cutoff"] == cutoff].copy()
  if task == "multiclass":
    sub = sub[sub["multiclass_target"] != -1].copy()
    target_col = "multiclass_target"
  elif task == "binary":
    target_col = "at_risk"
  else:
    raise ValueError(f"Unsupported task: {task}")

  X = sub[feature_cols].copy()
  y = sub[target_col].copy()
  return X, y


if __name__ == "__main__":
  combined_path = PROCESSED / "combined_features.csv"
  if combined_path.exists():
    data = pd.read_csv(combined_path)
    drop_cols = {
        "code_module",
        "code_presentation",
        "id_student",
        "cutoff",
        "at_risk",
        "multiclass_target",
        "has_history",
    }
    numeric_feature_cols = [
        c
        for c in data.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])
    ]

    summary_rows = []
    for cutoff in CUTOFFS:
      X_c, y_c = prepare_cutoff_xy(
          data, cutoff, numeric_feature_cols, task="multiclass"
      )
      for model_key in ("rf", "xgb"):
        trials_df, _ = run_cv_trials(
            X_c, y_c, model_name=model_key, task="multiclass", n_trials=25
        )
        summary_rows.append({
            "cutoff": cutoff,
            "model": model_key,
            "precision_mean": trials_df["precision"].mean(),
            "precision_std": trials_df["precision"].std(),
            "recall_mean": trials_df["recall"].mean(),
            "recall_std": trials_df["recall"].std(),
            "f1_mean": trials_df["f1"].mean(),
            "f1_std": trials_df["f1"].std(),
            "roc_auc_mean": trials_df["roc_auc"].mean(),
            "roc_auc_std": trials_df["roc_auc"].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import time

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CUTOFFS = [20, 40, 60, 80, 100]


def make_smote_for_fold(y_train: np.ndarray, random_state: int) -> SMOTE | None:
  """Choose SMOTE settings from this fold's smallest class.

  Return None when a minority class is too small to synthesize safely.
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
  """Create the requested classifier with task-appropriate settings.

  SVM calibration is adjusted per training fold inside `run_cv_trials`.
  """
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
  if name == "lr":
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )
  if name == "svm":
    return CalibratedClassifierCV(
        estimator=SVC(
            kernel="rbf",
            random_state=seed,
        ),
        cv=3,
        ensemble=False,
    )
  if name == "nb":
    return GaussianNB()
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
  """Evaluate a classifier over repeated stratified train/validation folds.

  Fit preprocessing and apply SMOTE using each training fold only.
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

      # Learn missing-value replacements from training rows, then reuse them.
      # Validation values never influence imputation parameters.
      imputer = SimpleImputer(strategy="median")
      X_tr_imp = imputer.fit_transform(X_tr_raw)
      X_va_imp = imputer.transform(X_va_raw)

      if model_name.lower() in ("lr", "svm"):
        # Scale only the imputed training fold and apply the same transform.
        # This keeps validation statistics out of preprocessing.
        scaler = StandardScaler()
        X_tr_imp = scaler.fit_transform(X_tr_imp)
        X_va_imp = scaler.transform(X_va_imp)

      # Resample training data only; validation remains at its natural balance.
      smote = make_smote_for_fold(y_tr, random_state=trial_seed * 100 + fold_idx)
      if smote is not None:
        X_tr_fit, y_tr_fit = smote.fit_resample(X_tr_imp, y_tr)
      else:
        X_tr_fit, y_tr_fit = X_tr_imp, y_tr

      svm_uncalibrated = False
      if model_name.lower() == "svm":
        _, counts = np.unique(y_tr, return_counts=True)
        min_count = int(counts.min())

        if min_count < 2:
          # A singleton class cannot be split into calibration folds.
          # Use the SVM's decision margins directly for ROC-AUC in this fold.
          svm_uncalibrated = True
          clf = SVC(
              kernel="rbf",
              random_state=trial_seed * 100 + fold_idx,
          )
        else:
          calibration_folds = min(3, min_count)
          clf = CalibratedClassifierCV(
              estimator=SVC(
                  kernel="rbf",
                  random_state=trial_seed * 100 + fold_idx,
              ),
              cv=calibration_folds,
              ensemble=False,
          )
      else:
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
      y_prob = (
          clf.decision_function(X_va_imp)
          if svm_uncalibrated
          else clf.predict_proba(X_va_imp)
      )

      metrics = evaluate_fold_metrics(
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

  trials_df = pd.DataFrame(trial_rows)
  importances_df = pd.DataFrame(importances_records)
  return trials_df, importances_df


def evaluate_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    is_multiclass: bool,
    classes: np.ndarray,
    scores_are_margins: bool = False,
) -> dict[str, float]:
  """Compute threshold metrics and ROC-AUC for one validation fold.

  Margin scores are handled separately because they are not calibrated probabilities.
  """
  avg = "macro" if is_multiclass else "binary"
  precision = precision_score(
      y_true, y_pred, average=avg, zero_division=0
  )
  recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
  f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

  try:
    if is_multiclass:
      if scores_are_margins:
        # SVM margins are per-class scores, so compute one-vs-rest AUC per class.
        roc_auc = float(np.mean([
            roc_auc_score(y_true == class_id, y_prob[:, class_id])
            for class_id in classes
        ]))
      else:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    else:
      positive_scores = y_prob if scores_are_margins else y_prob[:, 1]
      roc_auc = roc_auc_score(y_true, positive_scores)
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
  """Select features and labels for one course cutoff and prediction task.

  Drop unlabeled rows for multiclass training while retaining binary labels.
  """
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


# To resolve feature groups, we define the expected columns for engagement, interaction, and assessment features.
# These lists are used to filter the DataFrame columns and create subsets for ablation experiments.
ASSESSMENT_FEATURE_COLS = [
    "assessment_count",
    "mean_score",
    "min_score",
    "max_score",
    "std_score",
    "latest_score",
    "first_score",
    "score_change",
    "weighted_mean_score",
    "late_submissions",
]

INTERACTION_FEATURE_COLS = [
    "interaction_clicks",
    "interaction_click_share",
    "vle_clicks_forumng",
    "vle_clicks_oucollaborate",
    "vle_clicks_ouwiki",
]

ENGAGEMENT_SUMMARY_COLS = [
    "total_vle_clicks",
    "vle_event_count",
    "active_vle_days",
    "engagement_clicks",
]


def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
  """Resolve the feature columns present in each domain group.

  Exclude identifiers, targets, and other non-predictor fields.
  """
  numeric_cols = set(
      c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
  )
  exclude_cols = {
      "code_module",
      "code_presentation",
      "id_student",
      "cutoff",
      "at_risk",
      "multiclass_target",
      "has_history",
      "module_presentation_length",
      "date_registration",
      "date_unregistration",
  }

  interaction_cols = [
      c for c in INTERACTION_FEATURE_COLS if c in numeric_cols and c not in exclude_cols
  ]
  assessment_cols = [
      c for c in ASSESSMENT_FEATURE_COLS if c in numeric_cols and c not in exclude_cols
  ]

  interaction_set = set(interaction_cols)
  assessment_set = set(assessment_cols)

  engagement_cols = [
      c
      for c in df.columns
      if c in numeric_cols
      and c not in exclude_cols
      and c not in interaction_set
      and c not in assessment_set
      and (c in ENGAGEMENT_SUMMARY_COLS or c.startswith("vle_clicks_"))
  ]

  return {
      "engagement": engagement_cols,
      "interaction": interaction_cols,
      "assessment": assessment_cols,
  }


def build_ablation_subsets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
  """Build single-group, paired-group, and combined feature subsets.

  Omit empty groups so ablation runs always receive usable inputs.
  """
  eng = groups["engagement"]
  inter = groups["interaction"]
  assess = groups["assessment"]

  subsets = {
      "engagement_only": eng,
      "interaction_only": inter,
      "assessment_only": assess,
      "engagement_interaction": list(dict.fromkeys(eng + inter)),
      "engagement_assessment": list(dict.fromkeys(eng + assess)),
      "interaction_assessment": list(dict.fromkeys(inter + assess)),
      "all_features": list(dict.fromkeys(eng + inter + assess)),
  }
  # Keep only non-empty subsets so runs don't fail if a group has 0 columns
  return {name: cols for name, cols in subsets.items() if len(cols) > 0}


# example usage: run_ablation_experiments(df, model_names=("rf", "xgb"), task="multiclass", n_trials=25, n_splits=10)

def run_ablation_experiments(
    df: pd.DataFrame,
    model_names: tuple[str, ...] = ("rf", "xgb"),
    task: str = "multiclass",
    n_trials: int = 25,
    n_splits: int = 10,
    base_seed: int = 42,
    n_estimators: int = 200,
    n_jobs: int = 1,
    summary_out_path: Path | None = None,
    importances_out_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compare feature-group subsets across all course cutoffs.

  Report progress and save accumulated summaries as each run completes.
  """
  groups = get_feature_groups(df)
  subsets = build_ablation_subsets(groups)

  summary_rows = []
  importance_rows = []
  total_steps = len(CUTOFFS) * len(subsets) * len(model_names)
  step_num = 0
  run_start = time.perf_counter()

  for cutoff in CUTOFFS:
    for subset_name, subset_cols in subsets.items():
      X_c, y_c = prepare_cutoff_xy(df, cutoff, subset_cols, task=task)
      if len(y_c) == 0:
        continue

      for model_name in model_names:
        step_num += 1
        step_start = time.perf_counter()

        trials_df, imp_df = run_cv_trials(
            X=X_c,
            y_raw=y_c,
            model_name=model_name,
            task=task,
            n_trials=n_trials,
            n_splits=n_splits,
            base_seed=base_seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )

        f1_mean = float(trials_df["f1"].mean())
        auc_mean = float(trials_df["roc_auc"].mean())

        summary_rows.append({
            "cutoff": cutoff,
            "feature_group": subset_name,
            "n_features": len(subset_cols),
            "model": model_name,
            "precision_mean": float(trials_df["precision"].mean()),
            "precision_std": float(trials_df["precision"].std()),
            "recall_mean": float(trials_df["recall"].mean()),
            "recall_std": float(trials_df["recall"].std()),
            "f1_mean": f1_mean,
            "f1_std": float(trials_df["f1"].std()),
            "roc_auc_mean": auc_mean,
            "roc_auc_std": float(trials_df["roc_auc"].std()),
        })

        if model_name == "rf" and not imp_df.empty:
          feat_cols = [c for c in subset_cols if c in imp_df.columns]
          mean_imps = imp_df[feat_cols].mean()
          std_imps = imp_df[feat_cols].std()
          for feat in feat_cols:
            importance_rows.append({
                "cutoff": cutoff,
                "feature_group": subset_name,
                "feature": feat,
                "importance_mean": mean_imps[feat],
                "importance_std": std_imps[feat],
            })

        summary_df = pd.DataFrame(summary_rows)
        imp_summary_df = pd.DataFrame(importance_rows)

        if summary_out_path is not None:
          summary_df.to_csv(summary_out_path, index=False)
        if importances_out_path is not None and not imp_summary_df.empty:
          imp_summary_df.to_csv(importances_out_path, index=False)

        step_elapsed = time.perf_counter() - step_start
        total_elapsed = time.perf_counter() - run_start
        print(
            f"[{step_num}/{total_steps}] cutoff={cutoff}% | group={subset_name} | "
            f"model={model_name} | F1={f1_mean:.4f} | AUC={auc_mean:.4f} | "
            f"step={step_elapsed:.1f}s | total={total_elapsed:.1f}s",
            flush=True,
        )

  return pd.DataFrame(summary_rows), pd.DataFrame(importance_rows)


if __name__ == "__main__":
  combined_path = PROCESSED / "combined_features.csv"
  if combined_path.exists():
    data = pd.read_csv(combined_path)
    summary_df, rf_imp_df = run_ablation_experiments(
        df=data,
        model_names=("rf", "xgb"),
        task="multiclass",
        n_trials=25,
        n_splits=10,
        base_seed=42,
        n_estimators=200,
        n_jobs=1,
        summary_out_path=PROCESSED / "multiclass_ablation_summary.csv",
        importances_out_path=PROCESSED / "rf_ablation_feature_importances.csv",
    )
    print("Finished full ablation experiment.")
    print("Summary shape:", summary_df.shape)
    print("RF importances shape:", rf_imp_df.shape)

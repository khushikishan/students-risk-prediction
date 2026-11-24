# SVM Optimization & Decoupled Ablation Pipeline Design

## Purpose
This document records the engineering rationale behind separating Support Vector Machine (SVM) execution from the faster classifiers (`rf`, `xgb`, `lr`, `nb`), switching the default SVM solver from a kernelized RBF `SVC` to a calibrated `LinearSVC`, organizing multiclass and binary execution into separate runners under `src/models/runners/`, and adding step-level CSV checkpointing (`src/models/fast_ablation_pipeline.py`).

## Computational Bottleneck Observed
During preliminary multiclass ablation runs (`n_trials=2, n_splits=5`) on OULAD across 5 cutoffs and 7 feature subsets, tree and linear models completed quickly per step (`rf`: ~62.7s, `xgb`: ~23.7s, `lr`: ~8.6s, `nb`: ~5.5s), whereas kernelized `SVC(kernel="rbf")` wrapped in `CalibratedClassifierCV(..., cv=3)` stalled on early steps.

Three compounding factors caused this bottleneck:
1. **Superlinear Kernel Scaling (`libsvm`):** `SVC(kernel="rbf")` constructs and solves a dense pairwise kernel matrix whose time and memory complexity scales roughly quadratically to cubically ($O(n^2)$ to $O(n^3)$) with training sample count.
2. **In-Fold SMOTE Expansion:** Applying SMOTE inside each outer training fold upsamples minority classes to match the majority class, substantially increasing $n$ before `SVC.fit()` runs.
3. **Nested Calibration Folds & Single-Threaded Execution:** Wrapping `SVC` in `CalibratedClassifierCV(..., cv=3)` fits multiple internal SVM models per outer fold, and `libsvm` does not parallelize single kernel fits across CPU cores via `n_jobs`. At full benchmark scale (`25 trials × 10 folds = 250 outer fits` per step across `35` cutoff/feature-group combinations = `8,750` outer fits), kernel RBF SVM creates an extreme wall-clock bottleneck on standard laptop hardware.

## Design Changes Implemented

### 1. Replacing Kernel RBF `SVC` With Calibrated `LinearSVC`
- **Change:** Default SVM training uses `LinearSVC` (`liblinear` coordinate descent) wrapped in `CalibratedClassifierCV(..., method="sigmoid", cv=min(2, min_count), ensemble=False)` after per-fold `StandardScaler` normalization (with `svm_kernel="rbf"` still available if needed).
- **Rationale:**
  - `LinearSVC` scales linearly ($O(n \cdot d)$) with sample size and feature count rather than quadratically, reducing per-step runtime to seconds while preserving linear maximum-margin classification on standardized features.
  - Using `cv=min(2, min_count)` with Platt scaling (`method="sigmoid"`) cuts internal calibration overhead relative to `cv=3` while still producing calibrated class probabilities (`predict_proba`) required for macro one-vs-rest and binary ROC-AUC evaluation.
  - When a training fold has fewer than 2 samples in its smallest class (`min_count < 2`), the pipeline safely falls back to `decision_function` margin scores so cross-validation never crashes on rare splits.

### 2. Separating Fast Classifiers From the SVM Track & Isolating Task Workflows
- **Change:** Decoupled model execution into `FAST_MODELS = ("rf", "xgb", "lr", "nb")` and `SVM_MODELS = ("svm",)`, with separate binary and multiclass runners in `src/models/runners/`.
- **Rationale:**
  - Prevents slow SVM sweeps from blocking completion or inspection of `rf`, `xgb`, `lr`, and `nb` results.
  - Preserves clear ownership and separate output files between `multiclass` (`multiclass_target`) and `binary` (`at_risk`) experiments.

### 3. Step-by-Step Incremental Saving & Automatic CSV Resume
- **Change:** `run_ablation_with_resume` writes accumulated metrics to disk after every `(cutoff, feature_group, model)` step and skips any `(task, cutoff, feature_group, model)` tuple already recorded in the target CSV on restart.
- **Rationale:** Protects long sweeps against interruptions, laptop sleep/restarts, or manual stops without losing completed steps or corrupting existing output files.

## How Results Are Saved & Resumed
- **Summary CSVs:** Each run writes incremental results after every completed `(cutoff, feature_group, model)` step to `data/processed/{task}_{track}_ablation_summary.csv` (for example, `multiclass_fast_ablation_summary.csv`, `multiclass_svm_ablation_summary.csv`, `binary_fast_ablation_summary.csv`, and `binary_svm_ablation_summary.csv`).
- **Random Forest Feature Importances:** When `rf` runs, mean and standard deviation feature importances per cutoff, feature group, and feature are written incrementally to `data/processed/{task}_rf_feature_importances.csv`.
- **Automatic Resume:** On startup, `run_ablation_with_resume` reads existing rows from `summary_out_path` and skips any `(task, cutoff, feature_group, model)` combination already present, allowing interrupted or intentionally paused Python runs to resume seamlessly when re-invoked with the same output path.

## Alignment With Project Requirements
All optimizations preserve the experimental protocol required for both binary and multiclass evaluation:
- Evaluates all five required model families (`rf`, `xgb`, `lr`, `svm`, `nb`).
- Fits `SimpleImputer(strategy="median")`, `StandardScaler` (for `lr` and `svm`), and `SMOTE` strictly inside each training fold with zero validation leakage.
- Supports `10-fold` cross-validation repeated across `25` independent random shuffles (`n_trials=25, n_splits=10`), reporting mean and standard deviation for precision, recall, F1, and ROC-AUC across all 5 course cutoffs (`20%–100%`) and 7 feature-group subsets for both `multiclass` and `binary` tasks.

## Running the Multiclass Experiment (`src/models/runners/run_multiclass.py`)
Call `run_multiclass_experiment` directly in Python for `task="multiclass"`:

```python
from src.models.runners.run_multiclass import run_multiclass_experiment

# Quick sanity check across all 5 models
run_multiclass_experiment(
    track="all",
    n_trials=2,
    n_splits=5,
    n_estimators=50,
    n_jobs=1,
)

# Full multiclass sweep — fast models (rf, xgb, lr, nb)
run_multiclass_experiment(
    track="fast",
    n_trials=25,
    n_splits=10,
    n_estimators=200,
    n_jobs=1,
)

# Full multiclass sweep — SVM track
run_multiclass_experiment(
    track="svm",
    n_trials=25,
    n_splits=10,
    n_jobs=1,
    svm_kernel="linear",
)

```

## Running the Binary Experiment (`src/models/runners/run_binary.py`)
The binary runner uses `at_risk` as its target and has the same resumable `fast`, `svm`, and `all` tracks:

```python
from src.models.runners.run_binary import run_binary_experiment

# Full binary sweep — fast models (rf, xgb, lr, nb)
run_binary_experiment(
    track="fast",
    n_trials=25,
    n_splits=10,
    n_estimators=200,
    n_jobs=1,
)

# Full binary sweep — SVM track
run_binary_experiment(
    track="svm",
    n_trials=25,
    n_splits=10,
    n_jobs=1,
    svm_kernel="linear",
)
```

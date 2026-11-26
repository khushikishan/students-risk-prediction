"""Resumable ablation runs that can be split across machines and merged.

Each fold is one shard. A later run skips shards that are already complete,
so two machines can take disjoint cutoffs, models, or trial ranges and copy
the shard folders together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

import src.models.rf_xgboost_train as train_mod
from src.models.rf_xgboost_train import (
    CUTOFFS,
    PROCESSED,
    build_ablation_subsets,
    get_feature_groups,
    prepare_cutoff_xy,
)

FAST_MODELS = ("rf", "xgb", "lr", "nb")
SVM_MODELS = ("svm",)
ALL_MODELS = FAST_MODELS + SVM_MODELS
FEATURE_GROUP_ORDER = (
    "engagement_only",
    "interaction_only",
    "assessment_only",
    "engagement_interaction",
    "engagement_assessment",
    "interaction_assessment",
    "all_features",
)
MODEL_ORDER = ("rf", "xgb", "lr", "nb", "svm", "svm_linear")
METRIC_NAMES = ("precision", "recall", "f1", "roc_auc")
RUNS_ROOT = PROCESSED.parent / "runs"
CELL_KEYS = (
    "task",
    "cutoff",
    "feature_group",
    "model",
    "svm_kernel",
    "n_features",
    "n_trials",
    "n_splits",
)


def models_for_track(track: str) -> tuple[str, ...]:
  if track == "fast":
    return FAST_MODELS
  if track == "svm":
    return SVM_MODELS
  if track == "all":
    return ALL_MODELS
  raise ValueError(f"Unsupported track: {track}")


def file_sha256(path: Path) -> str:
  """Hash the dataset with Windows line endings normalized to LF.

  The same combined_features.csv then names the same shard folder on this
  machine and on the machine running SVM.
  """
  digest = hashlib.sha256()
  pending_cr = False
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      if pending_cr:
        chunk = b"\r" + chunk
        pending_cr = False
      if chunk.endswith(b"\r"):
        pending_cr = True
        chunk = chunk[:-1]
      digest.update(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
  if pending_cr:
    digest.update(b"\n")
  return digest.hexdigest()


def resolve_workers(n_jobs: int) -> int:
  """Turn ``n_jobs`` into a count of folds running at the same time."""
  if n_jobs is None or n_jobs < 0:
    return max(1, os.cpu_count() or 1)
  return max(1, n_jobs)


def protocol_name(
    task: str,
    n_trials: int,
    n_splits: int,
    n_estimators: int,
    base_seed: int,
    fingerprint: str,
) -> str:
  return (
      f"{task}_t{n_trials}_s{n_splits}_e{n_estimators}"
      f"_seed{base_seed}_{fingerprint[:12]}"
  )


def default_parts_dir(
    task: str,
    n_trials: int,
    n_splits: int,
    n_estimators: int,
    base_seed: int,
    fingerprint: str,
) -> Path:
  name = protocol_name(
      task, n_trials, n_splits, n_estimators, base_seed, fingerprint
  )
  return RUNS_ROOT / name / "parts"


def parse_trial_spec(spec: str | None, n_trials: int) -> tuple[int, ...]:
  if spec is None or spec.strip() == "":
    return tuple(range(n_trials))
  trials: list[int] = []
  for piece in spec.split(","):
    piece = piece.strip()
    if "-" in piece:
      start_text, stop_text = piece.split("-", 1)
      start, stop = int(start_text), int(stop_text)
      if stop < start:
        raise ValueError(f"Trial range '{piece}' is reversed")
      trials.extend(range(start, stop + 1))
    else:
      trials.append(int(piece))
  unique = tuple(dict.fromkeys(trials))
  invalid = [trial for trial in unique if trial < 0 or trial >= n_trials]
  if invalid:
    raise ValueError(
        f"Trials {invalid} are outside 0..{n_trials - 1} for n_trials={n_trials}"
    )
  return unique


def _model_folder(model_name: str, svm_kernel: str) -> str:
  if model_name == "svm" and svm_kernel != "rbf":
    return f"svm_{svm_kernel}"
  return model_name


def _shard_path(
    parts_dir: Path,
    cutoff: int,
    feature_group: str,
    model_name: str,
    svm_kernel: str,
    trial: int,
    fold: int,
) -> Path:
  folder = _model_folder(model_name, svm_kernel)
  return (
      parts_dir
      / f"c{cutoff}"
      / feature_group
      / folder
      / f"t{trial:02d}_f{fold:02d}.json"
  )


def _write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
  temporary.write_text(json.dumps(payload), encoding="utf-8")
  temporary.replace(path)


def _read_shard(path: Path) -> dict | None:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  if not payload.get("complete"):
    return None
  return payload


def _shard_matches(path: Path, fingerprint: str) -> bool:
  payload = _read_shard(path)
  return payload is not None and payload.get("dataset_sha256") == fingerprint


def _fold_indices(
    X_arr: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    base_seed: int,
    trial: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
  splitter = StratifiedKFold(
      n_splits=n_splits,
      shuffle=True,
      random_state=base_seed + trial,
  )
  for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_arr, y)):
    if fold_idx == fold:
      return train_idx, val_idx
  raise RuntimeError(f"Fold {fold} is outside n_splits={n_splits}")


def _model_thread_count(workers: int, model_name: str) -> int:
  # Folds already occupy the cores. A forest or booster then stays on one core
  # so the threads do not oversubscribe. A single-fold run can use every core.
  if workers > 1:
    return 1
  if model_name in ("rf", "xgb"):
    return -1
  return 1


def _plain(value):
  if isinstance(value, str):
    return value
  item = getattr(value, "item", None)
  if item is not None:
    try:
      return item()
    except (ValueError, AttributeError):
      return value
  return value


def _cell_identity(row) -> tuple:
  return tuple(_plain(row[key]) for key in CELL_KEYS)


def _load_parts(parts_dir: Path) -> list[dict]:
  if not parts_dir.exists():
    return []
  records = []
  patterns = ("t*_f*.json", "imported_t*.json", "imported_cell.json")
  seen: set[Path] = set()
  for pattern in patterns:
    for path in parts_dir.rglob(pattern):
      if path in seen:
        continue
      seen.add(path)
      payload = _read_shard(path)
      if payload is not None:
        records.append(payload)
  return records


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
  if frame.empty:
    return frame
  group_rank = {name: index for index, name in enumerate(FEATURE_GROUP_ORDER)}
  model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
  ordered = frame.assign(
      _group_rank=frame["feature_group"].map(lambda name: group_rank.get(name, 99)),
      _model_rank=frame["model"].map(lambda name: model_rank.get(name, 99)),
  )
  ordered = ordered.sort_values(
      ["task", "cutoff", "_group_rank", "_model_rank"]
  )
  return ordered.drop(columns=["_group_rank", "_model_rank"]).reset_index(drop=True)


def merge_shards(
    parts_dir: Path,
    summary_out_path: Path | None = None,
    importances_out_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate finished folds into trial means, then into the decision table.

  A cell is included only when every trial and fold for its recorded budget
  is present, so a 2-trial pilot cannot be read as a 25-trial result.
  """
  records = _load_parts(parts_dir)
  fingerprints = {record.get("dataset_sha256") for record in records}
  fingerprints.discard(None)
  if len(fingerprints) > 1:
    raise ValueError(
        "These shards come from more than one dataset. Keep each "
        "combined_features.csv in its own parts folder. Hashes: "
        + ", ".join(sorted(fingerprints))
    )
  fold_records = [
      record for record in records if record.get("aggregation", "fold") == "fold"
  ]
  trial_imports = [
      record for record in records if record.get("aggregation") == "trial"
  ]
  cell_imports = [
      record for record in records if record.get("aggregation") == "cell"
  ]
  progress_rows = []
  trial_rows = []
  complete_cells: set[tuple] = set()
  importance_pool: dict[tuple, list[dict]] = {}

  if fold_records:
    folds = pd.DataFrame(fold_records)
    for cell_key, cell in folds.groupby(list(CELL_KEYS), dropna=False):
      expected_folds = int(cell["n_trials"].iloc[0]) * int(cell["n_splits"].iloc[0])
      present = cell.drop_duplicates(["trial", "fold"])
      trials_done = 0
      for trial_id, trial in present.groupby("trial"):
        if trial["fold"].nunique() < int(cell["n_splits"].iloc[0]):
          continue
        trials_done += 1
        trial_rows.append({
            **{key: trial.iloc[0][key] for key in CELL_KEYS},
            "trial": int(trial_id),
            **{
                metric: float(np.nanmean(trial[metric].to_numpy(dtype=float)))
                for metric in METRIC_NAMES
            },
        })
      cell_complete = trials_done == int(cell["n_trials"].iloc[0])
      identity = _cell_identity(cell.iloc[0])
      if cell_complete:
        complete_cells.add(identity)
        for payload in cell.to_dict(orient="records"):
          if payload.get("model") == "rf" and payload.get("importances"):
            importance_pool.setdefault(identity, []).append(payload["importances"])
      progress_rows.append({
          **{key: cell.iloc[0][key] for key in CELL_KEYS},
          "folds_complete": int(len(present)),
          "folds_expected": expected_folds,
          "trials_complete": trials_done,
          "trials_expected": int(cell["n_trials"].iloc[0]),
          "complete": cell_complete,
      })

  for record in trial_imports:
    if _cell_identity(pd.Series(record)) in complete_cells:
      continue
    trial_rows.append({
        **{key: record.get(key) for key in CELL_KEYS},
        "trial": int(record["trial"]),
        **{metric: float(record[metric]) for metric in METRIC_NAMES},
    })

  if trial_imports:
    imported_trials = pd.DataFrame(trial_imports)
    for _, group in imported_trials.groupby(list(CELL_KEYS), dropna=False):
      identity = _cell_identity(group.iloc[0])
      if identity in complete_cells:
        continue
      expected = int(group["n_trials"].iloc[0])
      done = int(group["trial"].nunique())
      if done == expected:
        complete_cells.add(identity)
      progress_rows.append({
          **{key: group.iloc[0][key] for key in CELL_KEYS},
          "folds_complete": done * int(group["n_splits"].iloc[0]),
          "folds_expected": expected * int(group["n_splits"].iloc[0]),
          "trials_complete": done,
          "trials_expected": expected,
          "complete": done == expected,
      })

  trials_df = pd.DataFrame(trial_rows)
  summary_rows = []
  if not trials_df.empty:
    for identity, trial_group in trials_df.groupby(list(CELL_KEYS), dropna=False):
      if _cell_identity(trial_group.iloc[0]) not in complete_cells:
        continue
      summary_rows.append({
          **{key: trial_group.iloc[0][key] for key in CELL_KEYS},
          **{
              f"{metric}_mean": float(trial_group[metric].mean())
              for metric in METRIC_NAMES
          },
          **{
              f"{metric}_std": float(trial_group[metric].std())
              for metric in METRIC_NAMES
          },
      })

  for record in cell_imports:
    identity = _cell_identity(pd.Series(record))
    if identity in complete_cells:
      continue
    complete_cells.add(identity)
    summary_rows.append({key: record.get(key) for key in CELL_KEYS} | {
        f"{metric}_mean": float(record[f"{metric}_mean"])
        for metric in METRIC_NAMES
    } | {
        f"{metric}_std": float(record[f"{metric}_std"])
        for metric in METRIC_NAMES
    })
    progress_rows.append({
        **{key: record.get(key) for key in CELL_KEYS},
        "folds_complete": int(record["n_trials"]) * int(record["n_splits"]),
        "folds_expected": int(record["n_trials"]) * int(record["n_splits"]),
        "trials_complete": int(record["n_trials"]),
        "trials_expected": int(record["n_trials"]),
        "complete": True,
    })

  summary_df = _sort_frame(pd.DataFrame(summary_rows))
  progress_df = _sort_frame(pd.DataFrame(progress_rows))
  importance_rows = []
  for identity, payloads in importance_pool.items():
    identity_map = dict(zip(CELL_KEYS, identity))
    features = sorted({name for payload in payloads for name in payload})
    for feature in features:
      values = [payload[feature] for payload in payloads if feature in payload]
      importance_rows.append({
          "cutoff": identity_map["cutoff"],
          "feature_group": identity_map["feature_group"],
          "model": identity_map["model"],
          "feature": feature,
          "importance_mean": float(np.mean(values)),
          "importance_std": float(pd.Series(values).std()),
      })
  importances_df = pd.DataFrame(importance_rows)

  run_dir = parts_dir.parent if parts_dir.name == "parts" else parts_dir
  run_dir.mkdir(parents=True, exist_ok=True)
  summary_df.to_csv(run_dir / "summary.csv", index=False)
  progress_df.to_csv(run_dir / "progress.csv", index=False)
  if not importances_df.empty:
    importances_df.to_csv(run_dir / "rf_feature_importances.csv", index=False)
  if summary_out_path is not None:
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out_path, index=False)
  if importances_out_path is not None and not importances_df.empty:
    importances_out_path.parent.mkdir(parents=True, exist_ok=True)
    importances_df.to_csv(importances_out_path, index=False)

  complete = int(progress_df["complete"].sum()) if not progress_df.empty else 0
  started = int(len(progress_df))
  print(
      f"Merged {complete} complete cell(s) out of {started} started. "
      f"Summary: {run_dir / 'summary.csv'}",
      flush=True,
  )
  return summary_df, importances_df


def run_ablation_with_resume(
    df: pd.DataFrame,
    model_names: Sequence[str] = FAST_MODELS,
    task: str = "multiclass",
    cutoffs: Sequence[int] = tuple(CUTOFFS),
    n_trials: int = 25,
    n_splits: int = 10,
    n_estimators: int = 200,
    n_jobs: int = -1,
    svm_kernel: str = "rbf",
    summary_out_path: Path | None = None,
    importances_out_path: Path | None = None,
    base_seed: int = 42,
    feature_groups: Sequence[str] | None = None,
    trials: Sequence[int] | None = None,
    parts_dir: Path | None = None,
    dataset_fingerprint: str = "unspecified",
    svm_cache_size: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Train the requested slices, skip finished folds, and merge what is complete."""
  workers = resolve_workers(n_jobs)
  selected_trials = tuple(trials) if trials is not None else tuple(range(n_trials))
  subsets = build_ablation_subsets(get_feature_groups(df))
  if feature_groups is None:
    selected_groups = tuple(subsets)
  else:
    missing = [name for name in feature_groups if name not in subsets]
    if missing:
      raise ValueError(
          f"Unknown feature groups {missing}. Available: {list(subsets)}"
      )
    selected_groups = tuple(feature_groups)

  if parts_dir is None:
    parts_dir = default_parts_dir(
        task, n_trials, n_splits, n_estimators, base_seed, dataset_fingerprint
    )
  parts_dir.mkdir(parents=True, exist_ok=True)
  _write_json(
      parts_dir.parent / "protocol.json",
      {
          "task": task,
          "n_trials": n_trials,
          "n_splits": n_splits,
          "n_estimators": n_estimators,
          "base_seed": base_seed,
          "dataset_sha256": dataset_fingerprint,
          "svm_kernel": svm_kernel,
          "models": list(model_names),
          "workers": workers,
      },
  )

  prepared: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
  pending: list[tuple[int, str, str, int, int]] = []
  skipped = 0
  for cutoff in cutoffs:
    for group_name in selected_groups:
      columns = subsets[group_name]
      X_frame, y_raw = prepare_cutoff_xy(df, int(cutoff), columns, task=task)
      if len(y_raw) == 0:
        continue
      encoder = LabelEncoder()
      y = encoder.fit_transform(y_raw.to_numpy())
      prepared[(int(cutoff), group_name)] = (
          X_frame.to_numpy(dtype=float),
          y,
          np.arange(len(encoder.classes_)),
          list(X_frame.columns),
      )
      for model_name in model_names:
        folder = _shard_path(
            parts_dir, int(cutoff), group_name, model_name, svm_kernel, 0, 0
        ).parent
        if _imported_budget_matches(
            folder, n_trials, n_splits, n_estimators, dataset_fingerprint
        ):
          skipped += n_splits * len(selected_trials)
          print(
              f"SKIP imported: task={task} | cutoff={cutoff}% | "
              f"group={group_name} | model={_model_folder(model_name, svm_kernel)}",
              flush=True,
          )
          continue
        for trial in selected_trials:
          for fold in range(n_splits):
            path = _shard_path(
                parts_dir, int(cutoff), group_name, model_name,
                svm_kernel, trial, fold,
            )
            if _shard_matches(path, dataset_fingerprint):
              skipped += 1
              continue
            pending.append((int(cutoff), group_name, model_name, trial, fold))

  # Default (-1) gives each model the schedule measured on this CPU.
  # Forests and XGBoost are faster as one fold using every core. SVM,
  # logistic regression, and naive Bayes are single-threaded, so several
  # folds run together. An explicit worker count keeps that many folds in flight.
  adaptive = n_jobs is None or n_jobs < 0
  tree_names = {"rf", "xgb"}
  if adaptive:
    tree_units = [unit for unit in pending if unit[2] in tree_names]
    other_units = [unit for unit in pending if unit[2] not in tree_names]
  else:
    tree_units = []
    other_units = list(pending)
  print(
      f"Resume {task}: {skipped} folds already stored, {len(pending)} left. "
      f"Shards: {parts_dir}",
      flush=True,
  )
  if tree_units:
    print(
        f"Random forest and XGBoost: {len(tree_units)} folds, one at a time, "
        "using every core inside the fold.",
        flush=True,
    )
  if other_units:
    print(
        f"Other models: {len(other_units)} folds, {workers} at a time.",
        flush=True,
    )
  started = time.perf_counter()
  print_lock = Lock()
  batch_workers = {"n": workers}

  def execute(unit: tuple[int, str, str, int, int]) -> str:
    cutoff, group_name, model_name, trial, fold = unit
    X_arr, y, classes, feature_names = prepared[(cutoff, group_name)]
    train_idx, val_idx = _fold_indices(
        X_arr, y, n_splits, base_seed, trial, fold
    )
    fold_started = time.perf_counter()
    in_flight = batch_workers["n"]
    fit_kwargs = dict(
        model_name=model_name,
        is_multiclass=task == "multiclass",
        classes=classes,
        feature_names=feature_names,
        seed=(base_seed + trial) * 100 + fold,
        n_estimators=n_estimators,
        n_jobs=_model_thread_count(in_flight, model_name),
        svm_kernel=svm_kernel,
        svm_cache_size=svm_cache_size,
    )
    if in_flight > 1:
      with threadpool_limits(limits=1):
        metrics, importances = train_mod.fit_and_score_fold(
            X_arr[train_idx], y[train_idx], X_arr[val_idx], y[val_idx], **fit_kwargs
        )
    else:
      metrics, importances = train_mod.fit_and_score_fold(
          X_arr[train_idx], y[train_idx], X_arr[val_idx], y[val_idx], **fit_kwargs
      )
    stored_model = _model_folder(model_name, svm_kernel)
    _write_json(
        _shard_path(
            parts_dir, cutoff, group_name, model_name, svm_kernel, trial, fold
        ),
        {
            "complete": True,
            "task": task,
            "cutoff": cutoff,
            "feature_group": group_name,
            "model": stored_model,
            "svm_kernel": svm_kernel if model_name == "svm" else "",
            "trial": trial,
            "fold": fold,
            "n_trials": n_trials,
            "n_splits": n_splits,
            "n_estimators": n_estimators,
            "base_seed": base_seed,
            "n_features": len(feature_names),
            "dataset_sha256": dataset_fingerprint,
            "fit_seconds": round(time.perf_counter() - fold_started, 3),
            **metrics,
            "importances": importances if stored_model == "rf" else None,
        },
    )
    return (
        f"cutoff={cutoff}% group={group_name} model={stored_model} "
        f"trial={trial} fold={fold} F1={metrics['f1']:.4f} "
        f"AUC={metrics['roc_auc']:.4f}"
    )

  errors: list[BaseException] = []
  counter = {"n": 0}

  def drain(units: list[tuple[int, str, str, int, int]], in_flight: int) -> None:
    if not units:
      return
    batch_workers["n"] = in_flight
    with ThreadPoolExecutor(max_workers=in_flight) as pool:
      futures = {pool.submit(execute, unit): unit for unit in units}
      for future in as_completed(futures):
        try:
          message = future.result()
        except BaseException as exc:
          errors.append(exc)
          continue
        with print_lock:
          counter["n"] += 1
          elapsed = time.perf_counter() - started
          print(
              f"[{skipped + counter['n']}/{skipped + len(pending)}] {message} | "
              f"elapsed={elapsed:.1f}s",
              flush=True,
          )

  drain(tree_units, 1)
  drain(other_units, workers)

  summary_df, importances_df = merge_shards(
      parts_dir, summary_out_path, importances_out_path
  )
  if errors:
    print(
        f"Stopped with {len(errors)} fold error(s). "
        f"Completed shards were kept. First error: {errors[0]}",
        flush=True,
    )
    raise errors[0]
  return summary_df, importances_df


def _budget_matches(payload: dict | None, n_trials: int, n_splits: int, n_estimators: int, fingerprint: str) -> bool:
  if payload is None or not payload.get("complete"):
    return False
  return (
      int(payload.get("n_trials", -1)) == n_trials
      and int(payload.get("n_splits", -1)) == n_splits
      and int(payload.get("n_estimators", -1)) == n_estimators
      and payload.get("dataset_sha256") == fingerprint
  )


def _imported_budget_matches(
    folder: Path,
    n_trials: int,
    n_splits: int,
    n_estimators: int,
    fingerprint: str,
) -> bool:
  """A cell recorded from an older CSV is finished when its budget matches."""
  cell_payload = _read_shard(folder / "imported_cell.json")
  if _budget_matches(cell_payload, n_trials, n_splits, n_estimators, fingerprint):
    return True
  matched_trials: set[int] = set()
  if not folder.exists():
    return False
  for path in folder.glob("imported_t*.json"):
    payload = _read_shard(path)
    if _budget_matches(payload, n_trials, n_splits, n_estimators, fingerprint):
      matched_trials.add(int(payload["trial"]))
  return len(matched_trials) >= n_trials


def _dataset_hashes(path: Path) -> set[str]:
  raw = path.read_bytes()
  normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
  return {
      hashlib.sha256(raw).hexdigest(),
      hashlib.sha256(normalized).hexdigest(),
  }


def import_recorded_results(
    dataset_path: Path,
    fingerprint: str,
    n_trials: int = 25,
    n_splits: int = 10,
    n_estimators: int = 200,
    base_seed: int = 42,
    summary_paths: Sequence[Path] | None = None,
    binary_dir: Path | None = None,
    runs_root: Path = RUNS_ROOT,
) -> pd.DataFrame:
  """Load intact result CSVs into the resume store.

  A row is carried forward only when it used the requested trial and fold
  budget. Shorter pilot rows stay on disk and are not treated as finished.
  """
  dataset_path = Path(dataset_path)
  accepted_hashes = _dataset_hashes(dataset_path)
  frame = pd.read_csv(dataset_path)
  feature_count = len(build_ablation_subsets(get_feature_groups(frame))["all_features"])
  imported_rows: list[dict] = []
  pilot_rows = 0

  summaries = list(summary_paths) if summary_paths is not None else [
      PROCESSED / "multiclass_fast_ablation_summary.csv",
      PROCESSED / "multiclass_all_ablation_summary.csv",
  ]
  for summary_path in summaries:
    if not summary_path.exists():
      continue
    recorded = pd.read_csv(summary_path)
    if not {"cutoff", "feature_group", "model"}.issubset(recorded.columns):
      print(f"Skip {summary_path.name}: missing cutoff, feature group, or model.", flush=True)
      continue
    has_budget = {"n_trials", "n_splits"}.issubset(recorded.columns)
    for row in recorded.to_dict(orient="records"):
      row_trials = int(row["n_trials"]) if has_budget and not pd.isna(row.get("n_trials")) else None
      row_splits = int(row["n_splits"]) if has_budget and not pd.isna(row.get("n_splits")) else None
      if row_trials != n_trials or row_splits != n_splits:
        pilot_rows += 1
        continue
      task = str(row.get("task") or "multiclass")
      model_name = str(row["model"]).lower()
      kernel = "linear" if model_name == "svm_linear" else ""
      if model_name == "svm":
        kernel = str(row.get("svm_kernel") or "rbf")
      stored_model = _model_folder(model_name, kernel)
      parts_dir = runs_root / protocol_name(
          task, n_trials, n_splits, n_estimators, base_seed, fingerprint
      ) / "parts"
      payload = {
          "complete": True,
          "imported": True,
          "aggregation": "cell",
          "task": task,
          "cutoff": int(row["cutoff"]),
          "feature_group": str(row["feature_group"]),
          "model": stored_model,
          "svm_kernel": kernel if model_name.startswith("svm") else "",
          "n_features": int(row.get("n_features") or feature_count),
          "n_trials": n_trials,
          "n_splits": n_splits,
          "n_estimators": n_estimators,
          "base_seed": base_seed,
          "dataset_sha256": fingerprint,
          "source": str(summary_path),
      }
      for metric in METRIC_NAMES:
        payload[f"{metric}_mean"] = float(row[f"{metric}_mean"])
        payload[f"{metric}_std"] = float(row[f"{metric}_std"])
      destination = (
          parts_dir / f"c{payload['cutoff']}" / payload["feature_group"]
          / stored_model / "imported_cell.json"
      )
      _write_json(destination, payload)
      imported_rows.append(payload)

  experiment_dir = Path(binary_dir) if binary_dir is not None else PROCESSED / "binary_experiments"
  if experiment_dir.exists():
    for config_path in sorted(experiment_dir.glob("*_config.json")):
      config = json.loads(config_path.read_text(encoding="utf-8"))
      if config.get("data_hash") not in accepted_hashes:
        print(f"Skip {config_path.name}: dataset hash does not match {dataset_path.name}.", flush=True)
        continue
      if int(config.get("n_trials", -1)) != n_trials or int(config.get("n_splits", -1)) != n_splits:
        continue
      if int(config.get("n_estimators", n_estimators)) != n_estimators:
        continue
      model_name = str(config["model"]).lower()
      kernel = str(config.get("svm_kernel") or "")
      stored_model = _model_folder(model_name, kernel or "rbf")
      trials_path = config_path.with_name(config_path.name.replace("_config.json", ".csv"))
      trials = pd.read_csv(trials_path)
      if len(trials) != n_trials:
        print(f"Skip {trials_path.name}: found {len(trials)} trials, expected {n_trials}.", flush=True)
        continue
      parts_dir = runs_root / protocol_name(
          "binary", n_trials, n_splits, n_estimators, base_seed, fingerprint
      ) / "parts"
      for trial_row in trials.to_dict(orient="records"):
        payload = {
            "complete": True,
            "imported": True,
            "aggregation": "trial",
            "task": "binary",
            "cutoff": int(config["cutoff"]),
            "feature_group": "all_features",
            "model": stored_model,
            "svm_kernel": kernel if model_name == "svm" else "",
            "trial": int(trial_row["trial"]),
            "n_features": feature_count,
            "n_trials": n_trials,
            "n_splits": n_splits,
            "n_estimators": int(config.get("n_estimators", n_estimators)),
            "base_seed": base_seed,
            "dataset_sha256": fingerprint,
            "source": str(trials_path),
            "precision": float(trial_row["precision"]),
            "recall": float(trial_row["recall"]),
            "f1": float(trial_row["f1"]),
            "roc_auc": float(trial_row["roc_auc"]),
        }
        destination = (
            parts_dir / f"c{payload['cutoff']}" / "all_features" / stored_model
            / f"imported_t{payload['trial']:02d}.json"
        )
        _write_json(destination, payload)
        imported_rows.append(payload)

  print(
      f"Imported {len(imported_rows)} finished record(s). "
      f"Left {pilot_rows} shorter pilot row(s) out of the 25x10 resume set.",
      flush=True,
  )
  return pd.DataFrame(imported_rows)


def assignment_status(
    runs_root: Path = RUNS_ROOT,
    n_trials: int = 25,
    n_splits: int = 10,
    n_estimators: int = 200,
) -> pd.DataFrame:
  """Count finished folds against the 2-person assignment grid.

  The grid is binary and multiclass, five course cutoffs, seven feature
  subsets, and all five classifiers, each with 25 trials of 10 folds.
  """
  expected_folds = n_trials * n_splits
  rows = []
  records: list[dict] = []
  if runs_root.exists():
    for parts_dir in runs_root.glob("*/parts"):
      for record in _load_parts(parts_dir):
        if int(record.get("n_trials", -1)) != n_trials:
          continue
        if int(record.get("n_splits", -1)) != n_splits:
          continue
        if int(record.get("n_estimators", -1)) != n_estimators:
          continue
        records.append(record)

  counted: dict[tuple, set[tuple[int, int]]] = {}
  for record in records:
    model_name = str(record["model"])
    if model_name == "svm_linear":
      model_name = "svm"
    key = (
        str(record["task"]),
        int(record["cutoff"]),
        str(record["feature_group"]),
        model_name,
    )
    bucket = counted.setdefault(key, set())
    aggregation = record.get("aggregation", "fold")
    if aggregation == "cell":
      for trial in range(int(record["n_trials"])):
        for fold in range(int(record["n_splits"])):
          bucket.add((trial, fold))
    elif aggregation == "trial":
      for fold in range(int(record["n_splits"])):
        bucket.add((int(record["trial"]), fold))
    else:
      bucket.add((int(record["trial"]), int(record["fold"])))

  for task in ("binary", "multiclass"):
    for cutoff in CUTOFFS:
      for group in FEATURE_GROUP_ORDER:
        for model in ("rf", "xgb", "lr", "nb", "svm"):
          done = len(counted.get((task, int(cutoff), group, model), ()))
          rows.append({
              "task": task,
              "cutoff": int(cutoff),
              "feature_group": group,
              "model": model,
              "folds_complete": done,
              "folds_expected": expected_folds,
              "complete": done >= expected_folds,
          })
  return pd.DataFrame(rows)


def print_assignment_status(status: pd.DataFrame) -> None:
  total = len(status)
  complete = int(status["complete"].sum())
  print(
      f"Assignment grid: {total} cells "
      f"({status['task'].nunique()} tasks x {status['cutoff'].nunique()} cutoffs x "
      f"{status['feature_group'].nunique()} feature subsets x {status['model'].nunique()} models), "
      f"{int(status['folds_expected'].iloc[0])} folds each.",
      flush=True,
  )
  print(f"Complete cells: {complete}/{total}", flush=True)
  for task, task_rows in status.groupby("task", sort=False):
    task_complete = int(task_rows["complete"].sum())
    print(f"  {task}: {task_complete}/{len(task_rows)} cells", flush=True)
    for model, model_rows in task_rows.groupby("model", sort=False):
      folds = int(model_rows["folds_complete"].sum())
      expected = int(model_rows["folds_expected"].sum())
      print(f"    {model}: {folds}/{expected} folds", flush=True)


def _parse_csv_list(spec: str | None) -> tuple[str, ...] | None:
  if spec is None or spec.strip() == "":
    return None
  return tuple(piece.strip() for piece in spec.split(",") if piece.strip())


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Run the 25x10 ablation in resumable fold shards. "
          "Copy data/runs/<protocol>/parts from each machine into one folder, "
          "then run with --merge-only."
      ),
      epilog=(
          "Examples:\n"
          "  python run_fast_resume.py --task multiclass --track fast\n"
          "  python run_fast_resume.py --task binary --track fast\n"
          "  python run_fast_resume.py --task multiclass --track svm --cutoffs 20,40\n"
          "  python run_fast_resume.py --merge-only\n"
          "  python run_fast_resume.py --status\n\n"
          "The 2-person assignment needs both tasks and all five models. "
          "Colab can run a disjoint SVM slice on extra CPU cores. "
          "A Colab GPU does not train this scikit-learn RBF SVM."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--task", choices=("multiclass", "binary"), default="multiclass")
  parser.add_argument("--track", choices=("fast", "svm", "all"), default="fast")
  parser.add_argument("--models", default=None, help="Comma list. Overrides --track.")
  parser.add_argument("--cutoffs", default=None, help="Comma list. Default: 20,40,60,80,100.")
  parser.add_argument("--groups", default=None, help="Comma list of feature subsets.")
  parser.add_argument("--trials", default=None, help="0-12 or 0,1,5. Inclusive.")
  parser.add_argument("--n-trials", type=int, default=25)
  parser.add_argument("--n-splits", type=int, default=10)
  parser.add_argument("--n-estimators", type=int, default=200)
  parser.add_argument("--base-seed", type=int, default=42)
  parser.add_argument(
      "--workers",
      type=int,
      default=-1,
      help=(
          "Folds in flight. -1 runs trees one fold at a time on all cores, "
          "and runs SVM, logistic regression, and naive Bayes one fold per core."
      ),
  )
  parser.add_argument("--svm-kernel", default="rbf")
  parser.add_argument("--svm-cache-mb", type=int, default=1024)
  parser.add_argument("--merge-only", action="store_true")
  parser.add_argument("--import-recorded", action="store_true",
                      help="Load intact 25x10 CSV records into the resume store and exit.")
  parser.add_argument(
      "--status",
      action="store_true",
      help="Show which binary and multiclass cells of the assignment grid are finished.",
  )
  parser.add_argument("--parts-dir", type=Path, default=None)
  parser.add_argument(
      "--dataset",
      type=Path,
      default=PROCESSED / "combined_features.csv",
  )
  args = parser.parse_args(argv)

  if args.status:
    print_assignment_status(assignment_status())
    return

  if args.merge_only:
    if args.parts_dir is not None:
      merge_shards(args.parts_dir)
      return
    run_dirs = sorted(path for path in RUNS_ROOT.glob("*/parts") if path.is_dir())
    if not run_dirs:
      raise FileNotFoundError(f"No shard folders under {RUNS_ROOT}")
    for parts_dir in run_dirs:
      merge_shards(parts_dir)
    return

  if not args.dataset.exists():
    raise FileNotFoundError(f"Missing dataset at {args.dataset}")

  fingerprint = file_sha256(args.dataset)
  if args.import_recorded or not args.merge_only:
    import_recorded_results(
        dataset_path=args.dataset,
        fingerprint=fingerprint,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        base_seed=args.base_seed,
    )
  if args.import_recorded:
    print_assignment_status(assignment_status())
    return

  model_names = _parse_csv_list(args.models) or models_for_track(args.track)
  cutoff_text = _parse_csv_list(args.cutoffs)
  cutoffs = tuple(int(value) for value in cutoff_text) if cutoff_text else tuple(CUTOFFS)
  fingerprint = file_sha256(args.dataset)
  track_name = args.track if args.models is None else "custom"
  summary_path = PROCESSED / f"{args.task}_{track_name}_ablation_summary.csv"
  importances_path = PROCESSED / f"{args.task}_rf_feature_importances.csv"
  run_ablation_with_resume(
      df=pd.read_csv(args.dataset),
      model_names=model_names,
      task=args.task,
      cutoffs=cutoffs,
      n_trials=args.n_trials,
      n_splits=args.n_splits,
      n_estimators=args.n_estimators,
      n_jobs=args.workers,
      svm_kernel=args.svm_kernel,
      summary_out_path=summary_path,
      importances_out_path=importances_path,
      base_seed=args.base_seed,
      feature_groups=_parse_csv_list(args.groups),
      trials=parse_trial_spec(args.trials, args.n_trials),
      parts_dir=args.parts_dir,
      dataset_fingerprint=fingerprint,
      svm_cache_size=args.svm_cache_mb,
  )


if __name__ == "__main__":
  main()

"""Plot the assignment's course-progress curves from a merged summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


def plot_cutoff_curves(
    summary: pd.DataFrame,
    out_dir: Path,
    metric: str = "f1",
    stem: str = "cutoff_curves",
) -> list[Path]:
  """Write the time curve and the feature-group curve for one summary table.

  The x-axis is course progress. Error bars are the standard deviation across
  repeated cross-validation trials when that column is present.
  """
  if metric not in METRIC_LABELS:
    raise ValueError(f"Unsupported metric: {metric}")
  mean_col = f"{metric}_mean"
  std_col = f"{metric}_std"
  if mean_col not in summary.columns:
    raise ValueError(f"Summary is missing {mean_col}")

  frame = summary.copy()
  frame["cutoff"] = frame["cutoff"].astype(int)
  out_dir.mkdir(parents=True, exist_ok=True)
  written: list[Path] = []

  by_model = frame
  if "feature_group" in frame.columns and (frame["feature_group"] == "all_features").any():
    by_model = frame[frame["feature_group"] == "all_features"]
  model_path = out_dir / f"{stem}_{metric}_by_model.png"
  _write_lines(
      by_model,
      x_col="cutoff",
      line_col="model",
      mean_col=mean_col,
      std_col=std_col if std_col in frame.columns else None,
      title=f"{METRIC_LABELS[metric]} against course progress",
      ylabel=METRIC_LABELS[metric],
      path=model_path,
  )
  written.append(model_path)

  if "feature_group" in frame.columns and "model" in frame.columns:
    model_name = "xgb" if (frame["model"] == "xgb").any() else str(frame["model"].iloc[0])
    by_group = frame[frame["model"] == model_name]
    group_path = out_dir / f"{stem}_{metric}_{model_name}_by_feature_group.png"
    _write_lines(
        by_group,
        x_col="cutoff",
        line_col="feature_group",
        mean_col=mean_col,
        std_col=std_col if std_col in frame.columns else None,
        title=f"{METRIC_LABELS[metric]} by feature source ({model_name})",
        ylabel=METRIC_LABELS[metric],
        path=group_path,
    )
    written.append(group_path)
  return written


def _write_lines(
    frame: pd.DataFrame,
    x_col: str,
    line_col: str,
    mean_col: str,
    std_col: str | None,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
  fig, ax = plt.subplots(figsize=(8, 5))
  for name, group in frame.groupby(line_col, sort=False):
    group = group.sort_values(x_col)
    y = group[mean_col].to_numpy(dtype=float)
    yerr = None
    if std_col is not None:
      yerr = group[std_col].to_numpy(dtype=float)
    ax.errorbar(group[x_col], y, yerr=yerr, marker="o", capsize=3, label=str(name))
  ax.set_xlabel("Course progress (%)")
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.set_xticks(sorted(frame[x_col].unique()))
  ax.set_ylim(0, 1)
  ax.legend(loc="best", fontsize=8)
  ax.grid(True, axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description="Plot prediction metrics against course progress.")
  parser.add_argument("summary", type=Path)
  parser.add_argument("--out-dir", type=Path, default=Path("figures"))
  parser.add_argument("--metric", default="f1", choices=tuple(METRIC_LABELS))
  args = parser.parse_args()
  written = plot_cutoff_curves(
      pd.read_csv(args.summary),
      args.out_dir,
      metric=args.metric,
      stem=args.summary.stem,
  )
  for path in written:
    print(path)


if __name__ == "__main__":
  main()

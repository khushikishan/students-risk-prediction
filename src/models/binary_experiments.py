import argparse
from pathlib import Path
import time
import json
import hashlib

import pandas as pd

from src.models.rf_xgboost_train import (
    prepare_cutoff_xy,
    get_feature_groups,
    build_ablation_subsets,
    run_cv_trials,
)


DATA_PATH = Path("data/processed/combined_features.csv")
OUTPUT_DIR = Path("data/processed/binary_experiments")
MODELS = ("lr", "svm", "nb", "rf", "xgb")
DEFAULT_MODELS = ("lr", "nb", "rf", "xgb")
CUTOFFS = (20, 40, 60, 80, 100)
METRICS = ("precision", "recall", "f1", "roc_auc")


def run_binary_experiments(
    data_path=DATA_PATH,
    output_dir=OUTPUT_DIR,
    models=DEFAULT_MODELS,
    cutoffs=CUTOFFS,
    n_trials=25,
    n_splits=10,
    n_estimators=200,
    n_jobs=1,
):
    df = pd.read_csv(data_path)
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()

    groups = get_feature_groups(df)
    features = build_ablation_subsets(groups)["all_features"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []

    for cutoff in cutoffs:
        X, y = prepare_cutoff_xy(
            df,
            cutoff=cutoff,
            feature_cols=features,
            task="binary",
        )

        for model in models:
            output_file = output_dir / f"{model}_{cutoff}.csv"
            config_file = output_dir / f"{model}_{cutoff}_config.json"

            config = {
                "model": model,
                "cutoff": cutoff,
                "n_trials": n_trials,
                "n_splits": n_splits,
                "n_estimators": n_estimators,
                "n_jobs": n_jobs,
                "data_hash": data_hash,
            }

            if output_file.exists():
                if not config_file.exists():
                    raise ValueError(
                        f"Missing configuration for {output_file}. "
                        "Use a new output directory or remove the old results."
                    )

                saved_config = json.loads(config_file.read_text())

                if saved_config != config:
                    raise ValueError(
                        f"Configuration mismatch for {output_file}. "
                        "Use a different output directory."
                    )

                results = pd.read_csv(output_file)

                if len(results) != n_trials:
                    raise ValueError(
                        f"{output_file} contains {len(results)} trials, "
                        f"but {n_trials} were requested."
                    )

                print(f"Skipping completed: {model.upper()} | {cutoff}%", flush=True)
            else:
                print(
                    f"Running {model.upper()} | {cutoff}% | "
                    f"{n_trials} trials x {n_splits} folds",
                    flush=True,
                )

                start = time.perf_counter()

                results, _ = run_cv_trials(
                    X,
                    y,
                    model_name=model,
                    task="binary",
                    n_trials=n_trials,
                    n_splits=n_splits,
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                )

                results.insert(0, "model", model)
                results.insert(1, "cutoff", cutoff)

                results.to_csv(output_file, index=False)
                config_file.write_text(json.dumps(config, indent=2))

                elapsed = time.perf_counter() - start
                print(f"Completed in {elapsed:.1f}s", flush=True)

            all_results.append(results)

            summary = {
                "model": model,
                "cutoff": cutoff,
                "n_trials": len(results),
            }

            for metric in METRICS:
                summary[f"{metric}_mean"] = results[metric].mean()
                summary[f"{metric}_std"] = results[metric].std(ddof=0)

            summary_rows.append(summary)

    trials_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    trials_df.to_csv(output_dir / "all_trials.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    print("\nExperiment summary:", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    return trials_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        choices=CUTOFFS,
        default=list(CUTOFFS),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    args = parser.parse_args()

    run_binary_experiments(
        output_dir=args.output_dir,
        models=args.models,
        cutoffs=args.cutoffs,
        n_trials=args.trials,
        n_splits=args.folds,
        n_estimators=args.estimators,
        n_jobs=args.jobs,
    )

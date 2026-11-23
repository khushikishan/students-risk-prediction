import pandas as pd
import pytest
import json
import hashlib

from src.models import binary_experiments


def test_binary_experiment_outputs(tmp_path, monkeypatch):
    data = pd.DataFrame({
        "cutoff": [20, 20, 20, 20],
        "at_risk": [0, 1, 0, 1],
        "feature_a": [1, 2, 3, 4],
    })

    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    monkeypatch.setattr(
        binary_experiments,
        "get_feature_groups",
        lambda df: {"assessment": ["feature_a"]},
    )
    monkeypatch.setattr(
        binary_experiments,
        "build_ablation_subsets",
        lambda groups: {"all_features": ["feature_a"]},
    )
    monkeypatch.setattr(
        binary_experiments,
        "prepare_cutoff_xy",
        lambda df, cutoff, feature_cols, task: (
            df[feature_cols],
            df["at_risk"],
        ),
    )

    calls = []

    def fake_run_cv_trials(X, y, **kwargs):
        calls.append(kwargs)
        return (
            pd.DataFrame({
                "trial": [0, 1],
                "precision": [0.7, 0.8],
                "recall": [0.6, 0.7],
                "f1": [0.65, 0.75],
                "roc_auc": [0.75, 0.85],
            }),
            pd.DataFrame(),
        )

    monkeypatch.setattr(
        binary_experiments,
        "run_cv_trials",
        fake_run_cv_trials,
    )

    output_dir = tmp_path / "results"

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("lr",),
        cutoffs=(20,),
        n_trials=2,
        n_splits=3,
        n_jobs=1,
    )

    assert len(calls) == 1
    assert calls[0]["task"] == "binary"
    assert calls[0]["n_trials"] == 2
    assert calls[0]["n_splits"] == 3

    assert len(trials) == 2
    assert len(summary) == 1
    assert summary.iloc[0]["precision_mean"] == pytest.approx(0.75)
    assert summary.iloc[0]["f1_mean"] == pytest.approx(0.70)

    assert (output_dir / "lr_20.csv").exists()
    assert (output_dir / "all_trials.csv").exists()
    assert (output_dir / "summary.csv").exists()

    binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("lr",),
        cutoffs=(20,),
        n_trials=2,
        n_splits=3,
        n_jobs=1,
    )

    assert len(calls) == 1


def test_binary_experiment_trial_mismatch(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    pd.DataFrame({
        "trial": [0],
        "precision": [0.7],
        "recall": [0.6],
        "f1": [0.65],
        "roc_auc": [0.75],
    }).to_csv(output_dir / "lr_20.csv", index=False)

    data_path = tmp_path / "data.csv"
    pd.DataFrame({"at_risk": [0, 1]}).to_csv(data_path, index=False)

    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    config = {
        "model": "lr",
        "cutoff": 20,
        "n_trials": 2,
        "n_splits": 10,
        "n_estimators": 200,
        "n_jobs": -1,
        "data_hash": data_hash,
    }

    (output_dir / "lr_20_config.json").write_text(
        json.dumps(config, indent=2)
    )

    monkeypatch.setattr(
        binary_experiments,
        "get_feature_groups",
        lambda df: {"assessment": ["at_risk"]},
    )
    monkeypatch.setattr(
        binary_experiments,
        "build_ablation_subsets",
        lambda groups: {"all_features": ["at_risk"]},
    )
    monkeypatch.setattr(
        binary_experiments,
        "prepare_cutoff_xy",
        lambda df, cutoff, feature_cols, task: (
            df[feature_cols],
            df["at_risk"],
        ),
    )

    with pytest.raises(ValueError, match="1 trials"):
        binary_experiments.run_binary_experiments(
            data_path=data_path,
            output_dir=output_dir,
            models=("lr",),
            cutoffs=(20,),
            n_trials=2,
        )

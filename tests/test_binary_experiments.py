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
        "n_jobs": 1,
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


def test_binary_experiment_defaults(tmp_path, monkeypatch):
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

    binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 4
    assert [call["model_name"] for call in calls] == ["lr", "nb", "rf", "xgb"]
    assert all(call["n_jobs"] == 1 for call in calls)
    assert not (output_dir / "svm_20.csv").exists()

    binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 4


def test_binary_experiment_preserves_aggregate_results(tmp_path, monkeypatch):
    data = pd.DataFrame({
        "cutoff": [20, 20, 20, 20, 40, 40, 40, 40],
        "at_risk": [0, 1, 0, 1, 0, 1, 0, 1],
        "feature_a": [1, 2, 3, 4, 5, 6, 7, 8],
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
            df.loc[df["cutoff"] == cutoff, feature_cols],
            df.loc[df["cutoff"] == cutoff, "at_risk"],
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
    )

    assert len(calls) == 1
    assert len(trials) == 2
    assert len(summary) == 1

    original_results = (output_dir / "lr_20.csv").read_bytes()
    original_config = (output_dir / "lr_20_config.json").read_bytes()

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("nb",),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 2
    assert len(trials) == 4
    assert len(summary) == 2
    assert set(summary["model"]) == {"lr", "nb"}

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("lr",),
        cutoffs=(40,),
        n_trials=2,
    )

    assert len(calls) == 3
    assert len(trials) == 6
    assert len(summary) == 3
    assert set(summary["cutoff"]) == {20, 40}
    assert set(zip(summary["model"], summary["cutoff"])) == {
        ("lr", 20),
        ("nb", 20),
        ("lr", 40),
    }

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("nb",),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 3
    assert len(trials) == 6
    assert len(summary) == 3
    assert not trials.duplicated(subset=["model", "cutoff", "trial"]).any()
    assert not summary.duplicated(subset=["model", "cutoff"]).any()

    saved_trials = pd.read_csv(output_dir / "all_trials.csv")
    saved_summary = pd.read_csv(output_dir / "summary.csv")

    assert len(saved_trials) == 6
    assert len(saved_summary) == 3

    (output_dir / "all_trials.csv").unlink()
    (output_dir / "summary.csv").unlink()

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("lr",),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 3
    assert len(trials) == 6
    assert len(summary) == 3

    pd.testing.assert_frame_equal(
        saved_trials,
        pd.read_csv(output_dir / "all_trials.csv"),
    )
    pd.testing.assert_frame_equal(
        saved_summary,
        pd.read_csv(output_dir / "summary.csv"),
    )

    assert (output_dir / "lr_20.csv").read_bytes() == original_results
    assert (output_dir / "lr_20_config.json").read_bytes() == original_config


def test_binary_experiment_optimized_svm(tmp_path, monkeypatch):
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
    svm_calls = []

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

    def fake_run_cv_trials_fast(X, y, **kwargs):
        svm_calls.append(kwargs)
        return (
            pd.DataFrame({
                "trial": [0, 1],
                "precision": [0.8, 0.9],
                "recall": [0.7, 0.8],
                "f1": [0.75, 0.85],
                "roc_auc": [0.85, 0.95],
            }),
            pd.DataFrame(),
        )

    monkeypatch.setattr(
        binary_experiments,
        "run_cv_trials",
        fake_run_cv_trials,
    )
    monkeypatch.setattr(
        binary_experiments,
        "run_cv_trials_fast",
        fake_run_cv_trials_fast,
    )

    output_dir = tmp_path / "results"

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("lr", "nb", "rf", "xgb"),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 4
    assert len(svm_calls) == 0
    assert len(trials) == 8
    assert len(summary) == 4

    original_files = {}

    for model in ("lr", "nb", "rf", "xgb"):
        for filename in (f"{model}_20.csv", f"{model}_20_config.json"):
            original_files[filename] = (output_dir / filename).read_bytes()

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("svm",),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 4
    assert len(svm_calls) == 1
    assert svm_calls[0]["model_name"] == "svm"
    assert svm_calls[0]["task"] == "binary"
    assert svm_calls[0]["n_trials"] == 2
    assert svm_calls[0]["n_splits"] == 10
    assert svm_calls[0]["n_estimators"] == 200
    assert svm_calls[0]["n_jobs"] == 1
    assert svm_calls[0]["svm_kernel"] == "linear"
    assert svm_calls[0]["svm_calib_folds"] == 2

    assert len(trials) == 10
    assert len(summary) == 5
    assert set(summary["model"]) == {"lr", "nb", "rf", "xgb", "svm"}

    assert (output_dir / "svm_20.csv").exists()
    assert (output_dir / "svm_20_config.json").exists()

    svm_config = json.loads(
        (output_dir / "svm_20_config.json").read_text()
    )

    assert svm_config["svm_kernel"] == "linear"
    assert svm_config["svm_calib_folds"] == 2

    for model in ("lr", "nb", "rf", "xgb"):
        config = json.loads(
            (output_dir / f"{model}_20_config.json").read_text()
        )

        assert "svm_kernel" not in config
        assert "svm_calib_folds" not in config

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("svm",),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 4
    assert len(svm_calls) == 1
    assert len(trials) == 10
    assert len(summary) == 5
    assert not trials.duplicated(subset=["model", "cutoff", "trial"]).any()

    for filename, original_contents in original_files.items():
        assert (output_dir / filename).read_bytes() == original_contents


@pytest.mark.parametrize(
    "model,setting,value",
    [
        ("nb", "n_trials", 3),
        ("nb", "n_splits", 5),
        ("nb", "n_estimators", 50),
        ("nb", "n_jobs", 2),
        ("nb", "data_hash", "different_hash"),
        ("nb", "model", "rf"),
        ("nb", "cutoff", 40),
        ("svm", "svm_kernel", "rbf"),
        ("svm", "svm_calib_folds", 3),
    ],
)
def test_binary_experiment_aggregate_config_mismatch(
    tmp_path, monkeypatch, model, setting, value
):
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
    monkeypatch.setattr(
        binary_experiments,
        "run_cv_trials_fast",
        fake_run_cv_trials,
    )

    output_dir = tmp_path / "results"

    trials, summary = binary_experiments.run_binary_experiments(
        data_path=data_path,
        output_dir=output_dir,
        models=("lr", model),
        cutoffs=(20,),
        n_trials=2,
    )

    assert len(calls) == 2
    assert len(trials) == 4
    assert len(summary) == 2

    original_trials = (output_dir / "all_trials.csv").read_bytes()
    original_summary = (output_dir / "summary.csv").read_bytes()
    original_results = (output_dir / f"{model}_20.csv").read_bytes()

    config_file = output_dir / f"{model}_20_config.json"
    config = json.loads(config_file.read_text())
    config[setting] = value
    config_file.write_text(json.dumps(config, indent=2))

    with pytest.raises(ValueError, match=f"Configuration mismatch for .*{model}_20.csv"):
        binary_experiments.run_binary_experiments(
            data_path=data_path,
            output_dir=output_dir,
            models=("lr",),
            cutoffs=(20,),
            n_trials=2,
        )

    assert len(calls) == 2
    assert (output_dir / "all_trials.csv").read_bytes() == original_trials
    assert (output_dir / "summary.csv").read_bytes() == original_summary
    assert (output_dir / f"{model}_20.csv").read_bytes() == original_results


import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"

students = pd.read_csv(processed / "students_labeled.csv")
data = pd.read_csv(processed / "student_features.csv")

keys = ["code_module", "code_presentation", "id_student"]
cutoffs = [20, 40, 60, 80, 100]


# each enrollment should appear once per cutoff
def test_unique_rows():
    assert not data.duplicated(keys + ["cutoff"]).any()


# every student should be included at all five cutoffs
def test_student_coverage():
    assert set(data["cutoff"]) == set(cutoffs)

    for cutoff in cutoffs:
        current = data[data["cutoff"] == cutoff]

        assert len(current) == len(students)
        assert set(map(tuple, current[keys].to_numpy())) == set(
            map(tuple, students[keys].to_numpy())
        )


# binary labels should match final results
def test_binary_targets():
    expected = data["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    assert (data["at_risk"] == expected).all()


# multiclass labels should follow our definitions
def test_multiclass_targets():
    assert data["multiclass_target"].isin([-1, 0, 1, 2]).all()

    assert (
        data.loc[data["at_risk"] == 1, "multiclass_target"] == 2
    ).all()

    assert (
        data.loc[
            (data["at_risk"] == 0) & (~data["has_history"]),
            "multiclass_target"
        ] == -1
    ).all()

    assert (
        data.loc[
            (data["at_risk"] == 0) &
            (data["has_history"]) &
            (data["grade_drop"]),
            "multiclass_target"
        ] == 1
    ).all()

    assert (
        data.loc[
            (data["at_risk"] == 0) &
            (data["has_history"]) &
            (~data["grade_drop"]),
            "multiclass_target"
        ] == 0
    ).all()


# assessment counts should not decrease over time
def test_assessment_counts():
    ordered = data.sort_values(keys + ["cutoff"])

    differences = ordered.groupby(keys)["assessment_count"].diff()

    assert (differences.dropna() >= 0).all()


# students without assessments should not have invented scores
def test_missing_assessments():
    missing = data[data["assessment_count"] == 0]

    assert (missing["has_assessments"] == 0).all()
    assert missing["mean_score"].isna().all()
    assert missing["latest_score"].isna().all()


# assessment statistics should be within valid score limits
def test_score_ranges():
    score_columns = [
        "mean_score",
        "min_score",
        "max_score",
        "latest_score",
        "first_score",
        "weighted_mean_score"
    ]

    for column in score_columns:
        values = data[column].dropna()

        assert values.between(0, 100).all()


# score change should match latest minus first
def test_score_change():
    valid = data[data["assessment_count"] > 0]

    expected = valid["latest_score"] - valid["first_score"]

    assert (valid["score_change"] - expected).abs().lt(0.000001).all()


# check assessment counts against actual submissions
def test_assessment_cutoffs():
    raw = root / "data" / "raw"

    scores = pd.read_csv(raw / "studentAssessment.csv")
    assessments = pd.read_csv(raw / "assessments.csv")

    scores = scores.merge(
        assessments,
        on="id_assessment",
        how="left"
    )

    scores = scores[
        (scores["is_banked"] == 0) &
        (scores["score"].notna())
    ]

    course_lengths = students[
        ["code_module", "code_presentation", "module_presentation_length"]
    ].drop_duplicates()

    scores = scores.merge(
        course_lengths,
        on=["code_module", "code_presentation"],
        how="left"
    )

    for cutoff in cutoffs:
        current = scores[
            scores["date_submitted"] <=
            scores["module_presentation_length"] * cutoff / 100
        ]

        expected = current.groupby(keys).size().rename("expected_count")

        actual = data[data["cutoff"] == cutoff].set_index(keys)

        comparison = actual.join(expected)
        comparison["expected_count"] = comparison["expected_count"].fillna(0)

        assert (
            comparison["assessment_count"] == comparison["expected_count"]
        ).all()

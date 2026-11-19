
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
raw = root / "data" / "raw"
processed = root / "data" / "processed"

# loading data
students = pd.read_csv(processed / "students_labeled.csv")
assessments = pd.read_csv(raw / "assessments.csv")
scores = pd.read_csv(raw / "studentAssessment.csv")

keys = ["code_module", "code_presentation", "id_student"]

# matching the scores with the corresponding assessments
scores = scores.merge(
    assessments,
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

# keeps the scores that are valid and not banked
scores = scores[(scores["is_banked"] == 0) & (scores["score"].notna())].copy()

# adding the course length to each score record
course_lengths = students[["code_module", "code_presentation", "module_presentation_length"]].drop_duplicates()

scores = scores.merge(
    course_lengths,
    on=["code_module", "code_presentation"],
    how="left",
    validate="many_to_one"
)

print("Assessment records:", len(scores))
print("Students with assessments:", scores[keys].drop_duplicates().shape[0])
print("Missing course lengths:", scores["module_presentation_length"].isna().sum())


# calculating features at each course cutoff
all_features = []

for cutoff in [20, 40, 60, 80, 100]:
    cutoff_days = (scores["module_presentation_length"] * cutoff / 100)

    current = scores[scores["date_submitted"] <= cutoff_days].copy()

    current = current.sort_values(keys + ["date_submitted", "id_assessment"])

    # assessment statistics
    features = current.groupby(keys).agg(
        assessment_count=("score", "count"),
        mean_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        std_score=("score", "std"),
        latest_score=("score", "last"),
        first_score=("score", "first")
    ).reset_index()

    # score change
    features["score_change"] = (features["latest_score"] - features["first_score"])

    # weighted average score
    current["weighted_score"] = (current["score"] * current["weight"])

    weighted = current.groupby(keys).agg(weighted_sum=("weighted_score", "sum"), total_weight=("weight", "sum")).reset_index()

    weighted["weighted_mean_score"] = (weighted["weighted_sum"] / weighted["total_weight"].replace(0, float("nan")))

    features = features.merge(
        weighted[keys + ["weighted_mean_score"]],
        on=keys,
        how="left"
    )

    # late submissions
    current["is_late"] = (current["date"].notna() & (current["date_submitted"] > current["date"]))

    late = current.groupby(keys)["is_late"].sum().reset_index(name="late_submissions")

    features = features.merge(late, on=keys, how="left")

    features["cutoff"] = cutoff
    all_features.append(features)

# combining all the cutoff datasets
assessment_features = pd.concat(all_features, ignore_index=True)

print("\nAssessment features:", assessment_features.shape)
print("\nFeatures:")
print(assessment_features.columns.tolist())

print("\nStudents at each cutoff:")
print(assessment_features.groupby("cutoff").size())

# saving features
assessment_features.to_csv(processed / "assessment_features.csv", index=False)
print("\nSaved assessment_features.csv")

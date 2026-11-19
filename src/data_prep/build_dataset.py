
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
processed = root / "data" / "processed"

# loading data
students = pd.read_csv(processed / "students_labeled.csv")
features = pd.read_csv(processed / "assessment_features.csv")

keys = ["code_module", "code_presentation", "id_student"]

# creating a row for each student at every cutoff
cutoffs = pd.DataFrame({"cutoff": [20, 40, 60, 80, 100]})

students["merge_key"] = 1
cutoffs["merge_key"] = 1

data = students.merge(cutoffs, on="merge_key").drop(columns="merge_key")

# adding the assessment features
data = data.merge(
    features,
    on=keys + ["cutoff"],
    how="left",
    validate="one_to_one"
)

# students with no assessments
data["has_assessments"] = data["assessment_count"].notna().astype(int)

# no assessments means zero completed assessments
data["assessment_count"] = data["assessment_count"].fillna(0).astype(int)
data["late_submissions"] = data["late_submissions"].fillna(0).astype(int)

# one assessment has no score variation
data.loc[data["assessment_count"] == 1, "std_score"] = 0

print("Final dataset:", data.shape)

print("\nRows per cutoff:")
print(data.groupby("cutoff").size())

print("\nMissing assessment scores:")
print(data[["mean_score", "weighted_mean_score", "latest_score"]].isna().sum())

# saving final dataset
data.to_csv(processed / "student_features.csv", index=False)
print("\nSaved student_features.csv")

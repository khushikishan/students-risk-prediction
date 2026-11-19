
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
raw = root / "data" / "raw"
processed = root / "data" / "processed"

# loading data
students = pd.read_csv(processed / "students_clean.csv")
assessments = pd.read_csv(raw / "assessments.csv")
scores = pd.read_csv(raw / "studentAssessment.csv")

keys = ["code_module", "code_presentation", "id_student"]

# binary target
students["at_risk"] = students["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

print("Binary target distribution:")
print(students["at_risk"].value_counts())

# merging assessment scores with course details
scores = scores.merge(
    assessments,
    on="id_assessment",
    how="left",
    validate="many_to_one"
)

# removing banked assessments and missing scores
scores = scores[(scores["is_banked"] == 0) & (scores["score"].notna())].copy()

# sorting assessments by submission date
scores = scores.sort_values(keys + ["date_submitted", "id_assessment"])

# numbering assessments for each student
scores["assessment_order"] = scores.groupby(keys).cumcount()
assessment_counts = scores.groupby(keys).size().reset_index(name="assessment_count")

# first assessment score
first_scores = scores[scores["assessment_order"] == 0][keys + ["score"]].rename(columns={"score": "first_score"})

# lowest score after first assessment
later_scores = scores[scores["assessment_order"] > 0]

lowest_later = later_scores.groupby(keys)["score"].min().reset_index()
lowest_later = lowest_later.rename(columns={"score": "lowest_later_score"})

# merging assessment history
history = first_scores.merge(
    lowest_later,
    on=keys,
    how="left"
)

history = history.merge(
    assessment_counts,
    on=keys,
    how="left"
)

# checking if student has enough assessments
history["has_history"] = history["assessment_count"] >= 2

# checking for major grade drop
history["grade_drop"] = ((history["first_score"] >= 80) & (history["first_score"] - history["lowest_later_score"] >= 10))

# merging assessment history with students
students = students.merge(
    history[keys + ["has_history", "grade_drop"]],
    on=keys,
    how="left",
    validate="one_to_one"
)

students["has_history"] = students["has_history"].eq(True)
students["grade_drop"] = students["grade_drop"].eq(True)

# multiclass target
students["multiclass_target"] = 0

# class 1 : major grade drop
students.loc[students["grade_drop"], "multiclass_target"] = 1

# class 2 : fail or withdrawn
students.loc[students["at_risk"] == 1, "multiclass_target"] = 2

# class -1 : insufficient assessment history
students.loc[(students["at_risk"] == 0) & (~students["has_history"]), "multiclass_target"] = -1

print("\nMulticlass target distribution:")
print(students["multiclass_target"].value_counts().sort_index())

print("\nAssessment history by final result:")
print(pd.crosstab(students["final_result"], students["has_history"]))

# saving labeled dataset
students.to_csv(processed / "students_labeled.csv", index=False)
print("\nSaved students_labeled.csv")

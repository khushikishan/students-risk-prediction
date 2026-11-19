
import pandas as pd
from pathlib import Path

# defines the root, raw and preprocessed data directories
root = Path(__file__).resolve().parents[2]
raw = root / "data" / "raw"
processed = root / "data" / "processed"

# loading data
students = pd.read_csv(raw / "studentInfo.csv")
courses = pd.read_csv(raw / "courses.csv")
registration = pd.read_csv(raw / "studentRegistration.csv")

# columns identifying each enrollment
keys = ["code_module", "code_presentation", "id_student"]

# checks for duplicates in the datasets
print("Duplicate students:", students.duplicated(keys).sum())
print("Duplicate registrations:", registration.duplicated(keys).sum())

# handling missing demographic info
students["imd_band"] = students["imd_band"].fillna("Unknown")

# merging course information with student info
data = students.merge(
    courses,
    on=["code_module", "code_presentation"],
    how="left",
    validate="many_to_one"
)

# merging registration information
data = data.merge(
    registration,
    on=keys,
    how="left",
    validate="one_to_one"
)

# checking the merged data
print("\nMerged dataset:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

print("\nFinal results:")
print(data["final_result"].value_counts())

# saving processed data
processed.mkdir(exist_ok=True, parents=True)
data.to_csv(processed / "students_clean.csv", index=False)
print("\nSaved students_clean.csv")

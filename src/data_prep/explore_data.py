
import pandas as pd

# loading data
courses = pd.read_csv("data/raw/courses.csv")
student_info = pd.read_csv("data/raw/studentInfo.csv")
registration = pd.read_csv("data/raw/studentRegistration.csv")
assessments = pd.read_csv("data/raw/assessments.csv")
student_assessments = pd.read_csv("data/raw/studentAssessment.csv")
student_vle = pd.read_csv("data/raw/studentVle.csv")
vle = pd.read_csv("data/raw/vle.csv")

datasets = {
    "courses": courses,
    "studentInfo": student_info,
    "studentRegistration": registration,
    "assessments": assessments,
    "studentAssessment": student_assessments,
    "studentVle": student_vle,
    "vle": vle
}

# overview of the datasets
for name, df in datasets.items():
    print("\n", name)
    print("Rows and columns:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

print("\nFinal results:")
print(student_info["final_result"].value_counts())

print("\nAssessment types:")
print(assessments["assessment_type"].value_counts())

print("\nAssessment scores:")
print(student_assessments["score"].describe())

print("\nCourse lengths:")
print(courses[["code_module", "code_presentation", "module_presentation_length"]])

# assessment history per student
assessment_data = student_assessments.merge(
    assessments,
    on="id_assessment",
    how="left"
)

keys = ["code_module", "code_presentation", "id_student"]

assessment_counts = assessment_data.groupby(keys).size()

print("\nAssessment counts per enrollment:")
print(assessment_counts.describe())

print("\nEnrollments with only one assessment:")
print((assessment_counts == 1).sum())

print("\nEnrollments with two or more assessments:")
print((assessment_counts >= 2).sum())

print("\nMissing assessment deadlines:")
print(assessment_data["date"].isnull().sum())

print("\nBanked assessments:")
print(assessment_data["is_banked"].value_counts())

print("\nAssessment types with recorded scores:")
print(assessment_data["assessment_type"].value_counts())

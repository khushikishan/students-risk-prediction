# Student Risk Prediction

This project uses the Open University Learning Analytics Dataset (OULAD) to predict students who may be at risk of failing or withdrawing from a course. It uses student information, assessment performance, and learning activity to build classification models.

## Run and save on another machine

Check out this branch. Finished folds are already in `data/runs/`. Also keep `data/processed/combined_features.csv`. Stopping a run keeps every finished fold. The run folder name must contain `59d3fff37215`. A different name means the dataset file does not match.

From the project root, after `python -m pip install -r requirements.txt`:

```bash
python run_fast_resume.py --task multiclass --track fast
python run_fast_resume.py --task binary --track fast
```

SVM uses the linear kernel so it matches the finished binary SVM and skips those cells:

```bash
python run_fast_resume.py --task multiclass --track svm --svm-kernel linear
python run_fast_resume.py --task binary --track svm --svm-kernel linear
```

The same command skips folds already in `data/runs/`. Fast and SVM can run on two machines at the same time. Do not start the same command twice. Progress: `python run_fast_resume.py --status`.

Copy `data/runs/` back, then rebuild the tables with `python run_fast_resume.py --merge-only`. Each run folder gets `summary.csv`. A command also writes `data/processed/<task>_fast_ablation_summary.csv` or `<task>_svm_ablation_summary.csv` when it finishes.

## Dataset

The project uses the OULAD dataset, which contains student demographics, course registrations, assessment results, and virtual learning environment (VLE) interactions.

The raw CSV files should be placed in `data/raw/`. These files are not included in the repository.

## Data Preparation

The preprocessing pipeline consists of the following scripts:

- `src/data_prep/explore_data.py` — explores the datasets, missing values, duplicates, and assessment records.
- `src/data_prep/preprocess.py` — merges student information, course details, and registration data.
- `src/data_prep/targets.py` — creates binary and multiclass classification labels.
- `src/data_prep/assessment_features.py` — generates assessment features at different course cutoffs.
- `src/data_prep/engagement_interaction_features.py` — combines VLE and assessment features into `combined_features.csv`.
- `src/data_prep/build_dataset.py` — separately creates the assessment-focused `student_features.csv`.

### Target Labels

**Binary classification**

- `0`: Pass or Distinction
- `1`: Fail or Withdrawn

**Multiclass classification**

- `0`: Stable performance
- `1`: Major grade drop
- `2`: Fail or Withdrawn
- `-1`: Insufficient assessment history (excluded from multiclass training)

A major grade drop is defined as an initial assessment score of at least 80%, followed by a decrease of at least 10 percentage points on a later assessment. Banked assessments and missing scores are excluded from this calculation.

### Assessment Features

Assessment features are calculated at 20%, 40%, 60%, 80%, and 100% of each course's duration.

The features include assessment count, mean score, weighted mean score, minimum and maximum scores, standard deviation, first and latest scores, score change, and late submissions.

Only assessments submitted by each cutoff are included. Submission dates are used as an approximation of when assessment scores become available.

## Regenerating the Combined Feature Dataset

Place the OULAD source files in `data/raw/`: `studentInfo.csv`, `courses.csv`,
`studentRegistration.csv`, `assessments.csv`, `studentAssessment.csv`,
`studentVle.csv`, and `vle.csv`. From the project root, install dependencies
and run the complete pipeline in order:

```bash
python -m pip install -r requirements.txt
python src/data_prep/preprocess.py
python src/data_prep/targets.py
python src/data_prep/assessment_features.py
python src/data_prep/engagement_interaction_features.py
```

This creates `data/processed/combined_features.csv`, along with intermediate
files `students_clean.csv`, `students_labeled.csv`, `assessment_features.csv`,
and `engagement_interaction_features.csv`. Keep the same source data when
resuming experiments that validate results against a dataset hash.

`src/data_prep/build_dataset.py` separately creates the assessment-focused
`student_features.csv`; it is not part of the `combined_features.csv` pipeline.

The combined dataset contains student enrollment features across the five
course cutoffs: 20%, 40%, 60%, 80%, and 100%.

## Testing

Data validation tests are in `tests/test_data.py`.
Run the tests using:

```bash
python -m pytest tests/test_data.py -v
```

The tests check enrollment coverage, duplicate records, target labels, assessment counts, missing scores, score ranges, score changes, and assessment cutoff filtering.

## Project Structure

students-risk-prediction/
├── data/
│   ├── raw/                                      # Original OULAD CSVs (not committed)
│   └── processed/                                # Processed datasets and experiment results
├── docs/
│   ├── SVM_OPTIMIZATION_AND_PIPELINE_DESIGN.md
│   ├── feature_engineering_methodology_and_tradeoffs.md
│   └── feature_group_and_validation_methodology.md
├── src/
│   ├── data_prep/                                # Target and feature generation
│   └── models/
│       ├── fast_ablation_pipeline.py              # Cross-validation, checkpointing, and resume
│       ├── rf_xgboost_train.py                    # Shared model and feature-group helpers
│       └── runners/                               # Binary and multiclass entry points
└── tests/                                         # Data, model, and enrollment-isolation tests

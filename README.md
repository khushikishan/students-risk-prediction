# Student Risk Prediction

This project uses the Open University Learning Analytics Dataset (OULAD) to predict students who may be at risk of failing or withdrawing from a course. It uses student information, assessment performance, and learning activity to build classification models.

## Dataset

The project uses the OULAD dataset, which contains student demographics, course registrations, assessment results, and virtual learning environment (VLE) interactions.

The raw CSV files should be placed in `data/raw/`. These files are not included in the repository.

## Data Preparation

The preprocessing pipeline consists of the following scripts:

- `src/data_prep/explore_data.py` — explores the datasets, missing values, duplicates, and assessment records.
- `src/data_prep/preprocess.py` — merges student information, course details, and registration data.
- `src/data_prep/targets.py` — creates binary and multiclass classification labels.
- `src/data_prep/assessment_features.py` — generates assessment features at different course cutoffs.
- `src/data_prep/build_dataset.py` — combines student records and assessment features into one dataset.

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

## Running the Code

From the project root, run the scripts in this order for processing the datasets:

```bash
python src/data_prep/explore_data.py
python src/data_prep/preprocess.py
python src/data_prep/targets.py
python src/data_prep/assessment_features.py
python src/data_prep/build_dataset.py
```

The generated files are going to be saved in `data/processed/` as:

- `students_clean.csv`
- `students_labeled.csv`
- `assessment_features.csv`
- `student_features.csv`

The final assessment-feature dataset contains 32,593 student enrollments across five cutoffs, giving 162,965 rows.

## Testing

Data validation tests are in `tests/test_data.py`.
Run the tests using:

```bash
python -m pytest tests/test_data.py -v
```

The tests check enrollment coverage, duplicate records, target labels, assessment counts, missing scores, score ranges, score changes, and assessment cutoff filtering.

## Initial file structure (modified)

students-risk-prediction/
├── .env.example                                  # Template environment variables for collaborators
├── .gitignore                                    # Ignores .env, __pycache__, and large local CSV outputs
├── requirements.txt                              # Pinned Python dependencies
├── README.md                                     # Setup instructions and pipeline overview
├── PR_SUMMARY.md                                 # Pull request changelog and preliminary results
├── docs/
│   ├── FEATURE_GROUP_METHODOLOGY.md              # Engagement vs. interaction definitions & CV strategy
│   └── FEATURE_ENGINEERING_TRADEOFFS.md          # Feature engineering choices & methodological tradeoffs
├── data/
│   ├── raw/                                      # Original OULAD CSV tables (studentInfo, studentVle, etc.)
│   └── processed/
│       ├── combined_features.csv                 # Engineered feature dataset across 5 course cutoffs
│       ├── multiclass_ablation_summary.csv       # Incrementally saved CV metrics across all 70 steps
│       └── rf_ablation_feature_importances.csv   # Mean/std Random Forest feature importances per cutoff
├── src/
│   ├── features/
│   │   └── build_features.py                     # Raw OULAD aggregation and cutoff feature engineering
│   └── models/
│       └── rf_xgboost_train.py                   # Stratified/Group CV loop, SMOTE, and 70-step ablation
└── tests/
    ├── test_ablation_subsets.py                  # Verifies feature group partitioning and subsets
    ├── test_group_isolation.py                   # Verifies StratifiedGroupKFold enrollment isolation
    └── test_rf_xgboost_train.py                  # Unit tests for model training and fold metrics

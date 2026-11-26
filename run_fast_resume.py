

#with 100 estimators.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.runners.run_multiclass import run_multiclass_experiment
#resumes from any rows already saved in data/processed/multicalss_fast_ablation_summary.csv
# Uses n_estimators=200, n_jobs=1 to keep memory and CPU use on Windows manageable.
run_multiclass_experiment(
    track="fast",
    n_trials=25,
    n_splits=10,
    n_estimators=100,
    n_jobs=1,
)

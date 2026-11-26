"""Run the resumable ablation, or merge shards copied from other machines.

The 2-person assignment needs both tasks and all five models:

  python run_fast_resume.py --task multiclass --track fast
  python run_fast_resume.py --task binary --track fast
  python run_fast_resume.py --task multiclass --track svm --cutoffs 20,40
  python run_fast_resume.py --merge-only
  python run_fast_resume.py --status
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.models.fast_ablation_pipeline import main

if __name__ == "__main__":
  main()

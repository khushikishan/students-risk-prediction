import os
import pandas as pd


def apply_temporal_cutoff(student_vle_df, courses_df, cutoff_ratio):
  """Filters VLE interaction logs to only include activity occurring on or before

  the specified fraction (cutoff_ratio) of the module presentation length.
  """
  merged = pd.merge(
      student_vle_df,
      courses_df[['code_module', 'code_presentation', 'module_presentation_length']],
      on=['code_module', 'code_presentation'],
      how='inner',
  )
  cutoff_day = merged['module_presentation_length'] * cutoff_ratio
  filtered = merged[merged['date'] <= cutoff_day].copy()
  return filtered.drop(columns=['module_presentation_length'])


def generate_cutoff_datasets(raw_data_path, cutoffs=(0.2, 0.4, 0.6, 0.8, 1.0)):
  """Returns a dictionary mapping each cutoff percentage (e.g., 20) to its

  filtered VLE DataFrame.
  """
  student_vle = pd.read_csv(os.path.join(raw_data_path, 'studentVle.csv'))
  courses = pd.read_csv(os.path.join(raw_data_path, 'courses.csv'))

  cutoff_data = {}
  for ratio in cutoffs:
    pct = int(ratio * 100)
    cutoff_data[pct] = apply_temporal_cutoff(student_vle, courses, ratio)
  return cutoff_data

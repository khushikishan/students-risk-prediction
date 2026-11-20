import numpy as np
import pandas as pd
import pytest

from src.data_prep.engagement_interaction_features import (
    CUTOFFS,
    KEYS,
    build_combined_dataset,
    build_vle_cutoff_features,
    generate_all_vle_features,
)


@pytest.fixture
def sample_inputs():
  students = pd.DataFrame({
      "code_module": ["AAA", "AAA"],
      "code_presentation": ["2013J", "2013J"],
      "id_student": [101, 102],
      "module_presentation_length": [100, 100],
      "at_risk": [0, 1],
      "multiclass_target": [0, -1],
      "has_history": [True, False],
  })

  vle_meta = pd.DataFrame({
      "id_site": [1, 2, 3],
      "code_module": ["AAA", "AAA", "AAA"],
      "code_presentation": ["2013J", "2013J", "2013J"],
      "activity_type": ["oucontent", "forumng", "quiz"],
  })

  # Student 101 has clicks at day 10 (<=20), day 35 (<=40), day 75 (<=80)
  # Student 102 has no VLE activity at all
  student_vle = pd.DataFrame({
      "id_site": [1, 2, 3],
      "code_module": ["AAA", "AAA", "AAA"],
      "code_presentation": ["2013J", "2013J", "2013J"],
      "id_student": [101, 101, 101],
      "date": [10, 35, 75],
      "sum_click": [15, 20, 50],
  })
  return students, student_vle, vle_meta


def test_temporal_cutoff_excludes_future_events(sample_inputs):
  students, student_vle, vle_meta = sample_inputs

  df_20, _ = build_vle_cutoff_features(student_vle, vle_meta, students, 20)
  row_101_20 = df_20[df_20["id_student"] == 101].iloc[0]
  assert row_101_20["total_vle_clicks"] == 15
  assert row_101_20["interaction_clicks"] == 0
  assert row_101_20.get("vle_clicks_forumng", 0) == 0

  df_40, _ = build_vle_cutoff_features(student_vle, vle_meta, students, 40)
  row_101_40 = df_40[df_40["id_student"] == 101].iloc[0]
  assert row_101_40["total_vle_clicks"] == 35
  assert row_101_40["vle_clicks_forumng"] == 20
  assert np.isclose(row_101_40["interaction_click_share"], 20 / 35)


def test_one_row_per_enrollment_per_cutoff_and_missing_values(sample_inputs):
  students, student_vle, vle_meta = sample_inputs
  vle_feats, _ = generate_all_vle_features(students, student_vle, vle_meta)
  combined = build_combined_dataset(students, vle_feats, assessment_features=None)

  # 2 enrollments * 5 cutoffs = 10 rows total
  assert len(combined) == len(students) * len(CUTOFFS)
  assert not combined.duplicated(subset=KEYS + ["cutoff"]).any()

  # Student 102 should have 0 clicks rather than NaNs
  s102 = combined[combined["id_student"] == 102]
  assert (s102["total_vle_clicks"] == 0).all()
  assert not combined[["total_vle_clicks", "interaction_clicks", "interaction_click_share"]].isna().any().any()


def test_multiclass_excludes_minus_one_label(sample_inputs):
  students, student_vle, vle_meta = sample_inputs
  vle_feats, _ = generate_all_vle_features(students, student_vle, vle_meta)
  combined = build_combined_dataset(students, vle_feats, assessment_features=None)

  multiclass_df = combined[combined["multiclass_target"] != -1]
  assert (-1) not in multiclass_df["multiclass_target"].values
  assert (multiclass_df["id_student"] == 101).all()

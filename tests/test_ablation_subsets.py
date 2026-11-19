import pandas as pd

from src.models.rf_xgboost_train import (
    build_ablation_subsets,
    get_feature_groups,
)


def test_feature_groups_and_ablation_subsets():
  df = pd.DataFrame({
      "code_module": ["AAA"],
      "code_presentation": ["2013J"],
      "id_student": [101],
      "cutoff": [20],
      "at_risk": [0],
      "multiclass_target": [0],
      "total_vle_clicks": [120],
      "vle_event_count": [15],
      "active_vle_days": [5],
      "engagement_clicks": [90],
      "vle_clicks_oucontent": [90],
      "interaction_clicks": [30],
      "interaction_click_share": [0.25],
      "vle_clicks_forumng": [30],
      "assessment_count": [2],
      "mean_score": [78.5],
      "late_submissions": [0],
  })

  groups = get_feature_groups(df)
  assert "total_vle_clicks" in groups["engagement"]
  assert "vle_clicks_oucontent" in groups["engagement"]
  assert "interaction_clicks" in groups["interaction"]
  assert "vle_clicks_forumng" in groups["interaction"]
  assert "mean_score" in groups["assessment"]

  # Ensure groups are disjoint
  eng_set = set(groups["engagement"])
  inter_set = set(groups["interaction"])
  assess_set = set(groups["assessment"])
  assert eng_set.isdisjoint(inter_set)
  assert eng_set.isdisjoint(assess_set)
  assert inter_set.isdisjoint(assess_set)

  subsets = build_ablation_subsets(groups)
  assert set(subsets["engagement_only"]) == eng_set
  assert set(subsets["interaction_only"]) == inter_set
  assert set(subsets["assessment_only"]) == assess_set
  assert set(subsets["engagement_interaction"]) == eng_set | inter_set
  assert set(subsets["engagement_assessment"]) == eng_set | assess_set
  assert set(subsets["interaction_assessment"]) == inter_set | assess_set
  assert set(subsets["all_features"]) == eng_set | inter_set | assess_set

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

KEYS = ["code_module", "code_presentation", "id_student"]
CUTOFFS = [20, 40, 60, 80, 100]
INTERACTION_ACTIVITIES = {"forumng", "oucollaborate", "ouwiki"}
TARGET_COLS = ["at_risk", "multiclass_target", "has_history"]


def build_vle_cutoff_features(
    student_vle: pd.DataFrame,
    vle_meta: pd.DataFrame,
    course_lengths: pd.DataFrame,
    cutoff: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
  """Filters VLE events up to `cutoff` percent of module_presentation_length BEFORE

  aggregating, returning student-enrollment engagement & interaction features.
  """
  vle_len = student_vle.merge(
      course_lengths[
          ["code_module", "code_presentation", "module_presentation_length"]
      ].drop_duplicates(),
      on=["code_module", "code_presentation"],
      how="inner",
  )
  cutoff_days = vle_len["module_presentation_length"] * (cutoff / 100.0)
  current_vle = vle_len[vle_len["date"] <= cutoff_days].copy()

  merged = current_vle.merge(
      vle_meta[
          ["id_site", "code_module", "code_presentation", "activity_type"]
      ],
      on=["id_site", "code_module", "code_presentation"],
      how="left",
  )
  merged["activity_type"] = merged["activity_type"].fillna("unknown")

  base_agg = merged.groupby(KEYS, as_index=False).agg(
      total_vle_clicks=("sum_click", "sum"),
      vle_event_count=("sum_click", "count"),
      active_vle_days=("date", "nunique"),
  )

  pivot_clicks = (
      merged.groupby(KEYS + ["activity_type"], as_index=False)["sum_click"]
      .sum()
      .pivot_table(
          index=KEYS,
          columns="activity_type",
          values="sum_click",
          fill_value=0,
      )
      .reset_index()
  )
  pivot_clicks.columns.name = None

  activity_cols = [c for c in pivot_clicks.columns if c not in KEYS]
  rename_map = {c: f"vle_clicks_{c}" for c in activity_cols}
  pivot_clicks = pivot_clicks.rename(columns=rename_map)
  renamed_activity_cols = [rename_map[c] for c in activity_cols]

  df = base_agg.merge(pivot_clicks, on=KEYS, how="left")

  interaction_cols = [
      f"vle_clicks_{act}"
      for act in sorted(INTERACTION_ACTIVITIES)
      if f"vle_clicks_{act}" in df.columns
  ]
  df["interaction_clicks"] = (
      df[interaction_cols].sum(axis=1) if interaction_cols else 0
  )
  df["interaction_click_share"] = np.where(
      df["total_vle_clicks"] > 0,
      df["interaction_clicks"] / df["total_vle_clicks"],
      0.0,
  )

  engagement_activity_cols = [
      c for c in renamed_activity_cols if c not in set(interaction_cols)
  ]
  engagement_summary_cols = [
      "total_vle_clicks",
      "vle_event_count",
      "active_vle_days",
  ]
  df["engagement_clicks"] = (
      df[engagement_activity_cols].sum(axis=1)
      if engagement_activity_cols
      else 0
  )
  df["cutoff"] = cutoff

  groups = {
      "engagement": engagement_summary_cols
      + ["engagement_clicks"]
      + engagement_activity_cols,
      "interaction": interaction_cols
      + ["interaction_clicks", "interaction_click_share"],
  }
  return df, groups


def generate_all_vle_features(
    students: pd.DataFrame,
    student_vle: pd.DataFrame,
    vle_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
  per_cutoff = []
  merged_groups: dict[str, list[str]] = {"engagement": [], "interaction": []}

  for cutoff in CUTOFFS:
    df_c, groups_c = build_vle_cutoff_features(
        student_vle, vle_meta, students, cutoff
    )
    per_cutoff.append(df_c)
    for g in ("engagement", "interaction"):
      for col in groups_c[g]:
        if col not in merged_groups[g]:
          merged_groups[g].append(col)

  all_vle = pd.concat(per_cutoff, ignore_index=True)
  vle_feature_cols = merged_groups["engagement"] + merged_groups["interaction"]
  for col in vle_feature_cols:
    if col not in all_vle.columns:
      all_vle[col] = 0.0 if "share" in col else 0
  all_vle[vle_feature_cols] = all_vle[vle_feature_cols].fillna(0)
  return all_vle, merged_groups


def build_combined_dataset(
    students: pd.DataFrame,
    vle_features: pd.DataFrame,
    assessment_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
  """Builds one row per (code_module, code_presentation, id_student, cutoff)

  with targets and available feature groups.
  """
  keep_student_cols = KEYS + [
      c for c in TARGET_COLS if c in students.columns
  ]
  base_students = students[keep_student_cols].drop_duplicates(subset=KEYS)

  grid = pd.concat(
      [base_students.assign(cutoff=c) for c in CUTOFFS], ignore_index=True
  )
  join_keys = KEYS + ["cutoff"]

  df = grid.merge(vle_features, on=join_keys, how="left")
  vle_cols = [c for c in vle_features.columns if c not in join_keys]
  df[vle_cols] = df[vle_cols].fillna(0)

  if assessment_features is not None and not assessment_features.empty:
    df = df.merge(assessment_features, on=join_keys, how="left")
    for count_col in ("assessment_count", "late_submissions"):
      if count_col in df.columns:
        df[count_col] = df[count_col].fillna(0)

  return df


if __name__ == "__main__":
  students_df = pd.read_csv(PROCESSED / "students_labeled.csv")
  student_vle_df = pd.read_csv(RAW / "studentVle.csv")
  vle_meta_df = pd.read_csv(RAW / "vle.csv")

  vle_feats, feature_groups = generate_all_vle_features(
      students_df, student_vle_df, vle_meta_df
  )
  vle_feats.to_csv(
      PROCESSED / "engagement_interaction_features.csv", index=False
  )

  assess_path = PROCESSED / "assessment_features.csv"
  assess_feats = pd.read_csv(assess_path) if assess_path.exists() else None

  combined = build_combined_dataset(students_df, vle_feats, assess_feats)
  combined.to_csv(PROCESSED / "combined_features.csv", index=False)

  print("Saved engagement_interaction_features.csv:", vle_feats.shape)
  print("Saved combined_features.csv:", combined.shape)

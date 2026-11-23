# Feature Group & Validation Methodology

## 1. Feature Group Definitions

To isolate the predictive contribution of different behavioral and academic data sources across course progress cutoffs (`20%`, `40%`, `60%`, `80%`, `100%`), numeric features from `combined_features.csv` are partitioned into three disjoint groups:

### Engagement (`engagement_only`)
Captures general Virtual Learning Environment (VLE) content consumption and overall platform activity volume:
* **Aggregate Activity Features:** `total_vle_clicks`, `vle_event_count`, `active_vle_days`, `engagement_clicks`
* **Activity-Specific VLE Clicks:** All non-collaborative/content `vle_clicks_*` columns (e.g., `vle_clicks_oucontent`, `vle_clicks_resource`, `vle_clicks_homepage`, `vle_clicks_subpage`, `vle_clicks_url`, `vle_clicks_quiz`, etc.)
* **Scope Note:** Reflects logged platform interactions rather than offline study habits (such as reading notes offline, local coding, or in-person study groups).

### Interaction (`interaction_only`)
Captures peer-to-peer, discussion-forum, and collaborative VLE activity:
* **Aggregate Interaction Features:** `interaction_clicks`, `interaction_click_share`
* **Collaborative VLE Activity Types:** `vle_clicks_forumng`, `vle_clicks_oucollaborate`, `vle_clicks_ouwiki`
* **Scope Note:** Represents online forum and collaborative tool usage within the LMS and omits external or face-to-face peer discussions.

### Assessment (`assessment_only`)
Captures graded coursework performance and submission behavior engineered from student assessment records:
* **Assessment Features:** `assessment_count`, `mean_score`, `min_score`, `max_score`, `std_score`, `latest_score`, `first_score`, `score_change`, `weighted_mean_score`, `late_submissions`

### Ablation Subsets (`7` Total)
* **Single-Source:** `engagement_only`, `interaction_only`, `assessment_only`
* **Pairwise Combinations:** `engagement_interaction`, `engagement_assessment`, `interaction_assessment`
* **Full Feature Set:** `all_features` (`engagement` + `interaction` + `assessment`)

---

## 2. Cross-Validation & Data Leakage Prevention

### Per-Cutoff Evaluation (`StratifiedKFold`)
* When evaluating each cutoff (`20%`, `40%`, `60%`, `80%`, `100%`) separately, each row corresponds to a unique student enrollment (`code_module`, `code_presentation`, `id_student`) within that cutoff window.
* Repeated stratified $k$-fold cross-validation (`n_trials=25`, `n_splits=10` for final runs; `n_trials=2`, `n_splits=5` for preliminary runs) preserves multiclass target proportions across folds while applying median imputation and SMOTE strictly inside each training fold.

### Multi-Cutoff Combined Datasets (`StratifiedGroupKFold`)
* If records from multiple cutoffs are pooled into a single training/validation dataset, the same student enrollment appears across multiple cutoff rows.
* Passing an enrollment identifier (`enrollment_id = code_module + "_" + code_presentation + "_" + id_student`) via the `groups` parameter switches splitting to `StratifiedGroupKFold`, ensuring all cutoff records belonging to a given student enrollment stay strictly in either the training fold or the validation fold.
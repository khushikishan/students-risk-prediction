# Feature Engineering Methodology & Tradeoffs

## 1. Overview & Problem Framing

Predicting student academic risk in the Open University Learning Analytics Dataset (OULAD) requires transforming raw, time-stamped event logs (`studentVle.csv`, `vle.csv`, `studentAssessment.csv`, `assessments.csv`, `studentInfo.csv`) into fixed-length tabular feature vectors per student enrollment (`code_module`, `code_presentation`, `id_student`).

To support early-warning intervention while preventing temporal data leakage, features are engineered across **five cumulative course-progress cutoffs**: $20\%$, $40\%$, $60\%$, $80\%$, and $100\%$ of course duration. At each cutoff $c$, only records occurring on or before the cutoff day threshold are visible to the feature pipeline.

---

## 2. Engineered Feature Families

### A. Engagement Features (General VLE Content & Activity Volume)
Derived from `studentVle.csv` joined with `vle.csv` (`activity_type`). Activity-specific engagement counts and `engagement_clicks` exclude the collaborative tool types, while broad activity totals include all logged VLE records:
* **Aggregate Volume & Consistency:**
  * `total_vle_clicks`: Total clicks across all activity types, including forum and collaborative tools, up to cutoff $c$.
  * `vle_event_count`: Number of logged VLE records across all activity types up to cutoff $c$.
  * `active_vle_days`: Count of distinct course days with at least one VLE click of any type (captures regularity vs. cramming).
  * `engagement_clicks`: Sum of clicks across non-collaborative activity types only.
* **Activity-Type Granularity (`vle_clicks_<activity_type>`):**
  * Individual click totals per VLE resource category (e.g., `vle_clicks_oucontent`, `vle_clicks_resource`, `vle_clicks_homepage`, `vle_clicks_subpage`, `vle_clicks_url`, `vle_clicks_quiz`, `vle_clicks_glossary`, `vle_clicks_dataplus`, etc.).

### B. Interaction Features (Peer & Collaborative Activity)
Isolated from general VLE clicks to capture social and collaborative engagement within the LMS:
* **Aggregate Interaction Metrics:**
  * `interaction_clicks`: Sum of clicks on collaborative/discussion tools (`forumng`, `oucollaborate`, `ouwiki`).
  * `interaction_click_share`: Ratio of `interaction_clicks` to `total_vle_clicks` (normalizes social participation relative to overall platform usage).
* **Tool-Specific Counts:**
  * `vle_clicks_forumng`, `vle_clicks_oucollaborate`, `vle_clicks_ouwiki`.

### C. Assessment Features (Graded Mastery & Submission Timeliness)
Derived from `studentAssessment.csv` joined with `assessments.csv` (`date`, `weight`, `assessment_type`), filtered strictly to assessments due/submitted on or before cutoff $c$:
* **Performance Summary & Extremes:** `mean_score`, `min_score`, `max_score`, `std_score`, `weighted_mean_score` (incorporating assessment weights).
* **Temporal Trajectory:** `first_score`, `latest_score`, `score_change` (`latest_score - first_score`), and `assessment_count` (completed assessments up to cutoff $c$).
* **Submission Behavior:** `late_submissions` (count of assessments submitted after the scheduled deadline `date`).

---

## 3. Key Feature Engineering Tradeoffs

### 1. Intentional Granularity: Cumulative Aggregates vs. Atomic Click Streams
* **Decision:** Aggregate raw daily VLE clicks into cutoff-level summary totals, active day counts, and per-activity-type click counts rather than modeling raw event sequences or keystroke/page-transition logs.
* **Tradeoff:**
  * *Gain:* Produces a compact, fixed-width tabular representation compatible with tree ensembles (Random Forest, XGBoost), reduces noise from erratic single-day click spikes, and keeps training tractable across 32,560 enrollments per cutoff.
  * *Loss:* Discards fine-grained temporal ordering and intra-session dynamics (e.g., whether a student clicks a resource *before* vs. *after* failing a quiz, or dwell time per page).

### 2. Defining Engagement vs. Interaction From Click Logs Alone
* **Decision:** Separate activity-specific VLE click features into content/platform consumption (`engagement`) versus forum/wiki/collaborate tools (`interaction`), and compute `interaction_click_share`.
* **Tradeoff:**
  * *Gain:* Supports ablation comparisons (`engagement_only` vs. `interaction_only`) of content-specific activity and peer/forum tools.
  * *Caveat:* `total_vle_clicks`, `vle_event_count`, and `active_vle_days` aggregate all activity types, so the engagement-only group still carries some collaborative activity signal. The groups separate activity-specific columns, but they are not fully independent measures.
  * *Loss:* OULAD logs record *clicks*, not semantic actions (e.g., reading a forum thread vs. writing a substantive reply, or passive scrolling vs. active problem solving). Furthermore, LMS logs omit offline study (reading physical textbooks, local coding in an IDE, in-person peer help), so low click counts can reflect either disengagement or efficient offline study.

### 3. Temporal Cutoff Filtering vs. Early-Stage Missingness
* **Decision:** Enforce strict cutoff filtering (only assessments and VLE events $\le$ cutoff day are included) rather than imputing future course information or dropping students without early assessments.
* **Tradeoff:**
  * *Gain:* Eliminates future-information leakage, ensuring evaluation mirrors realistic early-warning deployment at $20\%$ or $40\%$ of the semester.
  * *Loss:* At early cutoffs ($20\%$), many modules have few or zero graded assignments due yet, resulting in structural `NaN` values for `mean_score`, `std_score`, and `score_change`—requiring careful median imputation inside each training fold and reducing early assessment coverage.

### 4. Handling Missing Values & Zero-Activity Encodings
* **Decision:** Fill unrecorded VLE activity categories with `0` (since absence of clicks in a log indicates zero recorded platform events), while leaving unattempted/unscheduled assessment statistics as `NaN` to be imputed via training-fold median (`SimpleImputer(strategy="median")`) during cross-validation.
* **Tradeoff:**
  * *Gain:* Distinguishes "zero clicks on a forum" (observed non-use) from "no assessment grade available yet" (unobserved mastery), and fitting the imputer strictly inside each training fold prevents validation leakage.
  * *Loss:* Median imputation compresses variance for students missing early assessments and can slightly attenuate tree split thresholds at the $20\%$ cutoff.

### 5. Multiclass Target Filtering (`-1` Exclusion) & Class Imbalance
* **Decision:** Exclude records labeled `-1` (invalid/unmapped outcomes for the multiclass task) and apply SMOTE oversampling strictly inside each training fold after imputation.
* **Tradeoff:**
  * *Gain:* Preserves genuine multiclass risk distinctions (rather than collapsing nuanced outcomes into a single binary flag) while keeping validation folds untouched and representative of true class distributions.
  * *Loss:* Removing `-1` rows slightly reduces sample size, and synthetic SMOTE interpolation in high-dimensional tabular space increases per-fold training runtime and can introduce synthetic boundary points if minority classes are sparse.

### 6. Pooling Cutoffs vs. Independent Per-Cutoff Datasets
* **Decision:** Train and evaluate models independently at each cutoff ($20\%, 40\%, 60\%, 80\%, 100\%$) using `StratifiedKFold`, while supporting `StratifiedGroupKFold` grouped by student enrollment if cutoffs are later pooled.
* **Tradeoff:**
  * *Gain:* Per-cutoff models adapt split thresholds to the exact feature availability at each stage of the term (e.g., relying more on `engagement` when `assessment` is sparse at $20\%$, and shifting weight to `assessment` by $60\%\text{--}100\%$).
  * *Loss:* Training 5 separate cutoff models multiplies experiment steps ($5 \times 7 \times 2 = 70$ ablation runs) compared to a single unified model, and requires strict enrollment-level grouping if cutoffs are ever combined.

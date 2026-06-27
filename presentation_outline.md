# Presentation Outline — LA Crime Classification

Section-level outline for the final team presentation, following the lecturer's
prescribed structure (deck 9 "Possible Outline") and the required content
(motivation → methods → key results → conclusion). Target **~12–15 slides /
10–12 min**. Fill in every `[placeholder]` before submitting.

> Course: Python & Advanced Data Science (TUM School of Management).
> Grading weight: presentation = 30%. Attendance obligatory.
> Rule: **put each contributor's name on the section(s) they present/built.**
> See `course_content/CLAUDE.md` for full formalities and lecturer expectations.

---

## 1. Title / Author / Affiliation — *1 slide*
- Project title (e.g. "Predicting Crime Categories in Los Angeles — A Multi-Class
  Classification Study").
- `[Team Name]`; members `[Member 1]`, `[Member 2]`, `[Member 3]`, `[Member 4]`.
- Affiliation: TUM School of Management — Python & Advanced Data Science, SS26.
- Dataset credit: LA crime data 2020–present (data.lacity.org).

## 2. Outline / Agenda — *1 slide*
- Background → Methods → Key Results → Conclusion (mirror this document).

## 3. Background — *3–4 slides*  ·  contributor: `[name]`
- **Introduction:** the dataset (LA crime records 2020–present; ~hundreds of
  thousands of rows); the **5-class target `category`** — Violent, Property,
  Sexual Assault, Vehicle, Other — collapsed from 140+ raw crime codes.
  Problem statement: *predict the crime category from time, location, victim
  demographics, and premise features.*
- **Motivation (why it matters):** public-safety relevance — patterns by time/
  place/victim profile inform resource allocation; a real, messy, large
  real-world dataset; multi-class + class-imbalance makes it a genuine ML
  challenge (not a toy binary problem).
- **Methods / approach (our framework):** present as a **CRISP-DM pipeline**
  (Data Understanding → Preparation → Modelling → Evaluation):
  - cleaning (`data_cleaning.py`): fixed bad timestamps/ages, `(0,0)` GPS,
    duplicates; expanded sex/descent codes; built the `category` target — *say
    why each fix*.
  - EDA (`dataset_overview.py`): visualization, correlation, interactive map.
  - feature engineering (`feature_engineering.py`): date/time parts, age groups,
    weapon/geo flags, grouped premises; dropped leakage (crime-code) columns.
  - **reproducible, fair comparison**: one shared 80/20 stratified split + one
    seed (`config.py`) reused by every model; categoricals kept as labels, each
    model does its own encoding.
- One slide = the pipeline diagram (cleaning → EDA → features → 5 models →
  comparison).

## 4. Results — *4–5 slides* (key insights only, NOT all results)  ·  contributor: `[name]`
- **2–3 EDA highlights** (pick the most striking; figures in `outputs/eda_outputs/`):
  e.g. temporal patterns (`03_temporal_patterns.png`), area distribution
  (`05_area_distribution.png`), the interactive map (`heatmap.html`),
  victim profiles (`06_victim_profiles.png`).
- **Model comparison table** — the 5 models, baseline vs tuned. Pull live numbers
  from `outputs/model_comparison_summary.csv`. Current standing:

  | Model | Accuracy | Macro-F1 | Macro ROC-AUC |
  |---|---|---|---|
  | Logistic Regression (tuned) | 0.65 | 0.40 | 0.81 |
  | Decision Tree (tuned) | 0.65 | 0.46 | 0.78 |
  | **XGBoost (tuned)** | **0.69** | **0.51** | **0.86** |
  | Random Forest | *TBD* | *TBD* | *TBD* |
  | Neural Network (PyTorch) | *TBD* | *TBD* | *TBD* |

  → **XGBoost is the best model so far.** (Update RF + NN before the talk.)
- **Why macro-F1 / ROC-AUC, not just accuracy:** the classes are imbalanced, so
  accuracy is misleading — emphasize macro-averaged metrics and per-class recall
  on the rare classes (Sexual Assault, Vehicle).
- **Tuning impact:** baseline → Grid Search CV lift (e.g. Decision Tree macro-F1
  0.43 → 0.47; XGBoost 0.48 → 0.51) — shows tuning helped, as the brief requires.
- **Explainability:** one XGBoost **SHAP** slide (`outputs/xgboost_output/
  shap_summary_bar.png` / a beeswarm) — which features drive predictions.
- Optional: one confusion-matrix slide for the best model to show *where* it
  confuses classes.

## 5. Conclusion — *1 slide*  ·  contributor: `[name]`
- What worked: tree ensembles (XGBoost) beat linear/single-tree baselines;
  shared-split framework made the comparison fair; tuning gave consistent lifts.
- Limitations: class imbalance hurts rare-class recall; geospatial/temporal
  signal only partly captured; no MO-code feature yet.
- Next steps: finish Random Forest + NN; address imbalance (class weights /
  resampling); revisit MO codes as features.

## 6. Backup slides — *as many as needed, shown only if asked*
- Full `model_comparison_summary.csv` (all baseline + tuned rows).
- Per-model confusion matrices, ROC curves, per-class precision/recall/F1.
- Feature list & engineering details; tuning grids (`grid_search_results.csv`).
- LogReg statistical diagnostics (VIF, MNLogit significance) — ties to the
  multicollinearity content from the ML lecture.
- Data-cleaning decisions table (what/why).

---

### Pre-submission checklist
- [ ] Replace all `[placeholders]` (team + member names, contributor tags).
- [ ] Refresh the results table from `outputs/model_comparison_summary.csv`.
- [ ] **Build the 5th model (Random Forest)** — required: ≥5 models incl. an
      ensemble; RF is configured (`config.py`) but `model_random_forest.py`
      doesn't exist yet. Add RF + NN numbers to the table.
- [ ] Every section labelled with the contributing member's name.
- [ ] Keep main deck to key results; push detail to backup.

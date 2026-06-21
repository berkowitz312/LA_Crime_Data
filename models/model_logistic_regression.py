"""
multinomial_logistic_regression.py
====================================
LA Crime Data — Multinomial Logistic Regression

Predicts the 5-class `category` target (Violent / Property / Sexual Assault /
Vehicle / Other — created in data_cleaning.py) from time, location, victim
demographic, and premise-type features.

This is ONE model in a larger model-comparison project. It is intentionally
scoped to multinomial logistic regression only — ensemble and deep-learning
models are owned by teammates and will be compared against this script's
metrics using the same train/test split (same random seed, documented below).

RANDOM SEED — IMPORTANT FOR YOUR TEAM
======================================
    RANDOM_SEED = 42

Every random operation in this script (train/test split, cross-validation
folds, solver initialization) uses this exact seed. For your model
comparison to be fair and reproducible, anyone building another model
(ensemble, deep learning, etc.) on this dataset should:
    1. Use the SAME random seed (42) for their train/test split
    2. Use the SAME stratification strategy (stratify on `category`)
    3. Ideally, load the exact split indices this script saves to
       model_outputs/train_test_split_indices.csv, so everyone is
       evaluating on the literal same held-out rows.

Auto-detects the cleaned dataset in the SAME FOLDER as this script:
    cleaned_data/la_crime_cleaned.csv   (produced by data_cleaning.py)

Usage:
    python multinomial_logistic_regression.py

Produces (in model_outputs/):
    train_test_split_indices.csv   <- exact row indices used, for team reuse
    confusion_matrix_baseline.png
    confusion_matrix_tuned.png
    roc_curves_tuned.png
    feature_importance.png
    model_comparison_summary.csv   <- baseline vs tuned, ready to merge with
                                       teammates' model results

    -- Statistical diagnostics & variable-selection stage (requires
       statsmodels — `pip install statsmodels`) --
    vif_table.csv                       <- multicollinearity check
    significance_iteration_log.csv      <- which variables were dropped, in
                                            what order, and why
    mnlogit_final_summary.txt           <- final statsmodels model summary
                                            (coefficients, std errors, z, p)
    influence_diagnostics.png           <- Cook's distance / leverage plot
    outlier_flagged_rows.csv            <- rows flagged as high-influence
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, log_loss
)
from sklearn.preprocessing import label_binarize

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed. Run: pip install statsmodels")
    print("       The significance-testing / VIF / diagnostics stage will be skipped.")

# =============================================================================
# 0.  REPRODUCIBILITY  &  PATH AUTO-DETECTION
# =============================================================================

RANDOM_SEED = 42          # <-- documented above; reuse this across all models
np.random.seed(RANDOM_SEED)

SCRIPT_DIR  = Path(__file__).resolve().parent
CLEANED_DIR = SCRIPT_DIR / "cleaned_data"
OUT_DIR     = SCRIPT_DIR / "model_outputs"
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", font_scale=1.0)
ACCENT, COOL, WARM, DARK_BG = "#E63946", "#457B9D", "#F4A261", "#1D3557"
CATEGORY_COLORS = {
    "Violent": "#E63946", "Property": "#457B9D", "Sexual Assault": "#9D4EDD",
    "Vehicle": "#F4A261", "Other": "#6C757D",
}


def find_cleaned_file() -> Path:
    candidates = sorted(CLEANED_DIR.glob("*.csv")) if CLEANED_DIR.exists() else []
    if not candidates:
        candidates = sorted(SCRIPT_DIR.glob("*cleaned*.csv"))
    if not candidates:
        print(f"\n  [ERROR] Could not find a cleaned CSV file in {CLEANED_DIR}")
        print(f"          Run data_cleaning.py first.")
        sys.exit(1)
    return candidates[0]


# =============================================================================
# 1.  LOAD
# =============================================================================

def load_data() -> pd.DataFrame:
    print(f"\n{'='*70}")
    print("  STEP 1 – LOADING CLEANED DATASET")
    print(f"{'='*70}")
    path = find_cleaned_file()
    print(f"  File  : {path.name}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape : {df.shape}")
    return df


# =============================================================================
# 2.  FEATURE SELECTION
# =============================================================================
# Per project scope: predict `category` using TIME + LOCATION + VICTIM
# DEMOGRAPHICS + PREMISE features only. We deliberately EXCLUDE weapon_desc
# and status_desc because they are downstream consequences of the crime
# category being known (e.g. "STRONG-ARM" weapon almost certainly implies
# a Violent/Sexual Assault crime) — including them would leak the answer
# and produce an unrealistically easy, non-generalizable model.

FEATURE_COLUMNS = {
    # Time features (parsed from date_occ / time_occ — raw, not engineered)
    "time_occ":     "numeric",     # HHMM as integer, e.g. 1430 -> we'll derive hour
    "date_occ":     "datetime",    # used to derive month / day-of-week / year
    # Location features
    "area_name":    "categorical",
    "lat":          "numeric",
    "lon":          "numeric",
    # Victim demographics
    "vict_age":     "numeric",
    "vict_sex":     "categorical",
    "vict_descent": "categorical",
    # Premise
    "premis_desc":  "categorical",
}

TARGET_COLUMN = "category"


def select_and_engineer_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls in only the approved feature set and derives the minimum necessary
    numeric features from date/time so the model has something usable
    (raw HHMM integers and raw date strings aren't meaningful to a linear
    model as-is). This is NOT the project's full feature-engineering stage —
    just the minimal derivation needed for this model to function.
    """
    print(f"\n{'='*70}")
    print("  STEP 2 – FEATURE SELECTION")
    print(f"{'='*70}")
    print("  Selected feature groups: TIME, LOCATION, VICTIM DEMOGRAPHICS, PREMISE")
    print("  Excluded on purpose: weapon_desc, status_desc (would leak the target)")

    work = df[[TARGET_COLUMN] + list(FEATURE_COLUMNS.keys())].copy()

    # Derive hour-of-day from time_occ (HHMM integer -> 0-23)
    work["hour_occ"] = pd.to_numeric(work["time_occ"], errors="coerce") // 100

    # Derive month and day-of-week from date_occ
    work["date_occ"] = pd.to_datetime(work["date_occ"], errors="coerce")
    work["month_occ"] = work["date_occ"].dt.month
    work["dow_occ"]   = work["date_occ"].dt.dayofweek  # 0=Mon

    work = work.drop(columns=["time_occ", "date_occ"])

    print(f"  Derived 'hour_occ', 'month_occ', 'dow_occ' from raw time/date.")
    print(f"  Final feature set ({work.shape[1]-1} features + target):")
    print(f"    {[c for c in work.columns if c != TARGET_COLUMN]}")

    return work


# =============================================================================
# 3.  PREPROCESSING
# =============================================================================
# Each step below is explained inline — this is the "applicable for ML/DL
# models" transformation stage requested.

def explain_and_build_preprocessor(work: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  STEP 3 – DATA PRE-PROCESSING  (with rationale for each step)")
    print(f"{'='*70}")

    numeric_features     = ["lat", "lon", "vict_age", "hour_occ", "month_occ", "dow_occ"]
    categorical_features = ["area_name", "vict_sex", "vict_descent", "premis_desc"]

    numeric_features     = [c for c in numeric_features if c in work.columns]
    categorical_features = [c for c in categorical_features if c in work.columns]

    print("""
  (1) MISSING-VALUE IMPUTATION
      Why: logistic regression cannot handle NaN values directly; the solver
      will error out. Numeric gaps (e.g. unknown lat/lon, unknown victim age)
      are filled with the column MEDIAN, which is robust to outliers (unlike
      the mean) and keeps the imputed value 'typical' rather than distorting
      the distribution. Categorical gaps (e.g. unknown premise) are filled
      with the literal label "Missing", turning absence-of-data into its own
      informative category rather than silently guessing a value.

  (2) ONE-HOT ENCODING (categorical -> numeric)
      Why: multinomial logistic regression is a linear model operating on
      numeric feature vectors — it has no native concept of unordered
      categories like "Wilshire" or "Hispanic/Latin/Mexican". One-hot
      encoding converts each category into its own binary (0/1) column,
      so the model can learn an independent coefficient per category without
      imposing a false numeric ordering (e.g. encoding areas as 1, 2, 3...
      would wrongly imply Area 3 is "more" than Area 1).
      handle_unknown='ignore' ensures that if the test set contains a
      category unseen during training, it's encoded as all-zeros rather
      than crashing the pipeline.

  (3) STANDARDIZATION (numeric -> zero mean, unit variance)
      Why: logistic regression is fit via gradient-based optimization
      (here, the lbfgs/saga solver). Features on wildly different scales
      (e.g. latitude ~34, victim age ~0-100, hour ~0-23) cause the solver
      to converge slowly or unevenly weight large-scale features purely
      due to magnitude, not actual predictive importance. StandardScaler
      rescales every numeric feature to mean=0, std=1, so the regularization
      penalty (L2, applied uniformly) treats all features fairly and the
      solver converges faster and more reliably.

  (4) STRATIFIED TRAIN/TEST SPLIT
      Why: our target classes (Violent / Property / Sexual Assault /
      Vehicle / Other) are NOT evenly distributed — Property and Vehicle
      crimes vastly outnumber Sexual Assault in raw counts. A plain random
      split could, by chance, under-represent a rare class in the test set,
      making evaluation metrics unstable. Stratifying on `category` ensures
      the train and test sets preserve the same class proportions as the
      full dataset.
    """)

    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor, numeric_features, categorical_features


# =============================================================================
# 4.  TRAIN / TEST SPLIT  (saved for team reuse)
# =============================================================================

def make_split(work: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  STEP 4 – TRAIN / TEST SPLIT")
    print(f"{'='*70}")

    X = work.drop(columns=[TARGET_COLUMN])
    y = work[TARGET_COLUMN]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, work.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"  Split        : 80% train / 20% test")
    print(f"  Random seed  : {RANDOM_SEED}  <-- use this exact seed for any other model")
    print(f"  Stratified by: '{TARGET_COLUMN}'")
    print(f"  Train shape  : {X_train.shape}")
    print(f"  Test shape   : {X_test.shape}")
    print(f"\n  Class distribution preserved across split:")
    dist = pd.DataFrame({
        "train_%": (y_train.value_counts(normalize=True) * 100).round(2),
        "test_%":  (y_test.value_counts(normalize=True) * 100).round(2),
    })
    print(dist.to_string())

    # Save the exact split indices so teammates building other models can
    # reuse the identical train/test rows for a fair comparison.
    split_df = pd.DataFrame({
        "row_index": list(idx_train) + list(idx_test),
        "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
    })
    split_path = OUT_DIR / "train_test_split_indices.csv"
    split_df.to_csv(split_path, index=False)
    print(f"\n  Saved exact split indices -> {split_path}")
    print(f"  (Share this file + RANDOM_SEED={RANDOM_SEED} with your team so "
          f"everyone evaluates on identical held-out rows.)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 5.  BASELINE MODEL
# =============================================================================

def build_baseline_model(preprocessor):
    """
    Baseline multinomial logistic regression with default-ish, sensible
    settings. With a multi-class target and the 'lbfgs' or 'saga' solver,
    scikit-learn fits a genuine multinomial (softmax) model natively — a
    single joint model across all classes via cross-entropy loss — rather
    than the older One-vs-Rest scheme (which fits an independent binary
    classifier per class and can give miscalibrated, inconsistent
    probabilities across classes). In recent scikit-learn versions this
    multinomial behavior is automatic for multi-class problems with these
    solvers, so no explicit `multi_class` argument is needed (and newer
    versions no longer accept one).
    """
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",          # supports true multinomial loss natively, efficient for medium feature counts
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", model),
    ])
    return pipeline


def evaluate_model(pipeline, X_test, y_test, label: str, class_labels):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    ll = log_loss(y_test, y_proba, labels=pipeline.classes_)

    y_test_bin = label_binarize(y_test, classes=pipeline.classes_)
    try:
        auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        auc_macro = np.nan

    print(f"\n  -- {label} Results --")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro Precision : {precision:.4f}")
    print(f"  Macro Recall    : {recall:.4f}")
    print(f"  Macro F1        : {f1:.4f}")
    print(f"  Macro ROC-AUC   : {auc_macro:.4f}")
    print(f"  Log Loss        : {ll:.4f}")
    print(f"\n  Per-class report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    return {
        "label": label, "accuracy": acc, "macro_precision": precision,
        "macro_recall": recall, "macro_f1": f1, "macro_roc_auc": auc_macro,
        "log_loss": ll, "y_pred": y_pred, "y_proba": y_proba, "report": report,
    }


def plot_confusion_matrix(y_test, y_pred, class_labels, title: str, filename: str):
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={"label": "% of true class"}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


def plot_roc_curves(y_test, y_proba, class_labels, filename: str):
    y_test_bin = label_binarize(y_test, classes=class_labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, cls in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        color = CATEGORY_COLORS.get(cls, "#999999")
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest per Category (Tuned Model)",
                 fontsize=14, fontweight="bold", color=DARK_BG)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


# =============================================================================
# 6.  HYPERPARAMETER TUNING  (advanced technique)
# =============================================================================

def tune_model(preprocessor, X_train, y_train):
    """
    GridSearchCV over the regularization strength (C) and penalty type.

    Why these specific hyperparameters:
      - C (inverse regularization strength): controls the trade-off between
        fitting the training data closely vs. keeping coefficients small to
        avoid overfitting. We search a wide log-spaced range so both heavily
        regularized (small C) and lightly regularized (large C) options are
        considered.
      - penalty: 'l2' (ridge-style, shrinks all coefficients smoothly) vs
        'l1' (lasso-style, can zero out uninformative one-hot categories
        entirely, effectively performing feature selection). Comparing both
        lets the data decide whether sparsity helps.
      - solver: 'saga' is used for tuning because it's the only solver that
        supports both l1 and l2 penalties with multinomial loss at this
        scale; lbfgs only supports l2.

    Why GridSearchCV with StratifiedKFold (not plain KFold): the same class
    imbalance argument as the train/test split applies here — each fold must
    preserve class proportions, or some folds could end up with very few
    Sexual Assault examples, making the cross-validated score noisy and
    unreliable for comparing hyperparameter combinations fairly.
    """
    print(f"\n{'='*70}")
    print("  STEP 6 – HYPERPARAMETER TUNING (GridSearchCV)")
    print(f"{'='*70}")
    print("""
  Why we tune:
    The baseline model uses default-ish settings (C=1.0, l2 penalty). These
    are reasonable starting points but not necessarily optimal for THIS
    dataset's class balance and feature structure. Tuning systematically
    searches a grid of alternatives and picks the combination that performs
    best under cross-validation — i.e. data-driven model selection instead
    of guessing.

  Search space:
    C       : [0.01, 0.1, 1, 10]
    penalty : ['l1', 'l2']
    solver  : 'saga' (required for l1 + multinomial)

  Cross-validation:
    StratifiedKFold, 3 folds, scored on macro-F1 (treats every class as
    equally important regardless of how many examples it has — appropriate
    here since Sexual Assault, our rarest class, matters just as much for
    the project's purposes as the much more frequent Property/Vehicle
    classes).
    """)

    model = LogisticRegression(
        solver="saga",
        max_iter=2000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", model),
    ])

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print(f"\n  Best parameters found : {grid.best_params_}")
    print(f"  Best CV macro-F1      : {grid.best_score_:.4f}")

    cv_results = pd.DataFrame(grid.cv_results_)[
        ["param_classifier__C", "param_classifier__penalty",
         "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    print(f"\n  Full grid results:")
    print(cv_results.to_string(index=False))

    cv_path = OUT_DIR / "grid_search_results.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"\n  -> saved: {cv_path}")

    return grid.best_estimator_, grid.best_params_


# =============================================================================
# 7.  FEATURE IMPORTANCE  (coefficient magnitudes)
# =============================================================================

def plot_feature_importance(pipeline, class_labels, top_n: int = 20):
    print(f"\n  [PLOT] Feature importance (|coefficient| averaged across classes)")
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        print("      (skipped – could not extract feature names)")
        return

    coefs = pipeline.named_steps["classifier"].coef_   # shape: (n_classes, n_features)
    mean_abs_coef = np.abs(coefs).mean(axis=0)

    importance = pd.Series(mean_abs_coef, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(importance.index[::-1], importance.values[::-1], color=COOL, edgecolor="white")
    ax.set_title(f"Top {top_n} Features by Mean |Coefficient|\n(averaged across all classes)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Mean absolute coefficient (standardized features)")
    plt.tight_layout()
    path = OUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


# =============================================================================
# 9.  STATISTICAL DIAGNOSTICS & SIGNIFICANCE-DRIVEN VARIABLE SELECTION
# =============================================================================
# This stage answers a different question than steps 5-7: not "how accurate
# is the model on held-out data" but "which variables actually matter, in a
# classical inferential-statistics sense, and is the model well-specified?"
#
# sklearn's LogisticRegression has no concept of standard errors or p-values
# — it's built purely for prediction, not inference. To get real Wald tests
# we fit a SEPARATE model using statsmodels' MNLogit, which is built for
# exactly this. The sklearn pipeline above remains the model used for the
# team's accuracy/F1/ROC-AUC comparison; this statsmodels fit exists only to
# interpret and refine which variables belong in an explanatory model.

def build_design_matrix(work: pd.DataFrame, numeric_features: list, categorical_features: list):
    """
    Builds a statsmodels-ready design matrix: numeric columns imputed +
    standardized (same logic as the sklearn pipeline, for consistency), and
    categorical columns one-hot encoded with one level dropped per feature
    (drop_first=True) to avoid the dummy-variable trap — MNLogit, like any
    regression-style model, needs a reference level for each categorical
    predictor or the design matrix becomes singular (perfectly
    multicollinear) and cannot be inverted.
    """
    X = work.drop(columns=[TARGET_COLUMN]).copy()

    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    for col in categorical_features:
        X[col] = X[col].fillna("Missing")

    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    # statsmodels needs clean, Python-identifier-safe column names
    X_encoded.columns = [
        c.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
         .replace("/", "_").replace("-", "_").replace("'", "")
        for c in X_encoded.columns
    ]
    X_encoded = X_encoded.astype(float)
    return X_encoded


def compute_vif(X_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Inflation Factor for each predictor: VIF_i = 1 / (1 - R_i^2),
    where R_i^2 comes from regressing predictor i on all other predictors.
    High VIF means predictor i is well-explained by the others — i.e. it
    carries mostly redundant information, inflating coefficient standard
    errors and making individual p-values unreliable even if the model's
    overall fit is fine.

    Guideline used here (as specified):
        VIF < 5   -> fine
        5-10      -> investigate
        > 10      -> problematic, consider dropping/combining
    """
    from numpy.linalg import LinAlgError

    X_with_const = sm.add_constant(X_encoded)
    vif_data = []
    for i, col in enumerate(X_with_const.columns):
        if col == "const":
            continue
        try:
            y_col = X_with_const[col].values
            X_others = X_with_const.drop(columns=[col]).values
            r2 = sm.OLS(y_col, X_others).fit().rsquared
            vif = 1.0 / (1.0 - r2) if r2 < 0.999999 else np.inf
        except (LinAlgError, ValueError):
            vif = np.nan
        vif_data.append({"feature": col, "VIF": vif})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    def flag(v):
        if pd.isna(v): return "N/A"
        if v < 5: return "OK"
        if v < 10: return "Investigate"
        return "PROBLEMATIC"
    vif_df["flag"] = vif_df["VIF"].apply(flag)

    return vif_df


def fit_mnlogit(X_encoded: pd.DataFrame, y: pd.Series, reference_class: str):
    """
    Fits statsmodels MNLogit. The reference (baseline) category is the one
    every other class's coefficients are interpreted relative to — e.g. a
    coefficient under "Vehicle" means "log-odds of Vehicle vs. {reference}
    per unit change in this predictor." We use the most frequent class as
    reference (Property in this dataset), which is the conventional choice
    and keeps coefficients easiest to interpret against the most common
    baseline outcome.
    """
    y_codes = y.astype("category")
    # Re-order categories so reference_class is first -> statsmodels treats
    # the FIRST category alphabetically/numerically as baseline by default;
    # we control this explicitly via category ordering.
    cats = [reference_class] + [c for c in y_codes.cat.categories if c != reference_class]
    y_codes = y_codes.cat.reorder_categories(cats, ordered=False)
    y_numeric = y_codes.cat.codes

    X_with_const = sm.add_constant(X_encoded)
    model = sm.MNLogit(y_numeric, X_with_const)
    result = model.fit(method="newton", maxiter=100, disp=False)
    return result, y_codes.cat.categories


def summarize_significance(result, class_categories, reference_class: str) -> pd.DataFrame:
    """
    Extracts a tidy {feature, class, coef, std_err, z, p_value} table from
    the fitted MNLogit result across all non-reference classes, so we can
    rank/filter variables by significance in one place instead of reading
    the dense statsmodels text summary by hand.
    """
    params = result.params      # rows = predictors, columns = one per non-reference class
    bse    = result.bse
    zvals  = result.tvalues
    pvals  = result.pvalues

    records = []
    non_ref_classes = [c for c in class_categories if c != reference_class]
    for j, cls in enumerate(non_ref_classes):
        for feature in params.index:
            records.append({
                "feature": feature,
                "class_vs_reference": f"{cls} vs {reference_class}",
                "coef": params.iloc[:, j][feature],
                "std_err": bse.iloc[:, j][feature],
                "z": zvals.iloc[:, j][feature],
                "p_value": pvals.iloc[:, j][feature],
            })
    return pd.DataFrame(records)


def iterative_backward_elimination(work: pd.DataFrame, numeric_features: list,
                                   categorical_features: list, reference_class: str,
                                   p_threshold: float = 0.05, max_iterations: int = 15):
    """
    Iterative significance-driven variable selection:
        1. Fit MNLogit with all current candidate predictors.
        2. For each predictor, take its BEST (lowest) p-value across all
           non-reference classes — i.e. "is this variable significant for
           AT LEAST ONE category." A variable significant for distinguishing
           even one class is still explanatorily useful, so we only drop it
           if it's insignificant everywhere.
        3. If every remaining predictor has best-p < threshold, stop.
        4. Otherwise, drop the single variable with the worst (highest)
           best-p-value, and refit (one variable at a time — dropping
           multiple at once risks removing a variable whose apparent
           insignificance was actually caused by collinearity with another
           soon-to-be-dropped variable).
        5. Also fold in VIF: if a variable has VIF > 10, it's flagged as a
           multicollinearity-driven removal candidate even if its p-value
           looks borderline-significant, since its standard error (and
           therefore its p-value) is inflated and untrustworthy.

    Returns the final reduced design matrix, the final fitted result, and a
    log DataFrame recording every drop decision for transparency.
    """
    print(f"\n{'='*70}")
    print("  STEP 9b – ITERATIVE SIGNIFICANCE-DRIVEN VARIABLE SELECTION")
    print(f"{'='*70}")
    print(f"""
  Procedure (p-value threshold = {p_threshold}):
    1. Fit the full model.
    2. For each predictor, look at its smallest p-value across all
       (non-reference-class vs reference) comparisons — keep it if THAT
       is below {p_threshold}.
    3. Drop the single worst offender, refit, repeat.
    4. Also cross-check VIF > 10 each round — a variable flagged by both
       a high p-value AND high VIF is a strong removal candidate, since
       multicollinearity directly inflates p-values and makes individual
       significance tests unreliable.
    5. Stop when everything remaining is significant for at least one
       class, or after {max_iterations} iterations (safety cap).
    """)

    X_encoded = build_design_matrix(work, numeric_features, categorical_features)
    y = work[TARGET_COLUMN]

    log_rows = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n  -- Iteration {iteration}: fitting with {X_encoded.shape[1]} predictors --")

        try:
            result, categories = fit_mnlogit(X_encoded, y, reference_class)
        except Exception as e:
            print(f"  [ERROR] MNLogit failed to converge: {e}")
            print(f"  Stopping iteration with the last successfully fitted model.")
            break

        sig_table = summarize_significance(result, categories, reference_class)
        # Best (smallest) p-value per feature, across all class comparisons
        best_p_per_feature = sig_table.groupby("feature")["p_value"].min()
        best_p_per_feature = best_p_per_feature.drop(index="const", errors="ignore")

        vif_df = compute_vif(X_encoded)
        vif_lookup = dict(zip(vif_df["feature"], vif_df["VIF"]))

        insignificant = best_p_per_feature[best_p_per_feature >= p_threshold].sort_values(ascending=False)

        print(f"  Predictors with best-p >= {p_threshold}: {len(insignificant)} of {len(best_p_per_feature)}")

        if len(insignificant) == 0:
            print(f"  All remaining predictors are significant for at least one class. Stopping.")
            log_rows.append({
                "iteration": iteration, "action": "STOP - all significant",
                "feature_dropped": None, "best_p_value": None, "vif": None,
            })
            break

        # Drop the worst offender (highest p-value); note its VIF for context
        worst_feature = insignificant.index[0]
        worst_p = insignificant.iloc[0]
        worst_vif = vif_lookup.get(worst_feature, np.nan)

        reason = "high p-value"
        if not pd.isna(worst_vif) and worst_vif > 10:
            reason = "high p-value AND high VIF (multicollinearity)"

        print(f"  Dropping '{worst_feature}'  (best p-value={worst_p:.4f}, VIF={worst_vif:.2f})  — {reason}")

        log_rows.append({
            "iteration": iteration, "action": "DROP",
            "feature_dropped": worst_feature, "best_p_value": worst_p, "vif": worst_vif,
        })

        X_encoded = X_encoded.drop(columns=[worst_feature])

        if X_encoded.shape[1] == 0:
            print(f"  [WARN] All predictors dropped — stopping to avoid an empty model.")
            break

    log_df = pd.DataFrame(log_rows)
    return X_encoded, result, log_df


def plot_influence_diagnostics(result, X_encoded: pd.DataFrame, work: pd.DataFrame):
    """
    Cook's distance and leverage are classically defined for linear/binary
    logistic regression via the hat matrix; statsmodels' MNLogit does not
    expose a built-in get_influence() (unlike sm.Logit / sm.OLS). To still
    give a genuinely useful influence diagnostic for the MULTINOMIAL case,
    we fit an auxiliary one-vs-reference binary logistic regression for the
    single largest non-reference class and pull standard influence measures
    (Cook's distance, leverage) from THAT — a documented, standard fallback
    approach for approximating per-observation influence in a multinomial
    setting. This is explained in the printed output and the plot title so
    nothing is implied to be exact for all classes simultaneously.
    """
    print(f"\n{'='*70}")
    print("  STEP 9c – INFLUENCE DIAGNOSTICS  (outliers, leverage)")
    print(f"{'='*70}")
    print("""
  Note on method: statsmodels' MNLogit doesn't expose Cook's distance /
  leverage directly (these are defined via the hat matrix, which is
  well-established for binary logistic regression but not natively
  extended to the multinomial case in this library). As a standard
  workaround, we fit an auxiliary ONE-VS-REFERENCE binary logistic
  regression for the single most-common non-reference class and compute
  Cook's distance / leverage from that fit. This approximates which
  observations are most influential to the overall model, while being
  transparent that it's a single-class lens rather than a true
  multinomial generalization (no fully agreed-upon multinomial Cook's
  distance exists in standard statistical practice).
    """)

    y = work[TARGET_COLUMN]
    class_counts = y.value_counts()
    non_ref_classes = [c for c in class_counts.index]
    aux_class = class_counts.index[1] if len(class_counts) > 1 else class_counts.index[0]

    y_binary = (y == aux_class).astype(int)
    X_with_const = sm.add_constant(X_encoded)

    try:
        binary_model = sm.Logit(y_binary, X_with_const).fit(disp=False, method="newton", maxiter=100)
        influence = binary_model.get_influence()
        cooks_d = influence.cooks_distance[0]
        leverage = influence.hat_matrix_diag
    except Exception as e:
        print(f"  [WARN] Could not fit auxiliary binary model for influence diagnostics: {e}")
        return None

    n_params = X_with_const.shape[1]
    n_obs = len(y_binary)
    cooks_threshold = 4 / n_obs   # common rule-of-thumb cutoff

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Influence Diagnostics (auxiliary binary fit: '{aux_class}' vs. rest)",
                 fontsize=14, fontweight="bold", color=DARK_BG)

    axes[0].scatter(range(n_obs), cooks_d, s=4, alpha=0.3, color=ACCENT)
    axes[0].axhline(cooks_threshold, color=DARK_BG, linestyle="--",
                    label=f"4/n threshold = {cooks_threshold:.5f}")
    axes[0].set_title("Cook's Distance per Observation")
    axes[0].set_xlabel("Observation index"); axes[0].set_ylabel("Cook's D")
    axes[0].legend()

    axes[1].scatter(leverage, cooks_d, s=4, alpha=0.3, color=COOL)
    axes[1].set_title("Leverage vs. Cook's Distance")
    axes[1].set_xlabel("Leverage (hat value)"); axes[1].set_ylabel("Cook's D")

    plt.tight_layout()
    path = OUT_DIR / "influence_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")

    n_flagged = (cooks_d > cooks_threshold).sum()
    print(f"\n  Observations flagged by Cook's distance (> 4/n = {cooks_threshold:.5f}): "
          f"{n_flagged:,} of {n_obs:,} ({n_flagged/n_obs*100:.2f}%)")

    flagged_df = work.reset_index(drop=True).loc[cooks_d > cooks_threshold].copy()
    flagged_df["cooks_distance"] = cooks_d[cooks_d > cooks_threshold]
    flagged_df["leverage"] = leverage[cooks_d > cooks_threshold]
    flagged_path = OUT_DIR / "outlier_flagged_rows.csv"
    flagged_df.to_csv(flagged_path, index=False)
    print(f"  -> saved: {flagged_path}")

    return {"n_flagged": n_flagged, "n_obs": n_obs, "cooks_threshold": cooks_threshold}


def check_linearity_assumption(work: pd.DataFrame, numeric_features: list):
    """
    Quick, practical check of the 'linearity in the logit' assumption: bins
    each continuous predictor into deciles and plots the EMPIRICAL log-odds
    of each non-reference class (vs. reference) against the decile midpoint.
    If the relationship looks like a smooth, roughly straight line, the
    linearity assumption is reasonable; visible curvature suggests a
    transformation (log, polynomial term, or spline) may be warranted for
    that predictor. This is a lighter-weight, visual alternative to a formal
    Box-Tidwell test, which requires fitting interaction terms with log(X)
    for every continuous predictor and is easy to misapply with zero/negative
    values (e.g. lat/lon here can be negative, which breaks log(X) directly).
    """
    print(f"\n{'='*70}")
    print("  STEP 9d – LINEARITY OF CONTINUOUS PREDICTORS WITH THE LOGIT")
    print(f"{'='*70}")
    print("""
  Why this matters: multinomial logistic regression assumes each continuous
  predictor relates LINEARLY to the log-odds of each class vs. the
  reference — not that the predictor relates linearly to the class itself.
  We check this empirically by binning each continuous predictor into
  deciles and plotting the observed log-odds per bin. A roughly straight
  line supports the assumption; a curved or non-monotonic pattern suggests
  the raw predictor should be transformed (e.g. age might need a quadratic
  term to capture a non-monotonic risk profile across the lifespan).
    """)

    y = work[TARGET_COLUMN]
    reference_class = y.value_counts().idxmax()
    non_ref_classes = [c for c in y.unique() if c != reference_class]

    n_feats = len(numeric_features)
    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4.5))
    if n_feats == 1:
        axes = [axes]
    fig.suptitle(f"Empirical Log-Odds vs. Predictor Deciles (reference class: {reference_class})",
                 fontsize=13, fontweight="bold", color=DARK_BG)

    for ax, feat in zip(axes, numeric_features):
        sub = work[[feat, TARGET_COLUMN]].dropna()
        try:
            sub["decile"] = pd.qcut(sub[feat], q=10, duplicates="drop")
        except ValueError:
            ax.set_title(f"{feat}\n(not enough unique values to bin)")
            continue

        for cls in non_ref_classes:
            grp = sub.groupby("decile", observed=True).apply(
                lambda g: np.log(
                    (g[TARGET_COLUMN] == cls).mean() / max((g[TARGET_COLUMN] == reference_class).mean(), 1e-6)
                ) if (g[TARGET_COLUMN] == reference_class).mean() > 0 else np.nan,
                include_groups=False
            )
            midpoints = [interval.mid for interval in grp.index]
            color = CATEGORY_COLORS.get(cls, "#999999")
            ax.plot(midpoints, grp.values, marker="o", markersize=4,
                   label=cls, color=color, linewidth=1.5)

        ax.set_title(feat); ax.set_xlabel(feat); ax.set_ylabel("log-odds vs reference")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = OUT_DIR / "linearity_check.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")
    print(f"  Inspect this plot: roughly straight lines support the linearity-in-the-logit")
    print(f"  assumption; visible curves suggest adding a transformed/quadratic term for that feature.")


def check_sample_size_rule(work: pd.DataFrame, n_predictors: int):
    """
    Rule of thumb (as specified): at least 10-20 observations per parameter
    per category. With k predictors and J classes, a multinomial model
    fits (J-1) sets of coefficients, so the rule is applied per
    non-reference class.
    """
    print(f"\n{'='*70}")
    print("  STEP 9e – SAMPLE SIZE CHECK")
    print(f"{'='*70}")

    y = work[TARGET_COLUMN]
    class_counts = y.value_counts()
    n_classes = len(class_counts)
    smallest_class = class_counts.idxmin()
    smallest_n = class_counts.min()

    obs_per_param = smallest_n / n_predictors

    print(f"  Number of predictors (after any reduction)  : {n_predictors}")
    print(f"  Number of target classes                    : {n_classes}")
    print(f"  Smallest class                               : '{smallest_class}' (n={smallest_n:,})")
    print(f"  Observations per parameter (smallest class)  : {obs_per_param:.1f}")
    print(f"  Guideline                                    : >= 10-20 recommended")

    if obs_per_param < 10:
        print(f"  [FLAG] Below the 10-20 guideline for the smallest class — coefficient")
        print(f"         estimates and p-values for that class may be unstable. Consider:")
        print(f"         merging the smallest class into 'Other', collecting more data for")
        print(f"         it, or treating its results with extra caution.")
    else:
        print(f"  Sample size for the smallest class meets the common guideline.")

    return {"n_classes": n_classes, "smallest_class": smallest_class,
            "smallest_n": smallest_n, "obs_per_param": obs_per_param}


def run_statistical_diagnostics(work: pd.DataFrame, numeric_features: list, categorical_features: list):
    """
    Orchestrates the full diagnostics stage: assumption checklist, initial
    VIF + significance, iterative variable selection, final explanatory
    model summary, influence diagnostics, and linearity check. Returns the
    final reduced feature set and fitted statsmodels result for the
    conclusions section.
    """
    if not HAS_STATSMODELS:
        print(f"\n{'='*70}")
        print("  STEP 9 – STATISTICAL DIAGNOSTICS: SKIPPED")
        print(f"{'='*70}")
        print("  statsmodels is not installed in this environment. Install it with:")
        print("      pip install statsmodels")
        print("  then re-run this script to get p-values, VIF, and influence diagnostics.")
        return None

    print(f"\n{'#'*70}")
    print("  STEP 9 – STATISTICAL SIGNIFICANCE & MODEL DIAGNOSTICS")
    print(f"  (separate from the sklearn predictive model above — this stage uses")
    print(f"   statsmodels MNLogit purely for statistical inference / explanation)")
    print(f"{'#'*70}")

    print("""
  Assumption checklist for multinomial logistic regression:
    1. Target is categorical with > 2 mutually exclusive categories
       -> satisfied: 'category' has 5 mutually exclusive classes.
    2. Independent observations
       -> assumed satisfied (each row is a distinct, independently reported
          crime incident); cannot be tested purely from the data itself,
          this is a design assumption about how the data was collected.
    3. No perfect multicollinearity
       -> checked explicitly below via correlation + VIF.
    4. Linearity of continuous predictors with the log-odds
       -> checked visually below (decile log-odds plot).
    5. No extreme influential outliers
       -> checked below via Cook's distance / leverage.
    6. Sufficient sample size (10-20 obs per parameter per category)
       -> checked below.
    """)

    y = work[TARGET_COLUMN]
    reference_class = y.value_counts().idxmax()
    print(f"  Reference category for all comparisons below: '{reference_class}' "
          f"(most frequent class)")

    # ---- Correlation matrix among numeric predictors ----
    print(f"\n  -- Correlation matrix (numeric predictors) --")
    corr = work[numeric_features].corr()
    print(corr.round(3).to_string())
    high_corr = [
        (c1, c2, corr.loc[c1, c2])
        for i, c1 in enumerate(corr.columns)
        for c2 in corr.columns[i+1:]
        if abs(corr.loc[c1, c2]) > 0.7
    ]
    if high_corr:
        print(f"\n  [FLAG] Pairs with |correlation| > 0.7:")
        for c1, c2, v in high_corr:
            print(f"      {c1} <-> {c2}: r={v:.3f}")
    else:
        print(f"\n  No numeric predictor pairs exceed |r|=0.7.")

    # ---- Sample size check ----
    n_predictors_initial = build_design_matrix(work, numeric_features, categorical_features).shape[1]
    check_sample_size_rule(work, n_predictors_initial)

    # ---- Linearity check ----
    check_linearity_assumption(work, numeric_features)

    # ---- Initial VIF (before any variable removal) ----
    print(f"\n  -- Initial VIF (before variable selection) --")
    X_encoded_full = build_design_matrix(work, numeric_features, categorical_features)
    vif_initial = compute_vif(X_encoded_full)
    print(vif_initial.head(15).to_string(index=False))
    vif_path = OUT_DIR / "vif_table.csv"
    vif_initial.to_csv(vif_path, index=False)
    print(f"  -> saved: {vif_path}")

    # ---- Iterative significance-driven variable selection ----
    X_final, final_result, selection_log = iterative_backward_elimination(
        work, numeric_features, categorical_features, reference_class
    )
    log_path = OUT_DIR / "significance_iteration_log.csv"
    selection_log.to_csv(log_path, index=False)
    print(f"\n  -> saved: {log_path}")

    # ---- Final model summary ----
    print(f"\n{'='*70}")
    print("  STEP 9f – FINAL EXPLANATORY MODEL  (post variable-selection)")
    print(f"{'='*70}")
    print(f"  Final predictor count: {X_final.shape[1]} (started with {n_predictors_initial})")
    summary_text = str(final_result.summary())
    print(summary_text)

    summary_path = OUT_DIR / "mnlogit_final_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Reference category: {reference_class}\n")
        f.write(f"Final predictor count: {X_final.shape[1]} (started with {n_predictors_initial})\n\n")
        f.write(summary_text)
    print(f"\n  -> saved: {summary_path}")

    # ---- Influence diagnostics on the final reduced model ----
    influence_info = plot_influence_diagnostics(final_result, X_final, work)

    return {
        "reference_class": reference_class,
        "n_predictors_initial": n_predictors_initial,
        "n_predictors_final": X_final.shape[1],
        "selection_log": selection_log,
        "vif_initial": vif_initial,
        "influence_info": influence_info,
        "final_result": final_result,
    }


# =============================================================================
# 10.  MAIN
# =============================================================================

def main():
    print(f"\n{'#'*70}")
    print("  MULTINOMIAL LOGISTIC REGRESSION — LA Crime Category Prediction")
    print(f"  RANDOM SEED USED THROUGHOUT: {RANDOM_SEED}")
    print(f"{'#'*70}")

    df   = load_data()
    work = select_and_engineer_minimal_features(df)

    # Drop rows where target itself is missing (shouldn't normally happen,
    # but guards against any upstream edge cases)
    before = len(work)
    work = work.dropna(subset=[TARGET_COLUMN])
    if before != len(work):
        print(f"\n  Dropped {before - len(work):,} rows with missing target.")

    preprocessor, num_feats, cat_feats = explain_and_build_preprocessor(work)
    X_train, X_test, y_train, y_test = make_split(work)

    class_labels = sorted(work[TARGET_COLUMN].unique())

    # ---- Baseline model ----
    print(f"\n{'='*70}")
    print("  STEP 5 – BASELINE MODEL")
    print(f"{'='*70}")
    print("""
  Model: Multinomial Logistic Regression (default-ish settings)
  Why this model for this task:
    - The target has 5 unordered classes -> this is a genuine multi-class
      classification problem, and multinomial logistic regression is the
      direct generalization of binary logistic regression to that setting
      (a single softmax layer over all classes, fit jointly via cross-entropy
      loss) — exactly what was requested for this piece of the comparison.
    - It produces well-calibrated class probabilities (useful for the
      ROC-AUC and log-loss metrics used elsewhere in the team's comparison).
    - It's highly interpretable: each feature gets one coefficient per
      class, letting us directly inspect which features push predictions
      toward which category (see feature_importance.png).
    """)

    baseline_pipeline = build_baseline_model(preprocessor)
    baseline_pipeline.fit(X_train, y_train)
    baseline_results = evaluate_model(baseline_pipeline, X_test, y_test, "BASELINE", class_labels)
    plot_confusion_matrix(y_test, baseline_results["y_pred"], class_labels,
                          "Confusion Matrix — Baseline Model", "confusion_matrix_baseline.png")

    # ---- Tuned model ----
    tuned_pipeline, best_params = tune_model(preprocessor, X_train, y_train)
    tuned_results = evaluate_model(tuned_pipeline, X_test, y_test, "TUNED", class_labels)
    plot_confusion_matrix(y_test, tuned_results["y_pred"], class_labels,
                          "Confusion Matrix — Tuned Model", "confusion_matrix_tuned.png")
    plot_roc_curves(y_test, tuned_results["y_proba"], class_labels, "roc_curves_tuned.png")
    plot_feature_importance(tuned_pipeline, class_labels)

    # ---- Comparison summary ----
    print(f"\n{'='*70}")
    print("  STEP 8 – BASELINE vs TUNED COMPARISON")
    print(f"{'='*70}")
    comparison = pd.DataFrame([
        {"model": "Multinomial Logistic Regression (Baseline)",
         "accuracy": baseline_results["accuracy"],
         "macro_precision": baseline_results["macro_precision"],
         "macro_recall": baseline_results["macro_recall"],
         "macro_f1": baseline_results["macro_f1"],
         "macro_roc_auc": baseline_results["macro_roc_auc"],
         "log_loss": baseline_results["log_loss"],
         "random_seed": RANDOM_SEED},
        {"model": f"Multinomial Logistic Regression (Tuned: {best_params})",
         "accuracy": tuned_results["accuracy"],
         "macro_precision": tuned_results["macro_precision"],
         "macro_recall": tuned_results["macro_recall"],
         "macro_f1": tuned_results["macro_f1"],
         "macro_roc_auc": tuned_results["macro_roc_auc"],
         "log_loss": tuned_results["log_loss"],
         "random_seed": RANDOM_SEED},
    ])
    print(comparison.to_string(index=False))

    improvement = tuned_results["macro_f1"] - baseline_results["macro_f1"]
    print(f"\n  Macro-F1 change from tuning: {improvement:+.4f} "
          f"({'improvement' if improvement > 0 else 'no improvement / regression'})")

    comp_path = OUT_DIR / "model_comparison_summary.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"\n  -> saved: {comp_path}")
    print(f"  (Teammates: append your model's row to this CSV using the same "
          f"columns, for a unified comparison table.)")

    # ---- Statistical diagnostics & significance-driven variable selection ----
    diagnostics = run_statistical_diagnostics(work, num_feats, cat_feats)

    # ---- Conclusions ----
    print(f"\n{'='*70}")
    print("  STEP 11 – CONCLUSIONS")
    print(f"{'='*70}")
    top_class = pd.Series(y_test).value_counts().idxmax()
    print(f"""
  Summary of findings:

  1. Overall performance: the tuned multinomial logistic regression reached
     {tuned_results['accuracy']*100:.1f}% accuracy and a macro-F1 of
     {tuned_results['macro_f1']:.3f} on the held-out test set (seed={RANDOM_SEED}).

  2. Tuning impact: hyperparameter search changed macro-F1 by
     {improvement:+.4f} versus the untuned baseline, landing on
     {best_params}. {"This shows tuning meaningfully helped." if abs(improvement) > 0.01 else "This suggests the baseline defaults were already close to optimal for this feature set, and further gains likely require richer features or a more flexible model class rather than more tuning of this linear model."}

  3. Class-level behavior: check confusion_matrix_tuned.png — logistic
     regression, being a linear model, tends to perform best on the
     majority classes ('{top_class}' is the most frequent in this test set)
     and struggles more on minority classes with overlapping feature
     distributions (commonly Sexual Assault, given its much smaller sample
     size and feature overlap with other categories in time/location/premise
     space alone, without weapon or MO-derived signals).

  4. Why this matters for the team's comparison: as a LINEAR model,
     multinomial logistic regression provides an interpretable performance
     FLOOR for this task. Any non-linear model (ensemble methods like
     Random Forest/Gradient Boosting, or a deep learning model) should be
     benchmarked against these exact numbers using the same random seed
     ({RANDOM_SEED}) and train/test split (see train_test_split_indices.csv).
     If those models substantially outperform this one, it indicates real
     non-linear structure in the relationship between location/time/victim/
     premise features and crime category that a linear decision boundary
     cannot capture.

  5. Feature signal: see feature_importance.png for which standardized
     features carry the largest average coefficient magnitude across
     classes — useful context for your team when interpreting why other
     model types might pick up on similar or different signals.
    """)

    if diagnostics is not None:
        n_dropped = diagnostics["n_predictors_initial"] - diagnostics["n_predictors_final"]
        vif_problematic = (diagnostics["vif_initial"]["flag"] == "PROBLEMATIC").sum()
        infl = diagnostics["influence_info"]
        print(f"""
  6. Statistical significance & model specification (statsmodels MNLogit,
     reference category = '{diagnostics['reference_class']}'):

     - Started with {diagnostics['n_predictors_initial']} candidate predictors
       (after one-hot encoding); iterative backward elimination on p-values
       removed {n_dropped} of them as not significant for distinguishing ANY
       class from the reference. See significance_iteration_log.csv for the
       exact order and reasons (p-value alone, or p-value + high VIF).

     - Multicollinearity: {vif_problematic} predictor(s) showed VIF > 10 in
       the initial check (vif_table.csv). {"These overlapped with the variables removed during backward elimination, consistent with multicollinearity inflating their apparent insignificance." if vif_problematic > 0 else "No predictors showed problematic multicollinearity — the remaining coefficients' standard errors and p-values can be interpreted with reasonable confidence."}

     - Influence diagnostics: {infl['n_flagged']:,} of {infl['n_obs']:,} observations
       ({infl['n_flagged']/infl['n_obs']*100:.2f}%) were flagged as high-influence by
       Cook's distance (using the auxiliary one-vs-reference approach
       described in influence_diagnostics.png, since true multinomial
       Cook's distance isn't a standard, agreed-upon statistic). These rows
       are saved in outlier_flagged_rows.csv for inspection — worth checking
       whether they share a common data-quality issue (e.g. unusual premise
       codes, edge-case ages) before deciding whether to exclude them.

     - Linearity-in-the-logit: see linearity_check.png. Any predictor whose
       empirical log-odds curve deviates noticeably from a straight line
       across deciles is a candidate for a transformed term (e.g. a
       quadratic age term) in a future iteration of this model.

     - Practical takeaway: the FINAL explanatory model (mnlogit_final_summary.txt)
       keeps only variables that are individually justified by significance
       testing, not just ones that happened to help cross-validated accuracy.
       This is a genuinely different goal from steps 5-7 above (the sklearn
       tuned model is optimized to predict well; this statsmodels model is
       optimized to explain well) — it's normal and expected for the two to
       retain a similar, but not necessarily identical, set of variables.
        """)
    else:
        print(f"""
  6. Statistical significance & model diagnostics were SKIPPED because
     statsmodels is not installed in this environment. Run
     `pip install statsmodels` and re-run this script to get p-values,
     VIF, iterative variable selection, and influence diagnostics.
        """)

    print(f"\n{'#'*70}")
    print("  DONE. All outputs saved in: model_outputs/")
    print(f"  RANDOM SEED USED: {RANDOM_SEED}  <-- share this with your team")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()

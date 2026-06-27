"""
model_random_forest.py
======================
LA Crime Data — Random Forest (bagging ensemble of decision trees)

Predicts the 5-class `category` target (Violent / Property / Sexual Assault /
Vehicle / Other — created in data_cleaning.py) from time, location, victim
demographic, and premise-type features.

This is ONE model in the 5-model comparison project (logistic regression,
decision tree, random forest, XGBoost, neural network). It plugs into the shared
architecture defined in src/config.py:

    * reads the engineered features   -> data/processed/la_crime_features.csv
    * reuses the SHARED train/test split (cfg.SPLIT_INDICES_PATH) so every model
      is evaluated on the identical held-out rows
    * pulls hyperparameters           -> cfg.MODEL_PARAMS["random_forest"]
    * writes all artifacts            -> outputs/random_forest_output/
    * upserts its metrics             -> outputs/model_comparison_summary.csv
                                         (keyed by model_key="random_forest")

It uses the SAME approved feature set and the SAME random seed
(cfg.RANDOM_SEED) as the other models, so the comparison is fair.

Random Forest is the project's BAGGING ensemble: many decision trees, each grown
on a bootstrap sample of the rows and a random subset of the features, whose
votes are averaged. Like a single decision tree it handles string targets
natively (no LabelEncoder) and needs no scaling, so this mirrors
model_decision_tree.py. Boosting-specific artifacts (training curve, SHAP) are
not applicable and are intentionally omitted; the random-forest-specific extra is
a feature-importance plot WITH error bars across the trees, which visualizes the
ensemble spread that the averaging smooths out.

Usage:
    python models/model_random_forest.py

Produces (in outputs/random_forest_output/):
    confusion_matrix_baseline.png
    confusion_matrix_tuned.png
    roc_curves_tuned.png
    feature_importance.png
    feature_importance_variability.png   <- random-forest-specific extra (mean ± std across trees)
    precision_recall_f1_by_class.png
    baseline_vs_tuned_metrics.png
    class_distribution.png
    grid_search_results.csv
    model_comparison_summary.csv         <- this model's rows (also upserted to the shared file)
    model_random_forest.joblib           <- serialized tuned pipeline
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, log_loss
)

# All paths / settings (seed, target, split, hyperparameters) come from src/config.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as cfg

# =============================================================================
# 0.  REPRODUCIBILITY  &  PATHS  (resolved from the central config)
# =============================================================================

MODEL_KEY   = "random_forest"
RANDOM_SEED = cfg.RANDOM_SEED    # shared across all 5 models for a fair comparison
np.random.seed(RANDOM_SEED)

FEATURES_PATH = cfg.FEATURES_DATA_PATH
OUT_DIR       = cfg.ensure_dir(cfg.MODEL_DIRS[MODEL_KEY])
PARAMS        = cfg.MODEL_PARAMS[MODEL_KEY]

sns.set_theme(style="darkgrid", font_scale=1.0)
ACCENT, COOL, WARM, DARK_BG = "#E63946", "#457B9D", "#F4A261", "#1D3557"
CATEGORY_COLORS = {
    "Violent": "#E63946", "Property": "#457B9D", "Sexual Assault": "#9D4EDD",
    "Vehicle": "#F4A261", "Other": "#6C757D",
}

# Approved feature set — IDENTICAL to model_decision_tree.py / model_xgboost.py /
# model_logistic_regression.py so the models are compared on the same inputs.
# weapon_desc / status_desc are excluded on purpose: they are downstream
# consequences of the crime category and would leak the target. Any column
# missing from the features file (e.g. disabled via config.FEATURE_CATALOGUE) is
# skipped so the model still runs.
NUMERIC_FEATURE_COLUMNS     = ["lat", "lon", "vict_age", "hour", "month", "day_of_week"]
CATEGORICAL_FEATURE_COLUMNS = ["area_name", "vict_sex", "vict_descent", "premis_group"]

TARGET_COLUMN = cfg.TARGET_COLUMN


def find_features_file() -> Path:
    """Return the engineered features dataset path from config, erroring if absent."""
    path = cfg.FEATURES_DATA_PATH
    if not path.exists():
        print(f"\n  [ERROR] Could not find the engineered features file at {path}")
        print(f"          Run data_cleaning.py then feature_engineering.py first.")
        sys.exit(1)
    return path


# =============================================================================
# 1.  LOAD
# =============================================================================

def load_data() -> pd.DataFrame:
    print(f"\n{'='*70}")
    print("  STEP 1 – LOADING ENGINEERED FEATURES DATASET")
    print(f"{'='*70}")
    path = find_features_file()
    print(f"  File  : {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape : {df.shape}")
    return df


# =============================================================================
# 2.  FEATURE SELECTION
# =============================================================================

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select this model's approved feature set from the already-engineered features
    dataset (feature_engineering.py already derived hour / month / day_of_week /
    premis_group, so nothing is re-derived here). Columns absent from the file
    are skipped so the model degrades gracefully.
    """
    print(f"\n{'='*70}")
    print("  STEP 2 – FEATURE SELECTION")
    print(f"{'='*70}")
    print("  Selected feature groups: TIME, LOCATION, VICTIM DEMOGRAPHICS, PREMISE")
    print("  Excluded on purpose: weapon_desc, status_desc (would leak the target)")

    wanted  = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"  [NOTE] Not in features file (skipped): {missing}")
        print(f"         (likely disabled via config.FEATURE_CATALOGUE upstream)")

    if TARGET_COLUMN not in df.columns:
        print(f"\n  [ERROR] Target column '{TARGET_COLUMN}' not found in features file.")
        sys.exit(1)

    work = df[[TARGET_COLUMN] + present].copy()
    print(f"  Final feature set ({len(present)} features + target):")
    print(f"    {present}")
    return work


# =============================================================================
# 3.  PREPROCESSING  (tree-appropriate: impute + one-hot, NO scaling)
# =============================================================================

def build_preprocessor(work: pd.DataFrame):
    """
    A random forest is built from decision trees, which are scale-invariant
    (splits are threshold-based), so — like the single tree and XGBoost, and
    unlike logistic regression — it needs NO feature scaling. We still:
      * impute numeric gaps with the column MEDIAN (robust to outliers), and
      * one-hot encode categoricals (handle_unknown='ignore' so unseen test
        categories become all-zeros instead of crashing).
    """
    print(f"\n{'='*70}")
    print("  STEP 3 – PRE-PROCESSING  (impute + one-hot, no scaling for trees)")
    print(f"{'='*70}")

    numeric_features     = [c for c in NUMERIC_FEATURE_COLUMNS if c in work.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in work.columns]

    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    print(f"  Numeric (median-imputed)   : {numeric_features}")
    print(f"  Categorical (one-hot)      : {categorical_features}")
    return preprocessor, numeric_features, categorical_features


# =============================================================================
# 4.  SHARED TRAIN / TEST SPLIT  (identical held-out rows across all models)
# =============================================================================

def load_or_create_split(work: pd.DataFrame):
    """
    Reuse the shared split saved by whichever model ran first, so every model in
    the comparison evaluates on the EXACT same held-out rows. If the shared file
    doesn't exist yet, create the split here (same seed / stratification as the
    other models) and save it for the rest of the team.
    """
    print(f"\n{'='*70}")
    print("  STEP 4 – TRAIN / TEST SPLIT")
    print(f"{'='*70}")

    X = work.drop(columns=[TARGET_COLUMN])
    y = work[TARGET_COLUMN]

    split_path = cfg.SPLIT_INDICES_PATH
    if split_path.exists():
        split = pd.read_csv(split_path)
        train_idx = [i for i in split.loc[split["split"] == "train", "row_index"] if i in work.index]
        test_idx  = [i for i in split.loc[split["split"] == "test",  "row_index"] if i in work.index]
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]
        print(f"  Reusing SHARED split        : {split_path}")
        print(f"  (identical held-out rows as every other model in the comparison)")
    else:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, work.index,
            test_size=cfg.TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y if cfg.STRATIFY else None,
        )
        cfg.ensure_dir(split_path.parent)
        pd.DataFrame({
            "row_index": list(idx_train) + list(idx_test),
            "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
        }).to_csv(split_path, index=False)
        print(f"  No shared split found — created one and saved -> {split_path}")

    print(f"  Split        : {int((1-cfg.TEST_SIZE)*100)}% train / {int(cfg.TEST_SIZE*100)}% test")
    print(f"  Random seed  : {RANDOM_SEED}")
    print(f"  Stratified by: '{TARGET_COLUMN}'" if cfg.STRATIFY else "  Stratified   : no")
    print(f"  Train shape  : {X_train.shape}")
    print(f"  Test shape   : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 5.  MODELS  (baseline + tuned)
# =============================================================================

def build_baseline_model(preprocessor):
    """
    Baseline RandomForestClassifier with the sensible defaults from
    config.MODEL_PARAMS.

    Why a random forest for this task:
      - It's the project's BAGGING ensemble (lecture 7): each tree is grown on a
        bootstrap sample of the rows and a random feature subset, and their votes
        are averaged. Averaging many de-correlated trees cuts the high VARIANCE
        of a single deep decision tree, usually generalising better.
      - Like the single tree it captures non-linear feature interactions and
        needs no scaling, so it sits naturally between the decision tree (its
        building block) and XGBoost (boosting) in the comparison.
      - Defaults use 300 trees with max_features='sqrt'; tuning then searches
        tree count / depth / leaf size / feature subset.
    """
    base = dict(PARAMS["baseline"])   # copy so we can safely augment
    model = RandomForestClassifier(**base, random_state=RANDOM_SEED)
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", model)])


def tune_model(preprocessor, X_train, y_train):
    """
    GridSearchCV over the config-defined search space, scored on macro-F1 with
    StratifiedKFold (each fold preserves class proportions — important given the
    imbalanced target, so the rare Sexual Assault class isn't lost in some folds).

    The forest itself runs single-threaded here (n_jobs=1) while GridSearchCV
    parallelizes across folds/candidates (n_jobs=-1); this avoids nested
    parallelism oversubscribing the CPU.
    """
    print(f"\n{'='*70}")
    print("  STEP 6 – HYPERPARAMETER TUNING (GridSearchCV)")
    print(f"{'='*70}")
    print(f"  Search space : {PARAMS['param_grid']}")
    print(f"  CV           : StratifiedKFold({cfg.CV_FOLDS}), scoring=f1_macro")

    model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classifier", model)])
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    grid = GridSearchCV(
        pipeline,
        param_grid=PARAMS["param_grid"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best parameters found : {grid.best_params_}")
    print(f"  Best CV macro-F1      : {grid.best_score_:.4f}")

    cv_results = pd.DataFrame(grid.cv_results_)
    keep = [c for c in cv_results.columns
            if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
    cv_results = cv_results[keep].sort_values("rank_test_score")
    cv_path = OUT_DIR / "grid_search_results.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"  -> saved: {cv_path}")

    return grid.best_estimator_, grid.best_params_


# =============================================================================
# 6.  EVALUATION
# =============================================================================

def evaluate_model(pipeline, X_test, y_test, label: str, class_labels):
    """Compute the shared metric set on the raw string class labels (so the
    comparison table matches the other models' label space)."""
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    ll = log_loss(y_test, y_proba, labels=class_labels)

    y_test_bin = label_binarize(y_test, classes=class_labels)
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
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "label": label, "accuracy": acc, "macro_precision": precision,
        "macro_recall": recall, "macro_f1": f1, "macro_roc_auc": auc_macro,
        "log_loss": ll, "y_test": np.asarray(y_test), "y_pred": y_pred, "y_proba": y_proba,
    }


# =============================================================================
# 7.  VISUALS  (all saved into outputs/random_forest_output/)
# =============================================================================

def _save(fig, filename: str):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


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
    _save(fig, filename)


def plot_roc_curves(y_test, y_proba, class_labels, filename: str):
    y_test_bin = label_binarize(y_test, classes=class_labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, cls in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})",
                color=CATEGORY_COLORS.get(cls, "#999999"), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest per Category (Tuned Random Forest)",
                 fontsize=14, fontweight="bold", color=DARK_BG)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, filename)


def plot_feature_importance(pipeline, top_n: int = 20, filename: str = "feature_importance.png"):
    print(f"\n  [PLOT] Feature importance (impurity/Gini-based, averaged over the forest)")
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        print("      (skipped – could not extract feature names)")
        return
    importances = pipeline.named_steps["classifier"].feature_importances_
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(imp.index[::-1], imp.values[::-1], color=COOL, edgecolor="white")
    ax.set_title(f"Top {top_n} Features by Importance (impurity decrease)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save(fig, filename)


def plot_per_class_metrics(y_test, y_pred, class_labels, filename: str):
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, zero_division=0
    )
    x = np.arange(len(class_labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, p, w, label="Precision", color=COOL, edgecolor="white")
    ax.bar(x,     r, w, label="Recall",    color=WARM, edgecolor="white")
    ax.bar(x + w, f1, w, label="F1",        color=ACCENT, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(class_labels, rotation=20)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 (Tuned Random Forest)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_baseline_vs_tuned(baseline, tuned, filename: str):
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc"]
    labels  = ["Accuracy", "Macro P", "Macro R", "Macro F1", "ROC-AUC"]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, [baseline[m] for m in metrics], w, label="Baseline", color=COOL, edgecolor="white")
    ax.bar(x + w/2, [tuned[m] for m in metrics],    w, label="Tuned",    color=ACCENT, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Baseline vs Tuned — Random Forest", fontsize=13, fontweight="bold", color=DARK_BG)
    for i, m in enumerate(metrics):
        ax.text(i - w/2, baseline[m] + 0.01, f"{baseline[m]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, tuned[m] + 0.01,    f"{tuned[m]:.2f}",    ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_class_distribution(work: pd.DataFrame, filename: str):
    counts = work[TARGET_COLUMN].value_counts()
    colors = [CATEGORY_COLORS.get(c, "#999999") for c in counts.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold", color=DARK_BG)
    ax.set_ylabel("Incidents"); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, filename)


def _clean_feature_names(names) -> list:
    """Strip the ColumnTransformer 'num__' / 'cat__' prefixes for readable plots."""
    cleaned = []
    for n in names:
        n = str(n)
        for prefix in ("num__", "cat__", "remainder__"):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break
        cleaned.append(n)
    return cleaned


def plot_feature_importance_variability(pipeline, top_n: int = 20,
                                        filename: str = "feature_importance_variability.png"):
    """
    Random-forest-specific extra (the template's tree_structure equivalent): the
    top `top_n` features by mean impurity importance, WITH ±1 std error bars
    computed across the forest's individual trees (clf.estimators_). The spread
    shows how much each feature's importance varies from tree to tree — the
    disagreement that bagging averages away. Tall bars with small error bars are
    the forest's most consistently useful features.
    """
    print(f"\n  [PLOT] Feature importance variability (mean ± std across the trees)")
    clf = pipeline.named_steps["classifier"]
    try:
        feature_names = _clean_feature_names(
            pipeline.named_steps["preprocess"].get_feature_names_out()
        )
    except Exception:
        print("      (skipped – could not extract feature names)")
        return

    mean_imp = clf.feature_importances_
    # Per-tree importances -> std across the ensemble.
    per_tree = np.array([tree.feature_importances_ for tree in clf.estimators_])
    std_imp  = per_tree.std(axis=0)

    imp = (pd.DataFrame({"mean": mean_imp, "std": std_imp}, index=feature_names)
           .sort_values("mean", ascending=False).head(top_n))

    fig, ax = plt.subplots(figsize=(11, 8))
    order = imp.iloc[::-1]   # smallest at bottom for a top-down ranking
    ax.barh(order.index, order["mean"], xerr=order["std"],
            color=COOL, edgecolor="white",
            error_kw={"ecolor": ACCENT, "elinewidth": 1.2, "capsize": 3})
    ax.set_title(f"Top {top_n} Features — Importance ± std across trees",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Mean impurity decrease (± 1 std over the forest)")
    plt.tight_layout()
    _save(fig, filename)


# =============================================================================
# 8.  COMPARISON SUMMARY  (per-model copy + upsert into the shared table)
# =============================================================================

def write_comparison(baseline, tuned, best_params):
    print(f"\n{'='*70}")
    print("  STEP 7 – BASELINE vs TUNED COMPARISON")
    print(f"{'='*70}")
    comparison = pd.DataFrame([
        {"model": "Random Forest (Baseline)",
         "accuracy": baseline["accuracy"], "macro_precision": baseline["macro_precision"],
         "macro_recall": baseline["macro_recall"], "macro_f1": baseline["macro_f1"],
         "macro_roc_auc": baseline["macro_roc_auc"], "log_loss": baseline["log_loss"],
         "random_seed": RANDOM_SEED},
        {"model": f"Random Forest (Tuned: {best_params})",
         "accuracy": tuned["accuracy"], "macro_precision": tuned["macro_precision"],
         "macro_recall": tuned["macro_recall"], "macro_f1": tuned["macro_f1"],
         "macro_roc_auc": tuned["macro_roc_auc"], "log_loss": tuned["log_loss"],
         "random_seed": RANDOM_SEED},
    ])
    print(comparison.to_string(index=False))

    improvement = tuned["macro_f1"] - baseline["macro_f1"]
    print(f"\n  Macro-F1 change from tuning: {improvement:+.4f} "
          f"({'improvement' if improvement > 0 else 'no improvement / regression'})")

    # Keep a copy in this model's own output folder...
    comparison.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)

    # ...and upsert into the SHARED comparison table all 5 models contribute to.
    # Re-running this model replaces just its own rows, so no duplicates accumulate.
    comparison = comparison.assign(model_key=MODEL_KEY)
    shared_path = cfg.COMPARISON_SUMMARY_PATH
    cfg.ensure_dir(shared_path.parent)
    if shared_path.exists():
        prior = pd.read_csv(shared_path)
        if "model_key" in prior.columns:
            prior = prior[prior["model_key"] != MODEL_KEY]
        combined = pd.concat([prior, comparison], ignore_index=True)
    else:
        combined = comparison
    combined.to_csv(shared_path, index=False)
    print(f"\n  -> saved (this model): {OUT_DIR / 'model_comparison_summary.csv'}")
    print(f"  -> updated (shared)  : {shared_path}")


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    print(f"\n{'#'*70}")
    print("  RANDOM FOREST — LA Crime Category Prediction")
    print(f"  RANDOM SEED USED THROUGHOUT: {RANDOM_SEED}")
    print(f"{'#'*70}")

    df   = load_data()
    work = select_features(df)

    # Drop rows with a missing target (guards against upstream edge cases) — the
    # SAME indexing path as the other models, so shared split indices align.
    before = len(work)
    work = work.dropna(subset=[TARGET_COLUMN])
    if before != len(work):
        print(f"\n  Dropped {before - len(work):,} rows with missing target.")

    plot_class_distribution(work, "class_distribution.png")

    # Sklearn forests handle string targets natively — no LabelEncoder needed.
    # Class label order follows the fitted classifier's .classes_ (set below) so
    # predict_proba columns line up with ROC / log-loss label arrays.
    class_labels = sorted(work[TARGET_COLUMN].unique())
    print(f"\n  Target classes ({len(class_labels)}): {class_labels}")

    preprocessor, _, _ = build_preprocessor(work)
    X_train, X_test, y_train, y_test = load_or_create_split(work)

    # ---- Baseline ----
    print(f"\n{'='*70}")
    print("  STEP 5 – BASELINE MODEL")
    print(f"{'='*70}")
    baseline_pipeline = build_baseline_model(preprocessor)
    baseline_pipeline.fit(X_train, y_train)
    # Use the fitted classifier's class order for all proba-based metrics/plots.
    class_labels = list(baseline_pipeline.named_steps["classifier"].classes_)
    baseline_results = evaluate_model(baseline_pipeline, X_test, y_test,
                                      "BASELINE", class_labels)
    plot_confusion_matrix(baseline_results["y_test"], baseline_results["y_pred"],
                          class_labels, "Confusion Matrix — Baseline Random Forest",
                          "confusion_matrix_baseline.png")

    # ---- Tuned ----
    tuned_pipeline, best_params = tune_model(preprocessor, X_train, y_train)
    class_labels = list(tuned_pipeline.named_steps["classifier"].classes_)
    tuned_results = evaluate_model(tuned_pipeline, X_test, y_test,
                                   "TUNED", class_labels)

    # ---- Visuals ----
    plot_confusion_matrix(tuned_results["y_test"], tuned_results["y_pred"],
                          class_labels, "Confusion Matrix — Tuned Random Forest",
                          "confusion_matrix_tuned.png")
    plot_roc_curves(tuned_results["y_test"], tuned_results["y_proba"],
                    class_labels, "roc_curves_tuned.png")
    plot_feature_importance(tuned_pipeline)
    plot_per_class_metrics(tuned_results["y_test"], tuned_results["y_pred"],
                           class_labels, "precision_recall_f1_by_class.png")
    plot_baseline_vs_tuned(baseline_results, tuned_results, "baseline_vs_tuned_metrics.png")

    # ---- Random-forest-specific extra: importance spread across the trees ----
    plot_feature_importance_variability(tuned_pipeline)

    # ---- Comparison summary ----
    write_comparison(baseline_results, tuned_results, best_params)

    # ---- Persist the tuned model ----
    model_path = OUT_DIR / "model_random_forest.joblib"
    joblib.dump({"pipeline": tuned_pipeline, "class_labels": class_labels}, model_path)
    print(f"\n  Serialized tuned model -> {model_path}")

    print(f"\n{'#'*70}")
    print(f"  DONE. All outputs saved in: {OUT_DIR}")
    print(f"  RANDOM SEED USED: {RANDOM_SEED}  <-- shared across all models")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()

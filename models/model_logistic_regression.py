"""
Multinomial logistic regression for the `category` target. One of the five
comparison models.

Reads data/processed/la_crime_features.csv, reuses the shared train/test split,
takes its hyperparameters from config, and writes plots, metrics, and the saved
model to outputs/logistic_regression_output/. Same feature set and seed as the
other models.

A second stage (statsmodels MNLogit) adds inferential diagnostics: VIF,
significance-driven variable selection, influence, and a linearity check. It
runs only when statsmodels is installed and is separate from the sklearn model
used for the comparison.
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

# All paths / settings (seed, target, split, hyperparameters) come from src/config.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as cfg

# =============================================================================
# 0.  REPRODUCIBILITY  &  PATHS  (resolved from the central config)
# =============================================================================

MODEL_KEY   = "logistic_regression"
RANDOM_SEED = cfg.RANDOM_SEED    # <-- documented above; reuse this across all models
np.random.seed(RANDOM_SEED)

# Input is the engineered features file (shared by all 5 models); each model's
# outputs land in its own folder under outputs/.
FEATURES_PATH = cfg.FEATURES_DATA_PATH
OUT_DIR       = cfg.ensure_dir(cfg.MODEL_DIRS[MODEL_KEY])
PARAMS        = cfg.MODEL_PARAMS[MODEL_KEY]

sns.set_theme(style="darkgrid", font_scale=1.0)
ACCENT, COOL, WARM, DARK_BG = "#E63946", "#457B9D", "#F4A261", "#1D3557"
CATEGORY_COLORS = {
    "Violent": "#E63946", "Property": "#457B9D", "Sexual Assault": "#9D4EDD",
    "Vehicle": "#F4A261", "Other": "#6C757D",
}


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

# Feature set shared by all models. weapon_desc / status_desc left out: they are
# consequences of the crime type and would leak the target. Missing columns
# (e.g. disabled in config) are skipped.
NUMERIC_FEATURE_COLUMNS = ["lat", "lon", "vict_age", "hour", "month", "day_of_week"]
CATEGORICAL_FEATURE_COLUMNS = ["area_name", "vict_sex", "vict_descent", "premis_group"]

TARGET_COLUMN = cfg.TARGET_COLUMN


def select_and_engineer_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the model's feature set from the engineered data (nothing is re-derived;
    missing columns are skipped)."""
    print(f"\n{'='*70}")
    print("  STEP 2 – FEATURE SELECTION")
    print(f"{'='*70}")
    print("  Groups: time, location, victim demographics, premise")
    print("  Left out: weapon_desc, status_desc (leak the target)")

    wanted = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"  [NOTE] Not in features file (skipped): {missing}")

    if TARGET_COLUMN not in df.columns:
        print(f"\n  [ERROR] Target column '{TARGET_COLUMN}' not found in features file.")
        sys.exit(1)

    work = df[[TARGET_COLUMN] + present].copy()

    print(f"  Final feature set ({len(present)} features + target):")
    print(f"    {present}")

    return work


# =============================================================================
# 3.  PREPROCESSING
# =============================================================================

def explain_and_build_preprocessor(work: pd.DataFrame):
    """Numeric: median-impute then standardize (logistic regression needs scaled
    inputs). Categorical: impute 'Missing' then one-hot encode."""
    print(f"\n{'='*70}")
    print("  STEP 3 – PRE-PROCESSING")
    print(f"{'='*70}")

    numeric_features     = [c for c in NUMERIC_FEATURE_COLUMNS if c in work.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in work.columns]

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
        test_size=cfg.TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if cfg.STRATIFY else None,
    )

    print(f"  Split        : {int((1-cfg.TEST_SIZE)*100)}% train / {int(cfg.TEST_SIZE*100)}% test")
    print(f"  Random seed  : {RANDOM_SEED}")
    print(f"  Stratified by: '{TARGET_COLUMN}'")
    print(f"  Train shape  : {X_train.shape}")
    print(f"  Test shape   : {X_test.shape}")
    print(f"\n  Class distribution preserved across split:")
    dist = pd.DataFrame({
        "train_%": (y_train.value_counts(normalize=True) * 100).round(2),
        "test_%":  (y_test.value_counts(normalize=True) * 100).round(2),
    })
    print(dist.to_string())

    # Save the split indices so every model trains/tests on the same rows.
    split_df = pd.DataFrame({
        "row_index": list(idx_train) + list(idx_test),
        "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
    })
    split_path = cfg.SPLIT_INDICES_PATH
    cfg.ensure_dir(split_path.parent)
    split_df.to_csv(split_path, index=False)
    print(f"\n  Saved split indices -> {split_path}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 5.  BASELINE MODEL
# =============================================================================

def build_baseline_model(preprocessor):
    """Baseline logistic regression using the defaults from config. With lbfgs/saga
    on a multi-class target, sklearn fits a true multinomial (softmax) model."""
    base = PARAMS["baseline"]
    model = LogisticRegression(
        penalty=base["penalty"],
        solver=base["solver"],   # lbfgs supports true multinomial loss natively
        C=base["C"],
        max_iter=base["max_iter"],
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
# 6.  HYPERPARAMETER TUNING
# =============================================================================

def tune_model(preprocessor, X_train, y_train):
    """GridSearchCV over C and penalty (l1/l2), scored on macro-F1 with stratified
    folds. Tuning uses the saga solver, which supports both penalties."""
    print(f"\n{'='*70}")
    print("  STEP 6 – HYPERPARAMETER TUNING (GridSearchCV)")
    print(f"{'='*70}")
    print(f"  Grid: {PARAMS['param_grid']}")
    print(f"  CV  : StratifiedKFold({cfg.CV_FOLDS}), scoring=f1_macro")

    model = LogisticRegression(
        solver=PARAMS.get("tuning_solver", "saga"),
        max_iter=PARAMS.get("tuning_max_iter", 2000),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", model),
    ])

    param_grid = PARAMS["param_grid"]   # from config.MODEL_PARAMS["logistic_regression"]

    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

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


# -----------------------------------------------------------------------------
# Generic comparison visuals — kept IDENTICAL (filenames, layout, palette) to the
# other models (e.g. model_xgboost.py) so the per-model output folders can be
# compared side-by-side.
# -----------------------------------------------------------------------------

def _save_fig(fig, filename: str):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


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
    ax.set_title("Per-Class Precision / Recall / F1 (Tuned Logistic Regression)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, filename)


def plot_baseline_vs_tuned(baseline, tuned, filename: str):
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc"]
    labels  = ["Accuracy", "Macro P", "Macro R", "Macro F1", "ROC-AUC"]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, [baseline[m] for m in metrics], w, label="Baseline", color=COOL, edgecolor="white")
    ax.bar(x + w/2, [tuned[m] for m in metrics],    w, label="Tuned",    color=ACCENT, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Baseline vs Tuned — Logistic Regression",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    for i, m in enumerate(metrics):
        ax.text(i - w/2, baseline[m] + 0.01, f"{baseline[m]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, tuned[m] + 0.01,    f"{tuned[m]:.2f}",    ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, filename)


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
    _save_fig(fig, filename)


# =============================================================================
# 9.  STATISTICAL DIAGNOSTICS & SIGNIFICANCE-DRIVEN VARIABLE SELECTION
# =============================================================================
# sklearn gives no p-values, so this stage fits a separate statsmodels MNLogit
# for inference: which variables are significant and whether the model is well
# specified. It does not affect the sklearn model used for the comparison.

def build_design_matrix(work: pd.DataFrame, numeric_features: list, categorical_features: list):
    """Build a statsmodels design matrix: numerics imputed and standardized,
    categoricals one-hot encoded with drop_first to avoid the dummy-variable trap."""
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
    """Variance Inflation Factor per predictor (1 / (1 - R^2) from regressing it on
    the others). High VIF flags multicollinearity. Bands: <5 ok, 5-10 watch, >10 bad."""
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
    """Fit statsmodels MNLogit with reference_class as the baseline category that all
    other classes' coefficients are read against."""
    y_codes = y.astype("category")
    # Put reference_class first so statsmodels treats it as the baseline.
    cats = [reference_class] + [c for c in y_codes.cat.categories if c != reference_class]
    y_codes = y_codes.cat.reorder_categories(cats, ordered=False)
    y_numeric = y_codes.cat.codes

    X_with_const = sm.add_constant(X_encoded)
    model = sm.MNLogit(y_numeric, X_with_const)
    result = model.fit(method="newton", maxiter=100, disp=False)
    return result, y_codes.cat.categories


def summarize_significance(result, class_categories, reference_class: str) -> pd.DataFrame:
    """Return a tidy {feature, class, coef, std_err, z, p_value} table from the
    fitted MNLogit result, across all non-reference classes."""
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
    """Backward elimination on p-values: each round drop the predictor with the
    worst best-across-classes p-value (keeping anything significant for at least one
    class), refitting until all remain significant or max_iterations is hit. VIF is
    logged alongside as context. Returns the reduced matrix, final fit, and a drop log."""
    print(f"\n{'='*70}")
    print("  STEP 9b – ITERATIVE SIGNIFICANCE-DRIVEN VARIABLE SELECTION")
    print(f"{'='*70}")
    print(f"  p-value threshold = {p_threshold}, max iterations = {max_iterations}")

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
    """Cook's distance / leverage for the multinomial model, approximated via an
    auxiliary one-vs-rest binary Logit on the largest non-reference class (MNLogit
    has no get_influence). Single-class lens, not an exact multinomial measure."""
    print(f"\n{'='*70}")
    print("  STEP 9c – INFLUENCE DIAGNOSTICS  (outliers, leverage)")
    print(f"{'='*70}")

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
    """Visual check of linearity-in-the-logit: bin each numeric predictor into
    deciles and plot the empirical log-odds (each class vs reference) per bin.
    Straight lines support the assumption; curves suggest a transform. A lighter
    alternative to Box-Tidwell, which breaks on the negative lat/lon values here."""
    print(f"\n{'='*70}")
    print("  STEP 9d – LINEARITY OF CONTINUOUS PREDICTORS WITH THE LOGIT")
    print(f"{'='*70}")

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
    """Check the rule of thumb of 10-20 observations per parameter, applied to the
    smallest class (the binding constraint in a multinomial model)."""
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
    """Run the diagnostics stage: correlation/VIF, sample-size and linearity checks,
    backward elimination, final MNLogit summary, and influence diagnostics. Returns
    a dict for the conclusions, or None if statsmodels is missing."""
    if not HAS_STATSMODELS:
        print(f"\n{'='*70}")
        print("  STEP 9 – STATISTICAL DIAGNOSTICS: SKIPPED")
        print(f"{'='*70}")
        print("  statsmodels is not installed in this environment. Install it with:")
        print("      pip install statsmodels")
        print("  then re-run this script to get p-values, VIF, and influence diagnostics.")
        return None

    print(f"\n{'#'*70}")
    print("  STEP 9 – STATISTICAL SIGNIFICANCE & MODEL DIAGNOSTICS (statsmodels MNLogit)")
    print(f"{'#'*70}")
    # Assumptions checked below: no perfect multicollinearity (correlation + VIF),
    # linearity in the logit, no extreme influential outliers, adequate sample size.

    y = work[TARGET_COLUMN]
    reference_class = y.value_counts().idxmax()
    print(f"  Reference category: '{reference_class}' (most frequent class)")

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

    # Drop any rows with a missing target.
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

    # ---- Generic comparison visuals (kept uniform with the other models) ----
    plot_class_distribution(work, "class_distribution.png")
    plot_per_class_metrics(y_test, tuned_results["y_pred"], class_labels,
                           "precision_recall_f1_by_class.png")
    plot_baseline_vs_tuned(baseline_results, tuned_results, "baseline_vs_tuned_metrics.png")

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

    # Local copy, plus an upsert into the shared table (re-running replaces this
    # model's own rows rather than duplicating them).
    comparison.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)
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

    # ---- Statistical diagnostics & significance-driven variable selection ----
    diagnostics = run_statistical_diagnostics(work, num_feats, cat_feats)

    # ---- Conclusions ----
    print(f"\n{'='*70}")
    print("  STEP 11 – CONCLUSIONS")
    print(f"{'='*70}")
    top_class = pd.Series(y_test).value_counts().idxmax()
    print(f"  Tuned model: {tuned_results['accuracy']*100:.1f}% accuracy, "
          f"macro-F1 {tuned_results['macro_f1']:.3f} (seed={RANDOM_SEED}).")
    print(f"  Tuning changed macro-F1 by {improvement:+.4f}; best params: {best_params}.")
    print(f"  Strongest on majority classes (e.g. '{top_class}'), weakest on the rare "
          f"ones (Sexual Assault) — see confusion_matrix_tuned.png.")
    print(f"  As a linear model, it serves as the performance floor for the comparison.")

    if diagnostics is not None:
        n_dropped = diagnostics["n_predictors_initial"] - diagnostics["n_predictors_final"]
        vif_problematic = (diagnostics["vif_initial"]["flag"] == "PROBLEMATIC").sum()
        infl = diagnostics["influence_info"]
        print(f"\n  Diagnostics (reference '{diagnostics['reference_class']}'): "
              f"backward elimination dropped {n_dropped} of "
              f"{diagnostics['n_predictors_initial']} predictors; "
              f"{vif_problematic} predictor(s) with VIF > 10.")
        if infl is not None:
            print(f"  Influence: {infl['n_flagged']:,} of {infl['n_obs']:,} rows flagged "
                  f"by Cook's distance ({infl['n_flagged']/infl['n_obs']*100:.2f}%).")
        print(f"  See mnlogit_final_summary.txt, vif_table.csv, "
              f"significance_iteration_log.csv, linearity_check.png.")
    else:
        print(f"\n  Diagnostics skipped (statsmodels not installed).")

    print(f"\n{'#'*70}")
    print(f"  DONE. All outputs saved in: {OUT_DIR}")
    print(f"  RANDOM SEED USED: {RANDOM_SEED}  <-- share this with your team")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()

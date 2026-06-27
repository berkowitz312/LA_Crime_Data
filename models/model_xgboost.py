"""
XGBoost classifier for the `category` target. One of the five comparison models.

Reads data/processed/la_crime_features.csv, reuses the shared train/test split,
takes its hyperparameters from config, and writes plots, metrics, SHAP outputs,
and the saved model to outputs/xgboost_output/. Same feature set and seed as the
other models so the results are comparable.
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

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, log_loss
)
from xgboost import XGBClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed. Run: pip install shap")
    print("       The SHAP explainability stage will be skipped.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as cfg

# =============================================================================
# 0.  SETUP
# =============================================================================

MODEL_KEY   = "xgboost"
RANDOM_SEED = cfg.RANDOM_SEED
np.random.seed(RANDOM_SEED)

FEATURES_PATH = cfg.FEATURES_DATA_PATH
OUT_DIR       = cfg.ensure_dir(cfg.MODEL_DIRS[MODEL_KEY])
PARAMS        = cfg.MODEL_PARAMS[MODEL_KEY]

sns.set_theme(style="darkgrid", font_scale=1.0)
ACCENT, COOL, WARM, DARK_BG = "#E63946", "#457B9D", "#F4A261", "#1D3557"
CATEGORY_COLORS = {
    "Violent": "#E63946", "Property": "#457B9D",
    "Vehicle": "#F4A261", "Other": "#6C757D",
}

# Feature set shared by all models. weapon_desc / status_desc are left out as they
# leak the target. Missing columns (e.g. disabled in config) are skipped.
NUMERIC_FEATURE_COLUMNS     = ["lat", "lon", "vict_age", "hour", "month", "day_of_week"]
CATEGORICAL_FEATURE_COLUMNS = ["area_name", "vict_sex", "vict_descent", "premis_group"]

TARGET_COLUMN = cfg.TARGET_COLUMN

# Rows sampled for SHAP. Kept small to keep beeswarm plots readable and fast.
SHAP_SAMPLE_SIZE = 2000


def find_features_file() -> Path:
    """Return the features dataset path, erroring if the pipeline hasn't run."""
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
    """Keep the model's feature set from the engineered data. Missing columns are
    skipped so a disabled feature doesn't break the run."""
    print(f"\n{'='*70}")
    print("  STEP 2 – FEATURE SELECTION")
    print(f"{'='*70}")
    print("  Groups: time, location, victim demographics, premise")
    print("  Left out: weapon_desc, status_desc (leak the target)")

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
    """Median-impute numerics and one-hot encode categoricals. No scaling: tree
    splits are scale-invariant. handle_unknown='ignore' covers unseen categories."""
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
    """Load the shared split if it exists so all models use the same held-out rows;
    otherwise create it (same seed/stratification) and save it."""
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
        print(f"  Reusing shared split        : {split_path}")
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

def fit_with_early_stopping(params: dict, preprocessor, X_train, y_train_enc, n_classes: int):
    """Fit an XGBoost model with early stopping and return (fitted_pipeline, best_iteration).

    A stratified validation set is carved from TRAIN; the preprocessor is fit on the
    train portion only (no leakage), and boosting stops once validation mlogloss
    stops improving for `early_stopping.rounds` rounds. The n_estimators value in
    `params` acts only as a ceiling.

    The returned Pipeline wraps the already-fitted preprocessor + booster; downstream
    code only calls predict / predict_proba / named_steps, so it is never refit.
    """
    es = PARAMS["early_stopping"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc,
        test_size=es["val_size"],
        random_state=RANDOM_SEED,
        stratify=y_train_enc,
    )

    pre = clone(preprocessor)
    Xtr = pre.fit_transform(X_tr)      # fit on train portion only
    Xval = pre.transform(X_val)

    model = XGBClassifier(
        **params,
        num_class=n_classes,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=es["rounds"],
    )
    model.fit(Xtr, y_tr, eval_set=[(Xval, y_val)], verbose=False)

    pipeline = Pipeline(steps=[("preprocess", pre), ("classifier", model)])
    return pipeline, model.best_iteration


def tune_model(preprocessor, X_train, y_train, n_classes: int):
    """GridSearchCV over the config grid (scored on macro-F1, stratified folds) to
    pick the regularization/structure params, then refit the FINAL tuned model with
    early stopping so its tree count is chosen on a validation set.

    Returns (final_pipeline, best_params, best_iteration). The grid itself runs at a
    fixed `tuning_n_estimators` (no early stopping inside the CV+Pipeline)."""
    print(f"\n{'='*70}")
    print("  STEP 6 – HYPERPARAMETER TUNING (GridSearchCV)")
    print(f"{'='*70}")
    print(f"  Search space : {PARAMS['param_grid']}")
    print(f"  CV           : StratifiedKFold({cfg.CV_FOLDS}), scoring=f1_macro")
    print(f"  Search trees : n_estimators={PARAMS['tuning_n_estimators']} (early stopping applied to the final refit)")

    model = XGBClassifier(
        n_estimators=PARAMS["tuning_n_estimators"],
        objective=PARAMS["baseline"].get("objective", "multi:softprob"),
        tree_method=PARAMS["baseline"].get("tree_method", "hist"),
        num_class=n_classes,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
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

    # Merge the grid's best params onto the regularized baseline (so reg_lambda /
    # reg_alpha / subsample / colsample carry over and n_estimators=800 becomes the
    # early-stopping ceiling), then refit the final model with early stopping.
    best_params = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}
    final_params = dict(PARAMS["baseline"]); final_params.update(best_params)
    print(f"\n  Refitting final tuned model with early stopping "
          f"(ceiling n_estimators={final_params['n_estimators']})...")
    final_pipeline, best_iter = fit_with_early_stopping(
        final_params, preprocessor, X_train, y_train, n_classes
    )
    print(f"  Early stopping chose {best_iter + 1} trees for the tuned model.")

    return final_pipeline, grid.best_params_, best_iter


# =============================================================================
# 6.  EVALUATION
# =============================================================================

def evaluate_model(pipeline, X_test, y_test_enc, label: str, class_labels, le):
    """Compute the shared metric set on decoded string labels (so the comparison
    table matches the other models' label space)."""
    y_pred_enc = pipeline.predict(X_test)
    y_proba    = pipeline.predict_proba(X_test)

    y_test = le.inverse_transform(y_test_enc)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    ll = log_loss(y_test_enc, y_proba, labels=np.arange(len(class_labels)))

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
        "log_loss": ll, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
    }


# =============================================================================
# 7.  VISUALS  (all saved into outputs/xgboost_output/)
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
    ax.set_title("ROC Curves — One-vs-Rest per Category (Tuned XGBoost)",
                 fontsize=14, fontweight="bold", color=DARK_BG)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, filename)


def plot_feature_importance(pipeline, top_n: int = 20, filename: str = "feature_importance.png"):
    print(f"\n  [PLOT] Feature importance (gain-based, from the tuned model)")
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        print("      (skipped – could not extract feature names)")
        return
    importances = pipeline.named_steps["classifier"].feature_importances_
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(imp.index[::-1], imp.values[::-1], color=COOL, edgecolor="white")
    ax.set_title(f"Top {top_n} Features by Importance (gain)",
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
    ax.set_title("Per-Class Precision / Recall / F1 (Tuned XGBoost)",
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
    ax.set_title("Baseline vs Tuned — XGBoost", fontsize=13, fontweight="bold", color=DARK_BG)
    for i, m in enumerate(metrics):
        ax.text(i - w/2, baseline[m] + 0.01, f"{baseline[m]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, tuned[m] + 0.01,    f"{tuned[m]:.2f}",    ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_training_curve(best_params, preprocessor, X_train, y_train_enc,
                        n_classes: int, filename: str):
    """Refit the tuned params with early stopping on a validation carve and plot
    train vs validation mlogloss per boosting round. A vertical line marks the
    early-stopping point; the train/val gap shows how well overfitting is controlled."""
    print(f"\n  [PLOT] Training curve (mlogloss vs boosting round, with early stopping)")
    clf_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
    base = dict(PARAMS["baseline"]); base.update(clf_params)
    # These are passed explicitly below; drop them from the config copy.
    for k in ("num_class", "random_state", "eval_metric"):
        base.pop(k, None)

    es = PARAMS["early_stopping"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc,
        test_size=es["val_size"],
        random_state=RANDOM_SEED,
        stratify=y_train_enc,
    )
    pre = clone(preprocessor)
    Xtr = pre.fit_transform(X_tr)          # fit on train portion only
    Xval = pre.transform(X_val)

    # Early stopping watches the LAST eval_set entry (validation) — never test.
    model = XGBClassifier(
        **base, num_class=n_classes, random_state=RANDOM_SEED,
        eval_metric="mlogloss", early_stopping_rounds=es["rounds"],
    )
    model.fit(Xtr, y_tr, eval_set=[(Xtr, y_tr), (Xval, y_val)], verbose=False)
    results = model.evals_result()

    train_ll = results["validation_0"]["mlogloss"]
    val_ll   = results["validation_1"]["mlogloss"]
    rounds   = range(1, len(train_ll) + 1)
    best_round = model.best_iteration + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, train_ll, label="Train", color=COOL, linewidth=2)
    ax.plot(rounds, val_ll,   label="Validation", color=ACCENT, linewidth=2)
    ax.axvline(best_round, color=DARK_BG, linestyle="--", linewidth=1.5,
               label=f"Early stop (round {best_round})")
    ax.set_xlabel("Boosting round"); ax.set_ylabel("Multi-class log-loss")
    ax.set_title("XGBoost Training Curve (tuned params, early stopping)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
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


# =============================================================================
# 7b.  SHAP EXPLAINABILITY
# =============================================================================
# Per-prediction feature attributions: how much, and in which direction, each
# feature pushes the model toward each class. Complements feature_importance.png.

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


def _shap_values_to_list(sv, n_classes: int) -> list:
    """Return SHAP values as a per-class list of (n_samples, n_features) arrays,
    handling the different shapes shap can return."""
    if isinstance(sv, list):
        return sv
    sv = np.asarray(sv)
    if sv.ndim == 3:
        if sv.shape[-1] == n_classes:
            return [sv[:, :, i] for i in range(n_classes)]
        if sv.shape[0] == n_classes:
            return [sv[i] for i in range(n_classes)]
    return [sv]   # binary / single-output fallback


def run_shap_analysis(pipeline, X_explain_df, class_labels, sample_size=SHAP_SAMPLE_SIZE):
    """SHAP values for the tuned model on a sample of held-out rows. Writes a global
    bar plot, a per-class beeswarm each, and a mean|SHAP| table to OUT_DIR."""
    if not HAS_SHAP:
        print(f"\n{'='*70}")
        print("  STEP 7b – SHAP EXPLAINABILITY: SKIPPED (shap not installed)")
        print(f"{'='*70}")
        print("  Install with: pip install shap, then re-run for SHAP visuals.")
        return

    print(f"\n{'='*70}")
    print("  STEP 7b – SHAP EXPLAINABILITY (TreeExplainer on the tuned model)")
    print(f"{'='*70}")

    pre = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["classifier"]

    n = min(sample_size, len(X_explain_df))
    Xs = X_explain_df.sample(n=n, random_state=RANDOM_SEED)
    Xt = pre.transform(Xs)
    if hasattr(Xt, "toarray"):     # OneHotEncoder output is sparse; SHAP needs dense
        Xt = Xt.toarray()
    feat_names = _clean_feature_names(pre.get_feature_names_out())
    Xt_df = pd.DataFrame(Xt, columns=feat_names)
    print(f"  Explaining {n:,} held-out rows across {Xt_df.shape[1]} encoded features.")

    explainer = shap.TreeExplainer(clf)
    sv_list = _shap_values_to_list(explainer.shap_values(Xt_df), len(class_labels))

    # ---- Global multiclass importance (bar) ----
    plt.figure()
    shap.summary_plot(sv_list, Xt_df, plot_type="bar", class_names=class_labels,
                      max_display=20, show=False)
    fig = plt.gcf(); fig.set_size_inches(11, 8)
    fig.suptitle("SHAP — Global Feature Importance (mean |SHAP|, all classes)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    _save(fig, "shap_summary_bar.png")

    # ---- Per-class beeswarm (direction + magnitude) ----
    for i, cls in enumerate(class_labels):
        plt.figure()
        shap.summary_plot(sv_list[i], Xt_df, max_display=20, show=False)
        fig = plt.gcf(); fig.set_size_inches(11, 8)
        plt.title(f"SHAP — Feature Impact on '{cls}'",
                  fontsize=13, fontweight="bold", color=DARK_BG)
        safe = cls.replace(" ", "_").replace("/", "_")
        _save(fig, f"shap_beeswarm_{safe}.png")

    # ---- Tabular ranking ----
    mean_abs = {cls: np.abs(sv_list[i]).mean(axis=0) for i, cls in enumerate(class_labels)}
    imp_df = pd.DataFrame(mean_abs, index=feat_names)
    imp_df.insert(0, "mean_abs_shap_overall", imp_df.mean(axis=1))
    imp_df = imp_df.sort_values("mean_abs_shap_overall", ascending=False)
    imp_df.index.name = "feature"
    csv_path = OUT_DIR / "shap_feature_importance.csv"
    imp_df.to_csv(csv_path)
    print(f"  -> saved: {csv_path}")
    print(f"\n  Top 10 features by overall mean|SHAP|:")
    print(imp_df["mean_abs_shap_overall"].head(10).round(4).to_string())


# =============================================================================
# 8.  COMPARISON SUMMARY  (per-model copy + upsert into the shared table)
# =============================================================================

def write_comparison(baseline, tuned, best_params, baseline_iter=None, tuned_iter=None):
    print(f"\n{'='*70}")
    print("  STEP 8 – BASELINE vs TUNED COMPARISON")
    print(f"{'='*70}")
    # Tree counts chosen by early stopping (best_iteration is 0-based).
    baseline_trees = f" [{baseline_iter + 1} trees]" if baseline_iter is not None else ""
    tuned_trees    = f", {tuned_iter + 1} trees" if tuned_iter is not None else ""
    comparison = pd.DataFrame([
        {"model": f"XGBoost (Baseline{baseline_trees})",
         "accuracy": baseline["accuracy"], "macro_precision": baseline["macro_precision"],
         "macro_recall": baseline["macro_recall"], "macro_f1": baseline["macro_f1"],
         "macro_roc_auc": baseline["macro_roc_auc"], "log_loss": baseline["log_loss"],
         "random_seed": RANDOM_SEED},
        {"model": f"XGBoost (Tuned: {best_params}{tuned_trees})",
         "accuracy": tuned["accuracy"], "macro_precision": tuned["macro_precision"],
         "macro_recall": tuned["macro_recall"], "macro_f1": tuned["macro_f1"],
         "macro_roc_auc": tuned["macro_roc_auc"], "log_loss": tuned["log_loss"],
         "random_seed": RANDOM_SEED},
    ])
    print(comparison.to_string(index=False))

    improvement = tuned["macro_f1"] - baseline["macro_f1"]
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


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    print(f"\n{'#'*70}")
    print("  XGBOOST — LA Crime Category Prediction")
    print(f"  RANDOM SEED USED THROUGHOUT: {RANDOM_SEED}")
    print(f"{'#'*70}")

    df   = load_data()
    work = select_features(df)

    # Drop missing-target rows the same way as the other models so the split aligns.
    before = len(work)
    work = work.dropna(subset=[TARGET_COLUMN])
    if before != len(work):
        print(f"\n  Dropped {before - len(work):,} rows with missing target.")

    plot_class_distribution(work, "class_distribution.png")

    # XGBoost needs integer labels; decode back for metrics/plots.
    le = LabelEncoder().fit(work[TARGET_COLUMN])
    class_labels = list(le.classes_)
    n_classes = len(class_labels)
    print(f"\n  Target classes ({n_classes}): {class_labels}")

    preprocessor, num_feats, cat_feats = build_preprocessor(work)
    X_train, X_test, y_train, y_test = load_or_create_split(work)
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    # ---- Baseline (regularized config defaults + early stopping) ----
    print(f"\n{'='*70}")
    print("  STEP 5 – BASELINE MODEL  (regularized + early stopping)")
    print(f"{'='*70}")
    baseline_pipeline, baseline_iter = fit_with_early_stopping(
        dict(PARAMS["baseline"]), preprocessor, X_train, y_train_enc, n_classes
    )
    print(f"  Early stopping chose {baseline_iter + 1} trees for the baseline model.")
    baseline_results = evaluate_model(baseline_pipeline, X_test, y_test_enc,
                                      "BASELINE", class_labels, le)
    plot_confusion_matrix(baseline_results["y_test"], baseline_results["y_pred"],
                          class_labels, "Confusion Matrix — Baseline XGBoost",
                          "confusion_matrix_baseline.png")

    # ---- Tuned ----
    tuned_pipeline, best_params, tuned_iter = tune_model(
        preprocessor, X_train, y_train_enc, n_classes
    )
    tuned_results = evaluate_model(tuned_pipeline, X_test, y_test_enc,
                                   "TUNED", class_labels, le)

    # ---- Visuals ----
    plot_confusion_matrix(tuned_results["y_test"], tuned_results["y_pred"],
                          class_labels, "Confusion Matrix — Tuned XGBoost",
                          "confusion_matrix_tuned.png")
    plot_roc_curves(tuned_results["y_test"], tuned_results["y_proba"],
                    class_labels, "roc_curves_tuned.png")
    plot_feature_importance(tuned_pipeline)
    plot_per_class_metrics(tuned_results["y_test"], tuned_results["y_pred"],
                           class_labels, "precision_recall_f1_by_class.png")
    plot_baseline_vs_tuned(baseline_results, tuned_results, "baseline_vs_tuned_metrics.png")
    plot_training_curve(best_params, preprocessor, X_train, y_train_enc,
                        n_classes, "training_curve.png")

    # ---- SHAP explainability (on the tuned model, held-out rows) ----
    run_shap_analysis(tuned_pipeline, X_test, class_labels)

    # ---- Comparison summary ----
    write_comparison(baseline_results, tuned_results, best_params,
                     baseline_iter, tuned_iter)

    # ---- Persist the tuned model (+ label encoder) ----
    model_path = OUT_DIR / "model_xgboost.joblib"
    joblib.dump({"pipeline": tuned_pipeline, "label_encoder": le}, model_path)
    print(f"\n  Serialized tuned model -> {model_path}")

    print(f"\n{'#'*70}")
    print(f"  DONE. All outputs saved in: {OUT_DIR}")
    print(f"  RANDOM SEED USED: {RANDOM_SEED}  <-- shared across all models")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()

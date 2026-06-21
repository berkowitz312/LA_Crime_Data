"""
config.py
=========
Central configuration for the LA Crime Data ML project.

This is the single source of truth for:
    * file paths           (raw dataset, processed CSVs, per-model output dirs, logs)
    * reproducibility       (RANDOM_SEED)
    * the modeling target   (TARGET_COLUMN)
    * the train/test split  (TEST_SIZE, STRATIFY, CV_FOLDS)
    * the feature catalogue (enable/disable EACH engineered feature individually)
    * per-model hyperparameters (baseline settings + tuning grids)

Every other script imports this module instead of hard-coding paths or settings.
Scripts in `src/` and `models/` reach it with a small sys.path shim, e.g.:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # models/*.py
    import config as cfg

(`src/*.py` use `Path(__file__).resolve().parent` instead.)

All paths are derived from the project root, so the pipeline works regardless of the
current working directory or the machine it runs on.
"""

from pathlib import Path

# =============================================================================
# PATHS  (everything is relative to the project root, not the CWD)
# =============================================================================

# This file lives in <root>/src/, so the project root is its parent's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Input data ----
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Crime_Data_from_2020_to_Present.csv"

# Optional MO-codes lookup. Only a PDF ships with the project today, so this is
# auto-detected (and treated as optional) by data_cleaning.py. If a CSV version
# is ever added under data/ (or the project root) it will be picked up.
MO_CODES_GLOBS = ["MO_CODES.csv", "mo_codes.csv", "*MO*CODE*.csv", "*mo*code*.csv"]

# ---- Processed data (pipeline outputs that feed the models) ----
PROCESSED_DIR      = DATA_DIR / "processed"
CLEANED_DATA_PATH  = PROCESSED_DIR / "la_crime_cleaned.csv"     # data_cleaning.py output
FEATURES_DATA_PATH = PROCESSED_DIR / "la_crime_features.csv"    # feature_engineering.py output

# Shared train/test split indices — written by the first model that runs, then
# reused by every other model so all 5 evaluate on the identical held-out rows.
SPLIT_INDICES_PATH = PROCESSED_DIR / "train_test_split_indices.csv"

# ---- Outputs ----
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR     = OUTPUTS_DIR / "eda_outputs"   # dataset_overview.py charts + heatmap.html
LOG_DIR     = OUTPUTS_DIR / "logs"          # run logs (log_dir)

# Shared comparison table — every model appends one row of metrics here.
COMPARISON_SUMMARY_PATH = OUTPUTS_DIR / "model_comparison_summary.csv"

# Per-model output folder (model_dir): each holds that model's plots, metric CSVs,
# and serialized trained model — everything for one model in one place.
MODEL_DIRS = {
    "logistic_regression": OUTPUTS_DIR / "logistic_regression_output",
    "decision_tree":       OUTPUTS_DIR / "decision_tree_output",
    "random_forest":       OUTPUTS_DIR / "random_forest_output",
    "xgboost":             OUTPUTS_DIR / "xgboost_output",
    "neural_network":      OUTPUTS_DIR / "neural_network_output",
}


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if missing, then return it. Handy inline:
        out = cfg.ensure_dir(cfg.MODEL_DIRS["xgboost"]) / "metrics.csv"
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_mo_codes() -> Path | None:
    """Optional: locate an MO_CODES.csv lookup under data/ or the project root.
    Returns the first match, or None if absent (the pipeline works without it)."""
    for base in (DATA_DIR, PROJECT_ROOT):
        for pattern in MO_CODES_GLOBS:
            matches = sorted(base.glob(pattern))
            if matches:
                return matches[0]
    return None


# =============================================================================
# REPRODUCIBILITY / TARGET / SPLIT
# =============================================================================

# Reuse this exact seed for EVERY random operation in EVERY model (train/test
# split, CV folds, solver/estimator init) so the comparison is fair & reproducible.
RANDOM_SEED = 42

# The multi-class target created in data_cleaning.py:
# Violent / Property / Sexual Assault / Vehicle / Other.
TARGET_COLUMN = "category"

# Train/test split shared by all models.
TEST_SIZE = 0.20    # 80% train / 20% test
STRATIFY  = True    # preserve class proportions (target is imbalanced)
CV_FOLDS  = 3       # folds for GridSearchCV / cross-validation


# =============================================================================
# FEATURE CATALOGUE
# =============================================================================
# Enable/disable EACH engineered feature individually. feature_engineering.py
# reads these flags and only writes a derived column when its flag is True, so
# you can turn any feature on/off here without touching the engineering code.
#
# The raw `vict_age` column is always kept (age_known / age_group are derived
# from it). Columns sourced straight from the cleaned data that aren't derived
# here (area_name, lat, lon, vict_sex, vict_descent, etc.) are not toggled here.

FEATURE_CATALOGUE = {
    # ---- derived from date_occ ----
    "year":           True,
    "month":          True,
    "day_of_week":    True,    # Monday=0 .. Sunday=6
    "day_name":       True,
    "is_weekend":     True,
    "quarter":        True,
    # ---- derived from time_occ ----
    "hour":           True,    # 0-23
    "time_of_day":    True,    # Night / Morning / Afternoon / Evening
    # ---- derived from vict_age (raw vict_age column is always kept) ----
    "age_known":      True,
    "age_group":      True,
    # ---- flags ----
    "weapon_present": True,    # from weapon_used_cd
    "geo_known":      True,    # from lat/lon
    # ---- premises ----
    "premis_group":   True,    # top-N premises kept, long tail -> "Other"
}

# How many of the most frequent premises to keep before bucketing the long tail
# into "Other" (only used when premis_group is enabled).
PREMIS_TOP_N = 20


# =============================================================================
# MODEL HYPERPARAMETERS  (baseline settings + tuning grids, one block per model)
# =============================================================================
# Each model file pulls its block from here. `baseline` = sensible defaults for
# the un-tuned run; `param_grid` = the GridSearchCV search space (keys are
# prefixed with `classifier__` to match an sklearn Pipeline step named
# "classifier"). Edit freely — these are starting points, not commitments.

MODEL_PARAMS = {
    "logistic_regression": {
        "baseline": {
            "penalty":  "l2",
            "solver":   "lbfgs",   # supports native multinomial loss
            "C":        1.0,
            "max_iter": 1000,
        },
        # Tuning uses solver="saga" (the only solver supporting l1 + multinomial).
        "tuning_solver":   "saga",
        "tuning_max_iter": 2000,
        "param_grid": {
            "classifier__C":       [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"],
        },
    },

    "decision_tree": {
        "baseline": {
            "criterion":         "gini",
            "max_depth":         None,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
        },
        "param_grid": {
            "classifier__max_depth":         [5, 10, 20, None],
            "classifier__min_samples_split": [2, 10, 50],
            "classifier__min_samples_leaf":  [1, 5, 20],
            "classifier__criterion":         ["gini", "entropy"],
        },
    },

    "random_forest": {
        "baseline": {
            "n_estimators":     300,
            "max_depth":        None,
            "min_samples_leaf": 1,
            "max_features":     "sqrt",
            "n_jobs":           -1,
        },
        "param_grid": {
            "classifier__n_estimators":     [100, 300],
            "classifier__max_depth":        [10, 20, None],
            "classifier__min_samples_leaf": [1, 5],
            "classifier__max_features":     ["sqrt", "log2"],
        },
    },

    "xgboost": {
        "baseline": {
            "n_estimators":     300,
            "max_depth":        6,
            "learning_rate":    0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "objective":        "multi:softprob",
            "tree_method":      "hist",
            "n_jobs":           -1,
        },
        "param_grid": {
            "classifier__max_depth":     [4, 6, 8],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__n_estimators":  [200, 400],
            "classifier__subsample":     [0.8, 1.0],
        },
    },

    "neural_network": {
        # Framework-agnostic spec (use Keras or PyTorch when implementing).
        "architecture": {
            "hidden_layers": [128, 64],
            "dropout":       0.3,
            "activation":    "relu",
            "output":        "softmax",
        },
        "training": {
            "epochs":                 50,
            "batch_size":             256,
            "learning_rate":          1e-3,
            "optimizer":              "adam",
            "early_stopping_patience": 5,
        },
    },
}

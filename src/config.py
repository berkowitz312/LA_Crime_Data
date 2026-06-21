"""
Shared settings for the LA Crime project: paths, random seed, target, split,
feature toggles, and per-model hyperparameters. Imported by every script so
nothing is hard-coded. Paths are built from the project root, so location and
working directory don't matter.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# config.py lives in <root>/src/, so the root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Input data ----
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Crime_Data_from_2020_to_Present.csv"

# Optional MO-codes lookup, auto-detected by data_cleaning.py if present.
MO_CODES_GLOBS = ["MO_CODES.csv", "mo_codes.csv", "*MO*CODE*.csv", "*mo*code*.csv"]

# ---- Processed data (pipeline outputs that feed the models) ----
PROCESSED_DIR      = DATA_DIR / "processed"
CLEANED_DATA_PATH  = PROCESSED_DIR / "la_crime_cleaned.csv"     # from data_cleaning.py
FEATURES_DATA_PATH = PROCESSED_DIR / "la_crime_features.csv"    # from feature_engineering.py

# Train/test split indices, written once and reused by every model so they all
# evaluate on the same held-out rows.
SPLIT_INDICES_PATH = PROCESSED_DIR / "train_test_split_indices.csv"

# ---- Outputs ----
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR     = OUTPUTS_DIR / "eda_outputs"
LOG_DIR     = OUTPUTS_DIR / "logs"

# Combined metrics table; each model appends its rows here.
COMPARISON_SUMMARY_PATH = OUTPUTS_DIR / "model_comparison_summary.csv"

# One output folder per model (plots, metric CSVs, saved model).
MODEL_DIRS = {
    "logistic_regression": OUTPUTS_DIR / "logistic_regression_output",
    "decision_tree":       OUTPUTS_DIR / "decision_tree_output",
    "random_forest":       OUTPUTS_DIR / "random_forest_output",
    "xgboost":             OUTPUTS_DIR / "xgboost_output",
    "neural_network":      OUTPUTS_DIR / "neural_network_output",
}


def ensure_dir(path: Path) -> Path:
    """Create the directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_mo_codes() -> Path | None:
    """Return the MO-codes CSV under data/ or the root, or None if there isn't one."""
    for base in (DATA_DIR, PROJECT_ROOT):
        for pattern in MO_CODES_GLOBS:
            matches = sorted(base.glob(pattern))
            if matches:
                return matches[0]
    return None


# =============================================================================
# REPRODUCIBILITY / TARGET / SPLIT
# =============================================================================

# Same seed everywhere so every model is reproducible and comparable.
RANDOM_SEED = 42

# Target column (created in data_cleaning.py):
# Violent / Property / Sexual Assault / Vehicle / Other.
TARGET_COLUMN = "category"

TEST_SIZE = 0.20    # 80/20 train/test
STRATIFY  = True    # keep class proportions (classes are imbalanced)
CV_FOLDS  = 3       # cross-validation folds


# =============================================================================
# FEATURE CATALOGUE
# =============================================================================
# Per-feature on/off switches read by feature_engineering.py: a derived column
# is written only if its flag is True. Raw columns kept as-is (area_name, lat,
# lon, vict_sex, vict_descent, vict_age) are not listed here.

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

# Number of most-frequent premises to keep before grouping the rest into "Other".
PREMIS_TOP_N = 20


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# One block per model. `baseline` = defaults for the untuned run; `param_grid` =
# GridSearchCV search space (keys prefixed `classifier__` to match the Pipeline step).

MODEL_PARAMS = {
    "logistic_regression": {
        "baseline": {
            "penalty":  "l2",
            "solver":   "lbfgs",   # handles multinomial loss
            "C":        1.0,
            "max_iter": 1000,
        },
        # saga is needed for the l1 penalty in the grid below.
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

    "neural_network": {   # PyTorch MLP (models/model_neural_network.py)
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
            # Use a CUDA GPU when available, else fall back to CPU. CUDA is NVIDIA-only
            # and needs a CUDA build of torch; it has no effect on AMD GPUs.
            "use_cuda":               True,
            "val_size":               0.1,   # train fraction held out for early stopping
        },
        # Small grid for the tuned run; best picked by validation macro-F1.
        "tuning": {
            "learning_rate": [1e-3, 5e-4],
            "hidden_layers": [[128, 64], [256, 128]],
        },
    },
}

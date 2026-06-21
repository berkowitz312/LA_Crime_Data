"""
feature_engineering.py
======================
LA Crime Data – Feature Engineering (stage 3 of the pipeline)

Reads the cleaned dataset produced by data_cleaning.py:
    cleaned_data/la_crime_cleaned.csv   (auto-detected next to this script)

Derives model-ready features for the multi-class classification task
(target = `category`: Violent / Property / Sexual Assault / Vehicle / Other)
and writes them to:
    cleaned_data/la_crime_features.csv

No file paths or flags need to be passed in. Just make sure you've run
data_cleaning.py first (it creates cleaned_data/la_crime_cleaned.csv), then run:

    python feature_engineering.py

Design notes
------------
* NO encoding here (team decision): categorical columns are kept as clean,
  human-readable labels. Each downstream model applies whatever encoding it
  needs (e.g. one-hot + scaling for Logistic Regression / KNN, ordinal for
  tree models). This keeps the features CSV interpretable.

* Target leakage is removed: `category` was derived in data_cleaning.py from
  `crm_cd` / `crm_cd_desc` (and `crm_cd_1` matches `crm_cd` ~99.8% of the time).
  Those columns are dropped from the feature set so models can't trivially
  recover the label.

* MO-CODE CAVEAT: The project requirements ask for MO codes to be used as a feature, but the
  per-incident `Mocodes` column does NOT survive cleaning — data_cleaning.py
  drops it after using it transiently. This script flags that loudly (see the
  warning printed at the end) and does NOT silently re-read the raw CSV. Adding
  an MO-code feature needs a team decision: re-introduce `Mocodes` in
  data_cleaning.py first.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 0.  PATH AUTO-DETECTION  (same convention as data_cleaning / dataset_overview)
# =============================================================================

SCRIPT_DIR  = Path(__file__).resolve().parent
CLEANED_DIR = SCRIPT_DIR / "cleaned_data"
OUT_DIR     = CLEANED_DIR            # write the next stage's output alongside input
OUT_DIR.mkdir(exist_ok=True)

INPUT_NAME  = "la_crime_cleaned.csv"
OUTPUT_NAME = "la_crime_features.csv"

# Columns dropped from the feature set.
#   - Identifiers carry no signal.
#   - crm_cd* / crm_cd_desc are the SOURCE of the `category` target -> leakage.
#   - raw text / superseded codes are replaced by derived flags or labels below.
LEAKAGE_COLS    = ["crm_cd", "crm_cd_1", "crm_cd_2", "crm_cd_3", "crm_cd_4", "crm_cd_desc"]
IDENTIFIER_COLS = ["dr_no"]
DROP_RAW_COLS   = ["date_rptd", "location", "cross_street",
                   "premis_cd", "weapon_used_cd", "area", "rpt_dist_no"]

# How many of the most frequent premises to keep as-is before bucketing the
# long tail into "Other" (premis_desc has ~306 distinct values).
PREMIS_TOP_N = 20


def find_cleaned_file() -> Path:
    """Locate la_crime_cleaned.csv next to this script (cleaned_data/ first)."""
    candidates = []
    if CLEANED_DIR.exists():
        exact = CLEANED_DIR / INPUT_NAME
        candidates = [exact] if exact.exists() else sorted(CLEANED_DIR.glob("*cleaned*.csv"))
    if not candidates:
        candidates = sorted(SCRIPT_DIR.glob("*cleaned*.csv"))
    if not candidates:
        print(f"\n  [ERROR] Could not find {INPUT_NAME}.")
        print(f"          Looked in: {CLEANED_DIR}")
        print(f"          Run data_cleaning.py first — it produces "
              f"cleaned_data/{INPUT_NAME}")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"  [WARN] Multiple cleaned CSVs found, using: {candidates[0].name}")
    return candidates[0]


# =============================================================================
# 1.  TEMPORAL FEATURES  (from date_occ and time_occ)
# =============================================================================

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """year, month, day_of_week, day_name, is_weekend, quarter from date_occ."""
    if "date_occ" not in df.columns:
        print("  [WARN] No 'date_occ' column — skipping date features.")
        return df

    dt = pd.to_datetime(df["date_occ"], errors="coerce")
    df["year"]        = dt.dt.year.astype("Int64")
    df["month"]       = dt.dt.month.astype("Int64")
    df["day_of_week"] = dt.dt.dayofweek.astype("Int64")        # Monday=0 .. Sunday=6
    df["day_name"]    = dt.dt.day_name()
    df["is_weekend"]  = dt.dt.dayofweek.isin([5, 6]).astype("Int64")
    df["quarter"]     = dt.dt.quarter.astype("Int64")
    print(f"  Date features added            : year, month, day_of_week, "
          f"day_name, is_weekend, quarter")
    return df


def _time_of_day(hour) -> str:
    """Bucket an hour (0-23) into a coarse part-of-day label."""
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if 0 <= h < 6:
        return "Night"        # 00:00–05:59
    if 6 <= h < 12:
        return "Morning"      # 06:00–11:59
    if 12 <= h < 18:
        return "Afternoon"    # 12:00–17:59
    return "Evening"          # 18:00–23:59


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour (0-23) and time_of_day bucket from the HHMM integer time_occ."""
    if "time_occ" not in df.columns:
        print("  [WARN] No 'time_occ' column — skipping time-of-day features.")
        return df

    t = pd.to_numeric(df["time_occ"], errors="coerce")
    df["hour"]        = (t // 100).astype("Int64")            # HHMM -> HH
    df["time_of_day"] = df["hour"].map(_time_of_day)
    print(f"  Time features added            : hour, time_of_day "
          f"(Night/Morning/Afternoon/Evening)")
    return df


# =============================================================================
# 2.  VICTIM AGE  (flag + keep raw + bucket; no imputation)
# =============================================================================

def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Victim age in the cleaned data uses 0 as an 'unknown' placeholder (~25% of
    rows) alongside a handful of true nulls. We keep the raw value, add an
    `age_known` flag so models can use the missingness signal, and add an
    `age_group` bucket. No value is imputed.
    """
    if "vict_age" not in df.columns:
        print("  [WARN] No 'vict_age' column — skipping age features.")
        return df

    age = pd.to_numeric(df["vict_age"], errors="coerce")
    known = age.notna() & (age > 0)
    df["age_known"] = known.astype("Int64")

    bins   = [0, 12, 17, 25, 35, 45, 55, 65, 120]
    labels = ["0-12", "13-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    grp = pd.cut(age.where(known), bins=bins, labels=labels, right=True)
    df["age_group"] = grp.astype("object").where(known, "Unknown")

    print(f"  Age features added             : age_known ({int(known.sum()):,} known / "
          f"{int((~known).sum()):,} unknown), age_group, raw vict_age kept")
    return df


# =============================================================================
# 3.  FLAGS  (weapon presence, geo availability)
# =============================================================================

def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """weapon_present (from weapon_used_cd) and geo_known (from lat/lon)."""
    if "weapon_used_cd" in df.columns:
        df["weapon_present"] = df["weapon_used_cd"].notna().astype("Int64")
        print(f"  Weapon flag added              : weapon_present "
              f"({int(df['weapon_present'].sum()):,} incidents with a weapon)")
    else:
        print("  [WARN] No 'weapon_used_cd' column — skipping weapon_present flag.")

    if "lat" in df.columns and "lon" in df.columns:
        geo = df["lat"].notna() & df["lon"].notna()
        df["geo_known"] = geo.astype("Int64")
        print(f"  Geo flag added                 : geo_known "
              f"({int((~geo).sum()):,} incidents missing coordinates)")
    else:
        print("  [WARN] No 'lat'/'lon' columns — skipping geo_known flag.")
    return df


# =============================================================================
# 4.  PREMISES  (tame high cardinality: keep top-N, bucket the rest)
# =============================================================================

def add_premises_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    premis_desc has ~306 distinct values. Keep the PREMIS_TOP_N most frequent as
    clean labels and collapse the long tail (plus nulls) into 'Other' so the
    column is usable by downstream encoders without exploding dimensionality.
    """
    if "premis_desc" not in df.columns:
        print("  [WARN] No 'premis_desc' column — skipping premises grouping.")
        return df

    top = df["premis_desc"].value_counts().head(PREMIS_TOP_N).index
    df["premis_group"] = df["premis_desc"].where(df["premis_desc"].isin(top), "Other")
    print(f"  Premises grouped               : premis_group "
          f"(top {PREMIS_TOP_N} kept, {df['premis_desc'].nunique() - PREMIS_TOP_N:,} "
          f"rarer types + nulls -> 'Other')")
    return df


# =============================================================================
# 5.  ASSEMBLE FINAL FEATURE SET
# =============================================================================

def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final feature frame: derived columns + clean categorical labels +
    the `category` target. Drops identifiers, leakage columns, and raw columns
    that have been replaced by derived features.
    """
    drop_cols = LEAKAGE_COLS + IDENTIFIER_COLS + DROP_RAW_COLS + ["premis_desc"]
    present_drop = [c for c in drop_cols if c in df.columns]
    out = df.drop(columns=present_drop, errors="ignore")

    # Sanity guard: never let a leakage column slip into the output.
    leaked = [c for c in LEAKAGE_COLS if c in out.columns]
    if leaked:
        print(f"  [ERROR] Leakage columns still present: {leaked} — dropping forcibly.")
        out = out.drop(columns=leaked, errors="ignore")

    print(f"\n  Dropped (id/leakage/raw)       : {len(present_drop)} cols "
          f"-> {sorted(present_drop)}")
    if "category" not in out.columns:
        print("  [WARN] Target column 'category' is missing from the output!")
    return out


# =============================================================================
# 6.  MAIN
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print("  STAGE 3 – FEATURE ENGINEERING")
    print(f"{'='*60}")
    print(f"  Script folder: {SCRIPT_DIR}")

    cleaned_path = find_cleaned_file()
    print(f"  Cleaned file : {cleaned_path.name}")
    df = pd.read_csv(cleaned_path, low_memory=False)
    print(f"  Raw shape    : {df.shape}")

    print(f"\n{'-'*60}")
    print("  DERIVING FEATURES")
    print(f"{'-'*60}")
    df = add_date_features(df)
    df = add_time_features(df)
    df = add_age_features(df)
    df = add_flag_features(df)
    df = add_premises_feature(df)

    features = select_feature_columns(df)

    out_csv = OUT_DIR / OUTPUT_NAME
    features.to_csv(out_csv, index=False)

    print(f"\n{'-'*60}")
    print("  OUTPUT SUMMARY")
    print(f"{'-'*60}")
    print(f"  Final shape  : {features.shape}")
    print(f"  Columns ({features.shape[1]}):")
    for c in features.columns:
        print(f"    - {c}")
    if "category" in features.columns:
        print(f"\n  Target 'category' distribution (unchanged):")
        print(features["category"].value_counts().to_string())
    print(f"\n  Feature set saved -> {out_csv}")

    # -------------------------------------------------------------------------
    # MO-CODE FLAG  (project requirements can't be met from cleaned data)
    # -------------------------------------------------------------------------
    mo_present = any("mocode" in c.lower() for c in df.columns)
    if not mo_present:
        print(f"\n{'='*60}")
        print("  [ACTION NEEDED] MO codes are NOT available as a feature")
        print(f"{'='*60}")
        print("  Project requirements ask for MO codes to be used during feature engineering,")
        print("  but the per-incident 'Mocodes' column does not survive cleaning —")
        print("  data_cleaning.py drops it after using it transiently for category")
        print("  assignment. This script does NOT silently re-read the raw CSV.")
        print("  TEAM DECISION required: re-introduce 'Mocodes' in data_cleaning.py")
        print("  (keep it in la_crime_cleaned.csv) so an MO-code feature can be built.")

    print(f"\n{'='*60}")
    print("  FEATURE ENGINEERING COMPLETE")
    print("  Note: categorical columns are intentionally left UN-encoded —")
    print("  encoding is handled per-model in the modeling stage.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

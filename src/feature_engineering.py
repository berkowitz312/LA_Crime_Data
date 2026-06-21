"""
feature_engineering.py
======================
LA Crime Data – Feature Engineering (stage 3 of the pipeline)

Reads the cleaned dataset produced by data_cleaning.py:
    data/processed/la_crime_cleaned.csv   (path from src/config.py)

Derives model-ready features for the multi-class classification task
(target = `category`: Violent / Property / Sexual Assault / Vehicle / Other)
and writes them to:
    data/processed/la_crime_features.csv

Which features are derived is controlled per-feature by FEATURE_CATALOGUE in
src/config.py — flip any feature to False there to drop just that column.

No file paths or flags need to be passed in. Just make sure you've run
data_cleaning.py first, then run:

    python src/feature_engineering.py

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

# All paths / settings (incl. the feature catalogue toggles) come from src/config.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# =============================================================================
# 0.  PATHS & SETTINGS  (resolved from the central config — independent of CWD)
# =============================================================================

OUT_DIR = cfg.ensure_dir(cfg.PROCESSED_DIR)   # data/processed/

# Enable/disable each derived feature individually via config.FEATURE_CATALOGUE.
FEATURES = cfg.FEATURE_CATALOGUE

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
PREMIS_TOP_N = cfg.PREMIS_TOP_N


def find_cleaned_file() -> Path:
    """Return the cleaned dataset path from config, erroring if it's not there yet."""
    path = cfg.CLEANED_DATA_PATH
    if not path.exists():
        print(f"\n  [ERROR] Could not find the cleaned dataset.")
        print(f"          Looked for: {path}")
        print(f"          Run data_cleaning.py first — it produces "
              f"data/processed/la_crime_cleaned.csv")
        sys.exit(1)
    return path


# =============================================================================
# 1.  TEMPORAL FEATURES  (from date_occ and time_occ)
# =============================================================================

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """year, month, day_of_week, day_name, is_weekend, quarter from date_occ.
    Each column is added only when its FEATURE_CATALOGUE flag is enabled."""
    if "date_occ" not in df.columns:
        print("  [WARN] No 'date_occ' column — skipping date features.")
        return df

    dt = pd.to_datetime(df["date_occ"], errors="coerce")
    added = []
    if FEATURES.get("year"):
        df["year"] = dt.dt.year.astype("Int64");                       added.append("year")
    if FEATURES.get("month"):
        df["month"] = dt.dt.month.astype("Int64");                     added.append("month")
    if FEATURES.get("day_of_week"):
        df["day_of_week"] = dt.dt.dayofweek.astype("Int64")            # Monday=0 .. Sunday=6
        added.append("day_of_week")
    if FEATURES.get("day_name"):
        df["day_name"] = dt.dt.day_name();                             added.append("day_name")
    if FEATURES.get("is_weekend"):
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("Int64"); added.append("is_weekend")
    if FEATURES.get("quarter"):
        df["quarter"] = dt.dt.quarter.astype("Int64");                 added.append("quarter")
    print(f"  Date features added            : {added if added else '(none enabled)'}")
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
    """hour (0-23) and time_of_day bucket from the HHMM integer time_occ.
    Each column is added only when its FEATURE_CATALOGUE flag is enabled.
    Note: time_of_day derives from hour — if time_of_day is enabled but hour is
    disabled, hour is computed locally as a temp without being written out."""
    if "time_occ" not in df.columns:
        print("  [WARN] No 'time_occ' column — skipping time-of-day features.")
        return df

    t = pd.to_numeric(df["time_occ"], errors="coerce")
    hour = (t // 100).astype("Int64")            # HHMM -> HH (temp, may not be written)
    added = []
    if FEATURES.get("hour"):
        df["hour"] = hour;                                 added.append("hour")
    if FEATURES.get("time_of_day"):
        df["time_of_day"] = hour.map(_time_of_day);        added.append("time_of_day")
    print(f"  Time features added            : {added if added else '(none enabled)'}")
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
    added = []
    if FEATURES.get("age_known"):
        df["age_known"] = known.astype("Int64");           added.append("age_known")
    if FEATURES.get("age_group"):
        bins   = [0, 12, 17, 25, 35, 45, 55, 65, 120]
        labels = ["0-12", "13-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        grp = pd.cut(age.where(known), bins=bins, labels=labels, right=True)
        df["age_group"] = grp.astype("object").where(known, "Unknown")
        added.append("age_group")

    print(f"  Age features added             : {added if added else '(none enabled)'} "
          f"({int(known.sum()):,} known / {int((~known).sum()):,} unknown ages; raw vict_age kept)")
    return df


# =============================================================================
# 3.  FLAGS  (weapon presence, geo availability)
# =============================================================================

def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """weapon_present (from weapon_used_cd) and geo_known (from lat/lon).
    Each flag is added only when its FEATURE_CATALOGUE flag is enabled."""
    if FEATURES.get("weapon_present"):
        if "weapon_used_cd" in df.columns:
            df["weapon_present"] = df["weapon_used_cd"].notna().astype("Int64")
            print(f"  Weapon flag added              : weapon_present "
                  f"({int(df['weapon_present'].sum()):,} incidents with a weapon)")
        else:
            print("  [WARN] No 'weapon_used_cd' column — skipping weapon_present flag.")

    if FEATURES.get("geo_known"):
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
    if not FEATURES.get("premis_group"):
        return df
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
    print(f"  Project root : {cfg.PROJECT_ROOT}")
    print(f"  Output dir   : {OUT_DIR}")
    enabled  = [k for k, v in FEATURES.items() if v]
    disabled = [k for k, v in FEATURES.items() if not v]
    print(f"  Feature catalogue — enabled ({len(enabled)}): {enabled}")
    if disabled:
        print(f"  Feature catalogue — disabled ({len(disabled)}): {disabled}")

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

    out_csv = cfg.FEATURES_DATA_PATH
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

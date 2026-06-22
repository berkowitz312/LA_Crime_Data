"""
Derives the model-ready feature set from the cleaned data. Reads
data/processed/la_crime_cleaned.csv and writes data/processed/la_crime_features.csv.

Which columns are derived is controlled by FEATURE_CATALOGUE in config.py.
Categoricals stay as text labels (each model encodes them itself). Crime-code
columns are dropped to avoid leaking the target.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# =============================================================================
# 0.  PATHS & SETTINGS
# =============================================================================

OUT_DIR  = cfg.ensure_dir(cfg.PROCESSED_DIR)
FEATURES = cfg.FEATURE_CATALOGUE   # per-feature on/off switches

# Columns removed before saving: identifiers, the crime-code source of the
# target (leakage), and raw columns replaced by derived ones.
LEAKAGE_COLS    = ["crm_cd", "crm_cd_1", "crm_cd_2", "crm_cd_3", "crm_cd_4", "crm_cd_desc"]
IDENTIFIER_COLS = ["dr_no"]
DROP_RAW_COLS   = ["date_rptd", "location", "cross_street",
                   "premis_cd", "weapon_used_cd", "area", "rpt_dist_no"]

PREMIS_TOP_N = cfg.PREMIS_TOP_N


def find_cleaned_file() -> Path:
    """Return the cleaned dataset path, erroring if data_cleaning.py hasn't run."""
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
    """hour (0-23) and time_of_day bucket from time_occ. time_of_day needs hour,
    so hour is computed locally even when only time_of_day is enabled."""
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
    """age_known flag (0 is the 'unknown' placeholder) and age_group bins. The raw
    vict_age column is kept and nothing is imputed."""
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
    """Group premis_desc (~306 values) into the PREMIS_TOP_N most common labels,
    with everything else (and nulls) as 'Other'."""
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
    """Drop identifiers, leakage, and replaced raw columns; keep the derived
    features, categorical labels, and the target."""
    drop_cols = LEAKAGE_COLS + IDENTIFIER_COLS + DROP_RAW_COLS + ["premis_desc"]
    present_drop = [c for c in drop_cols if c in df.columns]
    out = df.drop(columns=present_drop, errors="ignore")

    # Guard against any leakage column slipping through.
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

    # Warn if no Mocodes column made it through cleaning (no MO-code feature possible).
    mo_present = any("mocode" in c.lower() for c in df.columns)
    if not mo_present:
        print(f"\n  [NOTE] No 'Mocodes' column in the cleaned data, so no MO-code "
              f"feature is built. Re-introduce it in data_cleaning.py to enable one.")

    print(f"\n{'='*60}")
    print("  FEATURE ENGINEERING COMPLETE  (categoricals left unencoded; each model encodes them)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
Cleans the raw LA crime CSV and adds a simplified `category` column. Reads
data/Crime_Data...csv and writes data/processed/la_crime_cleaned.csv.

MO_CODES.csv (optional) is only consulted to help assign `category`; it is never
merged in. The original `Mocodes` column is kept in the output for later use.
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
# 0.  PATHS
# =============================================================================

RAW_DATA_PATH = cfg.RAW_DATA_PATH
OUT_DIR       = cfg.ensure_dir(cfg.PROCESSED_DIR)


def find_input_files() -> tuple:
    """Return (crime_csv, mo_codes_csv_or_None). The MO-codes file is optional."""
    crime_path = cfg.RAW_DATA_PATH
    if not crime_path.exists():
        print(f"\n  [ERROR] Could not find the main crime data CSV at: {crime_path}")
        print(f"          Expected the raw dataset at cfg.RAW_DATA_PATH (data/).")
        sys.exit(1)

    mo_path = cfg.find_mo_codes()
    print(f"  Crime data file : {crime_path.name}")
    print(f"  MO codes file   : {mo_path.name if mo_path else '(not found — skipping, not required)'}")
    return crime_path, mo_path


# =============================================================================
# 1.  MO CODE LOOKUP  (only used to help assign `category`)
# =============================================================================

def load_mo_code_categories(path) -> dict:
    """Parse MO_CODES.csv into {code -> category}. Splits each line on the first
    and last comma only, so commas inside the description text don't break it.
    Returns {} when path is None."""
    if path is None:
        return {}

    code_to_cat = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_skipped = False
    for line in lines:
        line = line.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        if not header_skipped:
            header_skipped = True
            continue
        first_comma = line.index(",")
        code        = line[:first_comma].strip().zfill(4)
        rest        = line[first_comma + 1:]
        last_comma  = rest.rindex(",")
        category    = rest[last_comma + 1:].strip().strip('"')
        code_to_cat[code] = category

    print(f"  MO code categories loaded : {len(code_to_cat):,} entries")
    return code_to_cat


def mo_codes_to_category_hint(mo_field: str, code_to_cat: dict) -> str:
    """Return "Sex Related" if any code in a Mocodes string maps to that category,
    else "". Used as a hint when assigning `category`."""
    if not code_to_cat or pd.isna(mo_field) or str(mo_field).strip() == "":
        return ""
    codes = [c.strip().zfill(4) for c in str(mo_field).split() if c.strip()]
    cats  = {code_to_cat.get(c, "") for c in codes}
    if "Sex Related" in cats:
        return "Sex Related"
    return ""


# =============================================================================
# 2.  CRIME CODE -> SIMPLIFIED CATEGORY MAP
# =============================================================================
# Maps the dataset's ~140 distinct Crm Cd values down to 5 interpretable
# buckets: Violent, Property, Sexual Assault, Vehicle, Other.
# Codes not listed below default to "Other".

CRIME_CATEGORY_MAP = {
    # ---- Violent ----
    110:"Violent", 113:"Violent",                      # homicide
    210:"Violent", 220:"Violent",                       # robbery
    230:"Violent", 231:"Violent", 235:"Violent",
    236:"Violent", 250:"Violent", 251:"Violent",
    625:"Violent", 626:"Violent", 627:"Violent",
    231:"Violent", 235:"Violent", 236:"Violent",
    624:"Violent",                                       # battery - simple assault
    761:"Violent", 928:"Violent", 922:"Violent",
    # ---- Sexual Assault ----
    121:"Sexual Assault", 122:"Sexual Assault",
    805:"Sexual Assault", 806:"Sexual Assault",
    807:"Sexual Assault", 810:"Sexual Assault",
    811:"Sexual Assault", 812:"Sexual Assault",
    813:"Sexual Assault", 814:"Sexual Assault",
    815:"Sexual Assault", 820:"Sexual Assault",
    821:"Sexual Assault", 822:"Sexual Assault",
    860:"Sexual Assault", 921:"Sexual Assault",
    # ---- Vehicle ----
    330:"Vehicle",                                        # burglary from vehicle
    331:"Vehicle",
    410:"Vehicle", 420:"Vehicle", 421:"Vehicle",
    433:"Vehicle",
    480:"Vehicle",                                         # bike - stolen
    487:"Vehicle",
    510:"Vehicle", 520:"Vehicle",                          # vehicle - stolen
    # ---- Property ----
    310:"Property", 320:"Property",
    341:"Property", 343:"Property", 345:"Property",
    350:"Property", 351:"Property", 352:"Property",
    353:"Property", 354:"Property",                       # theft of identity
    440:"Property", 441:"Property", 442:"Property",
    443:"Property", 444:"Property", 445:"Property",
    450:"Property", 451:"Property",
    452:"Property", 453:"Property",
    471:"Property", 472:"Property", 473:"Property",
    474:"Property", 475:"Property",
    485:"Property",
    647:"Property",
    648:"Property",
    649:"Property",
    651:"Property",
    652:"Property",
    653:"Property",
    660:"Property",
    661:"Property",
    662:"Property",
    664:"Property",
    665:"Property",
    666:"Property",
}



def assign_crime_category(crm_cd: pd.Series,
                          crm_cd_desc: pd.Series = None,
                          mo_sex_hint: pd.Series = None) -> pd.Series:
    """Map crime codes to one of 5 categories. For codes not in CRIME_CATEGORY_MAP:
    try keyword matching on the description, then the Sex Related MO hint, else "Other"."""
    mapped = crm_cd.map(CRIME_CATEGORY_MAP)

    if crm_cd_desc is not None:
        unmapped = mapped.isna()
        if unmapped.any():
            desc_upper = crm_cd_desc.str.upper().fillna("")

            def keyword_fallback(desc: str) -> str:
                if any(k in desc for k in ["RAPE", "SEX", "SODOMY", "LEWD", "ORAL COP", "PENETRATION"]):
                    return "Sexual Assault"
                if any(k in desc for k in ["VEHICLE", "BIKE", "BOAT", " GTA", "DRIVING"]):
                    return "Vehicle"
                if any(k in desc for k in ["ASSAULT", "HOMICIDE", "MANSLAUGHTER", "BATTERY",
                                            "KIDNAPPING", "SHOTS FIRED", "WEAPON", "ROBBERY",
                                            "THREAT", "STALKING", "CHILD ABUSE", "LYNCHING"]):
                    return "Violent"
                if any(k in desc for k in ["THEFT", "BURGLARY", "STOLEN", "SHOPLIFT", "EMBEZZLE",
                                            "FORGERY", "FRAUD", "VANDALISM", "ARSON", "BUNCO",
                                            "IDENTITY", "BRIBERY", "EXTORTION", "TRESPASS"]):
                    return "Property"
                return ""   # unresolved -> leave blank so MO hint / Other can apply next

            fallback_result = desc_upper.loc[unmapped].apply(keyword_fallback)
            still_blank = fallback_result == ""

            # Use MO sex-related hint only for rows still unresolved after keywords
            if mo_sex_hint is not None and still_blank.any():
                hint_idx = fallback_result[still_blank].index
                hint_vals = mo_sex_hint.loc[hint_idx]
                fallback_result.loc[hint_idx] = hint_vals.where(hint_vals == "Sex Related", "")
                fallback_result.loc[hint_idx] = fallback_result.loc[hint_idx].replace(
                    {"Sex Related": "Sexual Assault"}
                )

            fallback_result = fallback_result.replace({"": np.nan})
            mapped.loc[unmapped] = fallback_result

    return mapped.fillna("Other")


# =============================================================================
# 3.  DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  STEP 2 – DATA CLEANING")
    print(f"{'='*60}")
    orig_rows = len(df)

    # Normalise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # Parse datetimes
    for col in ["date_rptd", "date_occ"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

    # TIME OCC – valid HHMM integer 0-2359
    if "time_occ" in df.columns:
        df["time_occ"] = pd.to_numeric(df["time_occ"], errors="coerce").astype("Int64")
        bad_time = ~df["time_occ"].between(0, 2359)
        df.loc[bad_time, "time_occ"] = pd.NA
        print(f"  Invalid TIME OCC nulled        : {bad_time.sum():,}")

    # Victim age 0-120
    if "vict_age" in df.columns:
        df["vict_age"] = pd.to_numeric(df["vict_age"], errors="coerce")
        bad_age = ~df["vict_age"].between(0, 120)
        df.loc[bad_age, "vict_age"] = np.nan
        print(f"  Invalid victim ages nulled     : {bad_age.sum():,}")

    # Victim Sex: expand abbreviations to full, readable labels
    if "vict_sex" in df.columns:
        df["vict_sex"] = df["vict_sex"].astype(str).str.strip().str.upper()
        sex_map = {
            "M": "Male",
            "F": "Female",
            "X": "Unknown",
            "H": "Unknown",   # rare data-entry artifact in some LAPD exports
            "N": "Unknown",
            "-": np.nan, "NAN": np.nan, "": np.nan,
        }
        df["vict_sex"] = df["vict_sex"].map(sex_map).fillna(df["vict_sex"])
        # Anything still not one of the expected labels becomes Unknown
        df.loc[~df["vict_sex"].isin(["Male", "Female", "Unknown"]), "vict_sex"] = "Unknown"
        print(f"  Vict Sex expanded to full labels (Male/Female/Unknown)")

    # Victim Descent: expand LAPD descent codes to full, readable labels
    if "vict_descent" in df.columns:
        df["vict_descent"] = df["vict_descent"].astype(str).str.strip().str.upper()
        descent_map = {
            "A": "Other Asian",
            "B": "Black",
            "C": "Chinese",
            "D": "Cambodian",
            "F": "Filipino",
            "G": "Guamanian",
            "H": "Hispanic/Latin/Mexican",
            "I": "American Indian/Alaskan Native",
            "J": "Japanese",
            "K": "Korean",
            "L": "Laotian",
            "O": "Other",
            "P": "Pacific Islander",
            "S": "Samoan",
            "U": "Hawaiian",
            "V": "Vietnamese",
            "W": "White",
            "X": "Unknown",
            "Z": "Asian Indian",
            "-": np.nan, "NAN": np.nan, "": np.nan,
        }
        df["vict_descent"] = df["vict_descent"].map(descent_map).fillna("Unknown")
        print(f"  Vict Descent expanded to full labels (Hispanic/White/Black/etc.)")

    # Lat/Lon – (0, 0) are unknown placeholders
    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "lat" in df.columns and "lon" in df.columns:
        bad_geo = (df["lat"] == 0) | (df["lon"] == 0)
        df.loc[bad_geo, ["lat", "lon"]] = np.nan
        print(f"  Zero-coordinate rows nulled    : {bad_geo.sum():,}")

    # Duplicates
    key = "dr_no" if "dr_no" in df.columns else None
    dups = df.duplicated(subset=key, keep="first") if key else df.duplicated()
    df = df[~dups].reset_index(drop=True)
    print(f"  Duplicate records removed      : {dups.sum():,}")

    # Future dates
    if "date_occ" in df.columns:
        future = df["date_occ"] > pd.Timestamp.now()
        df.loc[future, "date_occ"] = pd.NaT
        print(f"  Future DATE OCC nulled         : {future.sum():,}")

    # Missing-value summary
    miss     = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(1)
    summary  = pd.DataFrame({"missing": miss, "pct": miss_pct})
    print(f"\n  Missing-value summary (top 15 cols with nulls):")
    print(summary[summary["missing"] > 0].sort_values("pct", ascending=False).head(15).to_string())
    print(f"\n  Rows after cleaning : {len(df):,}  (removed {orig_rows - len(df):,})")
    return df


# =============================================================================
# 4.  CATEGORIZATION  (crime category only — no MO columns retained)
# =============================================================================

def categorize_data(df: pd.DataFrame, mo_code_to_cat: dict) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  STEP 3 – CATEGORIZATION")
    print(f"{'='*60}")

    if "crm_cd" not in df.columns:
        print("  [WARN] No 'crm_cd' column found – cannot assign category.")
        return df

    desc_col = df["crm_cd_desc"] if "crm_cd_desc" in df.columns else None

    # MO codes only feed the category hint; the raw Mocodes column is left untouched.
    mo_hint = None
    mo_col = next((c for c in df.columns if "mocode" in c or c == "mocodes"), None)
    if mo_col and mo_code_to_cat:
        mo_hint = df[mo_col].apply(lambda v: mo_codes_to_category_hint(v, mo_code_to_cat))

    df["category"] = assign_crime_category(df["crm_cd"], desc_col, mo_hint)

    print("  Crime 'category' assigned (Violent / Property / Sexual Assault / Vehicle / Other):")
    print(df["category"].value_counts().to_string())
    print(f"\n  Total columns after categorization : {df.shape[1]}")
    return df


# =============================================================================
# 5.  MAIN
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print("  STEP 1 – LOCATING & LOADING FILES")
    print(f"{'='*60}")
    print(f"  Project root : {cfg.PROJECT_ROOT}")
    print(f"  Raw data dir : {cfg.DATA_DIR}")
    print(f"  Output dir   : {OUT_DIR}")

    crime_path, mo_path = find_input_files()
    mo_code_to_cat = load_mo_code_categories(mo_path)

    df = pd.read_csv(crime_path, low_memory=False)
    print(f"  Raw shape : {df.shape}")

    df = clean_data(df)
    df = categorize_data(df, mo_code_to_cat)

    out_csv = cfg.CLEANED_DATA_PATH
    df.to_csv(out_csv, index=False)
    print(f"\n  Cleaned & categorized dataset saved -> {out_csv}")
    print(f"\n{'='*60}")
    print("  DATA CLEANING COMPLETE")
    print(f"  Next step: run dataset_overview.py to generate EDA & visuals")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

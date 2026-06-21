"""
data_cleaning.py
=================
LA Crime Data – Data Cleaning & Categorization

Auto-detects the input file in the SAME FOLDER as this script:
    - Crime_Data*.csv   (the main LA crime dataset)

The MO_CODES.csv lookup file is used ONLY to help assign the simplified
crime `category` column (see step 4 below) — the lookup table itself is
not loaded, merged, or otherwise referenced anywhere else. The raw
`Mocodes` column from the original crime data IS kept as-is in the
cleaned output (it's just an original column, never dropped), so it
remains available for downstream MO-code feature engineering.

No file paths or flags need to be passed in — just drop this script into
the same folder as your CSV file(s) and run:

    python data_cleaning.py

Produces:
    cleaned_data/la_crime_cleaned.csv   <- cleaned + categorized dataset
                                            ready for dataset_overview.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 0.  PATH AUTO-DETECTION  (works on any machine, any folder location)
# =============================================================================

# Folder this script lives in — NOT the current working directory.
# This means the script works correctly even if run from a different
# directory (e.g. `python C:\some\path\data_cleaning.py`).
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "cleaned_data"
OUT_DIR.mkdir(exist_ok=True)


def _find_file(patterns: list, label: str, required: bool = True):
    """Search SCRIPT_DIR for the first file matching any of the given glob patterns."""
    for pattern in patterns:
        matches = sorted(SCRIPT_DIR.glob(pattern))
        if matches:
            if len(matches) > 1:
                print(f"  [WARN] Multiple matches for {label}, using first: {matches[0].name}")
            return matches[0]
    if required:
        print(f"\n  [ERROR] Could not find {label} in: {SCRIPT_DIR}")
        print(f"          Looked for patterns: {patterns}")
        print(f"          Make sure the CSV file is in the SAME folder as this script.")
        sys.exit(1)
    return None


def find_input_files() -> tuple:
    crime_path = _find_file(
        ["Crime_Data_from_2020_to_Present.csv", "Crime_Data*.csv", "*crime*data*.csv", "*Crime*.csv"],
        "main crime data CSV", required=True
    )
    # MO_CODES.csv is optional — only used transiently to assist category
    # assignment. If absent, category assignment falls back to crime-code +
    # description matching only (still fully functional).
    mo_path = _find_file(
        ["MO_CODES.csv", "mo_codes.csv", "*MO*CODE*.csv", "*mo*code*.csv"],
        "MO_CODES.csv", required=False
    )
    print(f"  Crime data file : {crime_path.name}")
    print(f"  MO codes file   : {mo_path.name if mo_path else '(not found — skipping, not required)'}")
    return crime_path, mo_path


# =============================================================================
# 1.  MO CODE LOOKUP  (used ONLY to help assign the `category` column below;
#     no MO-derived columns are added to the cleaned output)
# =============================================================================

def load_mo_code_categories(path) -> dict:
    """
    Reads MO_CODES.csv and returns a simple {code -> mo_category} dict.
    Reads line-by-line, splitting on the FIRST and LAST comma only, so
    embedded commas inside description text (e.g. "SEX, UNLAWFUL...")
    don't break parsing. Returns {} if path is None.
    """
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
    """
    Given a raw Mocodes string (space-separated 4-digit codes), returns the
    single most useful category hint to help disambiguate crime category
    assignment — specifically, whether 'Sex Related' MO codes are present.
    This is the ONLY use of MO codes in this script: a one-shot signal
    consumed immediately during category assignment, never stored as a
    standalone column.
    """
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
    """
    Maps numeric crime codes to one of 5 simplified categories:
    Violent, Property, Sexual Assault, Vehicle, Other.

    Resolution order for any row not explicitly covered by
    CRIME_CATEGORY_MAP:
      1. Keyword match on Crm Cd Desc text.
      2. If still unresolved and a Sex Related MO-code hint is present,
         classify as Sexual Assault (helps catch ambiguous crime
         descriptions that don't contain an obvious keyword).
      3. Otherwise: Other.

    Note: MO codes are used here only as a transient hint during this
    one assignment step — no MO-derived column is added to the output.
    """
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

    # MO codes are consulted here ONLY as a transient hint to help resolve
    # ambiguous category cases. The MO_CODES.csv lookup table itself is not
    # merged into the dataset — but the raw `Mocodes` column from the
    # original data IS kept as-is in the output (see main()), so it's
    # available as a feature for downstream work.
    mo_hint = None
    mo_col = next((c for c in df.columns if "mocode" in c or c == "mocodes"), None)
    if mo_col and mo_code_to_cat:
        mo_hint = df[mo_col].apply(lambda v: mo_codes_to_category_hint(v, mo_code_to_cat))

    df["category"] = assign_crime_category(df["crm_cd"], desc_col, mo_hint)

    print("  Crime 'category' assigned (Violent / Property / Sexual Assault / Vehicle / Other):")
    print(df["category"].value_counts().to_string())
    print(f"\n  Total columns after categorization : {df.shape[1]}")
    print("  ('category' is the only new column added in this step. The raw "
          "'Mocodes' column from the original data is preserved as-is in the "
          "output for downstream MO-code feature engineering.)")
    return df


# =============================================================================
# 5.  MAIN
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print("  STEP 1 – LOCATING & LOADING FILES")
    print(f"{'='*60}")
    print(f"  Script folder: {SCRIPT_DIR}")

    crime_path, mo_path = find_input_files()

    # MO_CODES.csv is read only to build a temporary code->category lookup
    # used during category assignment. It is not merged into the dataset.
    mo_code_to_cat = load_mo_code_categories(mo_path)

    df = pd.read_csv(crime_path, low_memory=False)
    print(f"  Raw shape : {df.shape}")

    df = clean_data(df)
    df = categorize_data(df, mo_code_to_cat)

    # NOTE: the raw Mocodes column is intentionally KEPT in the cleaned
    # output (not dropped). It's still the original space-separated code
    # string from the source data — useful as a feature on its own, and
    # whoever owns feature engineering can parse it further downstream.
    # We only ever used MO_CODES.csv transiently above to help assign
    # `category`; that lookup itself is not merged into the dataset.

    out_csv = OUT_DIR / "la_crime_cleaned.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Cleaned & categorized dataset saved -> {out_csv}")
    print(f"\n{'='*60}")
    print("  DATA CLEANING COMPLETE")
    print(f"  Next step: run dataset_overview.py to generate EDA & visuals")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

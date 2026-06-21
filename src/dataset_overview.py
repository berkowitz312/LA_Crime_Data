"""
dataset_overview.py
====================
LA Crime Data – Statistical Summary & Exploratory Data Analysis

Reads the cleaned dataset produced by data_cleaning.py:
    data/processed/la_crime_cleaned.csv   (path from src/config.py)

This script uses ONLY columns present in the cleaned dataset itself. No
MO_CODES.csv lookup or any MO-derived columns are referenced here — MO codes
are consumed entirely within data_cleaning.py (only as a one-time hint to
help assign the `category` column) and are not needed again afterwards.

No file paths or flags need to be passed in — paths come from src/config.py.
Just make sure you've run data_cleaning.py first, then run:

    python src/dataset_overview.py

Produces:
    outputs/eda_outputs/*.png         <- static charts
    outputs/eda_outputs/heatmap.html  <- interactive, filterable crime heat-map
                                     (filter by category AND year, using
                                     every available geocoded point up to
                                     a generous safety cap)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# All paths come from the central config in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# =============================================================================
# 0.  PATHS  (resolved from the central config — independent of the CWD)
# =============================================================================

OUT_DIR = cfg.ensure_dir(cfg.EDA_DIR)   # outputs/eda_outputs/


def find_cleaned_file() -> Path:
    """Return the cleaned dataset path from config, erroring if it's not there yet."""
    path = cfg.CLEANED_DATA_PATH
    if not path.exists():
        print(f"\n  [ERROR] Could not find the cleaned CSV file.")
        print(f"          Looked for: {path}")
        print(f"          Run data_cleaning.py first — it produces "
              f"data/processed/la_crime_cleaned.csv")
        sys.exit(1)
    return path


# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
ACCENT  = "#E63946"
COOL    = "#457B9D"
WARM    = "#F4A261"
DARK_BG = "#1D3557"

CATEGORY_COLORS = {
    "Violent":        "#E63946",
    "Property":       "#457B9D",
    "Sexual Assault": "#9D4EDD",
    "Vehicle":        "#F4A261",
    "Other":          "#6C757D",
}


def _save(fig: plt.Figure, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


# =============================================================================
# 1.  STATISTICAL SUMMARY
# =============================================================================

def statistical_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  STATISTICAL SUMMARY")
    print(f"{'='*60}")

    print("\n  -- Numeric columns --")
    print(df.describe(include=[np.number]).T.round(2).to_string())

    print("\n  -- Top categoricals --")
    for col in ["crm_cd_desc", "area_name", "premis_desc",
                "category", "status_desc", "vict_sex", "vict_descent"]:
        if col in df.columns:
            vc = df[col].value_counts()
            print(f"\n  {col}  (unique={df[col].nunique():,})")
            print(vc.head(6).to_string())


# =============================================================================
# 2.  STATIC EDA PLOTS
# =============================================================================

def eda_overview(df: pd.DataFrame):
    print("\n  [EDA] 01 -- Overview")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold", color=DARK_BG)

    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(20)
    axes[0].barh(miss.index[::-1], miss.values[::-1], color=ACCENT, edgecolor="white")
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_title("Top-20 Columns – Missing Data")
    for i, v in enumerate(miss.values[::-1]):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    dtype_counts = df.dtypes.astype(str).value_counts()
    axes[1].pie(dtype_counts.values, labels=dtype_counts.index, autopct="%1.0f%%",
                colors=[ACCENT, COOL, WARM, DARK_BG, "#A8DADC"])
    axes[1].set_title("Column Data Types")

    plt.tight_layout()
    _save(fig, "01_overview")


def eda_crime_over_time(df: pd.DataFrame):
    print("\n  [EDA] 02 -- Crime over time")
    if "date_occ" not in df.columns:
        return
    df2 = df.copy()
    df2["date_occ"] = pd.to_datetime(df2["date_occ"], errors="coerce")
    df2 = df2.dropna(subset=["date_occ"])
    monthly = df2.groupby(df2["date_occ"].dt.to_period("M")).size().reset_index(name="count")
    monthly["date"] = monthly["date_occ"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Crime Frequency Over Time", fontsize=16, fontweight="bold", color=DARK_BG)

    axes[0].plot(monthly["date"], monthly["count"], color=ACCENT, linewidth=1.8)
    axes[0].fill_between(monthly["date"], monthly["count"], alpha=0.15, color=ACCENT)
    axes[0].set_title("Monthly Crime Count (all years)")
    axes[0].set_ylabel("Incidents")
    axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    axes[0].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    df2["year"]  = df2["date_occ"].dt.year
    df2["month"] = df2["date_occ"].dt.month
    ym = df2.groupby(["year", "month"]).size().reset_index(name="count")
    for yr, grp in ym.groupby("year"):
        axes[1].plot(grp["month"], grp["count"], marker="o",
                     label=str(yr), linewidth=1.5, markersize=4)
    axes[1].set_title("Year-over-Year Monthly Seasonality")
    axes[1].set_xlabel("Month"); axes[1].set_ylabel("Incidents")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"])
    axes[1].legend(title="Year", loc="upper right")

    plt.tight_layout()
    _save(fig, "02_crime_over_time")


def eda_temporal_patterns(df: pd.DataFrame):
    print("\n  [EDA] 03 -- Temporal patterns")
    df2 = df.copy()
    if "date_occ" in df2.columns:
        df2["date_occ"] = pd.to_datetime(df2["date_occ"], errors="coerce")
        df2["dow_name"] = df2["date_occ"].dt.strftime("%A")
        df2["dow"]      = df2["date_occ"].dt.dayofweek
    if "time_occ" in df2.columns:
        df2["hour"] = (pd.to_numeric(df2["time_occ"], errors="coerce") // 100)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Temporal Crime Patterns", fontsize=16, fontweight="bold", color=DARK_BG)

    if "dow_name" in df2.columns:
        dow_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_counts = df2["dow_name"].value_counts().reindex(dow_order).fillna(0)
        bars = axes[0].bar(dow_counts.index, dow_counts.values, color=COOL, edgecolor="white")
        for b in [5, 6]:
            bars[b].set_color(ACCENT)
        axes[0].set_title("By Day of Week")
        axes[0].set_ylabel("Incidents")
        axes[0].tick_params(axis="x", rotation=30)

    if "hour" in df2.columns:
        hc = df2["hour"].dropna().astype(int).value_counts().sort_index()
        axes[1].bar(hc.index, hc.values, color=WARM, edgecolor="white")
        axes[1].set_title("By Hour of Day")
        axes[1].set_xlabel("Hour (24h)"); axes[1].set_ylabel("Incidents")
        axes[1].set_xticks(range(0, 24, 2))

    if "hour" in df2.columns and "dow" in df2.columns:
        hm = (
            df2.dropna(subset=["hour","dow"])
            .assign(hour=lambda d: d["hour"].astype(int))
            .groupby(["dow","hour"]).size().unstack(fill_value=0)
        )
        hm.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(hm)]
        sns.heatmap(hm, ax=axes[2], cmap="YlOrRd", linewidths=0.3,
                    cbar_kws={"label": "Count"})
        axes[2].set_title("Heatmap: Day x Hour")
        axes[2].set_xlabel("Hour"); axes[2].set_ylabel("Day")

    plt.tight_layout()
    _save(fig, "03_temporal_patterns")


def eda_crime_types(df: pd.DataFrame):
    print("\n  [EDA] 04 -- Crime types & simplified category")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Crime Type Analysis", fontsize=16, fontweight="bold", color=DARK_BG)

    if "crm_cd_desc" in df.columns:
        top = df["crm_cd_desc"].value_counts().head(20)
        axes[0].barh(top.index[::-1], top.values[::-1], color=ACCENT, edgecolor="white")
        axes[0].set_title("Top 20 Crime Types (detailed)")
        axes[0].set_xlabel("Incidents")
        for i, v in enumerate(top.values[::-1]):
            axes[0].text(v * 1.005, i, f"{v:,}", va="center", fontsize=7.5)

    if "category" in df.columns:
        cat = df["category"].value_counts()
        colors = [CATEGORY_COLORS.get(c, "#999999") for c in cat.index]
        axes[1].pie(cat.values, labels=cat.index, autopct="%1.1f%%",
                    colors=colors, wedgeprops={"edgecolor": "white"}, startangle=140)
        axes[1].set_title("Simplified Crime Category\n(Violent / Property / Sexual Assault / Vehicle / Other)")

    plt.tight_layout()
    _save(fig, "04_crime_types")


def eda_geographic_static(df: pd.DataFrame):
    print("\n  [EDA] 05 -- Geographic distribution (static)")
    if "area_name" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    ac = df["area_name"].value_counts().sort_values()
    cvals = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(ac)))
    ax.barh(ac.index, ac.values, color=cvals, edgecolor="white")
    ax.set_title("Crime Incidents by LAPD Area", fontsize=14, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Incidents")
    for i, v in enumerate(ac.values):
        ax.text(v * 1.003, i, f"{v:,}", va="center", fontsize=8)
    plt.tight_layout()
    _save(fig, "05_area_distribution")


def eda_victim_profile(df: pd.DataFrame):
    print("\n  [EDA] 06 -- Victim profiles")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Victim Demographic Analysis", fontsize=16, fontweight="bold", color=DARK_BG)

    if "vict_age" in df.columns:
        ages = pd.to_numeric(df["vict_age"], errors="coerce").dropna()
        ages = ages[ages > 0]   # exclude unknown-age placeholder of 0
        axes[0,0].hist(ages, bins=40, color=COOL, edgecolor="white")
        axes[0,0].axvline(ages.median(), color=ACCENT, linewidth=2,
                          label=f"Median: {ages.median():.0f}")
        axes[0,0].set_title("Victim Age Distribution")
        axes[0,0].set_xlabel("Age"); axes[0,0].legend()

    if "vict_sex" in df.columns:
        # vict_sex already contains full labels (Male/Female/Unknown) from data_cleaning.py
        sx = df["vict_sex"].dropna().value_counts()
        bar_colors = [COOL, ACCENT, "#999999"][:len(sx)]
        axes[0,1].bar(sx.index, sx.values, color=bar_colors, edgecolor="white")
        axes[0,1].set_title("Victim Sex"); axes[0,1].set_ylabel("Count")

    if "vict_descent" in df.columns:
        # vict_descent already contains full labels from data_cleaning.py
        ds = df["vict_descent"].dropna().value_counts().head(10)
        axes[1,0].bar(ds.index, ds.values, color=DARK_BG, edgecolor="white")
        axes[1,0].set_title("Victim Descent (Top 10)")
        axes[1,0].tick_params(axis="x", rotation=40)

    if "category" in df.columns and "vict_age" in df.columns:
        df_age = df.copy()
        df_age["vict_age"] = pd.to_numeric(df_age["vict_age"], errors="coerce")
        df_age = df_age[df_age["vict_age"] > 0]
        sns.boxplot(data=df_age, x="category", y="vict_age", ax=axes[1,1],
                   palette=CATEGORY_COLORS)
        axes[1,1].set_title("Victim Age by Crime Category")
        axes[1,1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, "06_victim_profiles")


def eda_weapons_and_status(df: pd.DataFrame):
    print("\n  [EDA] 07 -- Weapons & case status")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Weapon Use & Case Status", fontsize=16, fontweight="bold", color=DARK_BG)

    if "weapon_desc" in df.columns:
        wep = df["weapon_desc"].dropna().value_counts().head(15)
        axes[0].barh(wep.index[::-1], wep.values[::-1], color=ACCENT, edgecolor="white")
        axes[0].set_title("Top 15 Weapon Types")
        axes[0].set_xlabel("Incidents")

    if "status_desc" in df.columns:
        st = df["status_desc"].value_counts()
        axes[1].pie(st.values, labels=st.index, autopct="%1.1f%%",
                    colors=[COOL, WARM, ACCENT, DARK_BG, "#A8DADC"],
                    wedgeprops={"edgecolor":"white"})
        axes[1].set_title("Case Status Distribution")

    plt.tight_layout()
    _save(fig, "07_weapons_status")


def eda_premises(df: pd.DataFrame):
    print("\n  [EDA] 08 -- Premises")
    if "premis_desc" not in df.columns:
        return
    top_prem = df["premis_desc"].value_counts().head(20)
    fig, ax  = plt.subplots(figsize=(14, 8))
    ax.barh(top_prem.index[::-1], top_prem.values[::-1], color=WARM, edgecolor="white")
    ax.set_title("Top 20 Crime Premises", fontsize=14, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Incidents")
    plt.tight_layout()
    _save(fig, "08_premises")


def eda_category_by_year(df: pd.DataFrame):
    print("\n  [EDA] 09 -- Crime category by year")
    if "date_occ" not in df.columns or "category" not in df.columns:
        return
    df2 = df.copy()
    df2["date_occ"] = pd.to_datetime(df2["date_occ"], errors="coerce")
    df2["year"] = df2["date_occ"].dt.year
    pivot = df2.dropna(subset=["year"]).groupby(["year","category"]).size().unstack(fill_value=0)
    colors = [CATEGORY_COLORS.get(c, "#999999") for c in pivot.columns]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(14, 7),
                    color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Crime Categories by Year", fontsize=14, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Year"); ax.set_ylabel("Incidents")
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    _save(ax.figure, "09_category_by_year")


def eda_correlation(df: pd.DataFrame):
    print("\n  [EDA] 10 -- Correlation matrix")
    numerics = df.select_dtypes(include=[np.number]).copy()
    drop_ids = ["dr_no","rpt_dist_no","crm_cd","premis_cd",
                "weapon_used_cd","crm_cd_1","crm_cd_2","crm_cd_3","crm_cd_4"]
    numerics.drop(columns=drop_ids, errors="ignore", inplace=True)
    numerics = numerics.dropna(thresh=int(0.4 * len(numerics)), axis=1)
    if numerics.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numerics.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", color=DARK_BG)
    plt.tight_layout()
    _save(fig, "10_correlation")


# =============================================================================
# 3.  INTERACTIVE FILTERABLE HEATMAP  (category + year filters)
# =============================================================================

def build_interactive_heatmap(df: pd.DataFrame, max_points: int = 300_000):
    """
    Builds a single self-contained HTML file with a Leaflet heat-map.
    Includes dropdown filters for:
        - Crime Category (Violent / Property / Sexual Assault / Vehicle / Other / All)
        - Year (each year present in the data, or All)

    `max_points` is a safety ceiling (default 300,000) to keep the HTML
    file from becoming unreasonably large/slow to open in a browser. If the
    dataset has fewer geocoded rows than this, ALL of them are used with no
    downsampling whatsoever. If downsampling is required, it is done in a
    stratified way (proportionally across category x year groups) so no
    category or year is ever dropped entirely from the map.

    No external API key needed (uses OpenStreetMap tiles + leaflet.heat,
    both loaded from public CDNs).
    """
    print("\n  [EDA] 11 -- Interactive filterable heat-map")

    needed = ["lat", "lon"]
    if not all(c in df.columns for c in needed):
        print("      (skipped – lat/lon columns not present)")
        return

    geo = df.dropna(subset=["lat", "lon"]).copy()

    if "category" not in geo.columns:
        geo["category"] = "Other"
    if "date_occ" in geo.columns:
        geo["date_occ"] = pd.to_datetime(geo["date_occ"], errors="coerce")
        geo["year"] = geo["date_occ"].dt.year
    else:
        geo["year"] = np.nan

    geo = geo.dropna(subset=["year"])
    geo["year"] = geo["year"].astype(int)

    total_available = len(geo)
    if total_available > max_points:
        # Stratified sample: each (category, year) group keeps the same
        # proportion of points, so every filter combination still has data.
        # Done via sampled index positions (robust across pandas versions,
        # rather than relying on groupby().apply() return semantics).
        frac = max_points / total_available
        sampled_indices = []
        for _, grp in geo.groupby(["category", "year"]):
            n_keep = max(1, int(round(len(grp) * frac)))
            sampled_indices.extend(
                grp.sample(n=min(n_keep, len(grp)), random_state=42).index.tolist()
            )
        geo = geo.loc[sampled_indices]
        print(f"  {total_available:,} geocoded points available; "
              f"sampled {len(geo):,} (proportional across category x year) "
              f"to stay under the {max_points:,} safety cap.")
    else:
        print(f"  Using all {total_available:,} geocoded points — under the "
              f"{max_points:,} cap, no downsampling needed.")

    categories = sorted(geo["category"].dropna().unique().tolist())
    years      = sorted(geo["year"].dropna().unique().tolist())

    # Build point payload grouped by (category, year) so the front-end
    # can instantly filter without re-querying anything server-side.
    points_by_group = {}
    for (cat, yr), grp in geo.groupby(["category", "year"]):
        key = f"{cat}||{yr}"
        points_by_group[key] = grp[["lat", "lon"]].values.round(5).tolist()

    points_json = json.dumps(points_by_group)
    categories_json = json.dumps(categories)
    years_json = json.dumps(years)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>LA Crime Heat-map — Filterable by Category &amp; Year</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; font-family: -apple-system, Segoe UI, Roboto, sans-serif; }}
  #map {{ position:absolute; top:64px; bottom:0; left:0; right:0; }}
  #controls {{
    position:absolute; top:0; left:0; right:0; height:64px;
    background:#1D3557; color:white; display:flex; align-items:center;
    gap:18px; padding:0 20px; box-shadow:0 2px 8px rgba(0,0,0,0.25); z-index:1000;
  }}
  #controls h1 {{ font-size:15px; margin:0; margin-right:8px; font-weight:600; white-space:nowrap; }}
  .ctrl-group {{ display:flex; align-items:center; gap:6px; }}
  .ctrl-group label {{ font-size:12px; opacity:0.85; }}
  select {{
    padding:6px 10px; border-radius:6px; border:none; font-size:13px;
    background:white; color:#1D3557; cursor:pointer;
  }}
  #count {{ margin-left:auto; font-size:13px; opacity:0.9; white-space:nowrap; }}
  .legend {{
    position:absolute; bottom:20px; right:20px; background:white;
    padding:10px 14px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.25);
    font-size:12px; color:#1D3557; z-index:999;
  }}
</style>
</head>
<body>

<div id="controls">
  <h1>LA Crime Heat-map</h1>
  <div class="ctrl-group">
    <label>Category</label>
    <select id="categorySelect"></select>
  </div>
  <div class="ctrl-group">
    <label>Year</label>
    <select id="yearSelect"></select>
  </div>
  <div id="count"></div>
</div>

<div id="map"></div>
<div class="legend">Heat intensity = incident density<br/>Use dropdowns to filter</div>

<script>
  const pointsByGroup = {points_json};
  const categories = {categories_json};
  const years = {years_json};

  const map = L.map('map').setView([34.05, -118.24], 10);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 19
  }}).addTo(map);

  let heatLayer = L.heatLayer([], {{ radius: 8, blur: 12, minOpacity: 0.35, maxZoom: 17 }}).addTo(map);

  const catSelect = document.getElementById('categorySelect');
  const yearSelect = document.getElementById('yearSelect');
  const countEl = document.getElementById('count');

  function addOption(select, value, label) {{
    const opt = document.createElement('option');
    opt.value = value; opt.textContent = label;
    select.appendChild(opt);
  }}

  addOption(catSelect, 'All', 'All Categories');
  categories.forEach(c => addOption(catSelect, c, c));

  addOption(yearSelect, 'All', 'All Years');
  years.forEach(y => addOption(yearSelect, y, y));

  function getFilteredPoints() {{
    const cat = catSelect.value;
    const yr = yearSelect.value;
    const catList = cat === 'All' ? categories : [cat];
    const yearList = yr === 'All' ? years : [parseInt(yr)];

    let pts = [];
    catList.forEach(c => {{
      yearList.forEach(y => {{
        const key = c + '||' + y;
        if (pointsByGroup[key]) {{
          pts = pts.concat(pointsByGroup[key]);
        }}
      }});
    }});
    return pts;
  }}

  function updateMap() {{
    const pts = getFilteredPoints();
    heatLayer.setLatLngs(pts);
    countEl.textContent = pts.length.toLocaleString() + ' incidents shown';
  }}

  catSelect.addEventListener('change', updateMap);
  yearSelect.addEventListener('change', updateMap);

  updateMap();
</script>

</body>
</html>
"""

    out_path = OUT_DIR / "heatmap.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  -> saved: {out_path}")
    print(f"  Open this file in any browser. Filter by Category and/or Year using "
          f"the dropdowns at the top.")


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print("  LOADING CLEANED DATASET")
    print(f"{'='*60}")
    print(f"  Project root : {cfg.PROJECT_ROOT}")
    print(f"  Output dir   : {OUT_DIR}")

    cleaned_path = find_cleaned_file()
    print(f"  Cleaned file : {cleaned_path.name}")
    df = pd.read_csv(cleaned_path, low_memory=False)
    print(f"  Shape        : {df.shape}")

    statistical_summary(df)

    print(f"\n{'='*60}")
    print("  EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"  Saving plots to: {OUT_DIR}/\n")

    eda_overview(df)
    eda_crime_over_time(df)
    eda_temporal_patterns(df)
    eda_crime_types(df)
    eda_geographic_static(df)
    eda_victim_profile(df)
    eda_weapons_and_status(df)
    eda_premises(df)
    eda_category_by_year(df)
    eda_correlation(df)
    build_interactive_heatmap(df)

    print(f"\n{'='*60}")
    print("  EDA COMPLETE")
    print(f"  All outputs in: {OUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
Exploratory data analysis on the cleaned dataset. Reads
data/processed/la_crime_cleaned.csv and writes static charts plus an
interactive heat-map to outputs/eda_outputs/.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# =============================================================================
# 0.  PATHS
# =============================================================================

OUT_DIR = cfg.ensure_dir(cfg.EDA_DIR)


def find_cleaned_file() -> Path:
    """Return the cleaned dataset path, erroring if data_cleaning.py hasn't run."""
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
    "Violent":  "#E63946",
    "Property": "#457B9D",
    "Vehicle":  "#F4A261",
    "Other":    "#6C757D",
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
        axes[1].set_title("Simplified Crime Category\n(Violent / Property / Vehicle / Other)")

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
# 3.  SCATTER & OUTLIER PLOTS  (feature-category relationships)
# =============================================================================

def eda_scatter_relationships(df: pd.DataFrame):
    """Four panels showing how key features separate the target categories —
    directly useful for assessing classification model feature value."""
    print("\n  [EDA] 11 -- Scatter plots: feature-category relationships")

    df2 = df.copy()
    # Derive hour / day_of_week locally so this works on the cleaned CSV directly
    if "hour" not in df2.columns and "time_occ" in df2.columns:
        df2["hour"] = (pd.to_numeric(df2["time_occ"], errors="coerce") // 100)
    if "day_of_week" not in df2.columns and "date_occ" in df2.columns:
        df2["day_of_week"] = pd.to_datetime(df2["date_occ"], errors="coerce").dt.dayofweek

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Feature Relationships by Crime Category\n(for classification model insight)",
                 fontsize=15, fontweight="bold", color=DARK_BG)

    # ── A: Geographic scatter ────────────────────────────────────────────────
    if {"lat", "lon", "category"}.issubset(df2.columns):
        geo = df2.dropna(subset=["lat", "lon", "category"])
        sample = geo.sample(min(40_000, len(geo)), random_state=42)
        for cat, grp in sample.groupby("category"):
            axes[0, 0].scatter(grp["lon"], grp["lat"], s=1, alpha=0.25,
                               color=CATEGORY_COLORS.get(cat, "#999999"), label=cat)
        axes[0, 0].set_title("A — Geographic Clusters by Category\n(sampled 40 k points)")
        axes[0, 0].set_xlabel("Longitude")
        axes[0, 0].set_ylabel("Latitude")
        axes[0, 0].legend(markerscale=6, title="Category", fontsize=8, loc="upper left")

    # ── B: Victim age vs Hour of day ─────────────────────────────────────────
    if {"vict_age", "hour", "category"}.issubset(df2.columns):
        tmp = df2.dropna(subset=["vict_age", "hour", "category"]).copy()
        tmp["vict_age"] = pd.to_numeric(tmp["vict_age"], errors="coerce")
        tmp = tmp[tmp["vict_age"] > 0]
        sample = tmp.sample(min(20_000, len(tmp)), random_state=42)
        rng = np.random.default_rng(42)
        for cat, grp in sample.groupby("category"):
            jitter = rng.uniform(-0.4, 0.4, len(grp))
            axes[0, 1].scatter(grp["hour"] + jitter, grp["vict_age"],
                               s=2, alpha=0.2,
                               color=CATEGORY_COLORS.get(cat, "#999999"), label=cat)
        axes[0, 1].set_title("B — Victim Age vs. Hour of Day by Category\n(jittered, sampled 20 k)")
        axes[0, 1].set_xlabel("Hour of Day (0–23)")
        axes[0, 1].set_ylabel("Victim Age")
        axes[0, 1].set_xticks(range(0, 24, 3))
        axes[0, 1].legend(markerscale=4, title="Category", fontsize=8)

    # ── C: Category composition per LAPD area (% heatmap) ───────────────────
    if {"area_name", "category"}.issubset(df2.columns):
        pivot = (df2.groupby(["area_name", "category"]).size()
                   .unstack(fill_value=0))
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
        sns.heatmap(pivot_pct, ax=axes[1, 0], cmap="YlOrRd",
                    annot=True, fmt=".0f", linewidths=0.4,
                    annot_kws={"size": 7},
                    cbar_kws={"label": "% of area's crimes"})
        axes[1, 0].set_title("C — Category Composition per LAPD Area (%)\n"
                             "(each row sums to 100 %)")
        axes[1, 0].set_xlabel("Crime Category")
        axes[1, 0].set_ylabel("LAPD Area")
        axes[1, 0].tick_params(axis="x", rotation=25, labelsize=8)
        axes[1, 0].tick_params(axis="y", labelsize=7)

    # ── D: Mean hour of crime by category × day of week ─────────────────────
    if {"hour", "day_of_week", "category"}.issubset(df2.columns):
        tmp = df2.dropna(subset=["hour", "day_of_week", "category"])
        pivot_hr = (tmp.groupby(["day_of_week", "category"])["hour"]
                       .mean().unstack())
        pivot_hr.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(pivot_hr)]
        sns.heatmap(pivot_hr, ax=axes[1, 1], cmap="coolwarm", center=12,
                    annot=True, fmt=".1f", linewidths=0.5,
                    annot_kws={"size": 9},
                    cbar_kws={"label": "Mean Hour of Day"})
        axes[1, 1].set_title("D — Mean Occurrence Hour: Day of Week × Category\n"
                             "(cooler = earlier in day; warmer = later)")
        axes[1, 1].set_xlabel("Crime Category")
        axes[1, 1].set_ylabel("Day of Week")
        axes[1, 1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    _save(fig, "11_scatter_relationships")


def eda_outlier_analysis(df: pd.DataFrame):
    """Four panels for identifying outliers in key numeric features, broken down
    by crime category where relevant."""
    print("\n  [EDA] 12 -- Outlier analysis")

    df2 = df.copy()
    if "hour" not in df2.columns and "time_occ" in df2.columns:
        df2["hour"] = (pd.to_numeric(df2["time_occ"], errors="coerce") // 100)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Outlier Analysis — Key Numeric Features",
                 fontsize=15, fontweight="bold", color=DARK_BG)

    # ── A: Boxplot victim age by category (flier dots visible) ───────────────
    if {"vict_age", "category"}.issubset(df2.columns):
        tmp = df2.copy()
        tmp["vict_age"] = pd.to_numeric(tmp["vict_age"], errors="coerce")
        tmp = tmp[tmp["vict_age"] > 0].dropna(subset=["category"])
        sns.boxplot(data=tmp, x="category", y="vict_age", ax=axes[0, 0],
                    palette=CATEGORY_COLORS,
                    flierprops={"marker": ".", "markersize": 2,
                                "alpha": 0.3, "markerfacecolor": ACCENT})
        axes[0, 0].set_title("A — Victim Age by Category\n(IQR box; dots = outlier observations)")
        axes[0, 0].set_xlabel("Category")
        axes[0, 0].set_ylabel("Age")
        axes[0, 0].tick_params(axis="x", rotation=20)

    # ── B: Violin of hour by category ────────────────────────────────────────
    if {"hour", "category"}.issubset(df2.columns):
        tmp = df2.dropna(subset=["hour", "category"])
        sns.violinplot(data=tmp, x="category", y="hour", ax=axes[0, 1],
                       palette=CATEGORY_COLORS, inner="box", cut=0)
        axes[0, 1].set_title("B — Hour of Occurrence by Category\n(violin shape + IQR box inside)")
        axes[0, 1].set_xlabel("Category")
        axes[0, 1].set_ylabel("Hour of Day")
        axes[0, 1].set_yticks(range(0, 24, 3))
        axes[0, 1].tick_params(axis="x", rotation=20)

    # ── C: IQR outlier rate per numeric feature ───────────────────────────────
    drop_ids = {"dr_no", "rpt_dist_no", "crm_cd", "premis_cd",
                "weapon_used_cd", "crm_cd_1", "crm_cd_2", "crm_cd_3", "crm_cd_4"}
    num_cols = [c for c in df2.select_dtypes(include=[np.number]).columns
                if c not in drop_ids]
    outlier_pct = {}
    for col in num_cols:
        s = df2[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        outlier_pct[col] = round(n_out / len(s) * 100, 2)

    if outlier_pct:
        oc = pd.Series(outlier_pct).sort_values(ascending=False)
        axes[1, 0].barh(oc.index[::-1], oc.values[::-1], color=WARM, edgecolor="white")
        axes[1, 0].set_title("C — IQR Outlier Rate per Numeric Feature\n"
                             "(% of non-null values outside 1.5 × IQR)")
        axes[1, 0].set_xlabel("Outlier %")
        for i, v in enumerate(oc.values[::-1]):
            axes[1, 0].text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=8)

    # ── D: Victim age histogram with ±3σ bounds ──────────────────────────────
    if "vict_age" in df2.columns:
        ages = pd.to_numeric(df2["vict_age"], errors="coerce").dropna()
        ages = ages[ages > 0]
        mean_a, std_a = ages.mean(), ages.std()
        lo, hi = mean_a - 3 * std_a, mean_a + 3 * std_a
        n_out = int(((ages < lo) | (ages > hi)).sum())

        axes[1, 1].hist(ages, bins=60, color=COOL, edgecolor="white", alpha=0.8)
        if lo > ages.min():
            axes[1, 1].axvspan(ages.min(), lo, alpha=0.15, color=ACCENT)
        axes[1, 1].axvspan(hi, ages.max(), alpha=0.15, color=ACCENT)
        axes[1, 1].axvline(lo, color=ACCENT, linestyle="--", linewidth=1.8,
                           label=f"−3σ  ({lo:.0f} yrs)")
        axes[1, 1].axvline(hi, color=ACCENT, linestyle="--", linewidth=1.8,
                           label=f"+3σ  ({hi:.0f} yrs)")
        axes[1, 1].axvline(mean_a, color=DARK_BG, linestyle="-", linewidth=1.5,
                           label=f"Mean  ({mean_a:.1f} yrs)")
        axes[1, 1].set_title(f"D — Victim Age Distribution with ±3σ Outlier Bounds\n"
                             f"({n_out:,} outliers — {n_out / len(ages) * 100:.1f}% of records)")
        axes[1, 1].set_xlabel("Age")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "12_outlier_analysis")


# =============================================================================
# 4.  INTERACTIVE FILTERABLE HEATMAP  (category + year filters)
# =============================================================================

def build_interactive_heatmap(df: pd.DataFrame, max_points: int = 300_000):
    """Write a self-contained Leaflet heat-map (HTML) with category and year
    filters. Caps the points at max_points, sampling proportionally across
    category x year so no group disappears. Uses public CDN tiles, no API key."""
    print("\n  [EDA] 13 -- Interactive filterable heat-map")

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
        # Proportional sample per (category, year) so every filter keeps points.
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

    # Group points by (category, year) so the page can filter client-side.
    points_by_group = {}
    for (cat, yr), grp in geo.groupby(["category", "year"]):
        key = f"{cat}||{yr}"
        points_by_group[key] = grp[["lat", "lon"]].values.round(5).tolist()

    points_json = json.dumps(points_by_group)
    categories_json = json.dumps(categories)
    years_json = json.dumps(years)

    boundary_json = "null"
    boundary_path = cfg.DATA_DIR / "City_Boundary.geojson"
    if boundary_path.exists():
        with open(boundary_path, "r", encoding="utf-8") as _f:
            boundary_json = _f.read()
        print(f"  City boundary loaded from: {boundary_path.name}")
    else:
        print(f"  [WARN] City_Boundary.geojson not found in data/ — boundary skipped")

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
<div class="legend">
  Heat intensity = incident density<br/>
  Use dropdowns to filter<br/>
  <span style="color:#FFE566;font-weight:bold;">— — —</span> City of Los Angeles boundary
</div>

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

  // City of Los Angeles boundary — embedded from City_Boundary.geojson
  (function() {{
    var boundaryData = {boundary_json};
    if (!boundaryData) return;
    // Glow layer — thick, low opacity
    L.geoJSON(boundaryData, {{
      style: {{
        color: '#FFFFFF',
        weight: 10,
        opacity: 0.18,
        fillOpacity: 0.04,
        fillColor: '#FFFFFF'
      }}
    }}).addTo(map);
    // Sharp border on top
    L.geoJSON(boundaryData, {{
      style: {{
        color: '#FFE566',
        weight: 3,
        opacity: 1,
        fillOpacity: 0,
        dashArray: '12 5'
      }}
    }}).addTo(map);
  }})();
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
    eda_scatter_relationships(df)
    eda_outlier_analysis(df)
    build_interactive_heatmap(df)

    print(f"\n{'='*60}")
    print("  EDA COMPLETE")
    print(f"  All outputs in: {OUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

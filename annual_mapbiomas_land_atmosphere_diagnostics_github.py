#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annual MapBiomas land-transformation diagnostics and spatial associations with
ERA5-derived heatwave and atmospheric-drying trends, 1990–2024.

Purpose
-------
Construct annual municipality-level land-transformation metrics from MapBiomas
Collection 10.1 and compare their spatial patterns with ERA5-derived heatwave
and atmospheric-drying trends.

ERA5 is treated as an atmospheric reanalysis product. The analysis does not
assume that ERA5 dynamically represents annual MapBiomas land-cover changes.
The annual land-cover record is used independently to compare cumulative
transformed-land state with time-varying transformation metrics.

Land metrics
------------
For each municipality:
  1. transformed/non-native land fraction in 1990;
  2. transformed/non-native land fraction in 2024;
  3. mean transformed fraction over 1990–2024;
  4. absolute transformed-land change, 2024 minus 1990;
  5. Theil–Sen trend in transformed land fraction;
  6. early-period transformation rate, 1990–2004;
  7. late-period transformation rate, 2010–2024.

Atmospheric and heatwave outcomes
---------------------------------
The workflow uses municipality-level Figure 03/02 products when available.
The DHW–HHW response must use the common-scale Tmax-only contrast:

    DHW_minus_HHW_TmaxOnly_intensity_trend_decade

or its municipality-table alias:

    dry_minus_humid_intensity_trend

Primary DHW and HHW regime-specific intensity trends are not subtracted.

Statistical analysis
--------------------
Controlled weighted least-squares (WLS) models use:

    outcome ~ z(land_predictor) + C(focus_region) + latitude + longitude

with weights proportional to the square root of municipal area and HC3
heteroscedasticity-robust standard errors. Weighted Spearman correlations use
the same square-root-area weights and two-sided permutation p-values.

These analyses quantify spatial associations and are not interpreted as causal
attribution, mediation, or land-cover sensitivity experiments.

Usage
-----
python annual_mapbiomas_land_atmosphere_diagnostics.py \
    --mapbiomas-coverage /path/to/mapbiomas_coverage.xlsx \
    --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
    --atmospheric-table /path/to/figure_03_municipality_mechanism_table_1990_2024.csv \
    --output-dir ./outputs/annual_mapbiomas_diagnostics

The atmospheric table may be a Figure 03 or Figure 02 municipality-level
CSV/XLSX/NetCDF product containing the common-scale DHW–HHW fields.
"""

import os
import re
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geopandas as gpd
from scipy import stats
from scipy.stats import theilslopes
import statsmodels.api as sm

try:
    import xarray as xr
except Exception:
    xr = None


# ============================================================
# Runtime paths
# ============================================================

COVERAGE_XLSX = None
MUNICIPALITY_SHP = None
ATMOSPHERIC_TABLE = None
OUT_DIR = None


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build annual MapBiomas land-transformation diagnostics and "
            "spatial associations with ERA5-derived outcomes."
        )
    )
    parser.add_argument(
        "--mapbiomas-coverage",
        required=True,
        help="MapBiomas Collection 10.1 municipality/state/biome workbook.",
    )
    parser.add_argument(
        "--municipality-shapefile",
        required=True,
        help="Brazilian municipality shapefile.",
    )
    parser.add_argument(
        "--atmospheric-table",
        required=True,
        help=(
            "Figure 03 or Figure 02 municipality-level CSV/XLSX/NetCDF table "
            "containing the common-scale DHW-HHW fields."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def configure_paths(args):
    global COVERAGE_XLSX, MUNICIPALITY_SHP, ATMOSPHERIC_TABLE, OUT_DIR

    COVERAGE_XLSX = str(Path(args.mapbiomas_coverage).expanduser().resolve())
    MUNICIPALITY_SHP = str(
        Path(args.municipality_shapefile).expanduser().resolve()
    )
    ATMOSPHERIC_TABLE = str(Path(args.atmospheric_table).expanduser().resolve())
    OUT_DIR = str(Path(args.output_dir).expanduser().resolve())

    required = [
        (COVERAGE_XLSX, "MapBiomas coverage workbook"),
        (MUNICIPALITY_SHP, "municipality shapefile"),
        (ATMOSPHERIC_TABLE, "atmospheric/heatwave municipality table"),
    ]
    for path, label in required:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Scientific configuration
# ============================================================

YEAR0, YEAR1 = 1990, 2024
YEARS = list(range(YEAR0, YEAR1 + 1))
EARLY0, EARLY1 = 1990, 2004
LATE0, LATE1 = 2010, 2024
MIN_AREA_HA = 1000.0
MIN_YEARS_FOR_LAND_TREND = 20

LON_MIN, LON_MAX = -75.5, -32.0
LAT_MIN, LAT_MAX = -35.5, 6.5

NATIVE_LEVEL1 = {"1. forest", "2. non forest natural formation"}
AGRO_LEVEL1 = {"3. farming"}
URBAN_LEVEL2 = {"4.2. urban area"}

MATOPIBA_STATES = {"MA", "TO", "PI", "BA"}
SEMIARID_STATES = {"AL", "BA", "CE", "PB", "PE", "PI", "RN", "SE", "MG"}
SOUTHEAST_STATES = {"SP", "RJ", "MG", "ES"}

REGION_ORDER = [
    "Amazon", "Cerrado", "MATOPIBA", "Semi-arid Northeast",
    "Urban Southeast", "Pantanal", "Atlantic Forest", "Pampa",
]

LAND_PREDICTORS = [
    "transformed_non_native_pct_1990",
    "transformed_non_native_pct_2024",
    "transformed_non_native_pct_mean_1990_2024",
    "transformed_non_native_change_1990_2024",
    "transformed_non_native_trend_decade",
    "transformed_non_native_early_trend_decade_1990_2004",
    "transformed_non_native_late_trend_decade_2010_2024",
]

LAND_PREDICTOR_LABELS = {
    "transformed_non_native_pct_1990": "1990 state",
    "transformed_non_native_pct_2024": "2024 state",
    "transformed_non_native_pct_mean_1990_2024": "1990–2024 mean state",
    "transformed_non_native_change_1990_2024": "absolute change",
    "transformed_non_native_trend_decade": "annual trend",
    "transformed_non_native_early_trend_decade_1990_2004": "early trend",
    "transformed_non_native_late_trend_decade_2010_2024": "late trend",
}


# Predictors shown in the supplementary figure. Early and late-period trends
# remain in the output tables, but are omitted from the figure to keep the
# visual comparison focused on the five metrics requested by the supplementary analysis.
PLOT_LAND_PREDICTORS = [
    "transformed_non_native_pct_1990",
    "transformed_non_native_pct_2024",
    "transformed_non_native_pct_mean_1990_2024",
    "transformed_non_native_change_1990_2024",
    "transformed_non_native_trend_decade",
]

OUTCOMES = [
    "dry_minus_humid_intensity_trend",
    "DHW_TmaxOnly_intensity_trend_decade",
    "VPD_trend_decade",
    "RH_trend_decade",
]

OUTCOME_LABELS = {
    "dry_minus_humid_intensity_trend": "DHW−HHW",
    "DHW_TmaxOnly_intensity_trend_decade": "DHW Tmax-only intensity",
    "VPD_trend_decade": "VPD trend",
    "RH_trend_decade": "RH trend",
}

RANDOM_SEED = 42
N_PERMUTATIONS = 999
FIG_DPI = 500


# ============================================================
# Text and data helpers
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_class(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_biome(x):
    s = normalize_text(x)
    mapping = {
        "AMAZONIA": "Amazon",
        "AMAZON": "Amazon",
        "CERRADO": "Cerrado",
        "CAATINGA": "Caatinga",
        "MATA ATLANTICA": "Atlantic Forest",
        "ATLANTIC FOREST": "Atlantic Forest",
        "PAMPA": "Pampa",
        "PAMPAS": "Pampa",
        "PANTANAL": "Pantanal",
    }
    return mapping.get(s, str(x))


def standardize_region_value(x):
    mapping = {
        "AMAZONIA": "Amazon", "AMAZON": "Amazon",
        "CERRADO": "Cerrado", "CAATINGA": "Caatinga",
        "MATA ATLANTICA": "Atlantic Forest", "ATLANTIC FOREST": "Atlantic Forest",
        "PAMPAS": "Pampa", "PAMPA": "Pampa",
        "PANTANAL": "Pantanal",
        "SEMIARIDO": "Semi-arid Northeast", "SEMI-ARID NORTHEAST": "Semi-arid Northeast",
        "SUDESTE URBANO": "Urban Southeast", "URBAN SOUTHEAST": "Urban Southeast",
        "MATOPIBA": "MATOPIBA",
    }
    key = normalize_text(x)
    return mapping.get(key, str(x))


def detect_column(columns, candidates):
    norm_map = {normalize_text(c): c for c in columns}
    for cand in candidates:
        nc = normalize_text(cand)
        if nc in norm_map:
            return norm_map[nc]
    for col in columns:
        nc = normalize_text(col)
        for cand in candidates:
            if normalize_text(cand) in nc:
                return col
    return None


def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where((den > 0) & np.isfinite(den), num / den, np.nan)


def save_table(df, stem):
    csv = os.path.join(OUT_DIR, f"{stem}.csv")
    xlsx = os.path.join(OUT_DIR, f"{stem}.xlsx")
    df.to_csv(csv, index=False)
    try:
        df.to_excel(xlsx, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save {xlsx}: {exc}")
    print(f"[OK] Saved table: {csv}")
    return csv, xlsx


def theil_sen_slope_decade(y, years):
    y = np.asarray(y, dtype=float)
    x = np.asarray(years, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < MIN_YEARS_FOR_LAND_TREND:
        return np.nan
    try:
        return float(theilslopes(y[ok], x[ok])[0] * 10.0)
    except Exception:
        return np.nan


def mann_kendall_p(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return np.nan
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(y[j] - y[i])
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return np.nan
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    values = values[ok]
    weights = weights[ok]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cdf, q)])


def weighted_spearman(
    x,
    y,
    w,
    n_permutations=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    """Weighted Spearman rho with a two-sided permutation p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())

    x = x[ok]
    y = y[ok]
    w = w[ok]

    xranks = stats.rankdata(x)
    yranks = stats.rankdata(y)

    def weighted_corr(a, b):
        ma = np.average(a, weights=w)
        mb = np.average(b, weights=w)
        cov = np.average((a - ma) * (b - mb), weights=w)
        va = np.average((a - ma) ** 2, weights=w)
        vb = np.average((b - mb) ** 2, weights=w)
        if va <= 0 or vb <= 0:
            return np.nan
        return float(cov / np.sqrt(va * vb))

    observed = weighted_corr(xranks, yranks)
    if not np.isfinite(observed):
        return np.nan, np.nan, int(ok.sum())

    rng = np.random.default_rng(seed)
    extreme = 0
    valid_perm = 0

    for _ in range(int(n_permutations)):
        permuted = rng.permutation(yranks)
        r_perm = weighted_corr(xranks, permuted)
        if not np.isfinite(r_perm):
            continue
        valid_perm += 1
        if abs(r_perm) >= abs(observed):
            extreme += 1

    p = (
        (extreme + 1.0) / (valid_perm + 1.0)
        if valid_perm > 0
        else np.nan
    )

    return float(observed), float(p), int(ok.sum())


# ============================================================
# Load MapBiomas annual record and build diagnostics
# ============================================================

def read_mapbiomas_annual():
    if not os.path.exists(COVERAGE_XLSX):
        raise FileNotFoundError(f"MapBiomas file not found: {COVERAGE_XLSX}")
    print(f"[INFO] Reading MapBiomas annual coverage: {COVERAGE_XLSX}")
    df = pd.read_excel(COVERAGE_XLSX, sheet_name="COVERAGE_10.1")

    year_cols = []
    for col in df.columns:
        try:
            y = int(col)
        except Exception:
            try:
                y = int(str(col))
            except Exception:
                continue
        if YEAR0 <= y <= YEAR1:
            year_cols.append(col)

    if len(year_cols) < 20:
        raise ValueError(f"Expected annual MapBiomas columns for {YEAR0}-{YEAR1}; found only {len(year_cols)}")

    required = ["state_acronym", "municipality", "biome", "class_level_1", "class_level_2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required MapBiomas columns: {missing}")

    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["municipality_norm"] = df["municipality"].map(normalize_text)
    df["state_acronym_norm"] = df["state_acronym"].map(normalize_text)
    df["biome_clean"] = df["biome"].map(normalize_biome)
    df["class1_norm"] = df["class_level_1"].map(normalize_class)
    df["class2_norm"] = df["class_level_2"].map(normalize_class)

    # Standardize year columns as int names for easier indexing.
    rename_years = {c: int(c) for c in year_cols}
    df = df.rename(columns=rename_years)

    return df, YEARS


def build_annual_transformation_metrics(df, years):
    print("[INFO] Building annual transformed/non-native land metrics by municipality.")
    keys = ["municipality_norm", "state_acronym_norm"]

    # Total mapped area by municipality and year.
    total = df.groupby(keys, dropna=False)[years].sum(min_count=1)

    native = (
        df.loc[df["class1_norm"].isin(NATIVE_LEVEL1)]
        .groupby(keys, dropna=False)[years]
        .sum(min_count=1)
    )
    native = native.reindex(total.index).fillna(0.0)

    agro = (
        df.loc[df["class1_norm"].isin(AGRO_LEVEL1)]
        .groupby(keys, dropna=False)[years]
        .sum(min_count=1)
    ).reindex(total.index).fillna(0.0)

    urban = (
        df.loc[df["class2_norm"].isin(URBAN_LEVEL2)]
        .groupby(keys, dropna=False)[years]
        .sum(min_count=1)
    ).reindex(total.index).fillna(0.0)

    transformed = 100.0 * (1.0 - native / total.replace(0, np.nan))
    native_pct = 100.0 * native / total.replace(0, np.nan)
    agro_pct = 100.0 * agro / total.replace(0, np.nan)
    urban_pct = 100.0 * urban / total.replace(0, np.nan)

    # Dominant biome in 2024.
    biome_area_2024 = (
        df.groupby(keys + ["biome_clean"], dropna=False)[YEAR1]
        .sum(min_count=1)
        .reset_index(name="area_2024_ha")
    )
    biome_area_2024 = biome_area_2024[biome_area_2024["area_2024_ha"] > 0]
    idx = biome_area_2024.groupby(keys)["area_2024_ha"].idxmax()
    dominant = biome_area_2024.loc[idx, keys + ["biome_clean", "area_2024_ha"]].rename(
        columns={"biome_clean": "dominant_biome", "area_2024_ha": "dominant_biome_area_2024_ha"}
    )

    # Basic municipality identity.
    ident = (
        df.groupby(keys, dropna=False)
        .agg(
            municipality=("municipality", "first"),
            state_acronym=("state_acronym", "first"),
        )
        .reset_index()
    )

    rows = []
    x_years = np.asarray(years, dtype=float)
    for idx_key in total.index:
        mun, uf = idx_key
        total_area_2024 = float(total.loc[idx_key, YEAR1]) if np.isfinite(total.loc[idx_key, YEAR1]) else np.nan
        if not np.isfinite(total_area_2024) or total_area_2024 < MIN_AREA_HA:
            continue

        tr = transformed.loc[idx_key, years].astype(float).values
        nat = native_pct.loc[idx_key, years].astype(float).values
        ag = agro_pct.loc[idx_key, years].astype(float).values
        urb = urban_pct.loc[idx_key, years].astype(float).values

        early_mask = (x_years >= EARLY0) & (x_years <= EARLY1)
        late_mask = (x_years >= LATE0) & (x_years <= LATE1)

        row = {
            "municipality_norm": mun,
            "state_acronym_norm": uf,
            "total_area_2024_ha": total_area_2024,
            "transformed_non_native_pct_1990": float(tr[0]),
            "transformed_non_native_pct_2024": float(tr[-1]),
            "transformed_non_native_pct_mean_1990_2024": float(np.nanmean(tr)),
            "transformed_non_native_change_1990_2024": float(tr[-1] - tr[0]),
            "transformed_non_native_trend_decade": theil_sen_slope_decade(tr, x_years),
            "transformed_non_native_trend_pvalue": mann_kendall_p(tr),
            "transformed_non_native_early_trend_decade_1990_2004": theil_sen_slope_decade(tr[early_mask], x_years[early_mask]),
            "transformed_non_native_late_trend_decade_2010_2024": theil_sen_slope_decade(tr[late_mask], x_years[late_mask]),
            "native_pct_1990": float(nat[0]),
            "native_pct_2024": float(nat[-1]),
            "agro_pct_1990": float(ag[0]),
            "agro_pct_2024": float(ag[-1]),
            "agri_gain_pct_points": float(ag[-1] - ag[0]),
            "urban_pct_1990": float(urb[0]),
            "urban_pct_2024": float(urb[-1]),
            "urban_gain_pct_points": float(urb[-1] - urb[0]),
        }

        for y, val in zip(years, tr):
            row[f"transformed_non_native_pct_{y}"] = float(val)
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.merge(ident, on=keys, how="left")
    out = out.merge(dominant, on=keys, how="left")
    out["dominant_biome"] = out["dominant_biome"].map(standardize_region_value)
    out = add_focus_regions(out)

    save_table(out, "Annual_MapBiomas_municipality_landmetrics_1990_2024")
    return out


def add_focus_regions(df):
    df = df.copy()
    state = df["state_acronym_norm"].astype(str).str.upper()
    dom = df["dominant_biome"].astype(str)

    se = state.isin(SOUTHEAST_STATES)
    urban_gain = pd.to_numeric(df.get("urban_gain_pct_points", np.nan), errors="coerce")
    urban_pct_2024 = pd.to_numeric(df.get("urban_pct_2024", np.nan), errors="coerce")

    se_gain_thr = np.nanpercentile(urban_gain[se], 75) if np.isfinite(urban_gain[se]).any() else np.nan
    se_urban_thr = np.nanpercentile(urban_pct_2024[se], 75) if np.isfinite(urban_pct_2024[se]).any() else np.nan

    is_southeast_urban = se & ((urban_gain >= se_gain_thr) | (urban_pct_2024 >= se_urban_thr))
    is_matopiba = state.isin(MATOPIBA_STATES) & dom.isin(["Cerrado", "Caatinga"])
    is_semiarid = state.isin(SEMIARID_STATES) & dom.eq("Caatinga")

    region = dom.copy().astype(object)
    region.loc[is_semiarid] = "Semi-arid Northeast"
    region.loc[is_matopiba] = "MATOPIBA"
    region.loc[is_southeast_urban] = "Urban Southeast"
    region = np.where(np.isin(region, REGION_ORDER), region, "Other")
    df["focus_region"] = region
    return df


# ============================================================
# Municipality geometry and atmospheric/heatwave data
# ============================================================

def find_municipality_shapefile():
    """Return the configured municipality shapefile."""
    if MUNICIPALITY_SHP and os.path.isfile(MUNICIPALITY_SHP):
        return MUNICIPALITY_SHP
    raise FileNotFoundError(f"Municipality shapefile not found: {MUNICIPALITY_SHP}")


def load_municipality_geometry():
    shp = find_municipality_shapefile()
    print(f"[INFO] Reading municipality shapefile: {shp}")
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    name_col = detect_column(gdf.columns, ["NM_MUN", "NM_MUNICIP", "NM_MUNICIPIO", "NOME", "MUNICIPIO", "municipality", "name"])
    state_col = detect_column(gdf.columns, ["SIGLA_UF", "UF", "state_acronym", "SIGLA"])
    if name_col is None or state_col is None:
        raise ValueError(f"Could not detect municipality/state columns. Columns: {list(gdf.columns)}")
    gdf["municipality_norm"] = gdf[name_col].map(normalize_text)
    gdf["state_acronym_norm"] = gdf[state_col].map(normalize_text)
    pts = gdf.geometry.representative_point()
    gdf["rep_lon"] = pts.x
    gdf["rep_lat"] = pts.y
    return gdf


def read_table_any(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    if path.endswith(".nc"):
        if xr is None:
            raise RuntimeError("xarray is required to read NetCDF tables.")
        return xr.open_dataset(path).to_dataframe().reset_index()
    raise ValueError(f"Unsupported table type: {path}")


def standardize_table_columns(df):
    """Standardize region labels and validate the common-scale DHW–HHW metric."""
    df = df.copy()

    for col in ["focus_region", "dominant_biome", "biome_clean", "region"]:
        if col in df.columns:
            df[col] = df[col].map(standardize_region_value)

    common_contrast = "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"

    if "dry_minus_humid_intensity_trend" not in df.columns:
        if common_contrast in df.columns:
            df["dry_minus_humid_intensity_trend"] = pd.to_numeric(
                df[common_contrast],
                errors="coerce",
            )
        else:
            raise ValueError(
                "The municipality table must contain either "
                "'dry_minus_humid_intensity_trend' or "
                f"'{common_contrast}'. Primary DHW and HHW intensity trends "
                "are not subtracted because their definitions are not "
                "directly commensurate."
            )

    if "DHW_TmaxOnly_intensity_trend_decade" not in df.columns:
        print(
            "[WARN] DHW_TmaxOnly_intensity_trend_decade is absent; "
            "the standalone DHW common-scale outcome will be skipped."
        )

    return df


def load_atmospheric_heatwave_table():
    """Read the configured Figure 03/02 municipality-level table."""
    print(f"[INFO] Reading atmospheric/heatwave municipality table: {ATMOSPHERIC_TABLE}")
    table = read_table_any(ATMOSPHERIC_TABLE)
    return standardize_table_columns(table), ATMOSPHERIC_TABLE


def merge_land_atmosphere(land, gdf):
    atm, atm_path = load_atmospheric_heatwave_table()
    keys = ["municipality_norm", "state_acronym_norm"]

    out = land.copy()
    if atm is not None:
        keep_cols = keys + [c for c in atm.columns if c not in keys]
        atm = atm[keep_cols].drop_duplicates(subset=keys)
        # Drop duplicated land columns from atmospheric table to avoid suffix clutter.
        duplicate_land_cols = [c for c in atm.columns if c in out.columns and c not in keys]
        atm = atm.drop(columns=duplicate_land_cols, errors="ignore")
        out = out.merge(atm, on=keys, how="left")

    coords = gdf[keys + ["rep_lat", "rep_lon"]].drop_duplicates()
    out = out.merge(coords, on=keys, how="left")

    # Guarantee focus/dominant region columns after merge.
    if "focus_region" not in out.columns:
        out = add_focus_regions(out)
    for c in ["focus_region", "dominant_biome"]:
        if c in out.columns:
            out[c] = out[c].map(standardize_region_value)

    save_table(out, "Annual_MapBiomas_merged_land_atmosphere_heatwave_metrics_1990_2024")

    missing_outcomes = [o for o in OUTCOMES if o not in out.columns]
    if missing_outcomes:
        print(f"[WARN] Missing outcomes in merged table: {missing_outcomes}")
        print("[WARN] Some requested outcomes are unavailable in the supplied table.")
    else:
        print("[OK] All requested outcomes are available in merged table.")
    return out


# ============================================================
# Region summaries and statistical comparisons
# ============================================================

def build_region_summary(df):
    print("[INFO] Building regional summary.")
    variables = LAND_PREDICTORS + [
        "native_pct_2024", "agro_pct_2024", "agri_gain_pct_points", "urban_pct_2024", "urban_gain_pct_points",
    ] + [o for o in OUTCOMES if o in df.columns]

    rows = []
    for region in ["Brazil"] + REGION_ORDER:
        if region == "Brazil":
            sub = df.copy()
        else:
            sub = df[df["focus_region"] == region].copy()
        if sub.empty:
            continue
        weights = pd.to_numeric(sub["total_area_2024_ha"], errors="coerce").values
        row = {
            "region": region,
            "n_municipalities": int(len(sub)),
            "area_2024_ha": float(np.nansum(weights)),
        }
        for v in variables:
            if v not in sub.columns:
                continue
            vals = pd.to_numeric(sub[v], errors="coerce").values
            row[f"{v}_median"] = weighted_quantile(vals, weights, 0.50)
            row[f"{v}_p25"] = weighted_quantile(vals, weights, 0.25)
            row[f"{v}_p75"] = weighted_quantile(vals, weights, 0.75)
            row[f"{v}_mean"] = weighted_mean(vals, weights)
        rows.append(row)

    out = pd.DataFrame(rows)
    save_table(out, "Annual_MapBiomas_region_summary_1990_2024")
    return out


def zscore_weighted(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    out = np.full_like(x, np.nan, dtype=float)
    if ok.sum() < 3:
        return out
    mu = np.average(x[ok], weights=w[ok])
    sd = np.sqrt(np.average((x[ok] - mu) ** 2, weights=w[ok]))
    if not np.isfinite(sd) or sd <= 0:
        return out
    out[ok] = (x[ok] - mu) / sd
    return out


def fit_wls_hc3(df, response, predictor):
    required = [response, predictor, "total_area_2024_ha", "focus_region", "rep_lat", "rep_lon"]
    d = df.copy()
    for c in [response, predictor, "total_area_2024_ha", "rep_lat", "rep_lon"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[c for c in required if c in d.columns]).copy()
    d = d[d["focus_region"].isin(REGION_ORDER)].copy()
    d = d[d["total_area_2024_ha"] > 0].copy()

    if len(d) < 80:
        return {
            "response": response, "predictor": predictor, "n": len(d),
            "coef_per_1sd": np.nan, "se_HC3": np.nan, "p_HC3": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "r2": np.nan,
            "adj_r2": np.nan, "aic": np.nan, "status": "insufficient_data",
        }

    # WLS weights are proportional to the square root of municipal area.
    d["weight"] = np.sqrt(d["total_area_2024_ha"])
    d["weight"] = d["weight"] / np.nanmean(d["weight"])
    d["x_z"] = zscore_weighted(d[predictor].values, d["weight"].values)
    d = d.dropna(subset=["x_z", response, "weight", "rep_lat", "rep_lon", "focus_region"]).copy()

    if len(d) < 80:
        return {
            "response": response, "predictor": predictor, "n": len(d),
            "coef_per_1sd": np.nan, "se_HC3": np.nan, "p_HC3": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "r2": np.nan,
            "adj_r2": np.nan, "aic": np.nan, "status": "insufficient_data_after_zscore",
        }

    # Main controlled spatial-consistency model.
    formula = f"{response} ~ x_z + C(focus_region) + rep_lat + rep_lon"
    try:
        fit = sm.WLS.from_formula(formula, data=d, weights=d["weight"]).fit(cov_type="HC3")
        ci = fit.conf_int().loc["x_z"] if "x_z" in fit.params.index else [np.nan, np.nan]
        return {
            "response": response,
            "response_label": OUTCOME_LABELS.get(response, response),
            "predictor": predictor,
            "predictor_label": LAND_PREDICTOR_LABELS.get(predictor, predictor),
            "n": int(fit.nobs),
            "coef_per_1sd": float(fit.params.get("x_z", np.nan)),
            "se_HC3": float(fit.bse.get("x_z", np.nan)),
            "t_HC3": float(fit.tvalues.get("x_z", np.nan)),
            "p_HC3": float(fit.pvalues.get("x_z", np.nan)),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "aic": float(fit.aic),
            "formula": formula,
            "status": "ok",
            "weighting": "weights proportional to sqrt(municipality area)",
            "covariance": "HC3 heteroscedasticity-robust standard errors",
            "interpretation_note": "Conditional spatial association; not causal attribution, mediation, or a land-cover sensitivity experiment.",
        }
    except Exception as exc:
        return {
            "response": response, "predictor": predictor, "n": len(d),
            "coef_per_1sd": np.nan, "se_HC3": np.nan, "p_HC3": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "r2": np.nan,
            "adj_r2": np.nan, "aic": np.nan, "status": f"fit_failed: {exc}",
        }


def run_predictor_comparisons(df):
    print("[INFO] Running WLS-HC3 predictor comparison models.")
    available_outcomes = [o for o in OUTCOMES if o in df.columns]
    rows = []
    for response in available_outcomes:
        for predictor in LAND_PREDICTORS:
            if predictor in df.columns:
                rows.append(fit_wls_hc3(df, response, predictor))
    out = pd.DataFrame(rows)
    save_table(out, "Annual_MapBiomas_predictor_comparison_WLS_HC3_1990_2024")

    print("[INFO] Running weighted Spearman diagnostics.")
    spear_rows = []
    # Weighted Spearman uses the same square-root-area weighting convention.
    w = np.sqrt(pd.to_numeric(df["total_area_2024_ha"], errors="coerce").values)
    for response in available_outcomes:
        y = pd.to_numeric(df[response], errors="coerce").values
        for predictor in LAND_PREDICTORS:
            if predictor not in df.columns:
                continue
            x = pd.to_numeric(df[predictor], errors="coerce").values
            rho, p, n = weighted_spearman(x, y, w)
            spear_rows.append({
                "response": response,
                "response_label": OUTCOME_LABELS.get(response, response),
                "predictor": predictor,
                "predictor_label": LAND_PREDICTOR_LABELS.get(predictor, predictor),
                "n": n,
                "weighted_spearman_rho": rho,
                "weighted_spearman_p_permutation": p,
                "n_permutations": N_PERMUTATIONS,
                "weighting": "weights proportional to sqrt(municipality area)",
                "interpretation_note": "Weighted monotonic spatial association; not causal attribution.",
            })
    spear = pd.DataFrame(spear_rows)
    save_table(spear, "Annual_MapBiomas_weighted_spearman_1990_2024")
    return out, spear


# ============================================================
# Plotting
# ============================================================

def add_colorbar(fig, ax, cmap, norm, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.03)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=12)


def plot_municipality_map(ax, gdf, column, title, cmap, norm, cbar_label):
    vals = pd.to_numeric(gdf[column], errors="coerce") if column in gdf.columns else pd.Series(np.nan, index=gdf.index)
    gdf.plot(ax=ax, color="0.93", edgecolor="none", linewidth=0, zorder=1)
    valid = gdf[np.isfinite(vals)].copy()
    if not valid.empty:
        valid.plot(ax=ax, column=column, cmap=cmap, norm=norm, linewidth=0.02, edgecolor="0.75", zorder=2)
    gdf.boundary.plot(ax=ax, color="0.70", linewidth=0.015, zorder=3)
    # outer Brazil boundary only, cleaner
    try:
        gpd.GeoSeries([gdf.geometry.unary_union], crs=gdf.crs).boundary.plot(ax=ax, color="0.15", linewidth=0.55, zorder=4)
    except Exception:
        pass
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", pad=5)
    add_colorbar(ax.figure, ax, cmap, norm, cbar_label)


def plot_adj_r2_heatmap(ax, stats_df):
    d = stats_df[stats_df["status"] == "ok"].copy()
    if d.empty:
        ax.text(0.5, 0.5, "No WLS results", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    outcomes = [o for o in OUTCOMES if o in d["response"].unique()]
    predictors = [p for p in PLOT_LAND_PREDICTORS if p in d["predictor"].unique()]
    mat = np.full((len(predictors), len(outcomes)), np.nan)
    for i, p in enumerate(predictors):
        for j, o in enumerate(outcomes):
            sub = d[(d["predictor"] == p) & (d["response"] == o)]
            if not sub.empty:
                mat[i, j] = float(sub["adj_r2"].iloc[0])
    vmax = np.nanpercentile(mat, 95) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 0.05)
    im = ax.imshow(mat, aspect="auto", cmap=plt.cm.YlGnBu, vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(outcomes)))
    ax.set_xticklabels([OUTCOME_LABELS.get(o, o) for o in outcomes], rotation=35, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(predictors)))
    ax.set_yticklabels([LAND_PREDICTOR_LABELS.get(p, p) for p in predictors], fontsize=12)
    ax.set_title("(c) Controlled model fit by land metric\nadjusted R²", fontsize=12, fontweight="bold", pad=6)
    for i in range(len(predictors)):
        for j in range(len(outcomes)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=12, color="black")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("adjusted R²", fontsize=12)
    cb.ax.tick_params(labelsize=12)


def plot_dryhot_coefficients(ax, stats_df):
    d = stats_df[(stats_df["status"] == "ok") & (stats_df["response"] == "dry_minus_humid_intensity_trend")].copy()
    if d.empty:
        ax.text(0.5, 0.5, "DHW−HHW results unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    d = d[d["predictor"].isin(PLOT_LAND_PREDICTORS)].copy()
    d["order"] = d["predictor"].map({p: i for i, p in enumerate(PLOT_LAND_PREDICTORS)})
    d = d.sort_values("order")
    y = np.arange(len(d))
    coef = pd.to_numeric(d["coef_per_1sd"], errors="coerce").values
    lo = pd.to_numeric(d["ci95_low"], errors="coerce").values
    hi = pd.to_numeric(d["ci95_high"], errors="coerce").values
    labels = [LAND_PREDICTOR_LABELS.get(p, p) for p in d["predictor"]]
    colors = ["#B2182B" if c >= 0 else "#2166AC" for c in coef]
    for yi, c, l, h, col in zip(y, coef, lo, hi, colors):
        if np.isfinite(c):
            ax.plot([l, h], [yi, yi], color=col, linewidth=1.5, alpha=0.8)
            ax.scatter(c, yi, color=col, s=38, edgecolor="0.15", linewidth=0.5, zorder=3)
    ax.axvline(0, color="0.20", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    ax.set_xlabel("Coefficient per 1 s.d. land metric\nDHW−HHW severity decade$^{-1}$", fontsize=12)
    ax.set_title(
        "(d) Land-metric associations with DHW–HHW contrast\n"
        "positive = stronger DHW−HHW at higher land-metric values",
        fontsize=12, fontweight="bold", pad=6
    )


def make_supplementary_figure(merged, gdf, stats_df):
    print("[INFO] Plotting supplementary annual MapBiomas diagnostics figure.")
    plot_gdf = gdf.merge(
        merged,
        on=["municipality_norm", "state_acronym_norm"],
        how="left",
        suffixes=("", "_tbl"),
    )

    # No suptitle or footnote is added to the figure canvas. The complete
    # explanation should be placed in the Supplementary Figure caption.
    # Wider spacing prevents panel-d y-axis labels from colliding with panel c.
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.055, right=0.985, top=0.965, bottom=0.075,
        wspace=0.38, hspace=0.38
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_municipality_map(
        ax_a, plot_gdf,
        "transformed_non_native_pct_2024",
        "(a) Cumulative transformed/non-native land fraction\nMapBiomas 2024",
        plt.cm.YlOrBr,
        Normalize(vmin=0, vmax=100),
        "% municipal area",
    )

    trend_vals = pd.to_numeric(plot_gdf["transformed_non_native_trend_decade"], errors="coerce")
    vmax = np.nanpercentile(np.abs(trend_vals[np.isfinite(trend_vals)]), 97) if np.isfinite(trend_vals).any() else 5
    vmax = max(vmax, 1.0)
    plot_municipality_map(
        ax_b, plot_gdf,
        "transformed_non_native_trend_decade",
        "(b) Annual transformed-land trend\nMapBiomas 1990–2024",
        plt.cm.RdBu_r,
        TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        "percentage points decade$^{-1}$",
    )

    plot_adj_r2_heatmap(ax_c, stats_df)
    plot_dryhot_coefficients(ax_d, stats_df)

    out_base = os.path.join(OUT_DIR, "Supplementary_Annual_MapBiomas_Diagnostics_1990_2024")
    fig.savefig(out_base + ".pdf", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".jpeg", dpi=FIG_DPI, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"[OK] Saved figure: {out_base}.pdf")
    print(f"[OK] Saved figure: {out_base}.jpeg")


# ============================================================
# Main workflow
# ============================================================

def main():
    args = parse_args()
    configure_paths(args)

    print(f"[START] Annual MapBiomas diagnostics | {pd.Timestamp.now().isoformat()}")
    print("[INFO] Permutation inference: 999 permutations")
    print("[INFO] Interpretation: spatial association diagnostics; not causal attribution.")

    df_raw, years = read_mapbiomas_annual()
    land = build_annual_transformation_metrics(df_raw, years)
    gdf = load_municipality_geometry()

    merged = merge_land_atmosphere(land, gdf)
    build_region_summary(merged)

    stats_df, spear_df = run_predictor_comparisons(merged)
    make_supplementary_figure(merged, gdf, stats_df)
    print("\n[SUMMARY] Available controlled associations with DHW−HHW:")
    if "dry_minus_humid_intensity_trend" in merged.columns:
        sub = stats_df[(stats_df["response"] == "dry_minus_humid_intensity_trend") & (stats_df["status"] == "ok")].copy()
        if not sub.empty:
            sub = sub.sort_values("adj_r2", ascending=False)
            for _, r in sub.iterrows():
                print(
                    f"  {r['predictor_label']:<24s} coef={r['coef_per_1sd']:+.3f}, "
                    f"p={r['p_HC3']:.3g}, adjR2={r['adj_r2']:.3f}"
                )
    print("\n[DONE] Outputs saved in:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

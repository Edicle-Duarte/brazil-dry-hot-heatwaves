#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 03 — Atmospheric drying and preferential dry-hot heatwave amplification,
1990–2024.

This workflow reproduces Figure 03 using outputs from Figures 01 and 02.

The cross-regime comparison uses the common-scale Figure 01 field
``DHW_minus_HHW_TmaxOnly_intensity_trend_decade``. Primary DHW and HHW
regime-specific intensity fields are not subtracted because they are not
directly commensurate.

ERA5 VPD and RH diagnostics are aggregated annually with at least 300 valid
daily values per grid-cell year. Trend magnitude is estimated with the
Theil–Sen median slope per decade. Significance is assessed with the original
two-sided Mann–Kendall test unless detrended lag-1 rank autocorrelation is
significant, in which case the Hamed–Rao variance-corrected Mann–Kendall test
with lag 1 is used.

Municipality-level ERA5 fields are aggregated with cos(latitude) grid-cell
weights. P-values are not averaged. LOWESS curves are descriptive smoothers.
Weighted Spearman correlations use two-sided permutation tests. Controlled
weighted least-squares (WLS) models use weights proportional to the square root
of municipal area and HC3 heteroscedasticity-robust standard errors. These
models quantify spatial associations and are not interpreted as causal or
mediation analyses.

The morphological opening/closing mask applied to panel (a) is used only for
visualization and does not affect municipality-level associations, regressions,
or reported quantitative summaries.

Usage
-----
python figure_03_atmospheric_drying.py \
    --figure1-output-dir /path/to/figure_01_outputs \
    --figure2-table /path/to/figure_02_municipality_dhw_hhw_contrast_1990_2024.csv \
    --mapbiomas-xlsx /path/to/mapbiomas_municipality_coverage.xlsx \
    --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
    --output-dir ./outputs/figure_03
"""

import os
import re
import sys
import json
import glob
import argparse
import warnings
import unicodedata
from pathlib import Path


import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy import stats
from scipy.stats import norm, theilslopes
from scipy.spatial import cKDTree
from shapely.prepared import prep
from shapely.geometry import Point
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    import pymannkendall as mk
    _HAS_PYMK = True
except Exception:
    _HAS_PYMK = False


# ============================================================
# Runtime paths and period
# ============================================================

FIGURE1_ROOT = None
DAILY_DIR = None
FIG1_TRENDS_NC = None
FIG2_TABLE = None
COVERAGE_XLSX = None
MUNICIPALITY_SHP = None
OUT_DIR = None

YEAR0, YEAR1 = 1990, 2024
BASELINE0, BASELINE1 = 1991, 2020
LON_MIN, LON_MAX = -75.5, -32.0
LAT_MIN, LAT_MAX = -35.5, 6.5


def parse_args():
    """Parse filesystem/recomputation options; scientific settings stay fixed."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Figure 03 using outputs from Figures 01 and 02."
        )
    )
    parser.add_argument(
        "--figure1-output-dir",
        required=True,
        help="Root output directory produced by figure_01_heatwave_trends.py.",
    )
    parser.add_argument(
        "--figure2-table",
        required=True,
        help="Figure 02 municipality CSV.",
    )
    parser.add_argument(
        "--mapbiomas-xlsx",
        required=True,
        help="MapBiomas Collection 10.1 coverage workbook.",
    )
    parser.add_argument(
        "--municipality-shapefile",
        required=True,
        help="Brazilian municipality shapefile.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Figure 03 products.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute annual and trend caches even when compatible files exist.",
    )
    return parser.parse_args()


def configure_paths(args):
    """Configure paths without embedding workstation/HPC-specific absolute paths."""
    global FIGURE1_ROOT, DAILY_DIR, FIG1_TRENDS_NC, FIG2_TABLE
    global COVERAGE_XLSX, MUNICIPALITY_SHP, OUT_DIR

    FIGURE1_ROOT = Path(args.figure1_output_dir).expanduser().resolve()
    DAILY_DIR = str(FIGURE1_ROOT / "cache" / "era5_daily")
    FIG1_TRENDS_NC = str(
        FIGURE1_ROOT / "data" / "figure_01_heatwave_trends_1990_2024.nc"
    )
    FIG2_TABLE = str(Path(args.figure2_table).expanduser().resolve())
    COVERAGE_XLSX = str(Path(args.mapbiomas_xlsx).expanduser().resolve())
    MUNICIPALITY_SHP = str(Path(args.municipality_shapefile).expanduser().resolve())
    OUT_DIR = str(Path(args.output_dir).expanduser().resolve())

    required = [
        (DAILY_DIR, "Figure 01 daily cache"),
        (FIG1_TRENDS_NC, "Figure 01 trend file"),
        (FIG2_TABLE, "Figure 02 municipality table"),
        (COVERAGE_XLSX, "MapBiomas workbook"),
        (MUNICIPALITY_SHP, "municipality shapefile"),
    ]

    for path, label in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Configuration (aligned with Figures 1–2)
# ============================================================

MIN_AREA_HA = 1000.0
MIN_YEARS_FOR_TREND = 20
N_BOOT = 300
N_PERMUTATIONS = 999
RANDOM_SEED = 42
P_STIPPLE = 0.05
MIN_VALID_DAYS_PER_YEAR = 300
AUTOCORR_ALPHA = 0.05
TREND_SIGNIFICANCE_VERSION = "Theil-Sen + conditional Hamed-Rao MK lag1 v1"


# Physical masking for dry-hot shift panel (a)
DRYHOT_MASK_MIN_SHIFT = 0.5  # ~10th percentile of significant positive shifts
DRYHOT_MASK_P = 0.05

# Stippling parameters (finer for VPD/RH maps to reduce visual clutter)
STIPPLE_SIZE_FINE = 0.06
STIPPLE_ALPHA_FINE = 0.18
STIPPLE_SIZE = 1.2
STIPPLE_ALPHA = 0.50

# Morphological filtering for panel (a): spatial coherence at mesoscale
DRYHOT_OPENING_STRUCTURE = np.ones((2, 2), dtype=bool)   # ~50 km scale
DRYHOT_CLOSING_STRUCTURE = np.ones((3, 3), dtype=bool)   # ~85 km scale

# Visualization smoothing (NOT used in trend estimation)
MAP_SMOOTH_SIGMA = 0.35

# Color scale limits based on empirical percentile diagnostics
DRYHOT_VMIN, DRYHOT_VMAX = 0.5, 10.0        # severity decade⁻¹ (sequencial, foca em shifts robustos)
VPD_VMIN, VPD_VCENTER, VPD_VMAX = -0.15, 0.0, 0.15  # kPa decade⁻¹; restored original figure range
RH_VMIN, RH_VCENTER, RH_VMAX = -3.0, 0.0, 3.0       # % decade⁻¹ (divergente, cobre -2.52 a +1.38)

REQUIRED_ANNUAL_VARS = ["VPD", "RH", "SMD", "SM1", "WS10"]
REQUIRED_TREND_VARS = [
    "VPD_trend_decade", "VPD_pvalue",
    "RH_trend_decade", "RH_pvalue",
    "SMD_trend_decade", "SMD_pvalue",
    "dry_minus_humid_intensity_trend",
    "DHW_TmaxOnly_intensity_trend_decade",
    "DHW_TmaxOnly_intensity_pvalue",
]

REGION_ORDER = [
    "Amazon", "Cerrado", "MATOPIBA", "Semi-arid Northeast",
    "Urban Southeast", "Pantanal", "Atlantic Forest", "Pampa",
]
#REGIONAL_RELATIONSHIP_REGIONS = ["Cerrado", "MATOPIBA"]  # dominant agricultural frontiers
#REGIONAL_RELATIONSHIP_REGIONS = [
#    "Amazônia",      # fronteira agrícola norte, desmatamento recente
#    "Pantanal",      # wetland sensível a mudanças hidrológicas
#    "Cerrado",       # hotspot de conversão agrícola
#    "MATOPIBA",      # fronteira agrícola em expansão acelerada
#    "Semiárido",     # vulnerável a secas atmosféricas
#    "Mata Atlântica" # paisagem fragmentada, urbanização
#]
REGIONAL_RELATIONSHIP_REGIONS = ["Amazon", "Cerrado", "MATOPIBA", "Atlantic Forest", "Urban Southeast"]
MATOPIBA_STATES = {"MA", "TO", "PI", "BA"}
SEMIARID_STATES = {"AL", "BA", "CE", "PB", "PE", "PI", "RN", "SE", "MG"}
SOUTHEAST_STATES = {"SP", "RJ", "MG", "ES"}

NATIVE_LEVEL1 = {"1. forest", "2. non forest natural formation"}
AGRO_LEVEL1 = {"3. farming"}
URBAN_LEVEL2 = {"4.2. urban area"}

# Software metadata for reproducibility (aligned with Figures 1–2)
SOFTWARE_VERSION = "2.0.0"
FIG_FONT_SIZE = 14
SOFTWARE_LICENSE = "MIT"

plt.rcParams.update({
    "font.size": FIG_FONT_SIZE,
    "axes.titlesize": FIG_FONT_SIZE,
    "axes.labelsize": FIG_FONT_SIZE,
    "xtick.labelsize": FIG_FONT_SIZE,
    "ytick.labelsize": FIG_FONT_SIZE,
    "legend.fontsize": FIG_FONT_SIZE,
    "figure.titlesize": FIG_FONT_SIZE,
})


# ============================================================
# Reproducibility utilities
# ============================================================

def get_software_metadata():
    """Collect runtime and methodological metadata."""
    return {
        "software_name": "figure_03_atmospheric_drying",
        "software_version": SOFTWARE_VERSION,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "xarray_version": xr.__version__,
        "matplotlib_version": matplotlib.__version__,
        "scipy_version": __import__("scipy").__version__,
        "geopandas_version": gpd.__version__,
        "statsmodels_version": __import__("statsmodels").__version__,
        "execution_timestamp": pd.Timestamp.now().isoformat(),
        "license": SOFTWARE_LICENSE,
        "figure1_dependency": os.path.basename(FIG1_TRENDS_NC) if FIG1_TRENDS_NC else "",
        "figure2_dependency": os.path.basename(FIG2_TABLE) if FIG2_TABLE else "",
        "trend_significance_version": TREND_SIGNIFICANCE_VERSION,
        "cross_regime_metric": "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "VPD_output_units": "kPa",
    }


def validate_figure1_compatibility(ds_fig1):
    """Require the Figure 01 common-scale HHW/DHW fields."""
    required_vars = [
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "DHW_TmaxOnly_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
        "DHW_TmaxOnly_intensity_pvalue",
    ]
    missing = [var for var in required_vars if var not in ds_fig1.data_vars]
    if missing:
        raise ValueError(
            "The supplied Figure 01 file is incompatible with this workflow. "
            "Missing variables: " + ", ".join(missing)
        )

    method = ds_fig1.attrs.get("main_trend_method", "")
    if method and method != "Theil-Sen median slope":
        raise ValueError(
            f"Figure 01 main trend method is '{method}', expected Theil-Sen median slope."
        )
    return True


def validate_figure2_compatibility(df_fig2):
    """
    Validate only the corrected Figure 02 fields actually required by Figure 03.

    The Figure 02 municipality-level significant-positive DHW fraction is used
    for Figure 02 hotspot classification, but Figure 03 does not require it for
    atmospheric-drying maps, LOWESS relationships, or controlled regressions.
    Therefore that field is optional here.
    """
    required_cols = [
        "municipality_norm",
        "state_acronym_norm",
        "dry_minus_humid_intensity_trend",
        "DHW_TmaxOnly_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
    ]

    missing = [col for col in required_cols if col not in df_fig2.columns]
    if missing:
        raise ValueError(
            "The supplied Figure 02 table is not compatible with corrected "
            "Figure 03. Missing required common-scale fields: "
            + ", ".join(missing)
        )

    if "DHW_TmaxOnly_significant_positive_fraction" not in df_fig2.columns:
        print(
            "[INFO] Optional Figure 02 hotspot-support fraction is absent; "
            "continuing because Figure 03 does not use it."
        )

    return True


# ============================================================
# NetCDF backend helpers
# ============================================================

def _available_netcdf_engines():
    engines = xr.backends.list_engines()
    return [name for name in ("h5netcdf", "netcdf4", "scipy") if name in engines]


def safe_open_dataset(path, *, decode_times=True, chunks=None, **kwargs):
    errors = []
    for engine in _available_netcdf_engines():
        try:
            return xr.open_dataset(
                path,
                engine=engine,
                decode_times=decode_times,
                chunks=chunks,
                **kwargs,
            )
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not open NetCDF file: {path}\n" + "\n".join(errors)
    )


def safe_to_netcdf(ds, path, *, encoding=None, **kwargs):
    errors = []
    engines = _available_netcdf_engines()

    for engine in ("h5netcdf", "netcdf4"):
        if engine not in engines:
            continue
        try:
            return ds.to_netcdf(path, engine=engine, encoding=encoding, **kwargs)
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")

    if "scipy" in engines:
        scipy_encoding = None
        if encoding:
            scipy_encoding = {
                var: {
                    k: v
                    for k, v in opts.items()
                    if k in {"dtype", "_FillValue", "scale_factor", "add_offset"}
                }
                for var, opts in encoding.items()
            }
        try:
            return ds.to_netcdf(
                path,
                engine="scipy",
                encoding=scipy_encoding,
                **kwargs,
            )
        except Exception as exc:
            errors.append(f"scipy: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not write NetCDF file: {path}\n" + "\n".join(errors)
    )


# ============================================================
# Text normalization helpers
# ============================================================

def normalize_text(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).upper()

def normalize_class(x):
    if pd.isna(x): return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s)

def normalize_biome(x):
    s = normalize_text(x)
    mapping = {
        "AMAZONIA": "Amazon", "AMAZON": "Amazon", "CERRADO": "Cerrado", "CAATINGA": "Caatinga",
        "MATA ATLANTICA": "Atlantic Forest", "ATLANTIC FOREST": "Atlantic Forest",
        "PAMPA": "Pampa", "PAMPAS": "Pampa", "PANTANAL": "Pantanal",
    }
    return mapping.get(s, str(x))

REGION_RENAME_EN = {
    "Amazônia": "Amazon", "Amazonia": "Amazon", "AMAZONIA": "Amazon",
    "Semiárido": "Semi-arid Northeast", "Semiarido": "Semi-arid Northeast", "SEMIARIDO": "Semi-arid Northeast",
    "Sudeste urbano": "Urban Southeast", "SUDESTE URBANO": "Urban Southeast",
    "Mata Atlântica": "Atlantic Forest", "Mata Atlantica": "Atlantic Forest", "MATA ATLANTICA": "Atlantic Forest",
    "Pampas": "Pampa", "Pampa": "Pampa", "PAMPAS": "Pampa", "PAMPA": "Pampa",
}

def standardize_region_columns(df):
    if df is None:
        return df
    df = df.copy()
    for col in ["dominant_biome", "focus_region", "biome_clean", "region", "stratum"]:
        if col in df.columns:
            df[col] = df[col].replace(REGION_RENAME_EN)
    return df

def save_table_dual(df, stem):
    """Save a table as CSV and XLSX inside Figure_3."""
    csv_path = os.path.join(OUT_DIR, f"{stem}.csv")
    xlsx_path = os.path.join(OUT_DIR, f"{stem}.xlsx")
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save {stem}.xlsx: {exc}")
    print(f"[OK] Saved {stem}: {csv_path}")
    return csv_path, xlsx_path

def build_supplementary_table_s9(mun):
    """Supplementary Table S9: municipal VPD and RH trend summary statistics."""
    df = standardize_region_columns(mun).copy()
    rows = []
    def add_rows(subset, region_name):
        for var, label, unit in [
            ("VPD_trend_decade", "VPD trend", "kPa decade^-1"),
            ("RH_trend_decade", "Relative humidity trend", "% decade^-1"),
        ]:
            if var not in subset.columns:
                continue
            vals = pd.to_numeric(subset[var], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append({
                "Region": region_name,
                "Variable": label,
                "Unit": unit,
                "N": int(len(vals)),
                "Median": float(np.nanmedian(vals)),
                "Q1": float(np.nanpercentile(vals, 25)),
                "Q3": float(np.nanpercentile(vals, 75)),
                "IQR": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
                "Mean": float(np.nanmean(vals)),
                "SD": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "Minimum": float(np.nanmin(vals)),
                "Maximum": float(np.nanmax(vals)),
            })
    add_rows(df, "Brazil")
    if "focus_region" in df.columns:
        for region in REGION_ORDER:
            sub = df[df["focus_region"] == region].copy()
            if len(sub) > 0:
                add_rows(sub, region)
    out = pd.DataFrame(rows)
    save_table_dual(out, "Supplementary_Table_S9")
    return out

def build_supplementary_table_s10(relationships):
    """Supplementary Table S10: nonlinear association between VPD trend and dry-hot transition."""
    res = relationships.get("vpd_to_dryhot") if isinstance(relationships, dict) else None
    rows = []
    if res is not None:
        x = np.asarray(res.get("x_obs", []), dtype=float)
        x = x[np.isfinite(x)]
        rows.append({
            "Relationship": "VPD trend vs dry-hot transition",
            "Region": "Brazil",
            "N": int(res.get("n", 0)),
            "Weighted_Spearman_rho": float(res.get("rho", np.nan)),
            "Permutation_P_value": float(res.get("p", np.nan)),
            "Observed_x_Q02": float(np.nanpercentile(x, 2)) if x.size else np.nan,
            "Observed_x_Q50": float(np.nanpercentile(x, 50)) if x.size else np.nan,
            "Observed_x_Q98": float(np.nanpercentile(x, 98)) if x.size else np.nan,
            "LOWESS_frac": 0.55,
            "Bootstrap_replicates": N_BOOT,
            "Spearman_permutations": N_PERMUTATIONS,
            "Interpretation": "Nonlinear diagnostic relationship; not a formal physical threshold estimate.",
        })
    out = pd.DataFrame(rows)
    save_table_dual(out, "Supplementary_Table_S10")
    return out

def build_supplementary_table_s11(relationships):
    """Supplementary Table S11: regional transformed-land diagnostics for panel e."""
    rows = []
    reg_dict = relationships.get("transformed_to_vpd_regions", {}) if isinstance(relationships, dict) else {}
    for region in REGIONAL_RELATIONSHIP_REGIONS:
        res = reg_dict.get(region)
        if res is None:
            rows.append({
                "Relationship": "Transformed/non-native land fraction vs VPD trend",
                "Region": region,
                "N": np.nan,
                "Weighted_Spearman_rho": np.nan,
                "P_value": np.nan,
                "Observed_x_Q02": np.nan,
                "Observed_x_Q50": np.nan,
                "Observed_x_Q98": np.nan,
                "LOWESS_frac": 0.65,
                "Bootstrap_replicates": N_BOOT,
            "Spearman_permutations": N_PERMUTATIONS,
                "Status": "Insufficient data",
            })
        else:
            x = np.asarray(res.get("x_obs", []), dtype=float)
            x = x[np.isfinite(x)]
            rows.append({
                "Relationship": "Transformed/non-native land fraction vs VPD trend",
                "Region": region,
                "N": int(res.get("n", 0)),
                "Weighted_Spearman_rho": float(res.get("rho", np.nan)),
                "Permutation_P_value": float(res.get("p", np.nan)),
                "Observed_x_Q02": float(np.nanpercentile(x, 2)) if x.size else np.nan,
                "Observed_x_Q50": float(np.nanpercentile(x, 50)) if x.size else np.nan,
                "Observed_x_Q98": float(np.nanpercentile(x, 98)) if x.size else np.nan,
                "LOWESS_frac": 0.65,
                "Bootstrap_replicates": N_BOOT,
            "Spearman_permutations": N_PERMUTATIONS,
                "Status": "OK",
            })
    out = pd.DataFrame(rows)
    save_table_dual(out, "Supplementary_Table_S11")
    return out

def detect_column(columns, candidates):
    norm_map = {normalize_text(c): c for c in columns}
    for cand in candidates:
        nc = normalize_text(cand)
        if nc in norm_map: return norm_map[nc]
    for col in columns:
        nc = normalize_text(col)
        for cand in candidates:
            if normalize_text(cand) in nc: return col
    return None

def find_municipality_shapefile():
    if MUNICIPALITY_SHP and os.path.isfile(MUNICIPALITY_SHP):
        return MUNICIPALITY_SHP
    raise FileNotFoundError(f"Municipality shapefile not found: {MUNICIPALITY_SHP}")

def safe_div(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    return np.where((den > 0) & np.isfinite(den), num / den, np.nan)

def weighted_mean_values(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if np.any(ok) else np.nan

def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok): return np.nan
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cdf, q)])

def _weighted_rank_correlation(x, y, w):
    xranks = stats.rankdata(x)
    yranks = stats.rankdata(y)

    mx = np.average(xranks, weights=w)
    my = np.average(yranks, weights=w)
    cov = np.average((xranks - mx) * (yranks - my), weights=w)
    vx = np.average((xranks - mx) ** 2, weights=w)
    vy = np.average((yranks - my) ** 2, weights=w)

    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def weighted_spearman(
    x,
    y,
    w,
    n_permutations=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    """Area-weighted Spearman rho with two-sided permutation p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 10:
        return np.nan, np.nan

    x = x[ok]
    y = y[ok]
    w = w[ok]

    observed = _weighted_rank_correlation(x, y, w)
    if not np.isfinite(observed):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    extreme = 0
    valid_perm = 0

    for _ in range(int(n_permutations)):
        r_perm = _weighted_rank_correlation(x, rng.permutation(y), w)
        if not np.isfinite(r_perm):
            continue
        valid_perm += 1
        if abs(r_perm) >= abs(observed):
            extreme += 1

    if valid_perm == 0:
        p = np.nan
    else:
        p = (extreme + 1.0) / (valid_perm + 1.0)

    return float(observed), float(p)


# ============================================================
# Meteorological calculations (ERA5 T2m/Td2m → VPD/RH)
# ============================================================

def saturation_vapour_pressure_hpa(temp_c):
    """Saturation vapour pressure over water [hPa] (Bolton 1980 Tetens approximation)."""
    temp_c = np.asarray(temp_c, dtype=float)
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

def calc_rh_vpd(t_k, td_k):
    """Compute relative humidity [%] and vapor pressure deficit [kPa] from ERA5 T2m/Td2m [K]."""
    t_c, td_c = t_k - 273.15, td_k - 273.15
    es = saturation_vapour_pressure_hpa(t_c)
    e = saturation_vapour_pressure_hpa(td_c)
    rh = np.clip(100.0 * safe_div(e, es), 0, 100)
    vpd_kpa = np.maximum(es - e, 0) / 10.0  # hPa → kPa
    return rh, vpd_kpa

def find_var(ds, candidates):
    for c in candidates:
        if c in ds.data_vars: return c
    lowered = {v.lower(): v for v in ds.data_vars}
    for c in candidates:
        if c.lower() in lowered: return lowered[c.lower()]
    return None

def standardize_lat_lon(ds):
    rename = {c: "lat" for c in ["latitude"] if c in ds.coords}
    rename.update({c: "lon" for c in ["longitude"] if c in ds.coords})
    if rename: ds = ds.rename(rename)
    if "lon" in ds.coords and float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    return ds


# ============================================================
# ERA5 annual mechanism metrics and trend estimation
# ============================================================

def _vpd_to_kpa(vpd_da, source_name):
    """
    Convert a precomputed VPD field to kPa.

    Corrected Figure 01 daily caches store VPDmean in hPa. Fields explicitly
    labelled kPa are retained unchanged.
    """
    units = str(vpd_da.attrs.get("units", "")).strip().lower()

    if "kpa" in units:
        return vpd_da.astype(float)

    if "hpa" in units or "hectopascal" in units:
        return (vpd_da.astype(float) / 10.0).assign_attrs(
            units="kPa",
            conversion="hPa / 10",
        )
    # metadata are incomplete.
    if source_name in {"VPDmean", "VPDmax"}:
        return (vpd_da.astype(float) / 10.0).assign_attrs(
            units="kPa",
            conversion="assumed Figure 01 VPD hPa / 10",
        )

    if "kpa" in source_name.lower():
        return vpd_da.astype(float).assign_attrs(units="kPa")

    raise ValueError(
        f"Ambiguous VPD units for variable '{source_name}' "
        f"(units='{vpd_da.attrs.get('units', '')}'). "
        "Explicit hPa or kPa metadata are required."
    )


def _annual_mean_min_valid(da, min_valid_days=MIN_VALID_DAYS_PER_YEAR):
    """Annual mean retained only with the required number of valid daily values."""
    valid_count = da.notnull().sum("time")
    mean = da.mean("time", skipna=True)
    return mean.where(valid_count >= min_valid_days)


def process_daily_file(path, year):
    """Read one Figure 01 daily cache and derive annual diagnostics."""
    ds = safe_open_dataset(path, decode_times=True, chunks=None)
    ds = standardize_lat_lon(ds)

    vpd_var = find_var(
        ds,
        ["VPDmean", "vpd_kpa", "VPD_kPa", "vpd", "VPD"],
    )
    rh_var = find_var(
        ds,
        ["RHmean", "rh", "RH", "relative_humidity"],
    )

    if vpd_var and rh_var:
        vpd = _vpd_to_kpa(ds[vpd_var], vpd_var)
        rh = ds[rh_var].astype(float)
    else:
        t_var = find_var(ds, ["t2m", "T2M", "t2m_mean", "Tmean", "var167"])
        td_var = find_var(ds, ["d2m", "D2M", "td2m", "dewpoint", "var168"])

        if t_var is None or td_var is None:
            ds.close()
            raise ValueError(
                f"Cannot compute VPD/RH in {path}. Available: {list(ds.data_vars)}"
            )

        t = ds[t_var].astype(float)
        td = ds[td_var].astype(float)

        # Tmean/Td fields may already be Celsius in Figure 01-derived products.
        t_units = str(t.attrs.get("units", "")).lower()
        td_units = str(td.attrs.get("units", "")).lower()

        t_vals = t if ("celsius" in t_units or "degc" in t_units) else (t - 273.15)
        td_vals = td if ("celsius" in td_units or "degc" in td_units) else (td - 273.15)

        es = saturation_vapour_pressure_hpa(t_vals)
        e = saturation_vapour_pressure_hpa(td_vals)
        rh = xr.DataArray(
            np.clip(100.0 * safe_div(e, es), 0.0, 100.0),
            coords=t.coords,
            dims=t.dims,
            name="RH",
            attrs={"units": "percent"},
        )
        vpd = xr.DataArray(
            np.maximum(es - e, 0.0) / 10.0,
            coords=t.coords,
            dims=t.dims,
            name="VPD",
            attrs={"units": "kPa"},
        )

    ws_var = find_var(ds, ["WS10mean", "ws10", "WS10", "wind_speed"])
    if ws_var:
        ws = ds[ws_var].astype(float)
    else:
        u_var = find_var(ds, ["u10", "U10", "var165"])
        v_var = find_var(ds, ["v10", "V10", "var166"])
        if u_var and v_var:
            ws = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2)
        else:
            ws = xr.full_like(vpd, np.nan)
        ws.name = "WS10"

    sm_var = find_var(
        ds,
        ["SM1mean", "SM1", "swvl1", "volumetric_soil_water_layer_1"],
    )
    sm1 = ds[sm_var].astype(float) if sm_var else xr.full_like(vpd, np.nan)
    sm1.name = "SM1"

    # A sign-inverted soil-moisture diagnostic is retained only as a simple
    # dryness proxy. It is not interpreted as a physically calibrated SMD.
    smd = (-1.0 * sm1).assign_attrs(
        description="Sign-inverted soil-moisture proxy; larger = drier",
        units=sm1.attrs.get("units", ""),
    )
    smd.name = "SMD"

    annual = xr.Dataset({
        "VPD": _annual_mean_min_valid(vpd),
        "RH": _annual_mean_min_valid(rh),
        "WS10": _annual_mean_min_valid(ws),
        "SM1": _annual_mean_min_valid(sm1),
        "SMD": _annual_mean_min_valid(smd),
    }).assign_coords(year=year)

    annual["VPD"].attrs.update(units="kPa")
    annual["RH"].attrs.update(units="percent")

    ds.close()
    return annual

def build_annual_mechanism_dataset(overwrite=False):
    out_nc = os.path.join(OUT_DIR, "figure_03_ERA5_annual_mechanism_metrics_1990_2024.nc")
    if os.path.exists(out_nc) and not overwrite:
        ds_cached = safe_open_dataset(out_nc, decode_times=False, chunks=None)
        if all(v in ds_cached.data_vars for v in REQUIRED_ANNUAL_VARS):
            print(f"[SKIP] Annual mechanism file exists: {os.path.basename(out_nc)}")
            return ds_cached
        print(f"[WARN] Existing file missing variables; rebuilding.")
        ds_cached.close()
    
    annual_list = []
    for year in range(YEAR0, YEAR1 + 1):
        pattern = os.path.join(DAILY_DIR, f"ERA5_daily_Brazil_{year}.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[WARN] Daily ERA5 file not found for {year}")
            continue
        print(f"[INFO] Processing mechanism metrics for {year}")
        annual_list.append(process_daily_file(files[0], year))
    
    if len(annual_list) < MIN_YEARS_FOR_TREND:
        raise RuntimeError(f"Only {len(annual_list)} annual files; need ≥{MIN_YEARS_FOR_TREND}")
    
    ds_ann = xr.concat(annual_list, dim="year")
    ds_ann.attrs.update(get_software_metadata())
    safe_to_netcdf(ds_ann, out_nc)
    print(f"[OK] Saved: {os.path.basename(out_nc)}")
    return ds_ann

def _mk_score_and_variance(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    if n < MIN_YEARS_FOR_TREND:
        return np.nan, np.nan

    s = sum(
        np.sign(y[j] - y[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
    )

    _, counts = np.unique(y, return_counts=True)
    tie_term = sum(
        c * (c - 1) * (2 * c + 5)
        for c in counts
    )
    var_s = (
        n * (n - 1) * (2 * n + 5) - tie_term
    ) / 18.0

    return float(s), float(var_s)


def _mk_p_from_s_var(s, var_s):
    if not np.isfinite(s) or not np.isfinite(var_s) or var_s <= 0:
        return np.nan

    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def mann_kendall_pvalue(y, min_valid_years=MIN_YEARS_FOR_TREND):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) < min_valid_years:
        return np.nan

    if _HAS_PYMK:
        try:
            return float(mk.original_test(y).p)
        except Exception:
            pass

    s, var_s = _mk_score_and_variance(y)
    return _mk_p_from_s_var(s, var_s)


def detrended_lag1_autocorrelation(
    y,
    years,
    alpha=AUTOCORR_ALPHA,
    min_valid_years=MIN_YEARS_FOR_TREND,
):
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=float)

    valid = np.isfinite(y) & np.isfinite(years)
    y = y[valid]
    years = years[valid]
    n = len(y)

    if n < min_valid_years or n < 4:
        return np.nan, np.nan, False

    critical = float(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n))

    if np.nanstd(y) <= 1e-12:
        return 0.0, critical, False

    sen = theilslopes(y, years, alpha=0.05)
    slope, intercept = float(sen[0]), float(sen[1])
    residual = y - (intercept + slope * years)

    ranks = pd.Series(residual).rank(method="average").to_numpy(dtype=float)

    if np.nanstd(ranks[:-1]) <= 1e-12 or np.nanstd(ranks[1:]) <= 1e-12:
        r1 = 0.0
    else:
        r1 = float(np.corrcoef(ranks[:-1], ranks[1:])[0, 1])

    significant = bool(np.isfinite(r1) and abs(r1) > critical)
    return r1, critical, significant


def hamed_rao_mk_pvalue_lag1(
    y,
    years,
    alpha=AUTOCORR_ALPHA,
    min_valid_years=MIN_YEARS_FOR_TREND,
):
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=float)

    valid = np.isfinite(y) & np.isfinite(years)
    y = y[valid]
    years = years[valid]

    if len(y) < min_valid_years:
        return np.nan

    if _HAS_PYMK:
        try:
            return float(
                mk.hamed_rao_modification_test(
                    y,
                    alpha=alpha,
                    lag=1,
                ).p
            )
        except Exception:
            pass

    s, var_s = _mk_score_and_variance(y)
    if not np.isfinite(var_s) or var_s <= 0:
        return np.nan

    n = len(y)
    sen = theilslopes(y, years, alpha=0.05)
    slope, intercept = float(sen[0]), float(sen[1])
    detrended = y - (intercept + slope * years)
    ranks = pd.Series(detrended).rank(method="average").to_numpy(dtype=float)

    if np.std(ranks[:-1]) <= 1e-12 or np.std(ranks[1:]) <= 1e-12:
        rho1 = 0.0
    else:
        rho1 = float(np.corrcoef(ranks[:-1], ranks[1:])[0, 1])

    critical = float(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n))
    if abs(rho1) <= critical:
        return _mk_p_from_s_var(s, var_s)

    correction = 1.0 + (2.0 * (n - 3) / n) * abs(rho1)
    correction = max(correction, 1.0)
    return _mk_p_from_s_var(s, var_s * correction)


def robust_trend_per_decade(
    y,
    years,
    min_valid_years=MIN_YEARS_FOR_TREND,
):
    """Theil-Sen magnitude + autocorrelation-aware MK significance."""
    y = np.asarray(y, dtype=float)
    years = np.asarray(years, dtype=float)

    valid = np.isfinite(y) & np.isfinite(years)
    yv = y[valid]
    xv = years[valid]

    if len(yv) < min_valid_years:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    sen = theilslopes(yv, xv, alpha=0.05)
    slope_decade = float(sen[0] * 10.0)

    original_p = mann_kendall_pvalue(
        yv,
        min_valid_years=min_valid_years,
    )

    lag1_r, lag1_critical, lag1_sig = detrended_lag1_autocorrelation(
        yv,
        xv,
        min_valid_years=min_valid_years,
    )

    if lag1_sig:
        selected_p = hamed_rao_mk_pvalue_lag1(
            yv,
            xv,
            min_valid_years=min_valid_years,
        )
        method_code = 1.0
    else:
        selected_p = original_p
        method_code = 0.0

    return (
        slope_decade,
        selected_p,
        lag1_r,
        lag1_critical,
        method_code,
    )

def calc_trends(ds_ann, overwrite=False):
    out_nc = os.path.join(
        OUT_DIR,
        "figure_03_ERA5_mechanistic_trends_1990_2024.nc",
    )

    if os.path.exists(out_nc) and not overwrite:
        ds_cached = safe_open_dataset(out_nc, decode_times=False, chunks=None)
        compatible = (
            all(var in ds_cached.data_vars for var in [
                "VPD_trend_decade", "VPD_pvalue",
                "RH_trend_decade", "RH_pvalue",
            ])
            and ds_cached.attrs.get("trend_significance_version")
            == TREND_SIGNIFICANCE_VERSION
            and ds_cached["VPD_trend_decade"].attrs.get("units") == "kPa decade-1"
        )
        if compatible:
            print(f"[SKIP] Compatible mechanistic trend file exists: {os.path.basename(out_nc)}")
            return ds_cached
        ds_cached.close()
        print("[WARN] Existing mechanistic trend cache is incompatible; recomputing.")

    years = ds_ann["year"].values.astype(float)
    out = xr.Dataset(
        coords={
            "lat": ds_ann["lat"],
            "lon": ds_ann["lon"],
        }
    )

    for var in ["VPD", "RH", "SMD", "SM1", "WS10"]:
        print(f"[INFO] Computing aligned Theil-Sen/autocorrelation-aware MK trends for {var}")

        arr = ds_ann[var].values
        _, nlat, nlon = arr.shape

        slope = np.full((nlat, nlon), np.nan, dtype=np.float32)
        pval = np.full_like(slope, np.nan)
        lag1 = np.full_like(slope, np.nan)
        lag1_critical = np.full_like(slope, np.nan)
        method = np.full_like(slope, np.nan)

        for i in range(nlat):
            if i % 20 == 0:
                print(f"       {var}: lat row {i + 1}/{nlat}")

            for j in range(nlon):
                (
                    sen,
                    p,
                    r1,
                    rcrit,
                    method_code,
                ) = robust_trend_per_decade(
                    arr[:, i, j],
                    years,
                )

                slope[i, j] = sen
                pval[i, j] = p
                lag1[i, j] = r1
                lag1_critical[i, j] = rcrit
                method[i, j] = method_code

        out[f"{var}_trend_decade"] = (("lat", "lon"), slope)
        out[f"{var}_pvalue"] = (("lat", "lon"), pval)
        out[f"{var}_lag1_autocorr_detrended"] = (("lat", "lon"), lag1)
        out[f"{var}_lag1_critical"] = (("lat", "lon"), lag1_critical)
        out[f"{var}_MK_method_code"] = (("lat", "lon"), method)

    out["VPD_trend_decade"].attrs["units"] = "kPa decade-1"
    out["RH_trend_decade"].attrs["units"] = "percent decade-1"

    out.attrs.update(get_software_metadata())
    out.attrs.update({
        "main_trend_method": "Theil-Sen median slope",
        "trend_significance_version": TREND_SIGNIFICANCE_VERSION,
        "MK_method_code_definition": (
            "0=original MK; 1=Hamed-Rao variance-corrected MK lag1"
        ),
        "minimum_valid_years": MIN_YEARS_FOR_TREND,
    })

    safe_to_netcdf(
        out,
        out_nc,
        encoding={
            var: {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
            for var in out.data_vars
        },
    )

    print(f"[OK] Saved: {os.path.basename(out_nc)}")
    return out

def add_dryhot_shift_to_mechanistic_trends(ds_trends, overwrite=False):
    """Attach the Figure 01 common-scale DHW–HHW contrast field."""
    out_nc = os.path.join(
        OUT_DIR,
        "figure_03_ERA5_mechanistic_trends_with_dryhot_1990_2024.nc",
    )

    if os.path.exists(out_nc) and not overwrite:
        ds_cached = safe_open_dataset(out_nc, decode_times=False, chunks=None)
        required = [
            "dry_minus_humid_intensity_trend",
            "DHW_TmaxOnly_intensity_trend_decade",
            "DHW_TmaxOnly_intensity_pvalue",
            "dryhot_morphological_mask",
            "dry_minus_humid_intensity_trend_masked",
        ]
        if (
            all(var in ds_cached.data_vars for var in required)
            and ds_cached.attrs.get("cross_regime_source")
            == "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
        ):
            print(f"[SKIP] Compatible mechanistic + dry-hot file exists: {os.path.basename(out_nc)}")
            return ds_cached
        ds_cached.close()
        print("[WARN] Existing dry-hot mechanism cache is incompatible; rebuilding.")

    f1 = safe_open_dataset(FIG1_TRENDS_NC, decode_times=False, chunks=None)
    f1 = standardize_lat_lon(f1)
    validate_figure1_compatibility(f1)
    # interpolation would conceal a pipeline mismatch, so exact/allclose
    # coordinate alignment is required.
    if (
        f1.sizes.get("lat") != ds_trends.sizes.get("lat")
        or f1.sizes.get("lon") != ds_trends.sizes.get("lon")
        or not np.allclose(f1["lat"].values, ds_trends["lat"].values)
        or not np.allclose(f1["lon"].values, ds_trends["lon"].values)
    ):
        f1.close()
        raise ValueError(
            "Figure 01 and Figure 03 grids are not aligned. Because both derive "
            "from the same Figure 01 daily cache, interpolation is not performed "
            "silently; check the input products."
        )

    ds_out = ds_trends.copy()

    ds_out["dry_minus_humid_intensity_trend"] = (
        f1["DHW_minus_HHW_TmaxOnly_intensity_trend_decade"]
        .astype("float32")
    )
    ds_out["DHW_TmaxOnly_intensity_trend_decade"] = (
        f1["DHW_TmaxOnly_intensity_trend_decade"]
        .astype("float32")
    )
    ds_out["DHW_TmaxOnly_intensity_pvalue"] = (
        f1["DHW_TmaxOnly_intensity_pvalue"]
        .astype("float32")
    )
    # positive common-scale DHW trend + minimum displayed magnitude.
    raw_mask = (
        (ds_out["dry_minus_humid_intensity_trend"] > DRYHOT_MASK_MIN_SHIFT)
        & (ds_out["DHW_TmaxOnly_intensity_trend_decade"] > 0)
        & (ds_out["DHW_TmaxOnly_intensity_pvalue"] <= DRYHOT_MASK_P)
    )

    raw_mask_np = (
        np.asarray(raw_mask.values, dtype=bool)
        & np.isfinite(ds_out["dry_minus_humid_intensity_trend"].values)
    )

    # Spatial-coherence filter for visualization only.
    morph_mask_np = binary_opening(
        raw_mask_np,
        structure=DRYHOT_OPENING_STRUCTURE,
    )
    morph_mask_np = (
        binary_closing(
            morph_mask_np,
            structure=DRYHOT_CLOSING_STRUCTURE,
        )
        & raw_mask_np
    )

    morph_mask = xr.DataArray(
        morph_mask_np,
        coords=ds_out["dry_minus_humid_intensity_trend"].coords,
        dims=ds_out["dry_minus_humid_intensity_trend"].dims,
        name="dryhot_morphological_mask",
    )

    ds_out["dryhot_morphological_mask"] = morph_mask
    ds_out["dry_minus_humid_intensity_trend_masked"] = (
        ds_out["dry_minus_humid_intensity_trend"].where(morph_mask)
    )

    ds_out["dry_minus_humid_intensity_trend"].attrs.update({
        "units": "standardized severity decade-1",
        "definition": (
            "DHW_TmaxOnly_intensity_trend_decade - "
            "HHW_TmaxOnly_intensity_trend_decade"
        ),
    })

    ds_out.attrs.update(get_software_metadata())
    ds_out.attrs.update({
        "dry_hot_transition_definition": (
            "Common-scale Tmax-only DHW minus HHW "
            "intensity-trend contrast"
        ),
        "cross_regime_source": (
            "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
        ),
        "visualization_minimum_shift": DRYHOT_MASK_MIN_SHIFT,
        "DHW_significance_threshold": DRYHOT_MASK_P,
        "visualization_filter_only": (
            "Morphological opening/closing is used only for panel-a display; "
            "not for municipality associations or regressions."
        ),
        "era5_source_file": os.path.basename(FIG1_TRENDS_NC),
    })

    f1.close()

    safe_to_netcdf(
        ds_out,
        out_nc,
        encoding={
            var: {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
            for var in ds_out.data_vars
            if ds_out[var].dtype.kind in {"f", "i", "u", "b"}
        },
    )

    print(f"[OK] Saved: {os.path.basename(out_nc)}")
    return ds_out


# ============================================================
# MapBiomas and municipality integration (aligned with Figure 2)
# ============================================================

def read_mapbiomas_landcover():
    print(f"[INFO] Reading MapBiomas coverage: {os.path.basename(COVERAGE_XLSX)}")
    df = pd.read_excel(COVERAGE_XLSX, sheet_name="COVERAGE_10.1")
    required = ["country", "biome", "state", "state_acronym", "municipality", "class_level_1", "class_level_2", YEAR0, YEAR1]
    if missing := [c for c in required if c not in df.columns]:
        raise ValueError(f"Missing columns in MapBiomas file: {missing}")
    for y in [YEAR0, YEAR1]: df[y] = pd.to_numeric(df[y], errors="coerce")
    df["municipality_norm"] = df["municipality"].map(normalize_text)
    df["state_acronym_norm"] = df["state_acronym"].map(normalize_text)
    df["biome_clean"] = df["biome"].map(normalize_biome)
    df["class1_norm"] = df["class_level_1"].map(normalize_class)
    df["class2_norm"] = df["class_level_2"].map(normalize_class)
    return df

def build_landcover_municipality_table(df):
    print(f"[INFO] Building municipality land-cover metrics for {YEAR0}–{YEAR1}.")
    id_cols = ["country", "biome", "biome_clean", "state", "state_acronym", "municipality", "municipality_norm", "state_acronym_norm"]
    total = df.groupby(id_cols, dropna=False)[[YEAR0, YEAR1]].sum().rename(columns={YEAR0: "total_area_1990_ha", YEAR1: "total_area_2024_ha"}).reset_index()
    
    def agg(mask, prefix):
        return df.loc[mask].groupby(id_cols, dropna=False)[[YEAR0, YEAR1]].sum().rename(columns={YEAR0: f"{prefix}_1990_ha", YEAR1: f"{prefix}_2024_ha"}).reset_index()
    
    out = total.copy()
    for prefix, mask in [("native", df["class1_norm"].isin(NATIVE_LEVEL1)), 
                        ("agro", df["class1_norm"].isin(AGRO_LEVEL1)), 
                        ("urban", df["class2_norm"].isin(URBAN_LEVEL2))]:
        out = out.merge(agg(mask, prefix), on=id_cols, how="left")
    
    area_cols = [c for c in out.columns if c.endswith("_ha")]
    out[area_cols] = out[area_cols].fillna(0.0)
    before = len(out)
    out = out[out["total_area_2024_ha"] >= MIN_AREA_HA].copy()
    print(f"[QC] Removed {before - len(out)} rows with area < {MIN_AREA_HA:.0f} ha.")
    
    den0, den1 = out["total_area_1990_ha"].values, out["total_area_2024_ha"].values
    for group in ["native", "agro", "urban"]:
        out[f"{group}_pct_1990"] = 100.0 * safe_div(out[f"{group}_1990_ha"].values, den0)
        out[f"{group}_pct_2024"] = 100.0 * safe_div(out[f"{group}_2024_ha"].values, den1)
    
    out["veg_loss_pct_points"] = out["native_pct_1990"] - out["native_pct_2024"]
    out["agri_gain_pct_points"] = out["agro_pct_2024"] - out["agro_pct_1990"]
    out["urban_gain_pct_points"] = out["urban_pct_2024"] - out["urban_pct_1990"]
    
    rows = []
    for keys, g in out.groupby(["municipality_norm", "state_acronym_norm"], dropna=False):
        total_area = g["total_area_2024_ha"].sum()
        if total_area <= 0: continue
        dominant = g.loc[g["total_area_2024_ha"].idxmax()]
        row = {"municipality_norm": keys[0], "state_acronym_norm": keys[1], "municipality": dominant["municipality"],
               "state_acronym": dominant["state_acronym"], "dominant_biome": dominant["biome_clean"], "total_area_2024_ha": total_area}
        for v in ["veg_loss_pct_points", "agri_gain_pct_points", "urban_gain_pct_points", "native_pct_1990", "native_pct_2024", "agro_pct_1990", "agro_pct_2024", "urban_pct_1990", "urban_pct_2024"]:
            row[v] = weighted_mean_values(g[v].values, g["total_area_2024_ha"].values)
        rows.append(row)
    
    mun = pd.DataFrame(rows)

    # Cumulative transformed-land state, consistent with the revised Figure 2.
    # This is saved explicitly because the mechanistic controls test whether
    # current transformed/non-native land fraction is associated with VPD and DHW−HHW.
    mun["transformed_non_native_pct_2024"] = 100.0 - pd.to_numeric(mun["native_pct_2024"], errors="coerce")
    mun["transformed_non_native_pct_1990"] = 100.0 - pd.to_numeric(mun["native_pct_1990"], errors="coerce")

    return add_focus_regions(mun)

def add_focus_regions(df):
    df = df.copy()
    state = df["state_acronym_norm"].astype(str).str.upper()
    dom = df["dominant_biome"].astype(str)
    se = state.isin(SOUTHEAST_STATES)
    urban_gain = pd.to_numeric(df["urban_gain_pct_points"], errors="coerce")
    urban_pct_2024 = pd.to_numeric(df.get("urban_pct_2024", np.nan), errors="coerce")
    se_gain_thr = np.nanpercentile(urban_gain[se], 75) if np.isfinite(urban_gain[se]).any() else np.nan
    se_urban_thr = np.nanpercentile(urban_pct_2024[se], 75) if np.isfinite(urban_pct_2024[se]).any() else np.nan
    is_southeast_urban = se & ((urban_gain >= se_gain_thr) | (urban_pct_2024 >= se_urban_thr))
    is_matopiba = state.isin(MATOPIBA_STATES) & dom.isin(["Cerrado", "Caatinga"])
    is_semiarid = state.isin(SEMIARID_STATES) & dom.eq("Caatinga")
    region = dom.copy().astype(object)
    region[is_semiarid.values], region[is_matopiba.values], region[is_southeast_urban.values] = "Semi-arid Northeast", "MATOPIBA", "Urban Southeast"
    df["focus_region"] = np.where(np.isin(region, REGION_ORDER), region, "Other")
    return df

def load_municipality_geometry():
    shp = find_municipality_shapefile()
    print(f"[INFO] Reading municipality shapefile: {os.path.basename(shp)}")
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    name_col = detect_column(gdf.columns, ["NM_MUN", "NM_MUNICIP", "NM_MUNICIPIO", "NOME", "MUNICIPIO", "municipality", "name"])
    state_col = detect_column(gdf.columns, ["SIGLA_UF", "UF", "state_acronym", "SIGLA"])
    if name_col is None or state_col is None:
        raise ValueError(f"Could not detect municipality/state columns. Columns: {list(gdf.columns)}")
    gdf["municipality_norm"] = gdf[name_col].map(normalize_text)
    gdf["state_acronym_norm"] = gdf[state_col].map(normalize_text)
    pts = gdf.geometry.representative_point()
    gdf["rep_lon"], gdf["rep_lat"] = pts.x, pts.y
    return gdf

def build_era5_point_gdf(ds):
    lon, lat = ds["lon"].values, ds["lat"].values
    if np.nanmax(lon) > 180: lon = ((lon + 180) % 360) - 180
    lon2d, lat2d = np.meshgrid(lon, lat)
    data = {"era5_lat": lat2d.ravel(), "era5_lon": lon2d.ravel()}
    for v in ds.data_vars: data[v] = ds[v].values.ravel()
    pts = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["era5_lon"], data["era5_lat"]), crs="EPSG:4326")
    return pts[(pts["era5_lon"] >= LON_MIN - 1) & (pts["era5_lon"] <= LON_MAX + 1) & (pts["era5_lat"] >= LAT_MIN - 1) & (pts["era5_lat"] <= LAT_MAX + 1)].copy()

def aggregate_grid_to_municipalities_zonal(gdf_mun, ds):
    """
    Aggregate mechanism trend fields with cos(latitude) grid-cell weights.

    P-values are not averaged. For VPD and RH, statistical support is stored as
    weighted fractions of contributing grid cells with the expected significant
    trend direction.
    """
    pts = build_era5_point_gdf(ds)
    keys = ["municipality_norm", "state_acronym_norm"]
    gdf_small = gdf_mun[
        keys + ["rep_lon", "rep_lat", "geometry"]
    ].copy()

    try:
        joined = gpd.sjoin(
            pts,
            gdf_small[keys + ["geometry"]],
            how="inner",
            predicate="within",
        )
    except Exception as exc:
        print(f"[WARN] Spatial join failed: {exc}. Falling back to nearest grid.")
        return aggregate_grid_to_municipalities_nearest(gdf_small, ds)

    joined["era5_area_weight"] = np.cos(
        np.deg2rad(pd.to_numeric(joined["era5_lat"], errors="coerce"))
    )

    trend_vars = [
        var
        for var in ds.data_vars
        if var.endswith("_trend_decade")
    ]

    rows = []

    for name, g in joined.groupby(keys, dropna=False):
        w = pd.to_numeric(
            g["era5_area_weight"],
            errors="coerce",
        ).to_numpy(dtype=float)

        row = {
            "municipality_norm": name[0],
            "state_acronym_norm": name[1],
            "era5_mechanism_method": "zonal_gridpoint_coslat_weighted_mean",
            "era5_mechanism_n_gridpoints": len(g),
            "era5_mechanism_uncertainty_flag": 0,
        }

        for var in trend_vars:
            row[var] = weighted_mean_values(
                pd.to_numeric(g[var], errors="coerce").values,
                w,
            )

        # Statistical-support fractions; p-values themselves are not averaged.
        support_specs = [
            (
                "VPD",
                "VPD_trend_decade",
                "VPD_pvalue",
                lambda trend: trend > 0,
                "VPD_significant_positive_fraction",
            ),
            (
                "RH",
                "RH_trend_decade",
                "RH_pvalue",
                lambda trend: trend < 0,
                "RH_significant_negative_fraction",
            ),
        ]

        for _, trend_var, p_var, direction, out_name in support_specs:
            if trend_var not in g.columns or p_var not in g.columns:
                row[out_name] = np.nan
                continue

            trend = pd.to_numeric(g[trend_var], errors="coerce").to_numpy(dtype=float)
            pval = pd.to_numeric(g[p_var], errors="coerce").to_numpy(dtype=float)

            valid = (
                np.isfinite(trend)
                & np.isfinite(pval)
                & np.isfinite(w)
                & (w > 0)
            )

            if valid.any():
                wv = w[valid]
                sig = direction(trend[valid]) & (pval[valid] <= P_STIPPLE)
                row[out_name] = float(np.sum(wv[sig]) / np.sum(wv))
            else:
                row[out_name] = np.nan

        rows.append(row)

    zonal = pd.DataFrame(rows)

    missing = (
        gdf_small[keys + ["rep_lon", "rep_lat"]]
        .merge(zonal[keys], on=keys, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )

    print(
        f"[INFO] Mechanism ERA5 zonal: {len(zonal)}; "
        f"fallback: {len(missing)} municipalities"
    )

    if len(missing) > 0:
        nearest = aggregate_grid_to_municipalities_nearest(
            gdf_small.merge(missing[keys], on=keys, how="inner"),
            ds,
        )
        return pd.concat([zonal, nearest], ignore_index=True)

    return zonal

def aggregate_grid_to_municipalities_nearest(gdf_mun, ds):
    """Nearest-grid fallback with explicit one-cell support fractions."""
    lon = ds["lon"].values
    lat = ds["lat"].values

    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    lon2d, lat2d = np.meshgrid(lon, lat)
    points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    tree = cKDTree(points)

    keys = ["municipality_norm", "state_acronym_norm"]
    out = gdf_mun[
        keys + ["rep_lon", "rep_lat"]
    ].drop_duplicates().copy()

    ok = out["rep_lon"].notna() & out["rep_lat"].notna()
    query = np.column_stack([
        out.loc[ok, "rep_lat"].values,
        out.loc[ok, "rep_lon"].values,
    ])
    _, idx = tree.query(query)

    out["era5_mechanism_method"] = "nearest_gridpoint_fallback"
    out["era5_mechanism_n_gridpoints"] = 1
    out["era5_mechanism_uncertainty_flag"] = 1

    lat_idx = np.array([], dtype=int)
    lon_idx = np.array([], dtype=int)

    if ok.any():
        nearest_lat = points[idx, 0]
        nearest_lon = points[idx, 1]
        lat_idx = np.array([
            np.argmin(np.abs(lat - value))
            for value in nearest_lat
        ])
        lon_idx = np.array([
            np.argmin(np.abs(lon - value))
            for value in nearest_lon
        ])

    for var in [
        v for v in ds.data_vars
        if v.endswith("_trend_decade")
    ]:
        arr = ds[var].values
        values = np.full(len(out), np.nan, dtype=float)
        if ok.any():
            values[ok.values] = arr[lat_idx, lon_idx]
        out[var] = values

    for trend_var, p_var, direction, out_name in [
        (
            "VPD_trend_decade",
            "VPD_pvalue",
            lambda t: t > 0,
            "VPD_significant_positive_fraction",
        ),
        (
            "RH_trend_decade",
            "RH_pvalue",
            lambda t: t < 0,
            "RH_significant_negative_fraction",
        ),
    ]:
        out[out_name] = np.nan
        if ok.any() and trend_var in ds and p_var in ds:
            trend = ds[trend_var].values[lat_idx, lon_idx]
            pval = ds[p_var].values[lat_idx, lon_idx]
            values = np.where(
                np.isfinite(trend) & np.isfinite(pval),
                (direction(trend) & (pval <= P_STIPPLE)).astype(float),
                np.nan,
            )
            out.loc[ok, out_name] = values

    return out

def read_corrected_figure2_table():
    """Read the explicitly supplied Figure 02 municipality table."""
    print(f"[INFO] Reading Figure 02 municipality table: {FIG2_TABLE}")

    if FIG2_TABLE.lower().endswith(".nc"):
        with safe_open_dataset(FIG2_TABLE, decode_times=False, chunks=None) as ds:
            df = ds.to_dataframe().reset_index()
    else:
        df = pd.read_csv(FIG2_TABLE)

    df = standardize_region_columns(df)
    validate_figure2_compatibility(df)
    return df


def get_dry_hot_shift_table():
    """
    Return only the corrected Figure 02 common-scale dry-hot-transition fields.

    No legacy fallback and no reconstruction from primary regime intensities are
    allowed, preventing accidental reintroduction of the old unit inconsistency.
    """
    f2 = read_corrected_figure2_table()

    keep = [
        "municipality_norm",
        "state_acronym_norm",
        "dry_minus_humid_intensity_trend",
        "DHW_TmaxOnly_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
    ]

    for optional in [
        "DHW_TmaxOnly_significant_positive_fraction",
        "dry_hot_hotspot",
        "dry_hot_hotspot_intensity",
    ]:
        if optional in f2.columns:
            keep.append(optional)

    return f2[keep].drop_duplicates(
        subset=["municipality_norm", "state_acronym_norm"]
    )


def ensure_dryhot_shift_columns(df):
    """Require the Figure 02 common-scale DHW–HHW contrast."""
    target = "dry_minus_humid_intensity_trend"

    if target not in df.columns:
        raise KeyError(
            "DHW–HHW contrast field is missing. "
            "Use the Figure 02 municipality table."
        )

    values = pd.to_numeric(df[target], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(
            "DHW–HHW contrast field contains no finite values."
        )

    out = df.copy()
    out[target] = values
    return out

def build_municipality_mechanism_table(ds_trends):
    raw = read_mapbiomas_landcover()
    mun = build_landcover_municipality_table(raw)
    gdf = load_municipality_geometry()
    mech = aggregate_grid_to_municipalities_zonal(gdf, ds_trends)
    dryhot = get_dry_hot_shift_table()
    out = mun.merge(mech, on=["municipality_norm", "state_acronym_norm"], how="left")
    out = out.merge(dryhot, on=["municipality_norm", "state_acronym_norm"], how="left", suffixes=("", "_dryhot"))
    # Representative coordinates are used only as simple spatial-gradient controls
    # in the supplementary/diagnostic regressions.
    coords = gdf[["municipality_norm", "state_acronym_norm", "rep_lat", "rep_lon"]].drop_duplicates()
    out = out.merge(coords, on=["municipality_norm", "state_acronym_norm"], how="left")
    out = ensure_dryhot_shift_columns(out)
    # Guarantee cumulative transformed-land metrics are present in the saved table
    # and in the controlled regressions. This is essential for consistency with
    # the revised Figure 2, where cumulative transformed/non-native land fraction
    # is the main land-transformation metric.
    if "transformed_non_native_pct_2024" not in out.columns and "native_pct_2024" in out.columns:
        out["transformed_non_native_pct_2024"] = 100.0 - pd.to_numeric(out["native_pct_2024"], errors="coerce")
    if "transformed_non_native_pct_1990" not in out.columns and "native_pct_1990" in out.columns:
        out["transformed_non_native_pct_1990"] = 100.0 - pd.to_numeric(out["native_pct_1990"], errors="coerce")

    out = standardize_region_columns(out)

    print("[DIAGNOSTICS] Dry-hot metric availability:")
    print(f"  dry_minus_humid_intensity_trend non-NaN: {int(out['dry_minus_humid_intensity_trend'].notna().sum())}/{len(out)}")
    
    # Save as NetCDF with full metadata (Nature requirement)
    out_file_nc = os.path.join(OUT_DIR, "figure_03_municipality_mechanism_table_1990_2024.nc")
    out_xr = out.set_index(["municipality_norm", "state_acronym_norm"]).to_xarray()
    out_xr.attrs.update(get_software_metadata())
    out_xr.attrs.update({
        "dry_hot_transition_definition": "Corrected common Tmax-only DHW minus HHW intensity-trend contrast",
        "hotspot_threshold": DRYHOT_MASK_MIN_SHIFT,
        "significance_threshold": P_STIPPLE,
        "era5_source_file": os.path.basename(FIG1_TRENDS_NC),
        "mapbiomas_source_file": os.path.basename(COVERAGE_XLSX),
        "figure2_dependency": os.path.basename(FIG2_TABLE),
    })
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out_xr.data_vars if out_xr[v].dtype in [np.float32, np.float64]}
    safe_to_netcdf(out_xr, out_file_nc, encoding=encoding)
    
    # Also save CSV for convenience
    out_file_csv = os.path.join(OUT_DIR, "figure_03_municipality_mechanism_table_1990_2024.csv")
    out.to_csv(out_file_csv, index=False)
    print(f"[OK] Saved municipality table (NetCDF): {os.path.basename(out_file_nc)}")
    print(f"[OK] Saved municipality table (CSV): {os.path.basename(out_file_csv)}")
    
    gdf_out = gdf.merge(out, on=["municipality_norm", "state_acronym_norm"], how="left")
    return out, gdf_out


# ============================================================
# Regression / LOWESS diagnostics (aligned with Figure 2)
# ============================================================

def prepare_relationship_data(df, x_col, y_col, regions=None):
    df = standardize_region_columns(ensure_dryhot_shift_columns(df))
    for col in [x_col, y_col]:
        if col not in df.columns:
            similar = [c for c in df.columns if c.startswith(col) or col in c]
            if similar: df[col] = pd.to_numeric(df[similar[0]], errors="coerce")
    cols = [x_col, y_col, "focus_region", "total_area_2024_ha"]
    if missing := [c for c in cols if c not in df.columns]:
        raise KeyError(f"Missing required columns: {missing}")
    d = df[cols].copy()
    if regions is not None:
        d = d[d["focus_region"].isin(regions)].copy()
    else:
        d = d[d["focus_region"].isin(REGION_ORDER)].copy()
    for col in [x_col, y_col, "total_area_2024_ha"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col, "total_area_2024_ha"])
    d = d[d["total_area_2024_ha"] > 0].copy()
    if len(d) > 20:
        for col in [x_col, y_col]:
            q1, q99 = np.nanpercentile(d[col], [2, 98])
            d = d[(d[col] >= q1) & (d[col] <= q99)]
    w = np.sqrt(d["total_area_2024_ha"].values)
    d["weight"] = w / np.nanmean(w)
    return d

def lowess_with_bootstrap(df, x_col, y_col, frac=0.65):
    x, y = df[x_col].values.astype(float), df[y_col].values.astype(float)
    q01, q99 = np.nanpercentile(x, [2, 98])
    xgrid = np.linspace(q01, q99, 180)
    order = np.argsort(x)
    fit = lowess(y[order], x[order], frac=frac, it=1, return_sorted=True)
    xu, idx = np.unique(fit[:, 0], return_index=True)
    yhat = np.interp(xgrid, xu, fit[:, 1][idx])
    rng = np.random.default_rng(RANDOM_SEED)
    boot = []
    for _ in range(N_BOOT):
        idxb = rng.integers(0, len(x), size=len(x))
        xb, yb = x[idxb], y[idxb]
        try:
            order_b = np.argsort(xb)
            fb = lowess(yb[order_b], xb[order_b], frac=frac, it=1, return_sorted=True)
            xbu, ib = np.unique(fb[:, 0], return_index=True)
            if len(xbu) >= 3: boot.append(np.interp(xgrid, xbu, fb[:, 1][ib]))
        except Exception: continue
    lo = np.nanpercentile(np.vstack(boot), 2.5, axis=0) if len(boot) >= 20 else np.full_like(yhat, np.nan)
    hi = np.nanpercentile(np.vstack(boot), 97.5, axis=0) if len(boot) >= 20 else np.full_like(yhat, np.nan)
    rho, p = weighted_spearman(x, y, df["weight"].values)
    return {"x": xgrid, "yhat": yhat, "lo": lo, "hi": hi, "rho": rho, "p": p, "n": len(df), "x_obs": x}


def zscore_column(d, col, weight_col="weight"):
    """Weighted z-score used for interpretable WLS coefficients."""
    x = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(d[weight_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.full(len(d), np.nan)
    mu = np.average(x[ok], weights=w[ok])
    sd = np.sqrt(np.average((x[ok] - mu) ** 2, weights=w[ok]))
    if not np.isfinite(sd) or sd <= 0:
        return np.full(len(d), np.nan)
    return (x - mu) / sd


def fit_controlled_wls(df, response, predictor, controls, model_name):
    """
    Fit weighted least squares (WLS) with HC3 heteroscedasticity-robust standard errors.

    These models are not used as formal attribution. They are diagnostic tests
    designed to evaluate whether the pathway remains directionally consistent
    after controlling for biome/focus-region structure and broad spatial gradients.
    """
    required = [response, predictor, "weight"] + controls
    d = df.copy()
    for c in [response, predictor, "weight", "rep_lat", "rep_lon"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[c for c in required if c in d.columns]).copy()
    d = d[d["weight"] > 0].copy()
    if len(d) < 80:
        return [{
            "model": model_name, "response": response, "predictor": predictor,
            "n": len(d), "term": "MODEL_NOT_FIT", "coef": np.nan,
            "se_HC3": np.nan, "p_HC3": np.nan, "r2": np.nan, "aic": np.nan,
        }]

    zcol = predictor + "_z"
    d[zcol] = zscore_column(d, predictor, weight_col="weight")
    d = d.dropna(subset=[zcol, response, "weight"]).copy()
    if len(d) < 80:
        return [{
            "model": model_name, "response": response, "predictor": predictor,
            "n": len(d), "term": "MODEL_NOT_FIT_AFTER_ZSCORE", "coef": np.nan,
            "se_HC3": np.nan, "p_HC3": np.nan, "r2": np.nan, "aic": np.nan,
        }]

    formula_terms = [zcol]
    for c in controls:
        if c == "dominant_biome":
            formula_terms.append("C(dominant_biome)")
        elif c == "focus_region":
            formula_terms.append("C(focus_region)")
        elif c in ["rep_lat", "rep_lon"] and c in d.columns:
            formula_terms.append(c)
    formula = f"{response} ~ " + " + ".join(formula_terms)

    try:
        fit = sm.WLS.from_formula(formula, data=d, weights=d["weight"]).fit(cov_type="HC3")
        return [{
            "model": model_name,
            "formula": formula,
            "response": response,
            "predictor": predictor,
            "n": int(fit.nobs),
            "term": zcol,
            "coef_per_1sd_predictor": float(fit.params.get(zcol, np.nan)),
            "se_HC3": float(fit.bse.get(zcol, np.nan)),
            "t_HC3": float(fit.tvalues.get(zcol, np.nan)),
            "p_HC3": float(fit.pvalues.get(zcol, np.nan)),
            "ci95_low": float(fit.conf_int().loc[zcol, 0]) if zcol in fit.params.index else np.nan,
            "ci95_high": float(fit.conf_int().loc[zcol, 1]) if zcol in fit.params.index else np.nan,
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "aic": float(fit.aic),
            "interpretation_note": "Conditional spatial association; not causal attribution or mediation.",
        }]
    except Exception as exc:
        return [{
            "model": model_name, "formula": formula, "response": response,
            "predictor": predictor, "n": len(d), "term": f"FIT_FAILED: {exc}",
            "coef_per_1sd_predictor": np.nan, "se_HC3": np.nan, "p_HC3": np.nan,
            "r2": np.nan, "aic": np.nan,
        }]


def run_controlled_mechanism_regressions(mun):
    """
    Supplementary diagnostic regressions for the mechanistic pathway.

    Core tests requested for Figure 3 robustness:
      1. agricultural expansion → VPD trend
      2. VPD trend → DHW−HHW
      3. agricultural expansion → DHW−HHW

    Additional cumulative-state tests are included because Figure 2 showed that
    the 2024 transformed-land state is more robust than recent expansion alone.
    """
    df = standardize_region_columns(ensure_dryhot_shift_columns(mun)).copy()
    needed = [
        "VPD_trend_decade", "dry_minus_humid_intensity_trend",
        "agri_gain_pct_points", "transformed_non_native_pct_2024",
        "dominant_biome", "focus_region", "total_area_2024_ha",
    ]
    for c in needed:
        if c not in df.columns:
            print(f"[WARN] Controlled regressions: missing {c}; models using it will be skipped.")
    # WLS weights are proportional to the square root of municipal area.
    df["weight"] = np.sqrt(pd.to_numeric(df["total_area_2024_ha"], errors="coerce"))
    df["weight"] = df["weight"] / np.nanmean(df["weight"])
    df = df[df["focus_region"].isin(REGION_ORDER)].copy()

    model_specs = [
        ("agri_to_vpd_biomeFE", "VPD_trend_decade", "agri_gain_pct_points", ["dominant_biome"]),
        ("agri_to_vpd_biomeFE_spatial", "VPD_trend_decade", "agri_gain_pct_points", ["dominant_biome", "rep_lat", "rep_lon"]),
        ("agri_to_vpd_focusFE", "VPD_trend_decade", "agri_gain_pct_points", ["focus_region"]),

        ("vpd_to_dryhot_biomeFE", "dry_minus_humid_intensity_trend", "VPD_trend_decade", ["dominant_biome"]),
        ("vpd_to_dryhot_biomeFE_spatial", "dry_minus_humid_intensity_trend", "VPD_trend_decade", ["dominant_biome", "rep_lat", "rep_lon"]),
        ("vpd_to_dryhot_focusFE", "dry_minus_humid_intensity_trend", "VPD_trend_decade", ["focus_region"]),

        ("agri_to_dryhot_biomeFE", "dry_minus_humid_intensity_trend", "agri_gain_pct_points", ["dominant_biome"]),
        ("agri_to_dryhot_biomeFE_spatial", "dry_minus_humid_intensity_trend", "agri_gain_pct_points", ["dominant_biome", "rep_lat", "rep_lon"]),
        ("agri_to_dryhot_focusFE", "dry_minus_humid_intensity_trend", "agri_gain_pct_points", ["focus_region"]),

        ("transformed_to_vpd_biomeFE", "VPD_trend_decade", "transformed_non_native_pct_2024", ["dominant_biome"]),
        ("transformed_to_vpd_biomeFE_spatial", "VPD_trend_decade", "transformed_non_native_pct_2024", ["dominant_biome", "rep_lat", "rep_lon"]),
        ("transformed_to_dryhot_biomeFE", "dry_minus_humid_intensity_trend", "transformed_non_native_pct_2024", ["dominant_biome"]),
        ("transformed_to_dryhot_biomeFE_spatial", "dry_minus_humid_intensity_trend", "transformed_non_native_pct_2024", ["dominant_biome", "rep_lat", "rep_lon"]),
    ]

    rows = []
    for model_name, response, predictor, controls in model_specs:
        if response not in df.columns or predictor not in df.columns:
            continue
        rows.extend(fit_controlled_wls(df, response, predictor, controls, model_name))

    out = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "figure_03_controlled_mechanism_regressions_1990_2024.csv")
    out_xlsx = os.path.join(OUT_DIR, "figure_03_controlled_mechanism_regressions_1990_2024.xlsx")
    out.to_csv(out_csv, index=False)
    try:
        out.to_excel(out_xlsx, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save xlsx controlled regressions: {exc}")
    save_table_dual(out, "Supplementary_Table_S12")
    print(f"[OK] Saved controlled mechanism regressions: {os.path.basename(out_csv)}")
    return out


def build_relationships(mun):
    relationships = {}
    rows = []
    
    # National: VPD trend → dry-hot shift
    d = prepare_relationship_data(mun, "VPD_trend_decade", "dry_minus_humid_intensity_trend", regions=REGION_ORDER)
    if len(d) >= 60:
        print(f"[INFO] LOWESS relationship vpd_to_dryhot: N={len(d)}")
        relationships["vpd_to_dryhot"] = lowess_with_bootstrap(d, "VPD_trend_decade", "dry_minus_humid_intensity_trend", frac=0.55)
        rows.append({"relationship": "vpd_to_dryhot", "region": "all", "n": len(d), 
                    "rho": relationships["vpd_to_dryhot"]["rho"], "p": relationships["vpd_to_dryhot"]["p"]})
    else:
        relationships["vpd_to_dryhot"] = None
    
    # Regional: cumulative transformed-land state → VPD trend.
    # This is the main panel-e diagnostic, aligned with Figure 2 v8.
    relationships["transformed_to_vpd_regions"] = {}
    relationships["agri_to_vpd_regions"] = {}  # retained as supplementary recent-change diagnostic

    for region in REGIONAL_RELATIONSHIP_REGIONS:
        dreg = prepare_relationship_data(mun, "transformed_non_native_pct_2024", "VPD_trend_decade", regions=[region])
        if len(dreg) < 60:
            print(f"[WARN] Insufficient data for transformed_to_vpd, {region}: N={len(dreg)} < 60; skipping LOWESS")
            relationships["transformed_to_vpd_regions"][region] = None
        else:
            print(f"[INFO] LOWESS relationship transformed_to_vpd, {region}: N={len(dreg)}")
            res = lowess_with_bootstrap(dreg, "transformed_non_native_pct_2024", "VPD_trend_decade", frac=0.65)
            relationships["transformed_to_vpd_regions"][region] = res
            rows.append({"relationship": "transformed_to_vpd", "region": region, "n": len(dreg),
                        "rho": res["rho"], "p": res["p"]})

        # Recent-change diagnostic kept for supplementary interpretation.
        dreg_agri = prepare_relationship_data(mun, "agri_gain_pct_points", "VPD_trend_decade", regions=[region])
        if len(dreg_agri) < 60:
            relationships["agri_to_vpd_regions"][region] = None
        else:
            res_agri = lowess_with_bootstrap(dreg_agri, "agri_gain_pct_points", "VPD_trend_decade", frac=0.65)
            relationships["agri_to_vpd_regions"][region] = res_agri
            rows.append({"relationship": "agri_to_vpd", "region": region, "n": len(dreg_agri),
                        "rho": res_agri["rho"], "p": res_agri["p"]})

    # Save summary
    summary = pd.DataFrame(rows)
    out_file = os.path.join(OUT_DIR, "figure_03_mechanism_regression_summary_1990_2024.csv")
    summary.to_csv(out_file, index=False)
    build_supplementary_table_s10(relationships)
    build_supplementary_table_s11(relationships)
    print(f"[OK] Saved regression summary: {os.path.basename(out_file)}")
    
    return relationships


# ============================================================
# Visualization helpers (publication style)
# ============================================================

def nan_gaussian_filter(arr, sigma=0.75):
    """NaN-aware Gaussian smoothing for visualization only (not trend estimation)."""
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid): return arr
    arr0 = np.where(valid, arr, 0.0)
    w = valid.astype(float)
    smooth = gaussian_filter(arr0, sigma=sigma, mode="nearest")
    weight = gaussian_filter(w, sigma=sigma, mode="nearest")
    out = np.full_like(arr, np.nan, dtype=float)
    ok = weight > 1e-6
    out[ok] = smooth[ok] / weight[ok]
    out[~valid & (weight < 0.35)] = np.nan
    return out

def print_colorbar_diagnostics(ds):
    """Print robust distribution summaries to verify colorbar ranges."""
    diagnostics = [
        ("DHW−HHW masked", "dry_minus_humid_intensity_trend_masked", "Tmax-standardized severity decade⁻¹"),
        ("VPD trend", "VPD_trend_decade", "kPa decade⁻¹"),
        ("RH trend", "RH_trend_decade", "% decade⁻¹"),
    ]
    print("\n[DIAGNOSTICS] Figure 3 map-variable percentiles")
    for label, var, unit in diagnostics:
        if var not in ds:
            print(f"  {label}: missing variable {var}")
            continue
        arr = np.asarray(ds[var].values, dtype=float)
        if "brazil_mask" in ds: arr = np.where(ds["brazil_mask"].values, arr, np.nan)
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            print(f"  {label}: no finite values")
            continue
        p = np.nanpercentile(vals, [1, 2, 5, 50, 95, 98, 99])
        print(f"  {label} ({unit}): p1={p[0]:.3g}, p2={p[1]:.3g}, p5={p[2]:.3g}, p50={p[3]:.3g}, p95={p[4]:.3g}, p98={p[5]:.3g}, p99={p[6]:.3g}")

def add_brazil_mask_to_dataset(ds, gdf):
    """Add boolean Brazil mask using municipality union (consistent with Figures 1–2)."""
    if "brazil_mask" in ds: return ds
    print("[INFO] Building Brazil raster mask for Figure 3 plotting.")
    geom = gdf.geometry.unary_union
    pgeom = prep(geom)
    lon, lat = ds["lon"].values, ds["lat"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            mask[i, j] = pgeom.contains(Point(float(lon2d[i, j]), float(lat2d[i, j])))
    ds = ds.copy()
    ds["brazil_mask"] = (("lat", "lon"), mask)
    return ds

def add_brazil_outer_boundary(ax, gdf, linewidth=0.55):
    """Plot only outer Brazil boundary (not full municipality mesh) for clean visualization."""
    boundary = gpd.GeoSeries([gdf.geometry.unary_union], crs=gdf.crs).boundary
    boundary.plot(ax=ax, color="0.18", linewidth=linewidth, zorder=10)

def add_colorbar(fig, ax, cmap, norm, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=FIG_FONT_SIZE)
    cb.ax.tick_params(labelsize=FIG_FONT_SIZE)

def add_stippling(ax, ds, pvar, size=STIPPLE_SIZE, alpha=STIPPLE_ALPHA):
    if pvar not in ds: return
    p = ds[pvar].values
    lon, lat = ds["lon"].values, ds["lat"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = np.isfinite(p) & (p <= P_STIPPLE)
    if "brazil_mask" in ds: mask = mask & ds["brazil_mask"].values
    subsample = np.zeros_like(mask, dtype=bool)
    subsample[::2, ::2] = True
    mask = mask & subsample
    ax.scatter(lon2d[mask], lat2d[mask], s=size, c="black", alpha=alpha, linewidths=0, zorder=5)

def plot_trend_map(ax, ds, var, title, cmap, norm, cbar_label, raster_alpha=1.0, show_stippling=True,
                  stipple_size=STIPPLE_SIZE, stipple_alpha=STIPPLE_ALPHA, pvar=None, smooth=True, clip_to_brazil=True):
    arr = np.asarray(ds[var].values, dtype=float)
    if clip_to_brazil and "brazil_mask" in ds: arr = np.where(ds["brazil_mask"].values, arr, np.nan)
    if smooth: arr = nan_gaussian_filter(arr, sigma=MAP_SMOOTH_SIGMA)
    lon, lat = ds["lon"].values, ds["lat"].values
    im = ax.pcolormesh(lon, lat, arr, cmap=cmap, norm=norm, shading="auto", alpha=raster_alpha, zorder=1)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=FIG_FONT_SIZE, fontweight="bold", pad=5)
    ax.set_xlabel("Longitude", fontsize=FIG_FONT_SIZE)
    ax.set_ylabel("Latitude", fontsize=FIG_FONT_SIZE)
    ax.tick_params(labelsize=FIG_FONT_SIZE)
    ax.grid(alpha=0.12, linewidth=0.30)
    if show_stippling:
        if pvar is None: pvar = var.replace("_trend_decade", "_pvalue")
        add_stippling(ax, ds, pvar, size=stipple_size, alpha=stipple_alpha)
    add_colorbar(ax.figure, ax, cmap, norm, cbar_label)
    return im

def plot_lowess_panel(ax, res, title, xlabel, ylabel, color):
    if res is None:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center", fontsize=FIG_FONT_SIZE)
        ax.set_title(title, fontsize=FIG_FONT_SIZE, fontweight="bold")
        return
    ax.plot(res["x"], res["yhat"], color=color, linewidth=2.3)
    if np.isfinite(res["lo"]).any() and np.isfinite(res["hi"]).any():
        ax.fill_between(res["x"], res["lo"], res["hi"], color=color, alpha=0.20)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="0.35", linestyle=":", linewidth=0.7)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_title(title, fontsize=FIG_FONT_SIZE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FIG_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FIG_FONT_SIZE)
    ax.tick_params(labelsize=FIG_FONT_SIZE)
    ymin, ymax = ax.get_ylim()
    y_rug = ymin + 0.035 * (ymax - ymin)
    x_obs = np.asarray(res["x_obs"], dtype=float)[np.isfinite(res["x_obs"])]
    if len(x_obs) > 2500:
        rng = np.random.default_rng(RANDOM_SEED)
        x_obs = rng.choice(x_obs, size=2500, replace=False)
    ax.plot(x_obs, np.full_like(x_obs, y_rug), "|", color=color, alpha=0.16, markersize=4, markeredgewidth=0.45)
    txt = f"N={res['n']}\nρs,w={res['rho']:.2f}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=FIG_FONT_SIZE, bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.25))

def plot_regional_lowess_small_multiples(fig, outer_spec, region_results, region_order):
    """
    publication-style regional diagnostic panel.

    Uses a 2 × 3 layout but plots only regions with a robust and interpretable
    transformed-land gradient. The Semi-arid region and Pantanal are retained in
    tables/diagnostics but not displayed in the main panel when the smoother is
    visually uninformative or sample size is too small.
    """
    n_rows, n_cols = 2, 3
    sub = GridSpecFromSubplotSpec(
        n_rows, n_cols,
        subplot_spec=outer_spec,
        wspace=0.40,
        hspace=0.68,
    )

    colors = {
        "Amazon": "#2C7BB6",
        "Cerrado": "#D95F02",
        "MATOPIBA": "#D95F02",
        "Semi-arid Northeast": "#B35806",
        "Pantanal": "#1B9E77",
        "Atlantic Forest": "#7570B3",
        "Pampa": "#666666",
        "Urban Southeast": "#C51B7D",
    }

    axes = []
    for idx, region in enumerate(region_order):
        ax = fig.add_subplot(sub[idx // n_cols, idx % n_cols])
        axes.append(ax)
        res = region_results.get(region)
        color = colors.get(region, "#444444")

        if res is None or res.get("n", 0) < 60:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FIG_FONT_SIZE, color="0.45")
            ax.set_title(region, fontsize=FIG_FONT_SIZE, fontweight="bold")
            ax.grid(alpha=0.18, linewidth=0.4)
            continue

        ax.plot(res["x"], res["yhat"], color=color, linewidth=2.1)
        if np.isfinite(res["lo"]).any() and np.isfinite(res["hi"]).any():
            ax.fill_between(res["x"], res["lo"], res["hi"], color=color, alpha=0.22)

        ax.axhline(0, color="0.40", linestyle="--", linewidth=0.75)
        ax.grid(alpha=0.20, linewidth=0.45)

        # Use robust local x limits to avoid one region compressing the others.
        x = np.asarray(res["x_obs"], dtype=float)
        x = x[np.isfinite(x)]
        if x.size > 10:
            x0, x1 = np.nanpercentile(x, [2, 98])
            pad = 0.05 * max(1e-6, x1 - x0)
            ax.set_xlim(max(0, x0 - pad), min(100, x1 + pad))
        ax.set_ylim(VPD_VMIN, VPD_VMAX)

        ax.set_title(region, fontsize=FIG_FONT_SIZE, fontweight="bold", pad=6)
        ax.tick_params(labelsize=FIG_FONT_SIZE)

        ymin, ymax = ax.get_ylim()
        y_rug = ymin + 0.025 * (ymax - ymin)
        x_obs = np.asarray(res["x_obs"], dtype=float)[np.isfinite(res["x_obs"])]
        if len(x_obs) > 450:
            rng = np.random.default_rng(RANDOM_SEED)
            x_obs = rng.choice(x_obs, size=450, replace=False)
        ax.plot(x_obs, np.full_like(x_obs, y_rug), "|", color=color, alpha=0.28,
                markersize=3.5, markeredgewidth=0.45)

        txt = f"N={res['n']}\nρ={res['rho']:.2f}"
        ax.text(0.05, 0.93, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=FIG_FONT_SIZE, bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.25, pad=1.8))

    # Hide unused cells if fewer than 6 regions are shown.
    n_total = n_rows * n_cols
    for idx in range(len(region_order), n_total):
        ax = fig.add_subplot(sub[idx // n_cols, idx % n_cols])
        ax.axis("off")

    # Axis labels only where needed to reduce clutter.
    for idx, ax in enumerate(axes):
        if idx % n_cols == 0:
            ax.set_ylabel("VPD trend\n(kPa decade$^{-1}$)", fontsize=FIG_FONT_SIZE)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Transformed/non-native land\n(% municipal area, 2024)", fontsize=FIG_FONT_SIZE)

    return axes


def plot_mechanism_schematic(ax):
    """Minimal publication-style vertical pathway schematic."""
    ax.axis("off")
    y_positions = [0.82, 0.62, 0.42, 0.22]
    labels = ["Land transformation", "Reduced evapotranspiration", "Atmospheric drying\n(VPD ↑ ; RH ↓)", "Dry-hot transition\n(corrected DHW−HHW ↑)"]
    for y, label in zip(y_positions, labels):
        ax.text(0.50, y, label, transform=ax.transAxes, ha="center", va="center", fontsize=FIG_FONT_SIZE, fontweight="bold", color="0.08", linespacing=1.18)
    for y0, y1 in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate("", xy=(0.50, y1 + 0.07), xytext=(0.50, y0 - 0.07), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", linewidth=1.9, color="0.10"))
    ax.text(0.50, 0.055, "transformed landscapes weaken evaporative cooling and increase atmospheric moisture demand", transform=ax.transAxes, ha="center", va="center", fontsize=FIG_FONT_SIZE, color="0.25", wrap=True)

def plot_figure(ds_trends, gdf, relationships):
    print("[INFO] Plotting publication-ready Figure 3 (publication style)")
    ds_trends = add_brazil_mask_to_dataset(ds_trends, gdf)
    print_colorbar_diagnostics(ds_trends)
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, left=0.040, right=0.985, top=0.905, bottom=0.090, wspace=0.35, hspace=0.42)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e_container = fig.add_subplot(gs[1, 1:3])
    
    # Panel (a): masked dry-hot shift (significant positive shifts only)
    plot_trend_map(ax_a, ds_trends, "dry_minus_humid_intensity_trend_masked", 
                  "(a) Preferential dry-hot amplification\nDHW − HHW", plt.cm.Reds, 
                  Normalize(vmin=DRYHOT_VMIN, vmax=DRYHOT_VMAX), "Tmax-standardized severity decade$^{-1}$",
                  show_stippling=False, smooth=True)
    ax_a.text(0.02, 0.02, f"shown where DHW−HHW > {DRYHOT_MASK_MIN_SHIFT:g}\npositive DHW; p ≤ {DRYHOT_MASK_P:g}; display filter",
             transform=ax_a.transAxes, fontsize=FIG_FONT_SIZE, ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.25, pad=1.5))
    add_brazil_outer_boundary(ax_a, gdf)
    
    # Panel (b): VPD trend (atmospheric drying)
    plot_trend_map(ax_b, ds_trends, "VPD_trend_decade", "(b) Atmospheric drying\nVPD trend", plt.cm.RdYlBu_r,
                  TwoSlopeNorm(vmin=VPD_VMIN, vcenter=VPD_VCENTER, vmax=VPD_VMAX), "kPa decade$^{-1}$",
                  stipple_size=STIPPLE_SIZE_FINE, stipple_alpha=STIPPLE_ALPHA_FINE)
    add_brazil_outer_boundary(ax_b, gdf)
    
    # Panel (c): RH trend (moisture availability)
    plot_trend_map(ax_c, ds_trends, "RH_trend_decade", "(c) Moisture availability\nrelative-humidity trend", plt.cm.RdBu,
                  TwoSlopeNorm(vmin=RH_VMIN, vcenter=RH_VCENTER, vmax=RH_VMAX), "% decade$^{-1}$",
                  raster_alpha=0.82, stipple_size=STIPPLE_SIZE_FINE, stipple_alpha=STIPPLE_ALPHA_FINE)
    add_brazil_outer_boundary(ax_c, gdf)
    
    # Panel (d): municipality-level association VPD → dry-hot shift
    plot_lowess_panel(ax_d, relationships.get("vpd_to_dryhot"), "(d) VPD trend and DHW−HHW contrast",
                     "VPD trend (kPa decade$^{-1}$)", "DHW − HHW intensity trend\n(Tmax-standardized severity decade$^{-1}$)", "#B2182B")
    
    # Panel (e): regional transformed-land → VPD diagnostics.
    # Use the same VPD range as panel (b) to preserve comparability with the original figure.
    # evidence, including Pantanal and other climatically important regions.
    ax_e_container.axis("off")
    ax_e_container.set_title("")
    fig.text(
        0.675, 0.505,
        "(e) Regional association between transformed land and VPD trend",
        fontsize=FIG_FONT_SIZE,
        fontweight="bold",
        ha="center",
        va="center",
    )
    plot_regional_lowess_small_multiples(
        fig,
        gs[1, 1:3],
        relationships.get("transformed_to_vpd_regions", {}),
        REGIONAL_RELATIONSHIP_REGIONS,
    )

    # Title and footer
    fig.suptitle("Atmospheric drying and preferential dry-hot heatwave amplification across Brazil (1990–2024)", fontsize=FIG_FONT_SIZE, fontweight="bold", y=0.975)
    # Explanatory caption text is intentionally not printed inside the figure.
    # Save outputs
    # Save outputs (vector PDF + high-resolution JPEG)
    out_base = os.path.join(OUT_DIR, "figure_03_atmospheric_drying")
    
    # PDF: vector format for submission
    fig.savefig(out_base + ".pdf", dpi=450, bbox_inches="tight", facecolor="white",
                metadata={"Creator": "figure_03_atmospheric_drying.py"})
    
    # JPEG: high-resolution preview
    fig.savefig(out_base + ".jpeg", dpi=450, bbox_inches="tight", facecolor="white", format="jpeg")
    
    plt.close(fig)
    print(f"[OK] Saved: {os.path.basename(out_base)}.pdf (vector, publication-ready)")
    print(f"[OK] Saved: {os.path.basename(out_base)}.jpeg (450 dpi preview)")


# ============================================================
# Main execution workflow
# ============================================================

def main():
    args = parse_args()
    configure_paths(args)

    print("=" * 86)
    print("FIGURE 03 — ATMOSPHERIC DRYING AND PREFERENTIAL DRY-HOT AMPLIFICATION")
    print("=" * 86)
    print(f"Figure 01 root : {FIGURE1_ROOT}")
    print(f"Figure 02 table: {FIG2_TABLE}")
    print(f"MapBiomas      : {COVERAGE_XLSX}")
    print(f"Output         : {OUT_DIR}")
    print("DHW-HHW metric : common Tmax-only standardized severity")
    print(f"Trend inference: {TREND_SIGNIFICANCE_VERSION}")
    print("VPD units      : kPa")
    print("=" * 86)
    with safe_open_dataset(
        FIG1_TRENDS_NC,
        decode_times=False,
        chunks=None,
    ) as f1_check:
        validate_figure1_compatibility(
            standardize_lat_lon(f1_check)
        )

    f2_check = read_corrected_figure2_table()
    validate_figure2_compatibility(f2_check)

    ds_ann = build_annual_mechanism_dataset(
        overwrite=args.recompute
    )

    ds_trends = calc_trends(
        ds_ann,
        overwrite=args.recompute,
    )

    ds_trends = add_dryhot_shift_to_mechanistic_trends(
        ds_trends,
        overwrite=args.recompute,
    )

    mun, gdf = build_municipality_mechanism_table(ds_trends)
    build_supplementary_table_s9(mun)

    relationships = build_relationships(mun)
    run_controlled_mechanism_regressions(mun)

    plot_figure(ds_trends, gdf, relationships)

    meta_file = os.path.join(OUT_DIR, "software_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(get_software_metadata(), handle, indent=2)

    ds_ann.close()
    ds_trends.close()

    print(f"\n[DONE] Figure 03 workflow completed | {pd.Timestamp.now().isoformat()}")
    print(f"[INFO] Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

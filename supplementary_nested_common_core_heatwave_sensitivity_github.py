#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary nested/common-core heatwave sensitivity analysis, 1990–2024.

Purpose
-------
Evaluate whether preferential dry-hot amplification is preserved when heatwave
days are restricted to a common Tmax-defined core and partitioned into mutually
exclusive thermodynamic states.

The primary manuscript comparison uses the common-scale Figure 01 field:

    DHW_minus_HHW_TmaxOnly_intensity_trend_decade

Primary regime-specific DHW and HHW intensity trends are not subtracted because
their intensity definitions are not directly commensurate.

Sensitivity framework
---------------------
1. Common heatwave core
       Tmax >= local day-of-year P95
   for at least 3 consecutive days.

2. Every valid day inside a persistent common-core event is assigned to one
   mutually exclusive state:
       DRY     : VPD >= P75 and Twbmax < P95
       HUMID   : Twbmax >= P95 and VPD < P75
       MIXED   : VPD >= P75 and Twbmax >= P95
       NEUTRAL : VPD < P75 and Twbmax < P95

   Days with missing Tmax, VPD, Twbmax, mapped thresholds, or Tmax
   climatological mean/standard deviation are not assigned to a state.

3. Annual metrics
       duration  = number of state days within persistent common-core events
       intensity = sum[z+(Tmax)] over those state days
       frequency = number of contiguous state episodes within common-core events

   CORE frequency counts parent common-core events. State frequency counts
   contiguous state episodes and therefore does not treat the same parent event
   as an independent event in multiple thermodynamic categories.

4. Trends
       Theil–Sen median slope per decade.

5. Significance
       Original two-sided Mann–Kendall unless detrended lag-1 rank
       autocorrelation is significant; Hamed–Rao variance-corrected
       Mann–Kendall with lag 1 otherwise.

6. Main nested contrast
       NESTED_DRY_intensity_trend_decade
       - NESTED_HUMID_intensity_trend_decade

   Both components use the same Tmax-standardized severity metric. The main
   contrast is a difference of trend slopes and has no standalone p-value.

A separate trend of the annual DRY-minus-HUMID intensity difference is retained
as a diagnostic and has its own Mann–Kendall p-value. It is not substituted for
the main difference-of-slopes contrast.

The workflow reuses the Figure 01 daily cache and threshold climatology without
recomputing percentile thresholds or spatially interpolating them. Spatial
summaries use cos(latitude) weights. Grid-cell comparisons use weighted
Spearman correlation with 999 two-sided permutations.

Usage
-----
python supplementary_nested_common_core_heatwave_sensitivity.py \
    --figure1-output-dir /path/to/figure_01_outputs \
    --brazil-shapefile /path/to/brazil_boundary.shp \
    --biomes-shapefile /path/to/brazil_biomes.shp \
    --south-america-shapefile /path/to/south_america_countries.shp \
    --output-dir ./outputs/nested_common_core

The biome and South America boundary files are optional.
Use ``--recompute`` to ignore existing row caches.
"""

import os
import argparse
import warnings
from pathlib import Path


import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from scipy import stats
from scipy.stats import theilslopes, norm
from shapely.geometry import Point

try:
    import pymannkendall as mk
    _HAS_PYMK = True
except Exception:
    _HAS_PYMK = False


# ============================================================
# Runtime paths
# ============================================================

FIGURE1_ROOT = None
DAILY_DIR = None
THRESHOLD_FILE = None
FIG1_TRENDS_NC = None
BRAZIL_SHP = None
BIOMES_SHP = None
SA_COUNTRIES_SHP = None
OUT_DIR = None
ROW_CACHE_DIR = None

START_YEAR = 1990
END_YEAR = 2024
MIN_LEN = 3
MIN_VALID_DAYS_PER_YEAR = 300
MIN_VALID_YEARS_FOR_TREND = 20
P_THRESHOLD = 0.05
AUTOCORR_ALPHA = 0.05
N_PERMUTATIONS = 999
RANDOM_SEED = 42

CORE_TMAX_P = 95.0
DRY_VPD_P = 75.0
HUMID_TWB_P = 95.0

Z_CLIP_MIN = -5.0
Z_CLIP_MAX = 5.0

LON_MIN, LON_MAX = -75.0, -32.0
LAT_MIN, LAT_MAX = -35.0, 6.0

SCRIPT_VERSION = "1.0.0"
TREND_SIGNIFICANCE_VERSION = "Theil-Sen + conditional Hamed-Rao MK lag1"

STATES = ["CORE", "DRY", "HUMID", "MIXED", "NEUTRAL"]
METRICS = ["frequency", "duration", "intensity"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# ============================================================
# CLI / path configuration
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Nested/common-core heatwave sensitivity analysis using "
            "Figure 01 daily and threshold products."
        )
    )
    p.add_argument(
        "--figure1-output-dir",
        required=True,
        help="Root directory produced by figure_01_heatwave_trends.py.",
    )
    p.add_argument(
        "--brazil-shapefile",
        required=True,
        help="Brazil boundary or state shapefile.",
    )
    p.add_argument(
        "--biomes-shapefile",
        default=None,
        help="Optional Brazilian biome shapefile.",
    )
    p.add_argument(
        "--south-america-shapefile",
        default=None,
        help="Optional South America country-boundary shapefile.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore row caches and recompute all nested metrics.",
    )
    return p.parse_args()

def configure_paths(args):
    global FIGURE1_ROOT, DAILY_DIR, THRESHOLD_FILE, FIG1_TRENDS_NC
    global BRAZIL_SHP, BIOMES_SHP, SA_COUNTRIES_SHP, OUT_DIR, ROW_CACHE_DIR

    FIGURE1_ROOT = Path(args.figure1_output_dir).expanduser().resolve()
    DAILY_DIR = FIGURE1_ROOT / "cache" / "era5_daily"
    THRESHOLD_FILE = (
        FIGURE1_ROOT / "cache" / "figure_01_heatwave_thresholds_1991_2020.nc"
    )
    FIG1_TRENDS_NC = (
        FIGURE1_ROOT / "data" / "figure_01_heatwave_trends_1990_2024.nc"
    )

    BRAZIL_SHP = Path(args.brazil_shapefile).expanduser().resolve()
    BIOMES_SHP = (
        Path(args.biomes_shapefile).expanduser().resolve()
        if args.biomes_shapefile else None
    )
    SA_COUNTRIES_SHP = (
        Path(args.south_america_shapefile).expanduser().resolve()
        if args.south_america_shapefile else None
    )
    OUT_DIR = Path(args.output_dir).expanduser().resolve()
    ROW_CACHE_DIR = OUT_DIR / "row_cache"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        (DAILY_DIR, "Figure 01 daily cache"),
        (THRESHOLD_FILE, "Figure 01 threshold file"),
        (FIG1_TRENDS_NC, "Figure 01 trend file"),
        (BRAZIL_SHP, "Brazil shapefile"),
    ]
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


# ============================================================
# NetCDF helpers
# ============================================================

def _available_netcdf_engines():
    engines = xr.backends.list_engines()
    return [e for e in ("h5netcdf", "netcdf4", "scipy") if e in engines]


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


def safe_to_netcdf(ds, path, *, encoding=None):
    errors = []
    engines = _available_netcdf_engines()

    for engine in ("h5netcdf", "netcdf4"):
        if engine not in engines:
            continue
        try:
            ds.to_netcdf(path, engine=engine, encoding=encoding)
            return
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")

    if "scipy" in engines:
        enc = None
        if encoding:
            enc = {
                var: {
                    k: v for k, v in opts.items()
                    if k in {"dtype", "_FillValue", "scale_factor", "add_offset"}
                }
                for var, opts in encoding.items()
            }
        try:
            ds.to_netcdf(path, engine="scipy", encoding=enc)
            return
        except Exception as exc:
            errors.append(f"scipy: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Could not write NetCDF file: {path}\n" + "\n".join(errors)
    )


# ============================================================
# Dataset/grid utilities
# ============================================================

def standardize_lat_lon(ds):
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)

    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(
            lon=((ds.lon + 180) % 360) - 180
        ).sortby("lon")

    if (
        "lat" in ds.coords
        and ds.lat.size > 1
        and float(ds.lat[0]) < float(ds.lat[-1])
    ):
        ds = ds.sortby("lat", ascending=False)

    return ds


def remove_feb29(ds):
    mask = ~((ds.time.dt.month == 2) & (ds.time.dt.day == 29))
    return ds.sel(time=mask)


def noleap_doy(times):
    dates = pd.to_datetime(times)
    out = np.empty(len(dates), dtype=np.int16)
    for i, d in enumerate(dates):
        value = (d - pd.Timestamp(f"{d.year}-01-01")).days + 1
        if d.is_leap_year and d.month > 2:
            value -= 1
        out[i] = value
    return out


def list_daily_files():
    files = []
    for year in range(START_YEAR, END_YEAR + 1):
        f = DAILY_DIR / f"ERA5_daily_Brazil_{year}.nc"
        if f.exists():
            files.append((year, f))
        else:
            print(f"[WARN] Missing daily file: {f}")
    if len(files) < MIN_VALID_YEARS_FOR_TREND:
        raise RuntimeError(
            f"Only {len(files)} Figure 1 daily files found; "
            f"need >= {MIN_VALID_YEARS_FOR_TREND}."
        )
    return files


def open_daily_template(daily_files):
    _, path = daily_files[0]
    ds = standardize_lat_lon(
        safe_open_dataset(path, decode_times=True, chunks=None)
    )
    ds = remove_feb29(ds)
    ds = ds.sel(
        lon=slice(LON_MIN, LON_MAX),
        lat=slice(LAT_MAX, LAT_MIN),
    )
    return ds


def make_brazil_mask(lat, lon):
    uf = gpd.read_file(BRAZIL_SHP).to_crs(epsg=4326)
    geom = uf.dissolve().geometry.iloc[0]

    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for iy, la in enumerate(lat):
        pts = [Point(float(lo), float(la)) for lo in lon]
        mask[iy, :] = [
            geom.contains(point) or geom.touches(point)
            for point in pts
        ]

    da = xr.DataArray(
        mask,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )
    return da, uf


def grid_area_weights(lat, lon):
    arr = np.repeat(
        np.cos(np.deg2rad(np.asarray(lat, float)))[:, None],
        len(lon),
        axis=1,
    )
    return xr.DataArray(
        arr,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )


def load_optional_boundaries():
    biomes = None
    countries = None

    if BIOMES_SHP is not None and BIOMES_SHP.exists():
        try:
            biomes = gpd.read_file(BIOMES_SHP).to_crs(epsg=4326)
        except Exception as exc:
            print(f"[WARN] Could not load biomes: {exc}")

    if SA_COUNTRIES_SHP is not None and SA_COUNTRIES_SHP.exists():
        try:
            countries = gpd.read_file(SA_COUNTRIES_SHP).to_crs(epsg=4326)
            if "CONTINENT" in countries.columns:
                countries = countries[
                    countries["CONTINENT"] == "South America"
                ].copy()
        except Exception as exc:
            print(f"[WARN] Could not load South America boundaries: {exc}")

    return biomes, countries


# ============================================================
# Exact Figure 1 threshold reuse
# ============================================================

def load_thresholds_exact(template):
    ds = standardize_lat_lon(
        safe_open_dataset(THRESHOLD_FILE, decode_times=False, chunks=None)
    )

    required = [
        "DHW_Tmax_P95",
        "DHW_VPDmean_P75",
        "HHW_Twbmax_P95",
        "CLIM_Tmax_mean",
        "CLIM_Tmax_std",
    ]
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        ds.close()
        raise RuntimeError(
            "Figure 01 threshold file is missing: "
            + ", ".join(missing)
        )

    # Require the daily domain to be an exact coordinate subset of Figure 1
    # thresholds. This is safer than spatial interpolation for percentile fields.
    try:
        aligned = ds[required].sel(
            lat=xr.DataArray(template.lat.values, dims="lat"),
            lon=xr.DataArray(template.lon.values, dims="lon"),
        ).load()
    except Exception as exc:
        ds.close()
        raise RuntimeError(
            "Figure 1 thresholds cannot be selected on the exact daily ERA5 grid. "
            "Do not interpolate percentile thresholds. "
            f"Details: {exc}"
        )

    if not np.allclose(aligned.lat.values, template.lat.values):
        ds.close()
        raise RuntimeError("Latitude coordinates are not exactly aligned.")
    if not np.allclose(aligned.lon.values, template.lon.values):
        ds.close()
        raise RuntimeError("Longitude coordinates are not exactly aligned.")

    ds.close()

    aligned.attrs.update({
        "source": str(THRESHOLD_FILE),
        "alignment": (
            "Exact Figure 01 threshold arrays selected on the daily-cache grid; "
            "no threshold recomputation or spatial interpolation."
        ),
    })
    print(f"[INFO] Reusing Figure 01 thresholds: {THRESHOLD_FILE}")
    return aligned


def threshold_row(ds_thr, source_var, doys, iy):
    da = ds_thr[source_var]
    mapped = da.sel(
        doy_noleap=xr.DataArray(doys, dims="time")
    )
    return (
        mapped.isel(lat=iy)
        .transpose("time", "lon")
        .values
        .astype(np.float32, copy=False)
    )


# ============================================================
# Row-wise daily data loading
# ============================================================

def read_row_for_year(path, iy):
    ds = standardize_lat_lon(
        safe_open_dataset(path, decode_times=True, chunks=None)
    )
    ds = remove_feb29(ds)
    ds = ds.sel(
        lon=slice(LON_MIN, LON_MAX),
        lat=slice(LAT_MAX, LAT_MIN),
    )

    required = ["Tmax", "VPDmean", "Twbmax"]
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        ds.close()
        raise RuntimeError(
            f"Daily Figure 1 cache {path} is missing {missing}"
        )

    arrays = {
        v: (
            ds[v].isel(lat=iy)
            .transpose("time", "lon")
            .values
            .astype(np.float32, copy=False)
        )
        for v in required
    }
    times = ds.time.values.copy()
    lon = ds.lon.values.copy()
    lat_value = float(ds.lat.values[iy])
    ds.close()
    return arrays, times, lon, lat_value


def concatenate_row(daily_files, iy):
    by_var = {"Tmax": [], "VPDmean": [], "Twbmax": []}
    times = []
    lon_ref = None
    lat_ref = None

    for _, path in daily_files:
        arrays, t, lon, lat_value = read_row_for_year(path, iy)
        if lon_ref is None:
            lon_ref = lon
            lat_ref = lat_value
        else:
            if not np.allclose(lon_ref, lon):
                raise RuntimeError(f"Longitude grid changed in {path}")
            if not np.isclose(lat_ref, lat_value):
                raise RuntimeError(f"Latitude row changed in {path}")

        for v in by_var:
            by_var[v].append(arrays[v])
        times.append(t)

    return (
        {v: np.concatenate(parts, axis=0) for v, parts in by_var.items()},
        np.concatenate(times),
    )


# ============================================================
# Event/state metrics
# ============================================================

def apply_min_duration(mask, min_len=MIN_LEN):
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    i = 0
    n = len(mask)

    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if (j - i + 1) >= min_len:
            out[i:j + 1] = True
        i = j + 1

    return out


def count_state_episodes(state_mask, core_mask, years):
    """Count contiguous state episodes without allowing runs across core gaps."""
    out = {y: 0.0 for y in range(START_YEAR, END_YEAR + 1)}
    state_mask = np.asarray(state_mask, bool)
    core_mask = np.asarray(core_mask, bool)
    episode_mask = state_mask & core_mask

    i = 0
    n = len(episode_mask)
    while i < n:
        if not episode_mask[i]:
            i += 1
            continue

        j = i
        while (
            j + 1 < n
            and episode_mask[j + 1]
            and core_mask[j + 1]
        ):
            j += 1

        yy = int(years[i])
        if START_YEAR <= yy <= END_YEAR:
            out[yy] += 1.0

        i = j + 1

    return out


def count_core_events(core_mask, years):
    out = {y: 0.0 for y in range(START_YEAR, END_YEAR + 1)}
    i = 0
    n = len(core_mask)

    while i < n:
        if not core_mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and core_mask[j + 1]:
            j += 1
        yy = int(years[i])
        if START_YEAR <= yy <= END_YEAR:
            out[yy] += 1.0
        i = j + 1

    return out


def annual_state_metrics(core_mask, state_masks, intensity, years):
    year_axis = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int16)
    output = {}

    for state in STATES:
        if state == "CORE":
            smask = core_mask
            frequency = count_core_events(core_mask, years)
        else:
            smask = state_masks[state]
            frequency = count_state_episodes(smask, core_mask, years)

        duration = {int(y): 0.0 for y in year_axis}
        severity = {int(y): 0.0 for y in year_axis}

        for k in np.where(smask)[0]:
            yy = int(years[k])
            if START_YEAR <= yy <= END_YEAR:
                duration[yy] += 1.0
                value = intensity[k]
                if np.isfinite(value) and value > 0:
                    severity[yy] += float(value)

        output[state] = {
            "frequency": frequency,
            "duration": duration,
            "intensity": severity,
        }

    return output


# ============================================================
# Trend methodology aligned with Figure 1
# ============================================================

def _mk_score_and_variance(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < MIN_VALID_YEARS_FOR_TREND:
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
    z = (
        0.0
        if s == 0
        else (s - np.sign(s)) / np.sqrt(var_s)
    )
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def mann_kendall_pvalue(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < MIN_VALID_YEARS_FOR_TREND:
        return np.nan

    if _HAS_PYMK:
        try:
            return float(mk.original_test(y).p)
        except Exception:
            pass

    s, var_s = _mk_score_and_variance(y)
    return _mk_p_from_s_var(s, var_s)


def detrended_lag1_autocorrelation(y, years):
    y = np.asarray(y, float)
    years = np.asarray(years, float)
    ok = np.isfinite(y) & np.isfinite(years)
    y, years = y[ok], years[ok]
    n = len(y)

    if n < MIN_VALID_YEARS_FOR_TREND or n < 4:
        return np.nan, np.nan, False

    critical = float(
        norm.ppf(1.0 - AUTOCORR_ALPHA / 2.0) / np.sqrt(n)
    )

    if np.nanstd(y) <= 1e-12:
        return 0.0, critical, False

    sen = theilslopes(y, years, alpha=0.05)
    residual = y - (
        float(sen[1]) + float(sen[0]) * years
    )
    ranks = pd.Series(residual).rank(
        method="average"
    ).to_numpy(float)

    if (
        np.std(ranks[:-1]) <= 1e-12
        or np.std(ranks[1:]) <= 1e-12
    ):
        r1 = 0.0
    else:
        r1 = float(
            np.corrcoef(ranks[:-1], ranks[1:])[0, 1]
        )

    return r1, critical, bool(
        np.isfinite(r1) and abs(r1) > critical
    )


def hamed_rao_pvalue_lag1(y, years):
    y = np.asarray(y, float)
    years = np.asarray(years, float)
    ok = np.isfinite(y) & np.isfinite(years)
    y, years = y[ok], years[ok]

    if len(y) < MIN_VALID_YEARS_FOR_TREND:
        return np.nan

    if _HAS_PYMK:
        try:
            return float(
                mk.hamed_rao_modification_test(
                    y,
                    alpha=AUTOCORR_ALPHA,
                    lag=1,
                ).p
            )
        except Exception:
            pass

    s, var_s = _mk_score_and_variance(y)
    if not np.isfinite(var_s) or var_s <= 0:
        return np.nan

    r1, critical, _ = detrended_lag1_autocorrelation(
        y, years
    )
    if not np.isfinite(r1) or abs(r1) <= critical:
        return _mk_p_from_s_var(s, var_s)

    # Conservative lag-1 fallback if pymannkendall is unavailable.
    n = len(y)
    correction = max(
        1.0 + 2.0 * (n - 3) / n * abs(r1),
        1.0,
    )
    return _mk_p_from_s_var(
        s,
        var_s * correction,
    )


def robust_trend(y, years):
    y = np.asarray(y, float)
    years = np.asarray(years, float)
    ok = np.isfinite(y) & np.isfinite(years)
    yv, xv = y[ok], years[ok]

    if len(yv) < MIN_VALID_YEARS_FOR_TREND:
        return (
            np.nan, np.nan,
            np.nan, np.nan, np.nan,
        )

    slope = float(
        theilslopes(yv, xv, alpha=0.05)[0] * 10.0
    )
    original_p = mann_kendall_pvalue(yv)
    r1, critical, lag1_sig = (
        detrended_lag1_autocorrelation(yv, xv)
    )

    if lag1_sig:
        selected_p = hamed_rao_pvalue_lag1(
            yv, xv
        )
        method = 1.0
    else:
        selected_p = original_p
        method = 0.0

    return slope, selected_p, r1, critical, method


# ============================================================
# Restartable row computation
# ============================================================

def row_cache_path(iy):
    return ROW_CACHE_DIR / f"nested_row_{iy:03d}.npz"


def expected_row_keys():
    keys = []
    for state in STATES:
        for metric in METRICS:
            keys.append(f"NESTED_{state}_{metric}_annual")
            keys.append(f"NESTED_{state}_{metric}_trend")
            keys.append(f"NESTED_{state}_{metric}_pvalue")
            keys.append(f"NESTED_{state}_{metric}_lag1")
            keys.append(f"NESTED_{state}_{metric}_lag1critical")
            keys.append(f"NESTED_{state}_{metric}_mkmethod")

    for metric in ["intensity", "duration", "frequency"]:
        keys.extend([
            f"NESTED_DRY_minus_HUMID_{metric}_contrast",
        ])

    keys.extend([
        "NESTED_DRY_minus_HUMID_annual_difference",
        "NESTED_DRY_minus_HUMID_annual_difference_trend",
        "NESTED_DRY_minus_HUMID_annual_difference_pvalue",
        "NESTED_DRY_minus_HUMID_annual_difference_lag1",
        "NESTED_DRY_minus_HUMID_annual_difference_lag1critical",
        "NESTED_DRY_minus_HUMID_annual_difference_mkmethod",
    ])
    return keys


def load_row_cache(iy, nyr, nx, recompute=False):
    path = row_cache_path(iy)
    if recompute or not path.exists():
        return None

    try:
        z = np.load(path, allow_pickle=False)
        if z["script_version"].item() != SCRIPT_VERSION:
            return None
        if int(z["nyr"]) != nyr or int(z["nx"]) != nx:
            return None
        missing = [
            k for k in expected_row_keys()
            if k not in z.files
        ]
        if missing:
            return None
        print(f"[SKIP] Reusing row cache {iy + 1}")
        return {k: z[k] for k in z.files}
    except Exception as exc:
        print(
            f"[WARN] Could not reuse row cache {path}: {exc}"
        )
        return None


def save_row_cache(iy, row):
    path = row_cache_path(iy)
    tmp = str(path) + ".tmp.npz"
    payload = dict(row)
    payload["script_version"] = np.asarray(SCRIPT_VERSION)
    payload["nyr"] = np.asarray(
        next(
            arr.shape[0]
            for key, arr in row.items()
            if key.endswith("_annual")
        )
    )
    payload["nx"] = np.asarray(
        next(
            arr.shape[-1]
            for key, arr in row.items()
            if isinstance(arr, np.ndarray)
        )
    )
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def compute_row(iy, daily_files, ds_thr, mask_row, recompute=False):
    year_axis = np.arange(
        START_YEAR, END_YEAR + 1, dtype=np.int16
    )
    nyr = len(year_axis)
    nx = len(mask_row)

    cached = load_row_cache(
        iy, nyr, nx, recompute=recompute
    )
    if cached is not None:
        return cached

    print(f"[INFO] Computing latitude row {iy + 1}")

    row_data, times = concatenate_row(
        daily_files, iy
    )
    tmax = row_data["Tmax"]
    vpd = row_data["VPDmean"]
    twb = row_data["Twbmax"]

    years = np.asarray(
        pd.to_datetime(times).year,
        dtype=np.int16,
    )
    doys = noleap_doy(times)
    year_indices = {
        int(y): np.where(years == int(y))[0]
        for y in year_axis
    }

    core_thr = threshold_row(
        ds_thr, "DHW_Tmax_P95", doys, iy
    )
    vpd_thr = threshold_row(
        ds_thr, "DHW_VPDmean_P75", doys, iy
    )
    twb_thr = threshold_row(
        ds_thr, "HHW_Twbmax_P95", doys, iy
    )
    clim_mean = threshold_row(
        ds_thr, "CLIM_Tmax_mean", doys, iy
    )
    clim_std = threshold_row(
        ds_thr, "CLIM_Tmax_std", doys, iy
    )

    valid_all = (
        np.isfinite(tmax)
        & np.isfinite(vpd)
        & np.isfinite(twb)
        & np.isfinite(core_thr)
        & np.isfinite(vpd_thr)
        & np.isfinite(twb_thr)
        & np.isfinite(clim_mean)
        & np.isfinite(clim_std)
        & (clim_std > 1e-6)
    )

    zt = (
        (tmax - clim_mean)
        / np.where(clim_std > 1e-6, clim_std, np.nan)
    )
    zt = np.clip(
        zt, Z_CLIP_MIN, Z_CLIP_MAX
    )
    zt_pos = np.where(
        zt > 0, zt, 0.0
    ).astype(np.float32)

    core_raw = valid_all & (tmax >= core_thr)
    dry_raw = valid_all & (vpd >= vpd_thr)
    humid_raw = valid_all & (twb >= twb_thr)

    row = {}

    for state in STATES:
        for metric in METRICS:
            row[f"NESTED_{state}_{metric}_annual"] = np.full(
                (nyr, nx), np.nan, np.float32
            )
            row[f"NESTED_{state}_{metric}_trend"] = np.full(
                nx, np.nan, np.float32
            )
            row[f"NESTED_{state}_{metric}_pvalue"] = np.full(
                nx, np.nan, np.float32
            )
            row[f"NESTED_{state}_{metric}_lag1"] = np.full(
                nx, np.nan, np.float32
            )
            row[f"NESTED_{state}_{metric}_lag1critical"] = np.full(
                nx, np.nan, np.float32
            )
            row[f"NESTED_{state}_{metric}_mkmethod"] = np.full(
                nx, np.nan, np.float32
            )

    for metric in ["intensity", "duration", "frequency"]:
        row[
            f"NESTED_DRY_minus_HUMID_{metric}_contrast"
        ] = np.full(nx, np.nan, np.float32)

    row["NESTED_DRY_minus_HUMID_annual_difference"] = np.full(
        (nyr, nx), np.nan, np.float32
    )
    for suffix in [
        "trend", "pvalue", "lag1",
        "lag1critical", "mkmethod",
    ]:
        row[
            f"NESTED_DRY_minus_HUMID_annual_difference_{suffix}"
        ] = np.full(nx, np.nan, np.float32)

    for ix in np.where(mask_row)[0]:
        valid_cell = valid_all[:, ix]
        valid_by_year = {
            int(y): int(
                np.sum(
                    valid_cell[
                        year_indices[int(y)]
                    ]
                )
            )
            for y in year_axis
        }

        if (
            sum(
                count >= MIN_VALID_DAYS_PER_YEAR
                for count in valid_by_year.values()
            )
            < MIN_VALID_YEARS_FOR_TREND
        ):
            continue

        core = apply_min_duration(
            core_raw[:, ix],
            min_len=MIN_LEN,
        )

        dry = (
            core
            & dry_raw[:, ix]
            & ~humid_raw[:, ix]
        )
        humid = (
            core
            & humid_raw[:, ix]
            & ~dry_raw[:, ix]
        )
        mixed = (
            core
            & dry_raw[:, ix]
            & humid_raw[:, ix]
        )
        neutral = (
            core
            & ~dry_raw[:, ix]
            & ~humid_raw[:, ix]
        )

        # Strong invariant: every valid core day belongs to exactly one state.
        partition_count = (
            dry.astype(np.int8)
            + humid.astype(np.int8)
            + mixed.astype(np.int8)
            + neutral.astype(np.int8)
        )
        if np.any(partition_count[core] != 1):
            raise RuntimeError(
                f"State partition failed at row={iy}, col={ix}."
            )

        state_masks = {
            "DRY": dry,
            "HUMID": humid,
            "MIXED": mixed,
            "NEUTRAL": neutral,
        }

        metrics = annual_state_metrics(
            core,
            state_masks,
            zt_pos[:, ix],
            years,
        )

        for state in STATES:
            for metric in METRICS:
                series = np.asarray(
                    [
                        (
                            metrics[state][metric][int(y)]
                            if valid_by_year[int(y)]
                            >= MIN_VALID_DAYS_PER_YEAR
                            else np.nan
                        )
                        for y in year_axis
                    ],
                    dtype=np.float32,
                )

                prefix = f"NESTED_{state}_{metric}"
                row[f"{prefix}_annual"][:, ix] = series

                sl, pv, r1, cr, method = robust_trend(
                    series, year_axis
                )
                row[f"{prefix}_trend"][ix] = sl
                row[f"{prefix}_pvalue"][ix] = pv
                row[f"{prefix}_lag1"][ix] = r1
                row[f"{prefix}_lag1critical"][ix] = cr
                row[f"{prefix}_mkmethod"][ix] = method

        # Same-unit difference-of-slopes contrasts.
        for metric in ["intensity", "duration", "frequency"]:
            dry_sl = row[
                f"NESTED_DRY_{metric}_trend"
            ][ix]
            humid_sl = row[
                f"NESTED_HUMID_{metric}_trend"
            ][ix]
            if np.isfinite(dry_sl) and np.isfinite(humid_sl):
                row[
                    f"NESTED_DRY_minus_HUMID_{metric}_contrast"
                ][ix] = dry_sl - humid_sl

        # Separate annual-difference trend with its OWN significance.
        dry_ann = row[
            "NESTED_DRY_intensity_annual"
        ][:, ix]
        humid_ann = row[
            "NESTED_HUMID_intensity_annual"
        ][:, ix]
        annual_diff = dry_ann - humid_ann
        row[
            "NESTED_DRY_minus_HUMID_annual_difference"
        ][:, ix] = annual_diff

        sl, pv, r1, cr, method = robust_trend(
            annual_diff, year_axis
        )
        row[
            "NESTED_DRY_minus_HUMID_annual_difference_trend"
        ][ix] = sl
        row[
            "NESTED_DRY_minus_HUMID_annual_difference_pvalue"
        ][ix] = pv
        row[
            "NESTED_DRY_minus_HUMID_annual_difference_lag1"
        ][ix] = r1
        row[
            "NESTED_DRY_minus_HUMID_annual_difference_lag1critical"
        ][ix] = cr
        row[
            "NESTED_DRY_minus_HUMID_annual_difference_mkmethod"
        ][ix] = method

    save_row_cache(iy, row)
    print(
        f"[OK] Saved restartable latitude-row cache {iy + 1}"
    )
    return row


def assemble_products(
    daily_files,
    template,
    ds_thr,
    mask,
    recompute=False,
):
    annual_path = (
        OUT_DIR / "Nested_CommonCore_annual_metrics_1990_2024.nc"
    )
    trends_path = (
        OUT_DIR / "Nested_CommonCore_trends_1990_2024.nc"
    )

    year_axis = np.arange(
        START_YEAR, END_YEAR + 1, dtype=np.int16
    )
    nyr = len(year_axis)
    lat = template.lat.values
    lon = template.lon.values
    ny, nx = len(lat), len(lon)

    annual_arrays = {
        f"NESTED_{state}_{metric}": np.full(
            (nyr, ny, nx),
            np.nan,
            np.float32,
        )
        for state in STATES
        for metric in METRICS
    }
    annual_arrays[
        "NESTED_DRY_minus_HUMID_annual_intensity_difference"
    ] = np.full(
        (nyr, ny, nx),
        np.nan,
        np.float32,
    )

    trend_arrays = {}

    for state in STATES:
        for metric in METRICS:
            prefix = f"NESTED_{state}_{metric}"
            for suffix in [
                "trend_decade",
                "pvalue",
                "lag1_autocorr_detrended",
                "lag1_critical",
                "MK_method_code",
            ]:
                trend_arrays[
                    f"{prefix}_{suffix}"
                ] = np.full(
                    (ny, nx), np.nan, np.float32
                )

    for metric in ["intensity", "duration", "frequency"]:
        trend_arrays[
            f"NESTED_DRY_minus_HUMID_{metric}_trend_decade"
        ] = np.full(
            (ny, nx), np.nan, np.float32
        )

    for name in [
        "NESTED_DRY_minus_HUMID_annual_difference_trend_decade",
        "NESTED_DRY_minus_HUMID_annual_difference_pvalue",
        "NESTED_DRY_minus_HUMID_annual_difference_lag1_autocorr_detrended",
        "NESTED_DRY_minus_HUMID_annual_difference_lag1_critical",
        "NESTED_DRY_minus_HUMID_annual_difference_MK_method_code",
    ]:
        trend_arrays[name] = np.full(
            (ny, nx), np.nan, np.float32
        )

    for iy in range(ny):
        row = compute_row(
            iy,
            daily_files,
            ds_thr,
            mask.values[iy, :].astype(bool),
            recompute=recompute,
        )

        for state in STATES:
            for metric in METRICS:
                prefix = f"NESTED_{state}_{metric}"
                annual_arrays[prefix][:, iy, :] = row[
                    f"{prefix}_annual"
                ]

                mapping = {
                    "trend_decade": "trend",
                    "pvalue": "pvalue",
                    "lag1_autocorr_detrended": "lag1",
                    "lag1_critical": "lag1critical",
                    "MK_method_code": "mkmethod",
                }
                for target_suffix, source_suffix in mapping.items():
                    trend_arrays[
                        f"{prefix}_{target_suffix}"
                    ][iy, :] = row[
                        f"{prefix}_{source_suffix}"
                    ]

        annual_arrays[
            "NESTED_DRY_minus_HUMID_annual_intensity_difference"
        ][:, iy, :] = row[
            "NESTED_DRY_minus_HUMID_annual_difference"
        ]

        for metric in ["intensity", "duration", "frequency"]:
            trend_arrays[
                f"NESTED_DRY_minus_HUMID_{metric}_trend_decade"
            ][iy, :] = row[
                f"NESTED_DRY_minus_HUMID_{metric}_contrast"
            ]

        annual_diff_map = {
            "NESTED_DRY_minus_HUMID_annual_difference_trend_decade":
                "NESTED_DRY_minus_HUMID_annual_difference_trend",
            "NESTED_DRY_minus_HUMID_annual_difference_pvalue":
                "NESTED_DRY_minus_HUMID_annual_difference_pvalue",
            "NESTED_DRY_minus_HUMID_annual_difference_lag1_autocorr_detrended":
                "NESTED_DRY_minus_HUMID_annual_difference_lag1",
            "NESTED_DRY_minus_HUMID_annual_difference_lag1_critical":
                "NESTED_DRY_minus_HUMID_annual_difference_lag1critical",
            "NESTED_DRY_minus_HUMID_annual_difference_MK_method_code":
                "NESTED_DRY_minus_HUMID_annual_difference_mkmethod",
        }
        for target, source in annual_diff_map.items():
            trend_arrays[target][iy, :] = row[source]

    annual = xr.Dataset(
        {
            name: (
                ("year", "lat", "lon"),
                values,
            )
            for name, values in annual_arrays.items()
        },
        coords={
            "year": year_axis,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "description": (
                "Mutually exclusive thermodynamic-state sensitivity within a "
                "common persistent Tmax-P95 heatwave core."
            ),
            "common_core_definition": (
                f"Tmax >= local P{CORE_TMAX_P:g} for >= {MIN_LEN} consecutive days"
            ),
            "dry_state_definition": (
                f"VPDmean >= P{DRY_VPD_P:g} and Twbmax < P{HUMID_TWB_P:g}"
            ),
            "humid_state_definition": (
                f"Twbmax >= P{HUMID_TWB_P:g} and VPDmean < P{DRY_VPD_P:g}"
            ),
            "mixed_state_definition": (
                f"VPDmean >= P{DRY_VPD_P:g} and Twbmax >= P{HUMID_TWB_P:g}"
            ),
            "neutral_state_definition": (
                f"VPDmean < P{DRY_VPD_P:g} and Twbmax < P{HUMID_TWB_P:g}"
            ),
            "frequency_definition": (
                "Number of contiguous thermodynamic-state episodes within "
                "persistent common-core events; CORE frequency counts parent core events."
            ),
            "intensity_definition": (
                "Sum z+(Tmax) over state days, using Figure 01 "
                "CLIM_Tmax_mean/std and z clipping [-5,5]."
            ),
            "threshold_source": str(THRESHOLD_FILE),
            "script_version": SCRIPT_VERSION,
        },
    )

    trends = xr.Dataset(
        {
            name: (
                ("lat", "lon"),
                values,
            )
            for name, values in trend_arrays.items()
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "description": (
                "Theil-Sen trends for nested/common-core sensitivity metrics."
            ),
            "trend_method": "Theil-Sen median slope per decade",
            "significance_test": (
                "Original MK unless detrended lag-1 rank autocorrelation is "
                "significant; Hamed-Rao MK lag1 otherwise."
            ),
            "trend_significance_version": TREND_SIGNIFICANCE_VERSION,
            "main_nested_contrast": (
                "NESTED_DRY_minus_HUMID_intensity_trend_decade"
            ),
            "main_nested_contrast_definition": (
                "NESTED_DRY_intensity_trend_decade - "
                "NESTED_HUMID_intensity_trend_decade"
            ),
            "main_nested_contrast_units": (
                "Tmax-standardized severity decade-1"
            ),
            "contrast_significance_note": (
                "No standalone p-value is assigned to the main difference-of-slopes "
                "contrast. A separately named trend of the annual state difference "
                "and its MK p-value are retained only as a sensitivity diagnostic."
            ),
            "script_version": SCRIPT_VERSION,
        },
    )

    encoding_ann = {
        v: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for v in annual.data_vars
    }
    encoding_tr = {
        v: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for v in trends.data_vars
    }

    safe_to_netcdf(
        annual, annual_path, encoding=encoding_ann
    )
    safe_to_netcdf(
        trends, trends_path, encoding=encoding_tr
    )

    print(f"[OK] Saved {annual_path}")
    print(f"[OK] Saved {trends_path}")
    return annual, trends


# ============================================================
# Figure 01 comparison
# ============================================================

def load_primary_common_contrast(trends_nested):
    f1 = standardize_lat_lon(
        safe_open_dataset(
            FIG1_TRENDS_NC,
            decode_times=False,
            chunks=None,
        )
    )

    required = [
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "DHW_TmaxOnly_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
    ]
    missing = [
        v for v in required
        if v not in f1.data_vars
    ]
    if missing:
        f1.close()
        raise RuntimeError(
            "Figure 01 common-scale fields are missing: "
            + ", ".join(missing)
        )

    primary = f1[
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    ]

    # Require exact coordinate selection from the Figure 01 grid.
    # Spatial interpolation is not used for this sensitivity comparison.
    try:
        primary = primary.sel(
            lat=xr.DataArray(
                trends_nested.lat.values,
                dims="lat",
            ),
            lon=xr.DataArray(
                trends_nested.lon.values,
                dims="lon",
            ),
        ).load()
    except Exception as exc:
        f1.close()
        raise RuntimeError(
            "The primary Figure 01 contrast cannot be selected on the exact "
            "nested-analysis grid. Spatial interpolation is not permitted for "
            f"this comparison. Details: {exc}"
        )

    if not np.allclose(primary.lat.values, trends_nested.lat.values):
        f1.close()
        raise RuntimeError("Primary and nested latitude coordinates are not exactly aligned.")
    if not np.allclose(primary.lon.values, trends_nested.lon.values):
        f1.close()
        raise RuntimeError("Primary and nested longitude coordinates are not exactly aligned.")

    primary.name = (
        "PRIMARY_DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    )
    f1.close()
    return primary


# ============================================================
# Weighted spatial statistics
# ============================================================

def weighted_quantile(values, weights, q):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not np.any(ok):
        return np.nan

    values = values[ok]
    weights = weights[ok]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(
        np.interp(q, cdf, values)
    )


def _weighted_rank_corr(x, y, w):
    xranks = stats.rankdata(x)
    yranks = stats.rankdata(y)
    mx = np.average(xranks, weights=w)
    my = np.average(yranks, weights=w)
    cov = np.average(
        (xranks - mx) * (yranks - my),
        weights=w,
    )
    vx = np.average(
        (xranks - mx) ** 2,
        weights=w,
    )
    vy = np.average(
        (yranks - my) ** 2,
        weights=w,
    )
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(
        cov / np.sqrt(vx * vy)
    )


def weighted_spearman(
    x, y, w,
    seed=RANDOM_SEED,
):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )
    if ok.sum() < 10:
        return np.nan, np.nan

    x, y, w = x[ok], y[ok], w[ok]
    rho = _weighted_rank_corr(x, y, w)
    if not np.isfinite(rho):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    extreme = 0
    valid_perm = 0

    for _ in range(N_PERMUTATIONS):
        rp = _weighted_rank_corr(
            x,
            rng.permutation(y),
            w,
        )
        if np.isfinite(rp):
            valid_perm += 1
            if abs(rp) >= abs(rho):
                extreme += 1

    p = (
        (extreme + 1.0)
        / (valid_perm + 1.0)
    )
    return float(rho), float(p)


def build_gridcell_comparison(
    trends,
    primary,
    mask,
    weights,
):
    nested = trends[
        "NESTED_DRY_minus_HUMID_intensity_trend_decade"
    ]

    lat2d, lon2d = np.meshgrid(
        trends.lat.values,
        trends.lon.values,
        indexing="ij",
    )

    x = primary.where(mask).values.ravel()
    y = nested.where(mask).values.ravel()
    w = weights.where(mask).values.ravel()
    latv = lat2d.ravel()
    lonv = lon2d.ravel()

    ok = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    df = pd.DataFrame({
        "lat": latv[ok],
        "lon": lonv[ok],
        "primary_common_TmaxOnly_contrast": x[ok],
        "nested_common_core_contrast": y[ok],
        "grid_area_weight": w[ok],
    })

    rho, p = weighted_spearman(
        df["primary_common_TmaxOnly_contrast"],
        df["nested_common_core_contrast"],
        df["grid_area_weight"],
    )

    same = (
        100.0
        * np.sum(
            df["grid_area_weight"].values[
                np.sign(
                    df["primary_common_TmaxOnly_contrast"].values
                )
                == np.sign(
                    df["nested_common_core_contrast"].values
                )
            ]
        )
        / np.sum(df["grid_area_weight"].values)
    )

    print(
        "[SUMMARY] Nested vs primary common-scale contrast: "
        f"cos(latitude)-weighted rho={rho:+.3f}, "
        f"permutation p={p:.3f}, "
        f"same-sign area={same:.1f}%"
    )

    df.attrs = {
        "coslat_weighted_spearman_rho": rho,
        "permutation_p_value": p,
        "same_sign_area_pct": same,
    }

    path = (
        OUT_DIR / "Nested_CommonCore_gridcell_comparison.csv"
    )
    df.to_csv(path, index=False)
    print(f"[OK] Saved grid-cell comparison: {path}")
    return df, rho, p, same


# ============================================================
# Regional summaries
# ============================================================

def assign_biomes(trends, mask, biomes, weights):
    nested = trends[
        "NESTED_DRY_minus_HUMID_intensity_trend_decade"
    ].where(mask)

    df = nested.to_dataframe(
        name="nested_contrast"
    ).reset_index()
    df = df[
        np.isfinite(df["nested_contrast"])
    ].copy()

    weight_df = weights.to_dataframe(
        name="grid_area_weight"
    ).reset_index()
    df = df.merge(
        weight_df,
        on=["lat", "lon"],
        how="left",
    )

    if biomes is None or df.empty:
        df["region"] = "Brazil"
        return df

    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["lon"], df["lat"]
        ),
        crs="EPSG:4326",
    )

    biome_col = None
    for c in [
        "nome", "BIOMA", "bioma", "Bioma", "NAME"
    ]:
        if c in biomes.columns:
            biome_col = c
            break

    if biome_col is None:
        df["region"] = "Brazil"
        return df

    joined = gpd.sjoin(
        pts,
        biomes[[biome_col, "geometry"]],
        how="left",
        predicate="within",
    )

    mapping = {
        "AMAZONIA": "Amazon",
        "AMAZÔNIA": "Amazon",
        "AMAZON": "Amazon",
        "CERRADO": "Cerrado",
        "CAATINGA": "Caatinga",
        "MATA ATLANTICA": "Atlantic Forest",
        "MATA ATLÂNTICA": "Atlantic Forest",
        "PAMPA": "Pampa",
        "PAMPAS": "Pampa",
        "PANTANAL": "Pantanal",
    }

    raw = joined[biome_col].astype(str)
    joined["region"] = (
        raw.str.upper()
        .map(mapping)
        .fillna(raw)
    )

    return pd.DataFrame(
        joined.drop(columns="geometry")
    )


def build_regional_summary(
    trends,
    primary,
    mask,
    weights,
    biomes,
):
    df = assign_biomes(
        trends, mask, biomes, weights
    )

    p_df = primary.where(mask).to_dataframe(
        name="primary_contrast"
    ).reset_index()
    df = df.merge(
        p_df[["lat", "lon", "primary_contrast"]],
        on=["lat", "lon"],
        how="left",
    )

    all_df = pd.concat(
        [
            df.assign(region="Brazil"),
            df,
        ],
        ignore_index=True,
    )

    rows = []
    for region, g in all_df.groupby("region"):
        v = pd.to_numeric(
            g["nested_contrast"],
            errors="coerce",
        ).to_numpy(float)
        p = pd.to_numeric(
            g["primary_contrast"],
            errors="coerce",
        ).to_numpy(float)
        w = pd.to_numeric(
            g["grid_area_weight"],
            errors="coerce",
        ).to_numpy(float)

        ok = (
            np.isfinite(v)
            & np.isfinite(w)
            & (w > 0)
        )
        if not np.any(ok):
            continue

        row = {
            "region": region,
            "n_gridcells": int(ok.sum()),
            "nested_median": weighted_quantile(
                v[ok], w[ok], 0.50
            ),
            "nested_q25": weighted_quantile(
                v[ok], w[ok], 0.25
            ),
            "nested_q75": weighted_quantile(
                v[ok], w[ok], 0.75
            ),
            "nested_positive_area_pct": (
                100.0
                * np.sum(w[ok][v[ok] > 0])
                / np.sum(w[ok])
            ),
        }

        okp = (
            np.isfinite(p)
            & np.isfinite(w)
            & (w > 0)
        )
        if np.any(okp):
            row.update({
                "primary_median": weighted_quantile(
                    p[okp], w[okp], 0.50
                ),
                "primary_q25": weighted_quantile(
                    p[okp], w[okp], 0.25
                ),
                "primary_q75": weighted_quantile(
                    p[okp], w[okp], 0.75
                ),
            })

        rows.append(row)

    out = pd.DataFrame(rows)
    path = (
        OUT_DIR / "Nested_CommonCore_regional_summary.csv"
    )
    out.to_csv(path, index=False)
    print(f"[OK] Saved regional summary: {path}")
    return out


# ============================================================
# Plotting
# ============================================================

def draw_boundaries(
    ax, uf,
    biomes=None,
    countries=None,
):
    if countries is not None:
        countries.boundary.plot(
            ax=ax,
            color="0.55",
            linewidth=0.35,
            zorder=3,
        )
    if biomes is not None:
        biomes.boundary.plot(
            ax=ax,
            color="0.25",
            linewidth=0.28,
            alpha=0.55,
            zorder=4,
        )
    uf.boundary.plot(
        ax=ax,
        color="0.05",
        linewidth=0.70,
        zorder=5,
    )


def plot_map(
    ax,
    da,
    title,
    norm,
    cbar_label,
    uf,
    mask,
    biomes=None,
    countries=None,
):
    arr = da.where(mask)

    im = ax.pcolormesh(
        arr.lon,
        arr.lat,
        arr,
        cmap=plt.cm.RdBu_r,
        norm=norm,
        shading="auto",
        rasterized=True,
        zorder=1,
    )

    draw_boundaries(
        ax, uf,
        biomes=biomes,
        countries=countries,
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(
        title,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(
        True,
        linestyle="--",
        alpha=0.15,
        linewidth=0.4,
    )

    cb = ax.figure.colorbar(
        im,
        ax=ax,
        shrink=0.78,
        pad=0.015,
    )
    cb.set_label(cbar_label)


def plot_scatter(
    ax,
    df,
    rho,
    p,
    same_sign,
):
    x = df[
        "primary_common_TmaxOnly_contrast"
    ].to_numpy(float)
    y = df[
        "nested_common_core_contrast"
    ].to_numpy(float)

    if len(x) < 10:
        ax.text(
            0.5, 0.5,
            "Insufficient paired cells",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    hb = ax.hexbin(
        x,
        y,
        gridsize=45,
        mincnt=1,
        cmap="Greys",
        linewidths=0.0,
    )

    lo, hi = np.nanpercentile(
        np.r_[x, y],
        [2, 98],
    )
    lim = max(abs(lo), abs(hi), 1.0)

    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#B2182B",
        linestyle="--",
        linewidth=1.2,
    )
    ax.axhline(
        0,
        color="0.35",
        linewidth=0.8,
    )
    ax.axvline(
        0,
        color="0.35",
        linewidth=0.8,
    )

    ax.text(
        0.04, 0.96,
        f"cos(latitude)-weighted ρ = {rho:.2f}\n"
        f"permutation p = {p:.3f}\n"
        f"same-sign area = {same_sign:.1f}%\n"
        f"N = {len(df):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="0.7",
            alpha=0.90,
        ),
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(
        "Primary DHW−HHW\n"
        "(Tmax-standardized severity decade$^{-1}$)"
    )
    ax.set_ylabel(
        "Mutually exclusive common-core DRY−HUMID\n"
        "(Tmax-standardized severity decade$^{-1}$)"
    )
    ax.set_title(
        "(c) Primary vs common-core sensitivity",
        fontweight="bold",
    )
    ax.grid(
        True,
        alpha=0.2,
        linewidth=0.4,
    )

    cb = ax.figure.colorbar(
        hb,
        ax=ax,
        shrink=0.78,
        pad=0.015,
    )
    cb.set_label("grid-cell count")


def plot_regional_summary(ax, summary):
    order = [
        "Brazil",
        "Amazon",
        "Cerrado",
        "Caatinga",
        "Pantanal",
        "Atlantic Forest",
        "Pampa",
    ]

    data = (
        summary.set_index("region")
        .reindex([
            r for r in order
            if r in set(summary["region"])
        ])
    )

    y = np.arange(len(data))
    med = data["nested_median"].to_numpy(float)
    q25 = data["nested_q25"].to_numpy(float)
    q75 = data["nested_q75"].to_numpy(float)

    ax.axvline(
        0,
        color="0.2",
        linewidth=0.8,
    )

    for yi, m, lo, hi in zip(
        y, med, q25, q75
    ):
        if not np.isfinite(m):
            continue
        color = (
            "#B2182B"
            if m >= 0
            else "#2166AC"
        )
        ax.plot(
            [lo, hi], [yi, yi],
            color=color,
            linewidth=7,
            alpha=0.22,
            solid_capstyle="round",
        )
        ax.plot(
            [lo, hi], [yi, yi],
            color=color,
            linewidth=1.4,
        )
        ax.scatter(
            m, yi,
            s=42,
            color=color,
            edgecolor="0.15",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data.index.tolist()
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Common-core DRY−HUMID contrast\n"
        "(Tmax-standardized severity decade$^{-1}$)"
    )
    ax.set_title(
        "(d) Area-weighted regional median and IQR",
        fontweight="bold",
    )
    ax.grid(
        axis="x",
        alpha=0.25,
        linewidth=0.4,
    )


def plot_figure(
    trends,
    primary,
    comparison_df,
    rho,
    p,
    same_sign,
    summary,
    uf,
    mask,
    biomes=None,
    countries=None,
):
    print(
        "[INFO] Plotting corrected nested/common-core sensitivity figure"
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15.5, 11.2),
    )
    plt.subplots_adjust(
        left=0.070,
        right=0.965,
        top=0.955,
        bottom=0.095,
        wspace=0.28,
        hspace=0.30,
    )

    nested = trends[
        "NESTED_DRY_minus_HUMID_intensity_trend_decade"
    ]
    diff = (
        nested - primary
    ).rename(
        "nested_minus_primary"
    )

    common_norm = TwoSlopeNorm(
        vmin=-6.0,
        vcenter=0.0,
        vmax=6.0,
    )
    common_norm2 = TwoSlopeNorm(
        vmin=-4.0,
        vcenter=0.0,
        vmax=4.0,
    )

    plot_map(
        axes[0, 0],
        nested,
        "(a) Mutually exclusive common-core DRY−HUMID contrast",
        common_norm,
        "Tmax-standardized severity decade$^{-1}$",
        uf,
        mask,
        biomes=biomes,
        countries=countries,
    )

    plot_map(
        axes[0, 1],
        diff,
        "(b) Common-core minus primary contrast",
        common_norm2,
        "Tmax-standardized severity decade$^{-1}$",
        uf,
        mask,
        biomes=biomes,
        countries=countries,
    )

    plot_scatter(
        axes[1, 0],
        comparison_df,
        rho,
        p,
        same_sign,
    )

    plot_regional_summary(
        axes[1, 1],
        summary,
    )


    out_base = (
        OUT_DIR
        / "Supplementary_Nested_CommonCore_Heatwave_Sensitivity"
    )

    fig.savefig(
        str(out_base) + ".pdf",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        str(out_base) + ".jpeg",
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"[OK] Saved {out_base}.pdf")
    print(f"[OK] Saved {out_base}.jpeg")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    configure_paths(args)

    print(
        f"[START] Nested/common-core heatwave sensitivity | "
        f"{pd.Timestamp.now().isoformat()}"
    )
    print(
        "[INFO] Primary comparison: "
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    )
    print(
        "[INFO] Direct subtraction of primary regime-specific DHW and HHW "
        "intensity trends is prohibited."
    )
    print(
        "[INFO] Figure 01 thresholds are reused exactly; "
        "they are not recomputed."
    )
    print(
        f"[INFO] Restartable row cache: {ROW_CACHE_DIR}"
    )

    daily_files = list_daily_files()
    print(
        f"[INFO] Figure 01 daily files found: {len(daily_files)}"
    )

    template = open_daily_template(
        daily_files
    )
    mask, uf = make_brazil_mask(
        template.lat.values,
        template.lon.values,
    )
    weights = grid_area_weights(
        template.lat.values,
        template.lon.values,
    )
    print(
        f"[INFO] Brazil mask grid cells: {int(mask.sum())}"
    )

    ds_thr = load_thresholds_exact(
        template
    )

    biomes, countries = (
        load_optional_boundaries()
    )

    annual, trends = assemble_products(
        daily_files,
        template,
        ds_thr,
        mask,
        recompute=args.recompute,
    )

    primary = load_primary_common_contrast(
        trends
    )

    (
        comparison_df,
        rho,
        p,
        same_sign,
    ) = build_gridcell_comparison(
        trends,
        primary,
        mask,
        weights,
    )

    summary = build_regional_summary(
        trends,
        primary,
        mask,
        weights,
        biomes,
    )

    plot_figure(
        trends,
        primary,
        comparison_df,
        rho,
        p,
        same_sign,
        summary,
        uf,
        mask,
        biomes=biomes,
        countries=countries,
    )

    template.close()
    ds_thr.close()
    annual.close()
    trends.close()

    print("\n[DONE] Outputs saved in:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

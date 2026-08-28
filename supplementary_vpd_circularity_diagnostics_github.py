#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary VPD circularity diagnostics, 1990–2024.

Purpose
-------
Evaluate whether preferential dry-hot amplification is preserved when VPD is
excluded from the cross-regime intensity metric.

Figure 01 provides the common-scale fields:

    DHW_TmaxOnly_intensity
    HHW_TmaxOnly_intensity
    DHW_minus_HHW_TmaxOnly_intensity_trend_decade

Both common-scale intensities use:

    sum[z+(Tmax)]

over persistent regime event days. VPD continues to classify DHW occurrence but
does not enter the common-scale intensity metric.

The workflow does not re-detect heatwaves or reconstruct Figure 01 metrics from
daily data. It reads the Figure 01 annual and trend products directly.

Additional diagnostics
----------------------
1. Common-scale DHW−HHW Tmax-only intensity-trend contrast.
2. Association between the primary DHW intensity trend and the DHW Tmax-only
   intensity trend. These fields are not subtracted because their definitions
   and units differ.
3. DHW−HHW duration-trend contrast.
4. DHW−HHW frequency-trend contrast.
5. Actual atmospheric vapour-pressure trend.
6. Dew-point trend.

Actual atmospheric vapour pressure is derived preferentially from dew point.
When dew point is unavailable, RH and Tmean are used as a documented fallback.

Scientific safeguards
---------------------
- Primary DHW and HHW intensity trends are never subtracted.
- No standalone p-value is assigned to a difference-of-slopes contrast.
- Trend magnitude uses Theil–Sen median slope per decade.
- Significance uses the original two-sided Mann–Kendall test unless detrended
  lag-1 rank autocorrelation is significant; Hamed–Rao lag-1 correction is used
  otherwise.
- Annual atmospheric diagnostics require at least 300 valid days.
- Trend estimation requires at least 20 valid years.
- Spatial summaries use cos(latitude) weights.
- Spatial rank associations use cos(latitude)-weighted Spearman correlation
  with 999 two-sided permutations.
- The analysis is a sensitivity/thermodynamic diagnostic and is not interpreted
  causally.

Usage
-----
python supplementary_vpd_circularity_diagnostics.py \
    --figure1-output-dir /path/to/figure_01_outputs \
    --brazil-shapefile /path/to/brazil_boundary.shp \
    --output-dir ./outputs/vpd_circularity

Use ``--recompute-moisture`` to rebuild the auxiliary atmospheric-moisture
diagnostics.
"""

import os
import glob
import argparse
import warnings
from pathlib import Path


import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from scipy import stats
from scipy.stats import theilslopes, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
FIG1_ANNUAL_NC = None
FIG1_TRENDS_NC = None
BRAZIL_SHP = None
OUT_DIR = None

START_YEAR = 1990
END_YEAR = 2024
MIN_VALID_DAYS_PER_YEAR = 300
MIN_VALID_YEARS_FOR_TREND = 20
AUTOCORR_ALPHA = 0.05
P_THRESHOLD = 0.05
N_PERMUTATIONS = 999
RANDOM_SEED = 42

LON_MIN, LON_MAX = -75.0, -32.0
LAT_MIN, LAT_MAX = -35.0, 6.0

SCRIPT_VERSION = "1.0.0"
TREND_SIGNIFICANCE_VERSION = "Theil-Sen + conditional Hamed-Rao MK lag1"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# ============================================================
# Arguments / paths
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Supplementary VPD circularity diagnostics using Figure 01 "
            "annual, trend, and daily products."
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
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    p.add_argument(
        "--recompute-moisture",
        action="store_true",
        help="Recompute auxiliary annual/trend atmospheric-moisture diagnostics.",
    )
    return p.parse_args()

def configure_paths(args):
    global FIGURE1_ROOT, DAILY_DIR, FIG1_ANNUAL_NC, FIG1_TRENDS_NC
    global BRAZIL_SHP, OUT_DIR

    FIGURE1_ROOT = Path(args.figure1_output_dir).expanduser().resolve()
    DAILY_DIR = FIGURE1_ROOT / "cache" / "era5_daily"
    FIG1_ANNUAL_NC = FIGURE1_ROOT / "data" / "figure_01_annual_heatwave_metrics_1990_2024.nc"
    FIG1_TRENDS_NC = FIGURE1_ROOT / "data" / "figure_01_heatwave_trends_1990_2024.nc"
    BRAZIL_SHP = Path(args.brazil_shapefile).expanduser().resolve()
    OUT_DIR = Path(args.output_dir).expanduser().resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        (DAILY_DIR, "Figure 01 daily cache"),
        (FIG1_ANNUAL_NC, "Figure 01 annual metrics"),
        (FIG1_TRENDS_NC, "Figure 01 trends"),
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
    raise RuntimeError(f"Could not open NetCDF: {path}\n" + "\n".join(errors))


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
        scipy_encoding = None
        if encoding:
            scipy_encoding = {
                v: {
                    k: val for k, val in opts.items()
                    if k in {"dtype", "_FillValue", "scale_factor", "add_offset"}
                }
                for v, opts in encoding.items()
            }
        try:
            ds.to_netcdf(path, engine="scipy", encoding=scipy_encoding)
            return
        except Exception as exc:
            errors.append(f"scipy: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Could not write NetCDF: {path}\n" + "\n".join(errors))


# ============================================================
# Generic utilities
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
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    if "lat" in ds.coords and ds.lat[0] < ds.lat[-1]:
        ds = ds.sortby("lat", ascending=False)
    return ds


def remove_feb29(ds):
    if "time" not in ds.coords:
        return ds
    mask = ~((ds.time.dt.month == 2) & (ds.time.dt.day == 29))
    return ds.sel(time=mask)


def find_var(ds, candidates):
    for c in candidates:
        if c in ds.data_vars:
            return c
    lower = {v.lower(): v for v in ds.data_vars}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def to_celsius(da):
    units = str(da.attrs.get("units", "")).lower()
    if "celsius" in units or "degc" in units or units.strip() in {"c", "°c"}:
        return da.astype(float)
    sample = da
    if "time" in sample.dims:
        sample = sample.isel(time=slice(0, min(30, sample.sizes["time"])))
    med = float(np.nanmedian(sample.values))
    return da.astype(float) - 273.15 if np.isfinite(med) and med > 100 else da.astype(float)


def saturation_vapour_pressure_kpa(temp_c):
    return 0.6112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def dewpoint_from_ea_kpa(ea_kpa):
    ea = np.maximum(ea_kpa, 1e-8)
    ln = np.log(ea / 0.6112)
    return (243.5 * ln) / (17.67 - ln)


def vpd_to_kpa(da, source_name):
    units = str(da.attrs.get("units", "")).lower().strip()
    if "kpa" in units:
        return da.astype(float)
    if "hpa" in units or "hectopascal" in units:
        return da.astype(float) / 10.0
    if source_name in {"VPDmean", "VPDmax"}:
        # Figure 01 daily-cache convention.
        return da.astype(float) / 10.0
    sample = da
    if "time" in sample.dims:
        sample = sample.isel(time=slice(0, min(30, sample.sizes["time"])))
    med = float(np.nanmedian(sample.values))
    if np.isfinite(med) and med > 5.0:
        return da.astype(float) / 10.0
    if "kpa" in source_name.lower():
        return da.astype(float)
    raise ValueError(
        f"Ambiguous VPD units for {source_name!r}; units={da.attrs.get('units', '')!r}"
    )


def make_brazil_mask(lat, lon):
    uf = gpd.read_file(BRAZIL_SHP).to_crs(epsg=4326)
    geom = uf.dissolve().geometry.iloc[0]
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for iy, la in enumerate(lat):
        pts = [Point(float(lo), float(la)) for lo in lon]
        mask[iy, :] = [geom.contains(p) or geom.touches(p) for p in pts]
    return (
        xr.DataArray(mask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")),
        uf,
    )


def grid_area_weights(lat, lon):
    w = np.repeat(np.cos(np.deg2rad(np.asarray(lat)))[:, None], len(lon), axis=1)
    return xr.DataArray(w, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def weighted_quantile(values, weights, q):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    values, weights = values[ok], weights[ok]
    idx = np.argsort(values)
    values, weights = values[idx], weights[idx]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cdf, values))


def weighted_mean(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if np.any(ok) else np.nan


# ============================================================
# Trend methodology — aligned with Figure 01
# ============================================================

def _mk_score_and_variance(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < MIN_VALID_YEARS_FOR_TREND:
        return np.nan, np.nan
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    tie_term = sum(c * (c - 1) * (2 * c + 5) for c in counts)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    return float(s), float(var_s)


def _mk_p_from_s_var(s, var_s):
    if not np.isfinite(s) or not np.isfinite(var_s) or var_s <= 0:
        return np.nan
    z = 0.0 if s == 0 else (s - np.sign(s)) / np.sqrt(var_s)
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
    critical = float(norm.ppf(1.0 - AUTOCORR_ALPHA / 2.0) / np.sqrt(n))
    if np.nanstd(y) <= 1e-12:
        return 0.0, critical, False

    sen = theilslopes(y, years, alpha=0.05)
    residual = y - (float(sen[1]) + float(sen[0]) * years)
    ranks = pd.Series(residual).rank(method="average").to_numpy(float)

    if np.std(ranks[:-1]) <= 1e-12 or np.std(ranks[1:]) <= 1e-12:
        r1 = 0.0
    else:
        r1 = float(np.corrcoef(ranks[:-1], ranks[1:])[0, 1])

    return r1, critical, bool(np.isfinite(r1) and abs(r1) > critical)


def hamed_rao_mk_pvalue_lag1(y, years):
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
                    y, alpha=AUTOCORR_ALPHA, lag=1
                ).p
            )
        except Exception:
            pass

    s, var_s = _mk_score_and_variance(y)
    if not np.isfinite(var_s) or var_s <= 0:
        return np.nan

    r1, critical, _ = detrended_lag1_autocorrelation(y, years)
    if not np.isfinite(r1) or abs(r1) <= critical:
        return _mk_p_from_s_var(s, var_s)

    n = len(y)
    correction = max(1.0 + 2.0 * (n - 3) / n * abs(r1), 1.0)
    return _mk_p_from_s_var(s, var_s * correction)


def robust_trend(y, years):
    y = np.asarray(y, float)
    years = np.asarray(years, float)
    ok = np.isfinite(y) & np.isfinite(years)
    yv, xv = y[ok], years[ok]
    if len(yv) < MIN_VALID_YEARS_FOR_TREND:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    slope = float(theilslopes(yv, xv, alpha=0.05)[0] * 10.0)
    original_p = mann_kendall_pvalue(yv)
    r1, critical, lag1_sig = detrended_lag1_autocorrelation(yv, xv)
    if lag1_sig:
        selected_p = hamed_rao_mk_pvalue_lag1(yv, xv)
        method_code = 1.0
    else:
        selected_p = original_p
        method_code = 0.0

    return slope, selected_p, r1, critical, method_code


# ============================================================
# Weighted spatial association
# ============================================================

def _weighted_rank_corr(x, y, w):
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


def weighted_spearman(x, y, w, seed=RANDOM_SEED):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
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
        rp = _weighted_rank_corr(x, rng.permutation(y), w)
        if np.isfinite(rp):
            valid_perm += 1
            if abs(rp) >= abs(rho):
                extreme += 1
    p = (extreme + 1.0) / (valid_perm + 1.0)
    return float(rho), float(p)


# ============================================================
# Figure 01 import — do not recompute heatwaves
# ============================================================

def load_figure1_products():
    annual = standardize_lat_lon(
        safe_open_dataset(FIG1_ANNUAL_NC, decode_times=False, chunks=None)
    )
    trends = standardize_lat_lon(
        safe_open_dataset(FIG1_TRENDS_NC, decode_times=False, chunks=None)
    )

    annual_required = [
        "DHW_TmaxOnly_intensity",
        "HHW_TmaxOnly_intensity",
        "DHW_frequency",
        "HHW_frequency",
        "DHW_duration",
        "HHW_duration",
    ]
    trend_required = [
        "DHW_TmaxOnly_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "DHW_intensity_trend_decade",
        "DHW_frequency_trend_decade",
        "HHW_frequency_trend_decade",
        "DHW_duration_trend_decade",
        "HHW_duration_trend_decade",
    ]

    missing_ann = [v for v in annual_required if v not in annual.data_vars]
    missing_tr = [v for v in trend_required if v not in trends.data_vars]

    if missing_ann or missing_tr:
        annual.close()
        trends.close()
        raise RuntimeError(
            "Figure 01 products are incomplete.\n"
            f"Missing annual: {missing_ann}\n"
            f"Missing trends: {missing_tr}"
        )

    method = trends.attrs.get("main_trend_method", "")
    if method and method != "Theil-Sen median slope":
        annual.close()
        trends.close()
        raise RuntimeError(
            f"Unexpected Figure 1 trend method: {method!r}; "
            "expected 'Theil-Sen median slope'."
        )

    # Build compact copies for SI output and use exact Figure 1 metrics.
    tonly_ann = annual[annual_required].load()
    tonly_tr = trends[trend_required].load()

    tonly_tr["DHW_minus_HHW_frequency_trend_decade"] = (
        tonly_tr["DHW_frequency_trend_decade"]
        - tonly_tr["HHW_frequency_trend_decade"]
    )
    tonly_tr["DHW_minus_HHW_duration_trend_decade"] = (
        tonly_tr["DHW_duration_trend_decade"]
        - tonly_tr["HHW_duration_trend_decade"]
    )

    tonly_ann.attrs.update({
        "source": str(FIG1_ANNUAL_NC),
        "alignment": "Exact subset of Figure 01 annual metrics.",
        "common_intensity": "sum[z+(Tmax)] over persistent regime event days",
        "script_version": SCRIPT_VERSION,
    })
    tonly_tr.attrs.update({
        "source": str(FIG1_TRENDS_NC),
        "alignment": "Exact subset and same-unit contrasts derived from Figure 01 trends.",
        "cross_regime_metric": "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "frequency_duration_note": (
            "Frequency and duration differences are same-unit differences of "
            "Figure 1 trend fields; no standalone p-value is assigned to the contrasts."
        ),
        "script_version": SCRIPT_VERSION,
    })

    annual.close()
    trends.close()
    return tonly_ann, tonly_tr


def save_figure1_subset_products(tonly_ann, tonly_tr):
    ann_path = OUT_DIR / "VPD_Circularity_Tonly_annual_metrics_1990_2024.nc"
    tr_path = OUT_DIR / "VPD_Circularity_Tonly_trends_1990_2024.nc"

    enc_ann = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for v in tonly_ann.data_vars
        if tonly_ann[v].dtype.kind in {"f", "i", "u"}
    }
    enc_tr = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for v in tonly_tr.data_vars
        if tonly_tr[v].dtype.kind in {"f", "i", "u"}
    }

    safe_to_netcdf(tonly_ann, ann_path, encoding=enc_ann)
    safe_to_netcdf(tonly_tr, tr_path, encoding=enc_tr)
    print(f"[OK] Saved exact Figure 1 annual subset: {ann_path}")
    print(f"[OK] Saved exact Figure 1 trend subset: {tr_path}")


# ============================================================
# Auxiliary atmospheric-moisture diagnostics from Figure 01 daily cache
# ============================================================

def list_daily_files():
    files = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = DAILY_DIR / f"ERA5_daily_Brazil_{year}.nc"
        if path.exists():
            files.append((year, path))
        else:
            print(f"[WARN] Missing daily file for {year}: {path}")
    if len(files) < MIN_VALID_YEARS_FOR_TREND:
        raise RuntimeError(
            f"Only {len(files)} Figure 01 daily files found."
        )
    return files


def process_year_moisture(path, year):
    ds = standardize_lat_lon(
        safe_open_dataset(path, decode_times=True, chunks=None)
    )
    ds = remove_feb29(ds)

    t_var = find_var(ds, ["Tmean", "tmean", "t2m_mean", "t2m", "var167"])
    rh_var = find_var(ds, ["RHmean", "rhmean", "RH", "relative_humidity"])
    vpd_var = find_var(ds, ["VPDmean", "vpdmean", "VPD", "vpd", "VPD_kPa", "vpd_kpa"])
    td_var = find_var(ds, ["Tdmean", "D2mean", "d2m_mean", "dewpoint_mean", "d2m", "var168"])

    if t_var is None:
        ds.close()
        raise ValueError(f"Tmean unavailable in {path}")

    tmean = to_celsius(ds[t_var])

    if td_var is not None:
        td = to_celsius(ds[td_var])
        ea = saturation_vapour_pressure_kpa(td)
        ea_source = "dewpoint"
    elif rh_var is not None:
        rh = ds[rh_var].astype(float)
        es = saturation_vapour_pressure_kpa(tmean)
        ea = es * (rh / 100.0)
        td = xr.apply_ufunc(dewpoint_from_ea_kpa, ea, dask="allowed")
        ea_source = "RHmean_Tmean_fallback"
    else:
        ds.close()
        raise ValueError(
            f"Neither dew point nor RH is available to derive actual atmospheric vapour pressure in {path}"
        )

    if rh_var is not None:
        rh = ds[rh_var].astype(float)
    else:
        es = saturation_vapour_pressure_kpa(tmean)
        rh = 100.0 * ea / es
        rh = rh.where((rh >= 0) & (rh <= 100))

    if vpd_var is not None:
        vpd = vpd_to_kpa(ds[vpd_var], vpd_var)
    else:
        vpd = saturation_vapour_pressure_kpa(tmean) - ea

    def annual_mean_qc(da):
        count = da.count("time")
        return da.mean("time", skipna=True).where(count >= MIN_VALID_DAYS_PER_YEAR)

    ann = xr.Dataset({
        "Tmean": annual_mean_qc(tmean),
        "RHmean": annual_mean_qc(rh),
        "VPD_kPa": annual_mean_qc(vpd),
        "ea_kPa": annual_mean_qc(ea),
        "Td": annual_mean_qc(td),
    }).expand_dims(year=[int(year)])

    ann.attrs["ea_source"] = ea_source
    ds.close()
    return ann, ea_source


def compute_or_load_moisture(recompute=False):
    ann_path = OUT_DIR / "VPD_Circularity_independent_moisture_annual_1990_2024.nc"
    tr_path = OUT_DIR / "VPD_Circularity_independent_moisture_trends_1990_2024.nc"

    required_ann = ["Tmean", "RHmean", "VPD_kPa", "ea_kPa", "Td"]
    required_tr = []
    for v in required_ann:
        required_tr.extend([
            f"{v}_trend_decade",
            f"{v}_pvalue",
            f"{v}_lag1_autocorr_detrended",
            f"{v}_lag1_critical",
            f"{v}_MK_method_code",
        ])

    if ann_path.exists() and tr_path.exists() and not recompute:
        try:
            ann = standardize_lat_lon(safe_open_dataset(ann_path, decode_times=False))
            tr = standardize_lat_lon(safe_open_dataset(tr_path, decode_times=False))
            ok_ann = all(v in ann.data_vars for v in required_ann)
            ok_tr = all(v in tr.data_vars for v in required_tr)
            if ok_ann and ok_tr and tr.attrs.get("trend_significance_version") == TREND_SIGNIFICANCE_VERSION:
                print("[SKIP] Reusing compatible atmospheric-moisture diagnostic caches.")
                return ann, tr
            ann.close()
            tr.close()
            print("[WARN] Existing atmospheric-moisture caches are incompatible or incomplete; rebuilding.")
        except Exception as exc:
            print(f"[WARN] Could not validate moisture caches ({exc}); rebuilding.")

    annual_parts = []
    ea_sources = []
    for year, path in list_daily_files():
        print(f"[INFO] Moisture annual diagnostics: {year}")
        ann, source = process_year_moisture(path, year)
        annual_parts.append(ann)
        ea_sources.append(source)

    annual = xr.concat(annual_parts, dim="year")
    years = annual.year.values.astype(float)
    lat = annual.lat.values
    lon = annual.lon.values
    ny, nx = len(lat), len(lon)

    mask, _ = make_brazil_mask(lat, lon)
    mask_np = mask.values.astype(bool)

    trends = xr.Dataset(coords={"lat": lat, "lon": lon})

    for var in required_ann:
        print(f"[INFO] Trend calculation: {var}")
        arr = annual[var].values
        slope = np.full((ny, nx), np.nan, np.float32)
        pval = np.full((ny, nx), np.nan, np.float32)
        lag1 = np.full((ny, nx), np.nan, np.float32)
        crit = np.full((ny, nx), np.nan, np.float32)
        method = np.full((ny, nx), np.nan, np.float32)

        for iy in range(ny):
            if iy % 20 == 0:
                print(f"       {var}: latitude row {iy+1}/{ny}")
            for ix in np.where(mask_np[iy])[0]:
                sl, pv, r1, cr, mc = robust_trend(arr[:, iy, ix], years)
                slope[iy, ix] = sl
                pval[iy, ix] = pv
                lag1[iy, ix] = r1
                crit[iy, ix] = cr
                method[iy, ix] = mc

        trends[f"{var}_trend_decade"] = (("lat", "lon"), slope)
        trends[f"{var}_pvalue"] = (("lat", "lon"), pval)
        trends[f"{var}_lag1_autocorr_detrended"] = (("lat", "lon"), lag1)
        trends[f"{var}_lag1_critical"] = (("lat", "lon"), crit)
        trends[f"{var}_MK_method_code"] = (("lat", "lon"), method)

    annual.attrs.update({
        "description": "Auxiliary atmospheric-moisture diagnostics from Figure 01 daily cache.",
        "ea_source_priority": "dew point; RHmean and Tmean only as fallback",
        "ea_sources_observed": ", ".join(sorted(set(ea_sources))),
        "annual_validity": f">={MIN_VALID_DAYS_PER_YEAR} valid days/year",
        "script_version": SCRIPT_VERSION,
    })
    trends.attrs.update({
        "description": "Robust trends of auxiliary atmospheric-moisture diagnostics.",
        "trend_method": "Theil-Sen median slope per decade",
        "trend_significance_version": TREND_SIGNIFICANCE_VERSION,
        "minimum_valid_years": MIN_VALID_YEARS_FOR_TREND,
        "interpretation": "Thermodynamic diagnostic; not causal attribution.",
        "script_version": SCRIPT_VERSION,
    })

    enc_ann = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in annual.data_vars}
    enc_tr = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in trends.data_vars}
    safe_to_netcdf(annual, ann_path, encoding=enc_ann)
    safe_to_netcdf(trends, tr_path, encoding=enc_tr)

    print(f"[OK] Saved moisture annual diagnostics: {ann_path}")
    print(f"[OK] Saved moisture trends: {tr_path}")
    return annual, trends


# ============================================================
# Summary
# ============================================================

def spatial_summary(da, mask, weights):
    vals = da.where(mask).values.ravel()
    w = weights.where(mask).values.ravel()
    ok = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return {
            "N": 0, "median": np.nan, "p25": np.nan, "p75": np.nan,
            "mean": np.nan, "positive_area_pct": np.nan,
        }
    vals, w = vals[ok], w[ok]
    return {
        "N": int(len(vals)),
        "median": weighted_quantile(vals, w, 0.50),
        "p25": weighted_quantile(vals, w, 0.25),
        "p75": weighted_quantile(vals, w, 0.75),
        "mean": weighted_mean(vals, w),
        "positive_area_pct": float(100.0 * np.sum(w[vals > 0]) / np.sum(w)),
    }


def build_summary(tonly_tr, moisture_tr, mask, weights):
    rows = []

    metrics = [
        (
            "DHW-HHW Tmax-only intensity-trend contrast",
            tonly_tr["DHW_minus_HHW_TmaxOnly_intensity_trend_decade"],
            "Tmax-standardized severity decade-1",
        ),
        (
            "DHW-HHW duration-trend contrast",
            tonly_tr["DHW_minus_HHW_duration_trend_decade"],
            "days decade-1",
        ),
        (
            "DHW-HHW frequency-trend contrast",
            tonly_tr["DHW_minus_HHW_frequency_trend_decade"],
            "events decade-1",
        ),
        (
            "VPD trend",
            moisture_tr["VPD_kPa_trend_decade"],
            "kPa decade-1",
        ),
        (
            "Actual atmospheric vapour pressure trend",
            moisture_tr["ea_kPa_trend_decade"],
            "kPa decade-1",
        ),
        (
            "Dew-point trend",
            moisture_tr["Td_trend_decade"],
            "degC decade-1",
        ),
        (
            "Relative humidity trend",
            moisture_tr["RHmean_trend_decade"],
            "% decade-1",
        ),
    ]

    for label, da, units in metrics:
        s = spatial_summary(da, mask, weights)
        s.update({"metric": label, "units": units})
        rows.append(s)

    # Scientifically valid circularity sensitivity:
    # compare the primary DHW trend with the DHW Tmax-only trend.
    x = tonly_tr["DHW_intensity_trend_decade"].where(mask).values.ravel()
    y = tonly_tr["DHW_TmaxOnly_intensity_trend_decade"].where(mask).values.ravel()
    w = weights.where(mask).values.ravel()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    rho, p_perm = weighted_spearman(x[ok], y[ok], w[ok], seed=RANDOM_SEED)
    same_sign = float(
        100.0 * np.sum(w[ok][np.sign(x[ok]) == np.sign(y[ok])]) / np.sum(w[ok])
    )
    rows.append({
        "metric": "Primary DHW vs DHW Tmax-only trend agreement",
        "units": "association across grid cells; no subtraction",
        "N": int(ok.sum()),
        "median": np.nan,
        "p25": np.nan,
        "p75": np.nan,
        "mean": np.nan,
        "positive_area_pct": np.nan,
        "coslat_weighted_spearman_rho": rho,
        "permutation_p_value": p_perm,
        "n_permutations": N_PERMUTATIONS,
        "same_sign_area_pct": same_sign,
    })

    out = pd.DataFrame(rows)
    for c in [
        "coslat_weighted_spearman_rho",
        "permutation_p_value",
        "n_permutations",
        "same_sign_area_pct",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    csv = OUT_DIR / "VPD_Circularity_summary_statistics_1990_2024.csv"
    xlsx = OUT_DIR / "VPD_Circularity_summary_statistics_1990_2024.xlsx"
    out.to_csv(csv, index=False)
    out.to_excel(xlsx, index=False)

    print(f"[OK] Saved summary: {csv}")
    print(f"[OK] Saved summary: {xlsx}")
    return out


# ============================================================
# Plotting
# ============================================================

def add_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    return cb


def plot_map(ax, da, uf, title, norm, label, pval_da=None):
    im = ax.pcolormesh(
        da.lon, da.lat, da,
        cmap=plt.cm.RdBu_r,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    uf.boundary.plot(ax=ax, color="0.15", linewidth=0.45, zorder=5)

    # Stippling is used only when p-value belongs to THIS plotted trend,
    # never for a difference-of-slopes contrast.
    if pval_da is not None:
        sig_arr = (
            (pval_da.values <= P_THRESHOLD)
            & np.isfinite(da.values)
        )
        lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
        sy = max(1, sig_arr.shape[0] // 55)
        sx = max(1, sig_arr.shape[1] // 55)
        sub = sig_arr[::sy, ::sx]
        ax.scatter(
            lon2d[::sy, ::sx][sub],
            lat2d[::sy, ::sx][sub],
            s=1.0, c="black", alpha=0.40, linewidths=0,
            zorder=6, rasterized=True,
        )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.25, alpha=0.18)
    ax.set_title(title, fontweight="bold", pad=5)
    add_colorbar(ax.figure, ax, im, label)


def plot_dhw_sensitivity_hexbin(ax, trends, mask, weights):
    x = trends["DHW_intensity_trend_decade"].where(mask).values.ravel()
    y = trends["DHW_TmaxOnly_intensity_trend_decade"].where(mask).values.ravel()
    w = weights.where(mask).values.ravel()

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]

    hb = ax.hexbin(
        x, y,
        gridsize=55,
        bins="log",
        cmap="Greys",
        mincnt=1,
    )
    rho, p_perm = weighted_spearman(x, y, w)
    same = 100.0 * np.sum(w[np.sign(x) == np.sign(y)]) / np.sum(w)

    ax.axhline(0, color="0.45", linewidth=0.7)
    ax.axvline(0, color="0.45", linewidth=0.7)
    ax.text(
        0.04, 0.96,
        f"cos(latitude)-weighted ρ = {rho:.2f}\n"
        f"permutation p = {p_perm:.3f}\n"
        f"same-sign area = {same:.1f}%\n"
        f"N = {len(x):,}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="0.65", alpha=0.88),
    )

    ax.set_xlabel(
        "Primary DHW intensity trend\n"
        "(regime-specific standardized severity decade$^{-1}$)"
    )
    ax.set_ylabel(
        "DHW Tmax-only intensity trend\n"
        "(Tmax-standardized severity decade$^{-1}$)"
    )
    ax.set_title(
        "(b) DHW intensity sensitivity to excluding VPD",
        fontweight="bold",
        pad=5,
    )
    ax.grid(True, alpha=0.20, linewidth=0.35)
    add_colorbar(ax.figure, ax, hb, "grid-cell count")


def plot_figure(tonly_tr, moisture_tr, mask, weights, uf):
    print("[INFO] Plotting VPD circularity diagnostics.")

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(
        2, 3,
        left=0.055, right=0.975, bottom=0.075, top=0.955,
        wspace=0.40, hspace=0.40,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    plot_map(
        ax_a,
        tonly_tr["DHW_minus_HHW_TmaxOnly_intensity_trend_decade"].where(mask),
        uf,
        "(a) Common-scale DHW−HHW\nTmax-only intensity-trend contrast",
        TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6),
        "Tmax-standardized severity decade$^{-1}$",
        pval_da=None,
    )

    plot_dhw_sensitivity_hexbin(ax_b, tonly_tr, mask, weights)

    plot_map(
        ax_c,
        tonly_tr["DHW_minus_HHW_duration_trend_decade"].where(mask),
        uf,
        "(c) DHW−HHW duration-trend contrast",
        TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10),
        "days decade$^{-1}$",
        pval_da=None,
    )

    plot_map(
        ax_d,
        tonly_tr["DHW_minus_HHW_frequency_trend_decade"].where(mask),
        uf,
        "(d) DHW−HHW frequency-trend contrast",
        TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3),
        "events decade$^{-1}$",
        pval_da=None,
    )

    plot_map(
        ax_e,
        moisture_tr["ea_kPa_trend_decade"].where(mask),
        uf,
        "(e) Actual atmospheric vapour pressure trend",
        TwoSlopeNorm(vmin=-0.10, vcenter=0, vmax=0.10),
        "kPa decade$^{-1}$",
        pval_da=moisture_tr["ea_kPa_pvalue"],
    )

    plot_map(
        ax_f,
        moisture_tr["Td_trend_decade"].where(mask),
        uf,
        "(f) Dew-point trend",
        TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0),
        "°C decade$^{-1}$",
        pval_da=moisture_tr["Td_pvalue"],
    )

    fig.text(
        0.5, 0.015,
        "DHW−HHW panels are difference-of-slopes diagnostics; no standalone "
        "significance test is assigned to these contrasts. Black stippling in "
        "panels (e–f) indicates p ≤ 0.05 for the plotted moisture trend itself.",
        ha="center", va="bottom", fontsize=9,
    )

    out_base = OUT_DIR / "Supplementary_VPD_Circularity_Diagnostics_1990_2024"
    fig.savefig(
        str(out_base) + ".pdf",
        dpi=450, bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        str(out_base) + ".jpeg",
        dpi=450, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    print(f"[OK] Saved figure: {out_base}.pdf")
    print(f"[OK] Saved figure: {out_base}.jpeg")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    configure_paths(args)

    print(f"[START] VPD circularity diagnostics | {pd.Timestamp.now().isoformat()}")
    print("[INFO] Figure 01 heatwave metrics are read directly; they are not re-detected.")
    print("[INFO] Primary DHW and HHW regime-specific intensity trends are never subtracted.")
    print("[INFO] Interpretation: sensitivity/thermodynamic diagnostics; not causal attribution.")

    tonly_ann, tonly_tr = load_figure1_products()
    save_figure1_subset_products(tonly_ann, tonly_tr)

    mask, uf = make_brazil_mask(tonly_tr.lat.values, tonly_tr.lon.values)
    weights = grid_area_weights(tonly_tr.lat.values, tonly_tr.lon.values)
    print(f"[INFO] Brazil mask grid cells: {int(mask.sum())}")

    moisture_ann, moisture_tr = compute_or_load_moisture(
        recompute=args.recompute_moisture
    )

    # Require exact grid alignment. These diagnostics are generated from the
    # same Figure 01 daily cache, so spatial interpolation should not be needed.
    if (
        moisture_tr.sizes.get("lat") != tonly_tr.sizes.get("lat")
        or moisture_tr.sizes.get("lon") != tonly_tr.sizes.get("lon")
        or not np.allclose(moisture_tr.lat.values, tonly_tr.lat.values)
        or not np.allclose(moisture_tr.lon.values, tonly_tr.lon.values)
    ):
        raise RuntimeError(
            "Atmospheric-moisture diagnostics are not exactly aligned with "
            "the Figure 01 grid. Spatial interpolation is not used in this "
            "sensitivity workflow."
        )

    summary = build_summary(tonly_tr, moisture_tr, mask, weights)

    print("\n[SUMMARY]")
    for _, row in summary.iterrows():
        metric = row["metric"]
        if metric == "Primary DHW vs DHW Tmax-only trend agreement":
            print(
                f"  {metric}: "
                f"rho={row['coslat_weighted_spearman_rho']:+.3f}, "
                f"p_perm={row['permutation_p_value']:.3f}, "
                f"same-sign area={row['same_sign_area_pct']:.1f}%"
            )
        else:
            print(
                f"  {metric}: median={row['median']:+.3f}, "
                f"positive area={row['positive_area_pct']:.1f}%"
            )

    plot_figure(tonly_tr, moisture_tr, mask, weights, uf)

    tonly_ann.close()
    tonly_tr.close()
    moisture_ann.close()
    moisture_tr.close()

    print("\n[DONE] Outputs saved in:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

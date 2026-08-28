#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary VPD decomposition diagnostics, 1990–2024.

Purpose
-------
Separate the thermodynamic components of vapour-pressure-deficit (VPD) change:

    VPD = es(T) - ea

where:

    es(T) = temperature-controlled saturation vapour pressure
    ea    = actual atmospheric vapour pressure

The analysis distinguishes changes in saturation demand from changes in actual
atmospheric moisture content and compares these diagnostics with the common-scale
Figure 01 DHW−HHW Tmax-only intensity-trend contrast.

The primary cross-regime comparison is:

    DHW_minus_HHW_TmaxOnly_intensity_trend_decade

Primary regime-specific DHW and HHW intensity trends are not subtracted.

Atmospheric-moisture derivation
-------------------------------
Actual atmospheric vapour pressure is derived from dew-point temperature when
available. If dew point is unavailable, RH and Tmean are used. ea is not derived
from VPD in this diagnostic because doing so would make the moisture component
algebraically dependent on the quantity being decomposed.

Annual diagnostics include direct VPD computed as es - ea and the VPD field
stored in the Figure 01 daily cache. Their difference is retained as a closure/
aggregation diagnostic.

Trend decomposition
-------------------
The annual-state identity VPD = es - ea is exact. Theil–Sen trend slopes are
nonlinear estimators, so:

    beta_VPD

is retained as the directly estimated VPD trend, while:

    beta_es - beta_ea

is reported separately as a component-slope approximation. Their difference is
stored as a trend-closure residual.

Statistical analysis
--------------------
Trend magnitude uses Theil–Sen median slope per decade. Significance uses the
original two-sided Mann–Kendall test unless detrended lag-1 rank
autocorrelation is significant; Hamed–Rao lag-1 correction is then used.
Annual values require at least 300 valid days and trends require at least
20 valid years.

Spatial summaries use cos(latitude) weights. Spatial rank associations use
cos(latitude)-weighted Spearman correlation with 999 two-sided permutations.

These diagnostics are descriptive thermodynamic analyses and are not interpreted
as causal attribution.

Usage
-----
python supplementary_vpd_decomposition_diagnostics.py \
    --figure1-output-dir /path/to/figure_01_outputs \
    --brazil-shapefile /path/to/brazil_boundary.shp \
    --biomes-shapefile /path/to/brazil_biomes.shp \
    --output-dir ./outputs/vpd_decomposition

The biome shapefile is optional. Use ``--recompute`` to rebuild annual and trend
products and ignore compatible caches.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
from scipy.stats import theilslopes, norm
from shapely.geometry import Point

try:
    import pymannkendall as mk
    _HAS_PYMK = True
except Exception:
    _HAS_PYMK = False

# ============================================================
# Runtime paths and configuration
# ============================================================

FIGURE1_ROOT = None
DAILY_DIR = None
FIG1_TRENDS_NC = None
BRAZIL_SHP = None
BIOMES_SHP = None
OUT_DIR = None
TREND_CACHE_DIR = None

YEAR0, YEAR1 = 1990, 2024
MIN_YEARS_FOR_TREND = 20
MIN_VALID_DAYS_YEAR = 300

LON_MIN, LON_MAX = -75.0, -32.0
LAT_MIN, LAT_MAX = -35.0, 6.0

FIG_DPI = 450
RANDOM_SEED = 42
N_PERMUTATIONS = 999
AUTOCORR_ALPHA = 0.05
TREND_SIGNIFICANCE_VERSION = "Theil-Sen + conditional Hamed-Rao MK lag1"

# Diagnostic tolerances [kPa decade-1].
EA_STABLE_TOL = 0.01
VPD_POS_TOL = 0.01

# Common colour scale for vapour-pressure trend components.
VP_TREND_VMIN = -0.20
VP_TREND_VCENTER = 0.0
VP_TREND_VMAX = 0.40


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Supplementary VPD decomposition diagnostics using Figure 01 "
            "daily and trend products."
        )
    )
    parser.add_argument(
        "--figure1-output-dir",
        required=True,
        help="Root directory produced by figure_01_heatwave_trends.py.",
    )
    parser.add_argument(
        "--brazil-shapefile",
        required=True,
        help="Brazil boundary or state shapefile.",
    )
    parser.add_argument(
        "--biomes-shapefile",
        default=None,
        help="Optional Brazilian biome shapefile.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rebuild annual and trend decomposition products.",
    )
    return parser.parse_args()


def configure_paths(args):
    global FIGURE1_ROOT, DAILY_DIR, FIG1_TRENDS_NC
    global BRAZIL_SHP, BIOMES_SHP, OUT_DIR, TREND_CACHE_DIR

    FIGURE1_ROOT = Path(args.figure1_output_dir).expanduser().resolve()
    DAILY_DIR = FIGURE1_ROOT / "cache" / "era5_daily"
    FIG1_TRENDS_NC = (
        FIGURE1_ROOT / "data" / "figure_01_heatwave_trends_1990_2024.nc"
    )
    BRAZIL_SHP = Path(args.brazil_shapefile).expanduser().resolve()
    BIOMES_SHP = (
        Path(args.biomes_shapefile).expanduser().resolve()
        if args.biomes_shapefile else None
    )
    OUT_DIR = Path(args.output_dir).expanduser().resolve()
    TREND_CACHE_DIR = OUT_DIR / "trend_variable_cache"

    required = [
        (DAILY_DIR, "Figure 01 daily cache"),
        (FIG1_TRENDS_NC, "Figure 01 trend file"),
        (BRAZIL_SHP, "Brazil shapefile"),
    ]
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if BIOMES_SHP is not None and not BIOMES_SHP.exists():
        raise FileNotFoundError(f"Biome shapefile not found: {BIOMES_SHP}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TREND_CACHE_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ============================================================
# NetCDF I/O helpers
# ============================================================

def _available_netcdf_engines():
    engines = xr.backends.list_engines()
    return [name for name in ("h5netcdf", "netcdf4", "scipy") if name in engines]

def safe_open_dataset(path, *, decode_times=True, chunks=None, **kwargs):
    errors=[]
    for engine in _available_netcdf_engines():
        try:
            return xr.open_dataset(path, engine=engine, decode_times=decode_times, chunks=chunks, **kwargs)
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not open NetCDF file: {path}\n" + "\n".join(errors))

def safe_to_netcdf(ds, path, *, encoding=None, **kwargs):
    errors=[]
    engines=_available_netcdf_engines()
    for engine in ("h5netcdf","netcdf4"):
        if engine not in engines:
            continue
        try:
            return ds.to_netcdf(path, engine=engine, encoding=encoding, **kwargs)
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    if "scipy" in engines:
        enc=None
        if encoding:
            enc={var:{k:v for k,v in opts.items() if k in {"dtype","_FillValue","scale_factor","add_offset"}} for var,opts in encoding.items()}
        try:
            return ds.to_netcdf(path, engine="scipy", encoding=enc, **kwargs)
        except Exception as exc:
            errors.append(f"scipy: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not write NetCDF file: {path}\n" + "\n".join(errors))

# ============================================================
# Utility functions
# ============================================================


def standardize_lat_lon(ds):
    ren = {}
    for c in ds.coords:
        if c == "latitude": ren[c] = "lat"
        if c == "longitude": ren[c] = "lon"
    for d in ds.dims:
        if d == "latitude": ren[d] = "lat"
        if d == "longitude": ren[d] = "lon"
    if ren:
        ds = ds.rename(ren)
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    if "lat" in ds.coords and ds.lat[0] < ds.lat[-1]:
        ds = ds.sortby("lat", ascending=False)
    return ds


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
    vals = da.values if da.size < 100 else da.isel({da.dims[0]: 0}).values if da.ndim > 0 else da.values
    try:
        med = float(np.nanmedian(vals))
    except Exception:
        med = np.nan
    if np.isfinite(med) and med > 100:
        return da - 273.15
    return da


def saturation_vapour_pressure_kpa(temp_c):
    """Saturation vapour pressure over water [kPa], Bolton/Tetens approximation."""
    return 0.6112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def dewpoint_from_ea_kpa(ea_kpa):
    """Dew-point temperature [C] from actual vapour pressure [kPa]."""
    ea = np.maximum(ea_kpa, 1e-6)
    ln = np.log(ea / 0.6112)
    return (243.5 * ln) / (17.67 - ln)


def infer_and_convert_vpd_to_kpa(vpd_da, source_name="VPD"):
    """Convert VPD to kPa using metadata first and Figure 01 conventions second."""
    units = str(vpd_da.attrs.get("units", "")).lower().strip()

    if "kpa" in units:
        return vpd_da.astype(float), "metadata_kPa"
    if "hpa" in units or "hectopascal" in units:
        return vpd_da.astype(float) / 10.0, "metadata_hPa_to_kPa"

    # Figure 01 daily-cache VPD variables are stored internally in hPa.
    if source_name in {"VPDmean", "VPDmax"}:
        return vpd_da.astype(float) / 10.0, "figure01_hPa_to_kPa"

    sample = vpd_da
    if "time" in sample.dims:
        sample = sample.isel(time=slice(0, min(30, sample.sizes["time"])))
    med = float(np.nanmedian(sample.values))
    if np.isfinite(med) and med > 5.0:
        return vpd_da.astype(float) / 10.0, "magnitude_hPa_to_kPa"

    if "kpa" in source_name.lower():
        return vpd_da.astype(float), "variable_name_kPa"

    raise ValueError(
        f"Ambiguous VPD units for {source_name!r}; "
        f"units={vpd_da.attrs.get('units', '')!r}"
    )


def make_brazil_mask_from_grid(lat, lon, shp_path):
    uf = gpd.read_file(shp_path).to_crs(epsg=4326)
    geom = uf.dissolve().geometry.iloc[0]
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, la in enumerate(lat):
        pts = [Point(float(lo), float(la)) for lo in lon]
        mask[i, :] = [geom.contains(p) or geom.touches(p) for p in pts]
    return xr.DataArray(mask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")), uf


def theil_sen_slope_decade(years, y):
    x=np.asarray(years,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<MIN_YEARS_FOR_TREND: return np.nan
    return float(theilslopes(y[ok],x[ok],alpha=0.05)[0]*10.0)

def _mk_score_and_variance(y):
    y=np.asarray(y,float); n=len(y)
    if n<2: return np.nan,np.nan
    s=0.0
    for k in range(n-1): s += np.sign(y[k+1:]-y[k]).sum()
    _,counts=np.unique(y,return_counts=True)
    tie=np.sum(counts*(counts-1)*(2*counts+5))
    var_s=(n*(n-1)*(2*n+5)-tie)/18.0
    return float(s),float(var_s)

def _mk_p_from_s_var(s,var_s):
    if not np.isfinite(s) or not np.isfinite(var_s) or var_s<=0: return np.nan
    z=(s-1)/np.sqrt(var_s) if s>0 else ((s+1)/np.sqrt(var_s) if s<0 else 0.0)
    return float(2*(1-norm.cdf(abs(z))))

def mann_kendall_pvalue(y,min_valid_years=MIN_YEARS_FOR_TREND):
    y=np.asarray(y,float); y=y[np.isfinite(y)]
    if len(y)<min_valid_years: return np.nan
    if _HAS_PYMK:
        try: return float(mk.original_test(y).p)
        except Exception: pass
    return _mk_p_from_s_var(*_mk_score_and_variance(y))

def detrended_lag1_autocorrelation(y,years,min_valid_years=MIN_YEARS_FOR_TREND,alpha=AUTOCORR_ALPHA):
    y=np.asarray(y,float); years=np.asarray(years,float)
    ok=np.isfinite(y)&np.isfinite(years); y,years=y[ok],years[ok]; n=len(y)
    if n<min_valid_years or n<4: return np.nan,np.nan,False
    crit=float(norm.ppf(1-alpha/2)/np.sqrt(n))
    if np.nanstd(y)<=1e-12: return 0.0,crit,False
    sen=theilslopes(y,years,alpha=0.05); slope,intercept=float(sen[0]),float(sen[1])
    residual=y-(intercept+slope*years)
    ranks=pd.Series(residual).rank(method="average").to_numpy(float)
    r1=0.0 if np.std(ranks[:-1])<=1e-12 or np.std(ranks[1:])<=1e-12 else float(np.corrcoef(ranks[:-1],ranks[1:])[0,1])
    return r1,crit,bool(np.isfinite(r1) and abs(r1)>crit)

def hamed_rao_mk_pvalue_lag1(y,years,min_valid_years=MIN_YEARS_FOR_TREND,alpha=AUTOCORR_ALPHA):
    y=np.asarray(y,float); years=np.asarray(years,float)
    ok=np.isfinite(y)&np.isfinite(years); y,years=y[ok],years[ok]
    if len(y)<min_valid_years: return np.nan
    if _HAS_PYMK:
        try: return float(mk.hamed_rao_modification_test(y,alpha=alpha,lag=1).p)
        except Exception: pass
    s,var_s=_mk_score_and_variance(y)
    if not np.isfinite(var_s) or var_s<=0: return np.nan
    n=len(y); sen=theilslopes(y,years,alpha=0.05); slope,intercept=float(sen[0]),float(sen[1])
    detrended=y-(intercept+slope*years)
    ranks=pd.Series(detrended).rank(method="average").to_numpy(float)
    rho1=0.0 if np.std(ranks[:-1])<=1e-12 or np.std(ranks[1:])<=1e-12 else float(np.corrcoef(ranks[:-1],ranks[1:])[0,1])
    critical=float(norm.ppf(1-alpha/2)/np.sqrt(n))
    if abs(rho1)<=critical: return _mk_p_from_s_var(s,var_s)
    correction=max(1.0+(2.0*(n-3)/n)*abs(rho1),1.0)
    return _mk_p_from_s_var(s,var_s*correction)

def robust_trend_per_decade(y,years):
    y=np.asarray(y,float); years=np.asarray(years,float)
    ok=np.isfinite(y)&np.isfinite(years); yv,xv=y[ok],years[ok]
    if len(yv)<MIN_YEARS_FOR_TREND: return np.nan,np.nan,np.nan,np.nan,np.nan
    slope=float(theilslopes(yv,xv,alpha=0.05)[0]*10.0)
    original_p=mann_kendall_pvalue(yv)
    r1,crit,sig=detrended_lag1_autocorrelation(yv,xv)
    p=hamed_rao_mk_pvalue_lag1(yv,xv) if sig else original_p
    return slope,p,r1,crit,(1.0 if sig else 0.0)

def _weighted_rank_correlation(x,y,w):
    xranks=stats.rankdata(x); yranks=stats.rankdata(y)
    mx=np.average(xranks,weights=w); my=np.average(yranks,weights=w)
    cov=np.average((xranks-mx)*(yranks-my),weights=w)
    vx=np.average((xranks-mx)**2,weights=w); vy=np.average((yranks-my)**2,weights=w)
    return np.nan if vx<=0 or vy<=0 else float(cov/np.sqrt(vx*vy))

def weighted_spearman(x,y,w,n_permutations=N_PERMUTATIONS,seed=RANDOM_SEED):
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    if ok.sum()<10: return np.nan,np.nan
    x,y,w=x[ok],y[ok],w[ok]; obs=_weighted_rank_correlation(x,y,w)
    if not np.isfinite(obs): return np.nan,np.nan
    rng=np.random.default_rng(seed); extreme=0; valid=0
    for _ in range(int(n_permutations)):
        rp=_weighted_rank_correlation(x,rng.permutation(y),w)
        if not np.isfinite(rp): continue
        valid+=1; extreme += int(abs(rp)>=abs(obs))
    return float(obs), float((extreme+1)/(valid+1)) if valid else np.nan

def weighted_quantile(values,weights,q):
    values=np.asarray(values,float); weights=np.asarray(weights,float)
    ok=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    if not np.any(ok): return np.nan
    values,weights=values[ok],weights[ok]; order=np.argsort(values); values,weights=values[order],weights[order]
    cdf=np.cumsum(weights)/np.sum(weights)
    return float(values[np.searchsorted(cdf,q)])


# ============================================================
# Annual VPD decomposition metrics
# ============================================================

def list_daily_files():
    files = []
    for year in range(YEAR0, YEAR1 + 1):
        f = os.path.join(DAILY_DIR, f"ERA5_daily_Brazil_{year}.nc")
        if os.path.exists(f):
            files.append((year, f))
        else:
            print(f"[WARN] Missing daily file for {year}: {f}")
    if len(files) < MIN_YEARS_FOR_TREND:
        raise RuntimeError(f"Only {len(files)} daily files found; need >= {MIN_YEARS_FOR_TREND}.")
    return files


def compute_annual_decomposition(overwrite=False):
    out_nc = os.path.join(OUT_DIR, "VPD_Decomposition_annual_metrics_1990_2024.nc")
    if os.path.exists(out_nc) and not overwrite:
        print(f"[INFO] Annual decomposition cache exists: {out_nc}")
        cached = safe_open_dataset(out_nc)
        required_base = [
            "Tmean_annual", "Tmax_annual", "es_annual", "ea_annual",
            "VPD_decomp_annual", "VPD_input_annual",
            "RH_component_annual", "Td_proxy_annual",
        ]
        missing_base = [v for v in required_base if v not in cached.data_vars]
        if missing_base:
            print(f"[WARN] Annual cache missing base variables {missing_base}; rebuilding from daily inputs.")
            cached.close()
            try:
                os.remove(out_nc)
            except OSError:
                pass
        else:
            if "VPD_decomp_minus_input_annual" not in cached.data_vars:
                print("[INFO] Upgrading annual cache in memory: deriving VPD_decomp_minus_input_annual.")
                cached = cached.assign(
                    VPD_decomp_minus_input_annual=(cached["VPD_decomp_annual"] - cached["VPD_input_annual"])
                )
                cached["VPD_decomp_minus_input_annual"].attrs.update({
                    "definition": "VPD_decomp_annual - VPD_input_annual",
                    "units": "kPa",
                    "purpose": "closure diagnostic",
                })
            print("[SKIP] Reusing validated/upgraded annual decomposition cache.")
            return cached

    files = list_daily_files()
    print(f"[INFO] Computing annual VPD decomposition from {len(files)} daily files")

    annual_list = []
    source_note = []
    vpd_unit_notes = []

    for year, path in files:
        print(f"       Annual decomposition metrics: {year}")
        ds = safe_open_dataset(path, decode_times=True, chunks=None)
        ds = standardize_lat_lon(ds)
        ds = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MAX, LAT_MIN))

        tmean_var = find_var(ds, ["Tmean", "tmean", "t2m_mean", "t2m", "var167"])
        tmax_var = find_var(ds, ["Tmax", "tmax"])
        rh_var = find_var(ds, ["RHmean", "rhmean", "RH", "relative_humidity"])
        vpd_var = find_var(ds, ["VPDmean", "vpdmean", "VPD", "vpd", "VPD_kPa", "vpd_kpa"])
        td_var = find_var(ds, ["Tdmean", "D2mean", "d2m_mean", "dewpoint_mean", "d2m", "var168"])

        if tmean_var is None:
            raise ValueError(f"No Tmean/t2m variable found in {path}. Available: {list(ds.data_vars)}")

        Tmean = to_celsius(ds[tmean_var])
        Tmax = to_celsius(ds[tmax_var]) if tmax_var else Tmean
        es = saturation_vapour_pressure_kpa(Tmean)
        es.name = "es"

        if td_var is not None:
            Td = to_celsius(ds[td_var])
            ea = saturation_vapour_pressure_kpa(Td)
            ea_source = "dewpoint"
        elif rh_var is not None:
            RH = ds[rh_var].astype(float)
            ea = es * (RH / 100.0)
            Td = xr.apply_ufunc(dewpoint_from_ea_kpa, ea, dask="allowed")
            ea_source = "RHmean_Tmean_fallback"
        else:
            ds.close()
            raise ValueError(
                f"Cannot derive actual atmospheric vapour pressure in {path}; "
                "dew point or RH is required. ea is not reconstructed from VPD "
                "in this decomposition diagnostic."
            )

        if vpd_var is not None:
            VPD_input, unit_note = infer_and_convert_vpd_to_kpa(
                ds[vpd_var],
                source_name=vpd_var,
            )
            vpd_unit_notes.append(unit_note)
        else:
            VPD_input = es - ea
            unit_note = "computed_from_es_minus_ea"
            vpd_unit_notes.append(unit_note)

        VPD_decomp = es - ea
        VPD_decomp = xr.where(VPD_decomp >= 0, VPD_decomp, 0.0)

        RH_from_components = 100.0 * ea / es
        RH_from_components = xr.where((RH_from_components >= 0) & (RH_from_components <= 100), RH_from_components, np.nan)

        # Annual means. A valid annual value requires at least MIN_VALID_DAYS_YEAR valid daily values.
        def annual_mean_with_qc(da):
            count = da.count("time")
            return da.mean("time", skipna=True).where(count >= MIN_VALID_DAYS_YEAR)

        ann = xr.Dataset({
            "Tmean_annual": annual_mean_with_qc(Tmean),
            "Tmax_annual": annual_mean_with_qc(Tmax),
            "es_annual": annual_mean_with_qc(es),
            "ea_annual": annual_mean_with_qc(ea),
            "VPD_decomp_annual": annual_mean_with_qc(VPD_decomp),
            "VPD_input_annual": annual_mean_with_qc(VPD_input),
            "VPD_decomp_minus_input_annual": annual_mean_with_qc(VPD_decomp) - annual_mean_with_qc(VPD_input),
            "RH_component_annual": annual_mean_with_qc(RH_from_components),
            "Td_proxy_annual": annual_mean_with_qc(Td),
        }).expand_dims(year=[year])

        annual_list.append(ann)
        source_note.append(ea_source)
        ds.close()

    out = xr.concat(annual_list, dim="year")
    out.attrs.update({
        "analysis": "VPD decomposition into saturation vapour pressure and actual vapour pressure components",
        "period": f"{YEAR0}-{YEAR1}",
        "units": "T variables in degC; vapour pressure variables in kPa",
        "ea_source_by_priority": "dew point if available; otherwise RHmean and Tmean",
        "ea_sources_observed": ", ".join(sorted(set(source_note))),
        "vpd_unit_handling_observed": ", ".join(sorted(set(vpd_unit_notes))),
        "interpretation": ("Grid-scale thermodynamic diagnostic; not causal attribution. "
            "Differences between VPD_decomp and the stored VPD field may arise from nonlinear es(T), daily aggregation, or source-variable definitions."),
    })
    comp = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars}
    safe_to_netcdf(out, out_nc, encoding=comp)
    print(f"[OK] Saved annual decomposition metrics: {out_nc}")
    return out


def _trend_cache_path(var):
    return os.path.join(TREND_CACHE_DIR, f"{var}_trend.nc")


def _load_valid_trend_cache(var, lat, lon):
    path = _trend_cache_path(var)
    if not os.path.exists(path):
        return None
    try:
        ds = safe_open_dataset(path, decode_times=False)
        stem = var.replace("_annual", "")
        required = [f"{stem}_trend_decade", f"{stem}_pvalue", f"{stem}_lag1_rank_autocorr", f"{stem}_lag1_critical", f"{stem}_mk_method_code"]
        missing = [v for v in required if v not in ds.data_vars]
        grid_ok = (ds.sizes.get("lat") == len(lat) and ds.sizes.get("lon") == len(lon)
                   and np.allclose(ds["lat"].values, lat, equal_nan=True)
                   and np.allclose(ds["lon"].values, lon, equal_nan=True))
        if missing or not grid_ok:
            ds.close()
            print(f"[WARN] Ignoring incompatible trend cache for {var}: missing={missing}, grid_ok={grid_ok}")
            return None
        print(f"[SKIP] Reusing completed trend cache for {var}")
        loaded = ds.load(); ds.close(); return loaded
    except Exception as exc:
        print(f"[WARN] Could not reuse trend cache for {var}: {exc}")
        return None


def _save_trend_cache(var, lat, lon, slope, pval, lag1, lag1crit, mkmethod):
    stem = var.replace("_annual", "")
    ds = xr.Dataset({
        f"{stem}_trend_decade": (("lat", "lon"), slope.astype(np.float32)),
        f"{stem}_pvalue": (("lat", "lon"), pval.astype(np.float32)),
        f"{stem}_lag1_rank_autocorr": (("lat", "lon"), lag1.astype(np.float32)),
        f"{stem}_lag1_critical": (("lat", "lon"), lag1crit.astype(np.float32)),
        f"{stem}_mk_method_code": (("lat", "lon"), mkmethod.astype(np.float32)),
    }, coords={"lat": lat, "lon": lon})
    ds.attrs.update({"source_variable": var, "trend_method": TREND_SIGNIFICANCE_VERSION,
                     "period": f"{YEAR0}-{YEAR1}", "minimum_valid_years": MIN_YEARS_FOR_TREND,
                     "cache_purpose": "per-variable restartable trend calculation"})
    path = _trend_cache_path(var); tmp = path + ".tmp"
    safe_to_netcdf(ds, tmp); os.replace(tmp, path); ds.close()
    print(f"[OK] Saved restartable trend cache: {path}")


def compute_trends(ds_ann, mask_brazil, overwrite=False):
    out_nc = os.path.join(OUT_DIR, "VPD_Decomposition_trends_1990_2024.nc")
    if os.path.exists(out_nc) and not overwrite:
        try:
            ds_cached = safe_open_dataset(out_nc)
            required_cached = [
                "brazil_mask",
                "VPD_increase_source_class",
                "es_trend_decade",
                "ea_trend_decade",
                "VPD_decomp_trend_decade",
                "VPD_component_sum_trend_decade",
                "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
            ]
            missing_cached = [v for v in required_cached if v not in ds_cached.data_vars]
            if missing_cached:
                print(
                    f"[WARN] Existing decomposition trends file is incomplete; "
                    f"missing {missing_cached}. Rebuilding."
                )
                ds_cached.close()
                os.remove(out_nc)
            else:
                print(f"[SKIP] Decomposition trends file exists and passed validation: {out_nc}")
                return ds_cached
        except Exception as exc:
            print(f"[WARN] Existing decomposition trend file could not be opened or validated ({exc}); rebuilding.")
            try:
                os.remove(out_nc)
            except OSError:
                pass

    years = ds_ann.year.values.astype(float)
    lat = ds_ann.lat.values
    lon = ds_ann.lon.values
    nlat, nlon = len(lat), len(lon)

    variables = [
        "Tmean_annual", "Tmax_annual", "es_annual", "ea_annual",
        "VPD_decomp_annual", "VPD_input_annual", "VPD_decomp_minus_input_annual",
        "RH_component_annual", "Td_proxy_annual"
    ]

    out = xr.Dataset(coords={"lat": lat, "lon": lon})

    for var in variables:
        if var not in ds_ann.data_vars:
            raise KeyError(f"Annual variable {var!r} unavailable after cache validation. Available: {list(ds_ann.data_vars)}")

        cached_var = None if overwrite else _load_valid_trend_cache(var, lat, lon)
        if cached_var is not None:
            for v in cached_var.data_vars:
                out[v] = cached_var[v]
            cached_var.close()
            continue

        print(f"[INFO] Trend calculation: {var}")
        arr = ds_ann[var].values
        slope=np.full((nlat,nlon),np.nan,np.float32); pval=np.full((nlat,nlon),np.nan,np.float32)
        lag1=np.full((nlat,nlon),np.nan,np.float32); lag1crit=np.full((nlat,nlon),np.nan,np.float32); mkmethod=np.full((nlat,nlon),np.nan,np.float32)
        for i in range(nlat):
            if i % 20 == 0: print(f"       {var}: latitude row {i+1}/{nlat}")
            for j in range(nlon):
                if not bool(mask_brazil.values[i,j]): continue
                sl,pv,r1,crit,method=robust_trend_per_decade(arr[:,i,j],years)
                slope[i,j]=sl; pval[i,j]=pv; lag1[i,j]=r1; lag1crit[i,j]=crit; mkmethod[i,j]=method

        _save_trend_cache(var, lat, lon, slope, pval, lag1, lag1crit, mkmethod)

        stem=var.replace("_annual","")
        out[f"{stem}_trend_decade"] = (("lat","lon"),slope)
        out[f"{stem}_pvalue"] = (("lat","lon"),pval)
        out[f"{stem}_lag1_rank_autocorr"] = (("lat","lon"),lag1)
        out[f"{stem}_lag1_critical"] = (("lat","lon"),lag1crit)
        out[f"{stem}_mk_method_code"] = (("lat","lon"),mkmethod)

    # The annual-state identity VPD_decomp = es - ea is exact, but Theil-Sen
    # slopes are nonlinear and therefore are not exactly additive. Preserve both
    # the directly estimated robust VPD trend and the component-slope approximation.
    out["VPD_component_sum_trend_decade"] = out["es_trend_decade"] - out["ea_trend_decade"]
    out["VPD_component_closure_residual_trend_decade"] = out["VPD_decomp_trend_decade"] - out["VPD_component_sum_trend_decade"]
    out["actual_moisture_contribution_trend_decade"] = -1.0 * out["ea_trend_decade"]

    # Classification of positive VPD trends.
    esb = out["es_trend_decade"].values
    eab = out["ea_trend_decade"].values
    vpdb = out["VPD_decomp_trend_decade"].values
    cls = np.full(esb.shape, 0, dtype=np.int16)
    # Positive VPD trends are partitioned by the sign of actual vapour pressure.
    pos = vpdb > VPD_POS_TOL
    cls[pos & (eab > EA_STABLE_TOL)] = 1
    cls[pos & (np.abs(eab) <= EA_STABLE_TOL)] = 2
    cls[pos & (eab < -EA_STABLE_TOL)] = 3
    cls[~mask_brazil.values] = -1
    out["VPD_increase_source_class"] = (("lat", "lon"), cls)
    # NetCDF attribute names cannot start with a minus sign or contain some special
    # characters. Use valid attribute keys and store the numeric class labels in values.
    out["VPD_increase_source_class"].attrs.update({
        "class_minus1": "-1 = outside Brazil mask",
        "class_0": "0 = no positive VPD increase or undefined",
        "class_1": "1 = VPD increases while ea also increases; saturation demand rises faster than moisture content",
        "class_2": "2 = VPD increases while ea is approximately stable",
        "class_3": "3 = VPD increases while ea decreases; actual atmospheric moisture content declines",
        "classification_note": "Thermodynamic diagnostic classes; not source attribution or causal attribution.",
    })

    # Add common-scale Tmax-only DHW-HHW contrast from Figure 1.
    if not os.path.exists(FIG1_TRENDS_NC):
        raise FileNotFoundError(f"Figure 01 trend file not found: {FIG1_TRENDS_NC}")
    f1 = standardize_lat_lon(safe_open_dataset(FIG1_TRENDS_NC))
    required_f1=["DHW_minus_HHW_TmaxOnly_intensity_trend_decade","DHW_TmaxOnly_intensity_trend_decade","HHW_TmaxOnly_intensity_trend_decade","DHW_TmaxOnly_intensity_pvalue"]
    missing=[v for v in required_f1 if v not in f1.data_vars]
    if missing:
        f1.close(); raise ValueError("Figure 01 product is not the common-scale workflow. Missing: "+", ".join(missing))
    try:
        f1_aligned = f1[required_f1].sel(
            lat=xr.DataArray(out.lat.values, dims="lat"),
            lon=xr.DataArray(out.lon.values, dims="lon"),
        ).load()
    except Exception as exc:
        f1.close()
        raise RuntimeError(
            "Figure 01 common-scale fields cannot be selected on the exact "
            "decomposition grid. Spatial interpolation is not used. "
            f"Details: {exc}"
        )

    if not np.allclose(f1_aligned.lat.values, out.lat.values):
        f1.close()
        raise RuntimeError("Figure 01 and decomposition latitude coordinates are not aligned.")
    if not np.allclose(f1_aligned.lon.values, out.lon.values):
        f1.close()
        raise RuntimeError("Figure 01 and decomposition longitude coordinates are not aligned.")

    contrast = f1_aligned[
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    ]
    out["DHW_minus_HHW_TmaxOnly_intensity_trend_decade"] = contrast
    out["dry_minus_humid_intensity_trend"] = contrast
    out["DHW_TmaxOnly_intensity_trend_decade"] = f1_aligned[
        "DHW_TmaxOnly_intensity_trend_decade"
    ]
    out["DHW_TmaxOnly_intensity_pvalue"] = f1_aligned[
        "DHW_TmaxOnly_intensity_pvalue"
    ]
    f1.close()

    out["brazil_mask"] = mask_brazil.astype(np.int8)
    out.attrs.update({
        "analysis": "VPD trend decomposition",
        "trend_method": TREND_SIGNIFICANCE_VERSION,
        "decomposition": "Annual-state VPD_decomp = es - ea; direct Theil-Sen VPD trend retained separately from beta_es - beta_ea because robust slopes are not exactly additive.",
        "interpretation": "es reflects temperature-controlled saturation demand; ea reflects actual atmospheric vapour-pressure change. Diagnostic, not causal attribution.",
        "units": "kPa decade-1 for vapour-pressure trends; degC decade-1 for T/Td trends; % decade-1 for RH; Tmax-standardized severity decade-1 for DHW-HHW.",
        "cross_regime_metric": "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "trend_significance_version": TREND_SIGNIFICANCE_VERSION,
    })
    # Save with conservative NetCDF-safe encoding. Compression is applied to all
    # variables, while native dtypes are preserved to avoid unintended casting.
    comp = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    safe_to_netcdf(out, out_nc, encoding=comp)
    print(f"[OK] Saved decomposition trends: {out_nc}")
    return out

# ============================================================
# Summaries and plotting
# ============================================================

def build_grid_dataframe(ds_trends):
    lon2d, lat2d = np.meshgrid(ds_trends.lon.values, ds_trends.lat.values)
    if "brazil_mask" in ds_trends.data_vars:
        mask = ds_trends["brazil_mask"].values.astype(bool)
    else:
        # Fallback only for incomplete cache files. The current workflow
        # should always include brazil_mask, but this prevents a hard crash if an
        # incomplete NetCDF is accidentally reused.
        print("[WARN] brazil_mask missing from trend dataset; using finite VPD_component_trend_decade as fallback mask.")
        if "VPD_decomp_trend_decade" in ds_trends:
            mask = np.isfinite(ds_trends["VPD_decomp_trend_decade"].values)
        else:
            mask = np.ones(lon2d.shape, dtype=bool)
    data = {
        "lat": lat2d[mask],
        "lon": lon2d[mask],
        "grid_area_weight": np.cos(np.deg2rad(lat2d[mask])),
    }
    for v in ds_trends.data_vars:
        if v == "brazil_mask":
            continue
        arr = ds_trends[v].values
        if arr.shape == mask.shape:
            data[v] = arr[mask]
    return pd.DataFrame(data)


def build_component_correlations(ds_trends):
    df=build_grid_dataframe(ds_trends); rows=[]
    target="DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    if target not in df.columns: return pd.DataFrame()
    predictors=[
        ("es_trend_decade","saturation vapour pressure trend, beta_es"),
        ("ea_trend_decade","actual atmospheric vapour pressure trend, beta_ea"),
        ("actual_moisture_contribution_trend_decade","actual-moisture contribution, -beta_ea"),
        ("VPD_decomp_trend_decade","direct decomposed-VPD trend"),
        ("VPD_component_sum_trend_decade","component-slope approximation, beta_es - beta_ea"),
        ("VPD_input_trend_decade","input VPD trend from Figure 01 daily cache"),
        ("RH_component_trend_decade","relative humidity trend"),
        ("Td_proxy_trend_decade","dew-point proxy trend"),
        ("Tmean_trend_decade","temperature trend"),
    ]
    for idx,(col,label) in enumerate(predictors):
        if col not in df.columns: continue
        x=pd.to_numeric(df[col],errors="coerce").to_numpy(float); y=pd.to_numeric(df[target],errors="coerce").to_numpy(float); w=pd.to_numeric(df["grid_area_weight"],errors="coerce").to_numpy(float)
        ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
        if ok.sum()<10: continue
        rho,p=weighted_spearman(x[ok],y[ok],w[ok],n_permutations=N_PERMUTATIONS,seed=RANDOM_SEED+idx)
        rows.append({"predictor":col,"description":label,"target":target,"n_gridcells":int(ok.sum()),"coslat_weighted_spearman_rho":rho,"permutation_p_value":p,"n_permutations":N_PERMUTATIONS,"predictor_coslat_weighted_median":weighted_quantile(x[ok],w[ok],0.50),"predictor_coslat_weighted_p25":weighted_quantile(x[ok],w[ok],0.25),"predictor_coslat_weighted_p75":weighted_quantile(x[ok],w[ok],0.75)})
    out=pd.DataFrame(rows)
    csv=os.path.join(OUT_DIR,"VPD_Decomposition_component_correlations_1990_2024.csv"); xlsx=os.path.join(OUT_DIR,"VPD_Decomposition_component_correlations_1990_2024.xlsx")
    out.to_csv(csv,index=False)
    try: out.to_excel(xlsx,index=False)
    except Exception as exc: print(f"[WARN] Could not save xlsx: {exc}")
    print(f"[OK] Saved component correlations: {csv}"); return out


def add_biome_labels(df, biomes_shp=None):
    df = df.copy()
    if biomes_shp is None:
        biomes_shp = BIOMES_SHP
    if biomes_shp is None or not os.path.exists(biomes_shp):
        df["region"] = "Brazil"
        return df
    try:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        bio = gpd.read_file(biomes_shp).to_crs(epsg=4326)
        name_col = None
        for c in ["nome", "BIOMA", "bioma", "Bioma", "NAME"]:
            if c in bio.columns:
                name_col = c
                break
        if name_col is None:
            bio["region"] = "Biome"
            name_col = "region"
        joined = gpd.sjoin(gdf, bio[[name_col, "geometry"]], how="left", predicate="within")
        mapping = {
            "AMAZONIA": "Amazon", "AMAZÔNIA": "Amazon", "AMAZON": "Amazon",
            "CERRADO": "Cerrado", "CAATINGA": "Caatinga",
            "MATA ATLANTICA": "Atlantic Forest", "MATA ATLÂNTICA": "Atlantic Forest",
            "PAMPA": "Pampa", "PAMPAS": "Pampa", "PANTANAL": "Pantanal",
        }
        raw = joined[name_col].astype(str).str.upper()
        joined["region"] = raw.map(mapping).fillna(joined[name_col].astype(str))
        return pd.DataFrame(joined.drop(columns="geometry"))
    except Exception as exc:
        print(f"[WARN] Biome join failed: {exc}")
        df["region"] = "Brazil"
        return df


def build_regional_summary(ds_trends):
    df=add_biome_labels(build_grid_dataframe(ds_trends)); rows=[]
    available=set(df.get("region",[])); regions=["Brazil"]+[r for r in ["Amazon","Cerrado","Caatinga","Pantanal","Atlantic Forest","Pampa"] if r in available]
    variables=["Tmean_trend_decade","es_trend_decade","ea_trend_decade","actual_moisture_contribution_trend_decade","VPD_decomp_trend_decade","VPD_component_sum_trend_decade","VPD_component_closure_residual_trend_decade","VPD_input_trend_decade","VPD_decomp_minus_input_trend_decade","RH_component_trend_decade","Td_proxy_trend_decade","DHW_minus_HHW_TmaxOnly_intensity_trend_decade"]
    for region in regions:
        sub=df if region=="Brazil" else df[df["region"]==region]
        if len(sub)==0: continue
        w=pd.to_numeric(sub["grid_area_weight"],errors="coerce").to_numpy(float); row={"region":region,"n_gridcells":int(len(sub))}
        for col in variables:
            if col not in sub.columns: continue
            vals=pd.to_numeric(sub[col],errors="coerce").to_numpy(float)
            row[f"{col}_median"]=weighted_quantile(vals,w,0.50); row[f"{col}_p25"]=weighted_quantile(vals,w,0.25); row[f"{col}_p75"]=weighted_quantile(vals,w,0.75)
        if "VPD_increase_source_class" in sub.columns:
            cls=pd.to_numeric(sub["VPD_increase_source_class"],errors="coerce").to_numpy(float); valid=np.isfinite(cls)&(cls>=0)&np.isfinite(w)&(w>0); denom=np.sum(w[valid])
            for code,name in [(0,"no_vpd_increase_or_undefined"),(1,"vpd_increase_with_ea_increase"),(2,"vpd_increase_with_stable_ea"),(3,"vpd_increase_with_ea_decline")]:
                row[f"class_{code}_{name}_area_percent"] = float(100*np.sum(w[valid&(cls==code)])/denom) if denom>0 else np.nan
        rows.append(row)
    out=pd.DataFrame(rows); csv=os.path.join(OUT_DIR,"VPD_Decomposition_regional_summary_1990_2024.csv"); xlsx=os.path.join(OUT_DIR,"VPD_Decomposition_regional_summary_1990_2024.xlsx")
    out.to_csv(csv,index=False)
    try: out.to_excel(xlsx,index=False)
    except Exception as exc: print(f"[WARN] Could not save xlsx: {exc}")
    print(f"[OK] Saved regional summary: {csv}"); return out


def add_colorbar(fig, ax, cmap, norm, label, ticks=None, ticklabels=None):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(sm, cax=cax, ticks=ticks)
    cb.set_label(label)
    if ticklabels is not None:
        cb.ax.set_yticklabels(ticklabels)
    return cb


def plot_map(ax, ds, var, uf, title, cmap, norm, label):
    mask_da = ds["brazil_mask"] if "brazil_mask" in ds.data_vars else xr.full_like(ds[var], 1)
    arr = ds[var].where(mask_da == 1)
    im = ax.pcolormesh(ds.lon, ds.lat, arr, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    uf.boundary.plot(ax=ax, color="0.15", linewidth=0.45, zorder=4)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontweight="bold", pad=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.12, linewidth=0.35)
    add_colorbar(ax.figure, ax, cmap, norm, label)
    return im


def plot_class_map(ax, ds, uf):
    mask_da = ds["brazil_mask"] if "brazil_mask" in ds.data_vars else xr.full_like(ds["VPD_increase_source_class"], 1)
    cls = ds["VPD_increase_source_class"].where(mask_da == 1)
    cmap = ListedColormap(["#F0F0F0", "#FDBE85", "#F46D43", "#A50026"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.pcolormesh(ds.lon, ds.lat, cls, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    uf.boundary.plot(ax=ax, color="0.15", linewidth=0.45, zorder=4)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title("(e) Thermodynamic composition of positive VPD trends", fontweight="bold", pad=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.12, linewidth=0.35)
    add_colorbar(
        ax.figure, ax, cmap, norm, "class",
        ticks=[0, 1, 2, 3],
        ticklabels=["no VPD↑", "ea↑", "ea stable", "ea↓"]
    )


def plot_component_correlation(ax, corr):
    if corr is None or corr.empty:
        ax.text(0.5, 0.5, "No DHW−HHW comparison available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    desired = [
        "saturation vapour pressure trend, beta_es",
        "actual atmospheric vapour pressure trend, beta_ea",
        "actual-moisture contribution, -beta_ea",
        "direct decomposed-VPD trend",
        "relative humidity trend",
        "dew-point proxy trend",
    ]
    d = corr[corr["description"].isin(desired)].copy()
    order = desired
    d["description"] = pd.Categorical(d["description"], categories=order, ordered=True)
    d = d.sort_values("description")
    labels = [
        r"$\beta_{e_s}$",
        r"$\beta_{e_a}$",
        r"$-\beta_{e_a}$",
        r"$\beta_{VPD}$",
        r"$\beta_{RH}$",
        r"$\beta_{T_d}$",
    ][:len(d)]
    vals = d["coslat_weighted_spearman_rho"].values
    y = np.arange(len(vals))
    colors = ["#B2182B" if v >= 0 else "#2166AC" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.82)
    ax.axvline(0, color="0.2", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("cos(latitude)-weighted Spearman ρ with DHW−HHW Tmax-only")
    ax.set_title("(f) Component association with DHW−HHW contrast", fontweight="bold", pad=4)
    for yi, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:.2f}", ha="left" if v >= 0 else "right", va="center", fontsize=8)


def plot_figure(ds_trends, uf, corr):
    print("[INFO] Plotting supplementary VPD decomposition figure")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.07, top=0.955, wspace=0.38, hspace=0.38)

    # Panels (a), (b), and (c) all show vapour-pressure trend components in
    # kPa decade-1. They therefore use one shared diverging colour scale.
    # This is scientifically preferable because it permits direct visual
    # comparison between beta_es, beta_ea, and beta_es - beta_ea.
    vp_cmap = plt.cm.RdBu_r
    vp_norm = TwoSlopeNorm(vmin=VP_TREND_VMIN, vcenter=VP_TREND_VCENTER, vmax=VP_TREND_VMAX)

    plot_map(
        axes[0, 0], ds_trends, "es_trend_decade", uf,
        r"(a) Saturation vapour pressure trend, $\beta_{e_s}$",
        vp_cmap, vp_norm,
        "kPa decade$^{-1}$"
    )
    plot_map(
        axes[0, 1], ds_trends, "ea_trend_decade", uf,
        r"(b) Actual atmospheric vapour pressure trend, $\beta_{e_a}$",
        vp_cmap, vp_norm,
        "kPa decade$^{-1}$"
    )
    plot_map(
        axes[0, 2], ds_trends, "VPD_decomp_trend_decade", uf,
        r"(c) Direct decomposed VPD trend",
        vp_cmap, vp_norm,
        "kPa decade$^{-1}$"
    )

    # Dew-point proxy is in degrees Celsius per decade, so it keeps a separate scale.
    plot_map(
        axes[1, 0], ds_trends, "Td_proxy_trend_decade", uf,
        r"(d) Dew-point proxy trend, $\beta_{T_d}$",
        plt.cm.RdBu_r, TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.8),
        "°C decade$^{-1}$"
    )
    plot_class_map(axes[1, 1], ds_trends, uf)
    plot_component_correlation(axes[1, 2], corr)

    out_base = os.path.join(OUT_DIR, "Supplementary_VPD_Decomposition_Diagnostics_1990_2024")
    fig.savefig(out_base + ".pdf", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".jpeg", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_base}.pdf")
    print(f"[OK] Saved figure: {out_base}.jpeg")

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    configure_paths(args)

    print(f"[START] VPD decomposition diagnostics | {pd.Timestamp.now().isoformat()}")
    print("[INFO] Interpretation: thermodynamic decomposition diagnostic; not causal attribution.")
    print(f"[INFO] Restartable trend cache: {TREND_CACHE_DIR}")

    files = list_daily_files()
    ds0 = safe_open_dataset(files[0][1], decode_times=True, chunks=None)
    ds0 = standardize_lat_lon(ds0).sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MAX, LAT_MIN))
    print(f"[INFO] Brazil shapefile: {BRAZIL_SHP}")
    mask_brazil, uf = make_brazil_mask_from_grid(
        ds0.lat.values,
        ds0.lon.values,
        BRAZIL_SHP,
    )
    print(f"[INFO] Brazil mask grid cells: {int(mask_brazil.sum())}")
    ds0.close()

    ds_ann = compute_annual_decomposition(overwrite=args.recompute)
    # Require exact grid alignment with the current Figure 01 daily-cache grid.
    if (
        ds_ann.sizes.get("lat") != mask_brazil.sizes["lat"]
        or ds_ann.sizes.get("lon") != mask_brazil.sizes["lon"]
        or not np.allclose(ds_ann.lat.values, mask_brazil.lat.values)
        or not np.allclose(ds_ann.lon.values, mask_brazil.lon.values)
    ):
        ds_ann.close()
        raise RuntimeError(
            "Cached annual decomposition metrics are not exactly aligned with "
            "the current Figure 01 grid. Re-run with --recompute."
        )

    ds_trends = compute_trends(ds_ann, mask_brazil, overwrite=args.recompute)
    corr = build_component_correlations(ds_trends)
    regional = build_regional_summary(ds_trends)
    plot_figure(ds_trends, uf, corr)

    print("\n[SUMMARY] National median trend components")
    if not regional.empty and "region" in regional.columns:
        brazil = regional[regional["region"] == "Brazil"]
        if len(brazil) > 0:
            b = brazil.iloc[0]
            for col, label in [
                ("es_trend_decade_median", "beta_es"),
                ("ea_trend_decade_median", "beta_ea"),
                ("actual_moisture_contribution_trend_decade_median", "-beta_ea"),
                ("VPD_decomp_trend_decade_median", "beta_VPD_direct"),
                ("Td_proxy_trend_decade_median", "beta_Td"),
                ("RH_component_trend_decade_median", "beta_RH"),
            ]:
                if col in b:
                    print(f"  {label:20s}: {b[col]:+.3f}")

    if not corr.empty:
        print("\n[SUMMARY] cos(latitude)-weighted Spearman correlations with DHW-HHW Tmax-only contrast")
        for _, r in corr.iterrows():
            print(f"  {r['predictor']:42s} rho={r['coslat_weighted_spearman_rho']:+.3f}, p={r['permutation_p_value']:.2e}")

    print("\n[DONE] Outputs saved in:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

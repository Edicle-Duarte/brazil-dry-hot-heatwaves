#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 01 — Trends in thermodynamic heatwave regimes across Brazil (1990–2024).

This script reproduces the main heatwave-regime analysis and Figure 01 for the
associated study. Hourly ERA5 single-level fields are converted to daily
thermodynamic diagnostics, local percentile thresholds are calculated from the
1991–2020 baseline, annual heatwave metrics are derived, and 1990–2024 trends
are estimated using Theil–Sen slopes with two-sided Mann–Kendall significance
tests.

Heatwave regimes
----------------
HW
    Tmean >= local day-of-year P90 for at least 3 consecutive days.
HHW
    Twbmax >= local day-of-year P95 for at least 3 consecutive days.
DHW
    Tmax >= local P95 and VPDmean >= local P75 for at least 3 consecutive days.
CHW
    Tmax >= local P95, VPDmean >= local P90, and either SM1mean <= local P10
    (when soil moisture is available) or WS10mean <= local P50 (fallback), for
    at least 3 consecutive days.

Inputs
------
--era5-dir
    Directory containing hourly ERA5 NetCDF files named approximately
    ``ERA5_single_levels_LatinAmerica_hourly_YYYY*.nc``. Required variables are
    2-m temperature and 2-m dew-point temperature. 10-m u/v wind components are
    required for the CHW low-wind fallback. Soil moisture layer 1 is optional.
--brazil-shapefile
    Brazil/UF boundary shapefile used to construct the national analysis mask.
--biomes-shapefile
    Optional Brazilian-biomes shapefile used only as a figure overlay.
--south-america-shapefile
    Optional South American country-boundary shapefile used only as a figure
    overlay.

Outputs
-------
Within ``--output-dir``:
    cache/era5_daily/ERA5_daily_Brazil_YYYY.nc
    cache/figure_01_heatwave_thresholds_1991_2020.nc
    data/figure_01_annual_heatwave_metrics_1990_2024.nc
    data/figure_01_heatwave_trends_1990_2024.nc
    figures/figure_01_heatwave_trends.pdf
    figures/figure_01_heatwave_trends.jpeg

Dependencies
------------
Required:
    numpy, pandas, xarray, scipy, geopandas, shapely, matplotlib
Optional:
    pymannkendall (accelerates Mann–Kendall significance testing)

Usage
-----
python figure_01_heatwave_trends.py \
    --era5-dir /path/to/ERA5_BR \
    --brazil-shapefile /path/to/UFEBRASIL.shp \
    --biomes-shapefile /path/to/lm_bioma_250.shp \
    --south-america-shapefile /path/to/ne_10m_admin_0_countries.shp \
    --output-dir /path/to/output

Reproducibility note
--------------------
The scientific defaults below correspond to the analysis used for the study.
The refactor intentionally changes file/path handling and naming only; it does
not silently change heatwave definitions, intensity calculations, thresholds,
quality-control criteria, trend estimators, or plotting scales.
"""

import os
import re
import glob
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
from matplotlib.lines import Line2D
from scipy.stats import linregress, norm, theilslopes  # Theil-Sen slope estimator
from shapely.geometry import Point

# Optional Mann-Kendall acceleration with pymannkendall
try:
    import pymannkendall as mk
    _HAS_PYMK = True
except ImportError:
    _HAS_PYMK = False
    print("[INFO] pymannkendall not installed; using pure Python Mann-Kendall. "
          "Install with: pip install pymannkendall for 10-30× speedup in significance testing.")


# ============================================================
# Runtime paths
# ============================================================
#
# These values are populated by ``configure_runtime_paths`` from command-line
# arguments. No workstation/HPC-specific absolute path is required by the
# repository version of the script.

ERA5_DIR = None
BRAZIL_SHP = None
BIOMES_SHP = None
SA_COUNTRIES_SHP = None

OUTPUT_ROOT = None
DATA_DIR = None
FIGURE_DIR = None
CACHE_DIR = None
DAILY_DIR = None
THRESHOLD_FILE = None
ANNUAL_FILE = None
TRENDS_FILE = None

FORCE_RECOMPUTE_THRESHOLDS = False


def configure_runtime_paths(args):
    """Populate and create runtime paths from command-line arguments."""
    global ERA5_DIR, BRAZIL_SHP, BIOMES_SHP, SA_COUNTRIES_SHP
    global OUTPUT_ROOT, DATA_DIR, FIGURE_DIR, CACHE_DIR, DAILY_DIR
    global THRESHOLD_FILE, ANNUAL_FILE, TRENDS_FILE
    global FORCE_RECOMPUTE_ANNUAL_TRENDS, FORCE_RECOMPUTE_THRESHOLDS

    ERA5_DIR = str(Path(args.era5_dir).expanduser().resolve())
    BRAZIL_SHP = str(Path(args.brazil_shapefile).expanduser().resolve())
    BIOMES_SHP = (
        str(Path(args.biomes_shapefile).expanduser().resolve())
        if args.biomes_shapefile else None
    )
    SA_COUNTRIES_SHP = (
        str(Path(args.south_america_shapefile).expanduser().resolve())
        if args.south_america_shapefile else None
    )

    OUTPUT_ROOT = Path(args.output_dir).expanduser().resolve()
    DATA_DIR = OUTPUT_ROOT / "data"
    FIGURE_DIR = OUTPUT_ROOT / "figures"
    CACHE_DIR = OUTPUT_ROOT / "cache"
    DAILY_DIR = CACHE_DIR / "era5_daily"

    for directory in (OUTPUT_ROOT, DATA_DIR, FIGURE_DIR, CACHE_DIR, DAILY_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    THRESHOLD_FILE = CACHE_DIR / "figure_01_heatwave_thresholds_1991_2020.nc"
    ANNUAL_FILE = DATA_DIR / "figure_01_annual_heatwave_metrics_1990_2024.nc"
    TRENDS_FILE = DATA_DIR / "figure_01_heatwave_trends_1990_2024.nc"

    FORCE_RECOMPUTE_ANNUAL_TRENDS = bool(args.recompute_annual_trends)
    FORCE_RECOMPUTE_THRESHOLDS = bool(args.recompute_thresholds)


# ============================================================
# Scientific configuration parameters
# ============================================================

# Temporal domain
START_YEAR = 1990
END_YEAR = 2024
BASE_START = 1991  # WMO-compliant baseline start
BASE_END = 2020    # WMO-compliant baseline end

# Event definition parameters
MIN_LEN = 3  # minimum consecutive days for heatwave event
ROLLING_WINDOW_DAYS = 31  # centred window for threshold climatology

# Heatwave-type percentile thresholds (local, day-of-year specific)
HHW_TWBMAX_P = 95.0  # humid heat: wet-bulb temperature extreme

DHW_TMAX_P = 95.0    # dry heat: temperature extreme
DHW_VPD_P = 75.0     # dry heat: atmospheric dryness (VPD) threshold

HW_TMEAN_P = 90.0    # general heat: mean temperature extreme

CHW_TMAX_P = 95.0    # compound: temperature extreme
CHW_VPD_P = 90.0     # compound: strong atmospheric dryness
CHW_SM_P = 10.0      # compound: soil moisture deficit (if available)
CHW_WS_P = 50.0      # compound: low-wind stagnation (fallback)

# Data quality thresholds
MIN_VALID_DAYS_PER_YEAR = 300   # minimum valid days/year per grid cell
MIN_VALID_YEARS_FOR_TREND = 20  # minimum years for robust trend estimation

# Spatial domain (Brazil focus: lat 6 to -35, lon -75 to -32)
LON_MIN, LON_MAX = -75.0, -32.0
LAT_MIN, LAT_MAX = -35.0, 6.0

# Color scale limits for trend visualization (optimized for Brazil's signal range)
FREQ_VMIN, FREQ_VMAX = -4.0, 4.0    # events decade⁻¹
DUR_VMIN, DUR_VMAX = -14.0, 14.0    # days decade⁻¹  
INT_VMIN, INT_VMAX = -10.0, 10.0    # severity decade⁻¹

# Statistical methodology configuration
P_THRESHOLD = 0.05                    # significance threshold for stippling

# Visual configuration requested for the manuscript figure
FIG_FONT_SIZE = 14
# Standardized intensity configuration (prevents outlier dominance)
Z_CLIP_MIN = -5.0   # lower bound for local z-score clipping
Z_CLIP_MAX = 5.0    # upper bound for local z-score clipping

# Equal-weighting for compound intensity components (avoids unit-mixing artifacts)
DHW_WEIGHT_TMAX = 0.5
DHW_WEIGHT_VPD = 0.5

CHW_WEIGHT_TMAX = 1.0 / 3.0
CHW_WEIGHT_VPD = 1.0 / 3.0  
CHW_WEIGHT_THIRD = 1.0 / 3.0

# Force recomputation to ensure methodology consistency with manuscript
FORCE_RECOMPUTE_ANNUAL_TRENDS = False


# ============================================================
# Utility functions
# ============================================================

def find_brazil_shapefile():
    """Validate and return the Brazil boundary shapefile."""
    if BRAZIL_SHP and os.path.exists(BRAZIL_SHP):
        return BRAZIL_SHP
    raise FileNotFoundError(
        "Brazil boundary shapefile not found. Provide a valid "
        "--brazil-shapefile path."
    )


def load_biomes_shapefile():
    """Load the optional Brazilian-biomes shapefile."""
    if not BIOMES_SHP:
        return None
    if not os.path.exists(BIOMES_SHP):
        print(f"[WARN] Biomes shapefile not found: {BIOMES_SHP}")
        return None
    try:
        biomes = gpd.read_file(BIOMES_SHP).to_crs(epsg=4326)
        print(f"[INFO] Loaded biomes: {os.path.basename(BIOMES_SHP)}")
        return biomes
    except Exception as exc:
        print(f"[WARN] Could not load biomes shapefile: {exc}")
        return None


def find_south_america_shapefile():
    """Return the optional South America boundary shapefile if available."""
    if SA_COUNTRIES_SHP and os.path.exists(SA_COUNTRIES_SHP):
        return SA_COUNTRIES_SHP
    return None


def list_era5_files_for_year(year):
    """List ERA5 hourly NetCDF files for a given year."""
    pattern = os.path.join(ERA5_DIR, f"ERA5_single_levels_LatinAmerica_hourly_{year}*.nc")
    return sorted(glob.glob(pattern))


def standardize_dataset(ds):
    """Standardize coordinate and variable names across ERA5 file versions."""
    rename_map = {}
    
    # Standardize coordinate names
    coord_aliases = {
        "valid_time": "time", "time": "time",
        "latitude": "lat", "lat": "lat", 
        "longitude": "lon", "lon": "lon"
    }
    for c in ds.coords:
        if c in coord_aliases:
            rename_map[c] = coord_aliases[c]
    for d in ds.dims:
        if d in coord_aliases:
            rename_map[d] = coord_aliases[d]
    
    ds = ds.rename(rename_map)

    # Standardize variable names (ERA5 parameter IDs and descriptive names)
    varmap = {
        "var167": "t2m", "var168": "d2m", "var165": "u10", "var166": "v10",
        "var39": "sm1", "swvl1": "sm1", 
        "volumetric_soil_water_layer_1": "sm1", "soil_moisture": "sm1",
    }
    ds = ds.rename({k: v for k, v in varmap.items() if k in ds.data_vars})

    # Validate required variables
    missing = [v for v in ["t2m", "d2m"] if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing required variables after renaming: {missing}")

    # Handle multi-experiment version files (expver dimension)
    for v in list(ds.data_vars):
        if "expver" in ds[v].dims:
            ds[v] = ds[v].max("expver", skipna=True)
    if "expver" in ds.dims:
        ds = ds.drop_dims("expver")

    # Normalize longitude to [-180, 180] and sort coordinates
    if ds.lon.max() > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    
    ds = ds.sortby("time")
    if ds.lat[0] < ds.lat[-1]:  # ensure north-to-south ordering
        ds = ds.sortby("lat", ascending=False)

    # Clip to Brazil domain (adjusted for new lat/lon bounds)
    ds = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MAX, LAT_MIN))
    
    return ds


def open_era5_year(files):
    """Open ERA5 files for a single year without dask (memory-efficient for daily aggregation)."""
    if len(files) == 1:
        ds = xr.open_dataset(files[0], decode_times=True, chunks=None)
        return standardize_dataset(ds)
    else:
        datasets = [standardize_dataset(xr.open_dataset(f, decode_times=True, chunks=None)) 
                   for f in files]
        return xr.concat(datasets, dim="time", data_vars="minimal", 
                        coords="minimal", compat="override").sortby("time")


def remove_feb29(ds):
    """Remove February 29 dates to create no-leap-year time coordinate."""
    mask = ~((ds.time.dt.month == 2) & (ds.time.dt.day == 29))
    return ds.sel(time=mask)


def add_noleap_doy(ds):
    """Add no-leap day-of-year coordinate (1–365) for threshold climatology."""
    dates = pd.to_datetime(ds.time.values)
    doy = []
    for d in dates:
        y0 = pd.Timestamp(f"{d.year}-01-01")
        val = (d - y0).days + 1
        if d.is_leap_year and d.month > 2:
            val -= 1  # adjust for removed Feb 29
        doy.append(val)
    return ds.assign_coords(doy_noleap=("time", np.asarray(doy, dtype=np.int16)))


# ============================================================
# Thermodynamic calculations
# ============================================================

def saturation_vapor_pressure_hpa(Tc):
    """Calculate saturation vapor pressure [hPa] from temperature [°C] (Tetens formula)."""
    return 6.112 * np.exp((17.62 * Tc) / (243.12 + Tc))


def relative_humidity_from_t_td(t_c, td_c):
    """Calculate relative humidity [%] from temperature and dewpoint [°C]."""
    e = saturation_vapor_pressure_hpa(td_c)
    es = saturation_vapor_pressure_hpa(t_c)
    rh = 100.0 * e / es
    return xr.where((rh >= 0) & (rh <= 100), rh, np.nan)


def vapor_pressure_deficit_hpa(t_c, td_c):
    """Calculate vapor pressure deficit [hPa] from temperature and dewpoint [°C]."""
    es = saturation_vapor_pressure_hpa(t_c)
    e = saturation_vapor_pressure_hpa(td_c)
    vpd = es - e
    return xr.where(vpd >= 0, vpd, 0.0)  # VPD cannot be negative


def wetbulb_stull(Tc, RH):
    """
    Calculate wet-bulb temperature [°C] using Stull (2011) approximation.
    Valid for RH ∈ [1, 100]% and Tc ∈ [−20, 50]°C; extrapolation clipped.
    
    Reference: Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity 
    and Air Temperature. J. Appl. Meteorol. Climatol., 50(11), 2267–2269.
    """
    RH = xr.where((RH >= 1) & (RH <= 100), RH, np.nan)
    
    Tw = (
        Tc * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(Tc + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )
    return Tw


# ============================================================
# Daily preprocessing pipeline
# ============================================================

def process_daily_year(year, overwrite=False):
    """Process hourly ERA5 to daily metrics for a single year."""
    out_file = os.path.join(str(DAILY_DIR), f"ERA5_daily_Brazil_{year}.nc")
    
    if os.path.exists(out_file) and not overwrite:
        print(f"[SKIP] Daily file exists: {os.path.basename(out_file)}")
        return out_file

    files = list_era5_files_for_year(year)
    if not files:
        print(f"[WARN] No ERA5 files found for {year}. Skipping.")
        return None

    print(f"[INFO] Processing {year}: {len(files)} file(s)")
    ds = open_era5_year(files)

    # Convert to °C for thermodynamic calculations
    t2_c = ds["t2m"] - 273.15
    td2_c = ds["d2m"] - 273.15

    # Calculate derived variables
    rh = relative_humidity_from_t_td(t2_c, td2_c)
    vpd = vapor_pressure_deficit_hpa(t2_c, td2_c)
    twb = wetbulb_stull(t2_c, rh)

    # Aggregate to daily metrics
    daily = xr.Dataset({
        "Tmean": t2_c.resample(time="1D").mean(skipna=True),
        "Tmax": t2_c.resample(time="1D").max(skipna=True),
        "Tmin": t2_c.resample(time="1D").min(skipna=True),
        "RHmean": rh.resample(time="1D").mean(skipna=True),
        "Twbmean": twb.resample(time="1D").mean(skipna=True),
        "Twbmax": twb.resample(time="1D").max(skipna=True),
        "VPDmean": vpd.resample(time="1D").mean(skipna=True),
        "VPDmax": vpd.resample(time="1D").max(skipna=True),
    })

    # Wind speed (required for CHW fallback)
    if "u10" in ds.data_vars and "v10" in ds.data_vars:
        ws10 = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
        daily["WS10mean"] = ws10.resample(time="1D").mean(skipna=True)
    else:
        daily["WS10mean"] = xr.full_like(daily["Tmean"], np.nan)

    # Soil moisture (optional, for primary CHW definition)
    if "sm1" in ds.data_vars:
        daily["SM1mean"] = ds["sm1"].resample(time="1D").mean(skipna=True)
    else:
        daily["SM1mean"] = xr.full_like(daily["Tmean"], np.nan)

    # Fail explicitly if neither CHW stressor is available.
    # This avoids silently generating an empty CHW diagnostic.
    if "sm1" not in ds.data_vars and not (
        "u10" in ds.data_vars and "v10" in ds.data_vars
    ):
        ds.close()
        raise ValueError(
            f"{year}: CHW cannot be evaluated because neither soil moisture "
            "(sm1/swvl1) nor both 10-m wind components (u10, v10) are available."
        )

    # Apply no-leap calendar.
    daily = remove_feb29(daily)
    daily = add_noleap_doy(daily)

    # Explicit units for intermediate products.
    variable_metadata = {
        "Tmean": ("degree_Celsius", "daily mean 2-m air temperature"),
        "Tmax": ("degree_Celsius", "daily maximum 2-m air temperature"),
        "Tmin": ("degree_Celsius", "daily minimum 2-m air temperature"),
        "RHmean": ("percent", "daily mean relative humidity"),
        "Twbmean": ("degree_Celsius", "daily mean wet-bulb temperature"),
        "Twbmax": ("degree_Celsius", "daily maximum wet-bulb temperature"),
        "VPDmean": ("hPa", "daily mean vapour-pressure deficit"),
        "VPDmax": ("hPa", "daily maximum vapour-pressure deficit"),
        "WS10mean": ("m s-1", "daily mean 10-m wind speed"),
        "SM1mean": ("m3 m-3", "daily mean volumetric soil water layer 1"),
    }
    for varname, (units, long_name) in variable_metadata.items():
        if varname in daily:
            daily[varname].attrs.update(units=units, long_name=long_name)

    daily.attrs.update({
        "source": "ERA5 hourly single-level data",
        "calendar_processing": (
            "February 29 removed; no-leap day-of-year coordinate added"
        ),
        "VPD_formula": "es(T2m) - e(Td2m), stored in hPa",
        "wet_bulb_method": (
            "Stull (2011) approximation from 2-m temperature and relative humidity"
        ),
    })

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in daily.data_vars
    }
    daily.to_netcdf(out_file, encoding=encoding)
    
    ds.close(); daily.close()
    print(f"[OK] Saved: {os.path.basename(out_file)}")
    return out_file


def open_daily_files_no_dask(files):
    """Concatenate daily files without dask (sufficient for Brazil domain)."""
    datasets = [xr.open_dataset(f, decode_times=True, chunks=None) for f in files]
    return xr.concat(datasets, dim="time", data_vars="minimal", 
                    coords="minimal", compat="override").sortby("time")


# ============================================================
# Spatial masking
# ============================================================

def make_brazil_mask(ds, shp_path):
    """Create boolean mask for Brazil territory from administrative shapefile."""
    uf = gpd.read_file(shp_path).to_crs(epsg=4326)
    brazil_geom = uf.dissolve().geometry.iloc[0]
    
    lons, lats = ds.lon.values, ds.lat.values
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    
    # Vectorized point-in-polygon test (efficient for Brazil's resolution)
    for i, lat in enumerate(lats):
        points = [Point(lon, lat) for lon in lons]
        mask[i, :] = [brazil_geom.contains(p) or brazil_geom.touches(p) for p in points]
    
    da = xr.DataArray(mask, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    return da, uf


# ============================================================
# Threshold climatology computation
# ============================================================

def compute_doy_quantile(base, varname, percentile, mask_brazil, outname):
    """Compute day-of-year-specific percentile threshold with 31-day moving window."""
    print(f"[INFO] {outname}: P{percentile} climatology ({BASE_START}–{BASE_END})")
    
    base = base.where(mask_brazil)
    thresholds = []
    half = ROLLING_WINDOW_DAYS // 2
    
    for doy in range(1, 366):
        # Build centred window with wrap-around for year boundaries
        window = [(doy + off - 1) % 365 + 1 for off in range(-half, half + 1)]
        subset = base[varname].where(base["doy_noleap"].isin(window), drop=True)
        thr = subset.quantile(percentile / 100.0, dim="time", skipna=True)
        thresholds.append(thr.drop_vars("quantile", errors="ignore"))
        if doy % 50 == 0:
            print(f"       {outname}: {doy}/365")
    
    out = xr.concat(thresholds, dim="doy_noleap")
    out = out.assign_coords(doy_noleap=np.arange(1, 366, dtype=np.int16))
    out.name = outname
    return out


def compute_doy_mean_std(base, varname, mask_brazil, mean_name, std_name):
    """Compute day-of-year-specific climatological mean and standard deviation."""
    print(f"[INFO] {varname} climatology: mean/std ({BASE_START}–{BASE_END})")
    
    base = base.where(mask_brazil)
    means, stds = [], []
    half = ROLLING_WINDOW_DAYS // 2
    
    for doy in range(1, 366):
        window = [(doy + off - 1) % 365 + 1 for off in range(-half, half + 1)]
        subset = base[varname].where(base["doy_noleap"].isin(window), drop=True)
        means.append(subset.mean(dim="time", skipna=True))
        stds.append(subset.std(dim="time", skipna=True))
        if doy % 50 == 0:
            print(f"       {varname}: {doy}/365")
    
    coords = np.arange(1, 366, dtype=np.int16)
    mean_da = xr.concat(means, dim="doy_noleap").assign_coords(doy_noleap=coords)
    std_da = xr.concat(stds, dim="doy_noleap").assign_coords(doy_noleap=coords)
    
    mean_da.name, std_da.name = mean_name, std_name
    return mean_da, std_da


def compute_or_load_thresholds(ds_daily, mask_brazil):
    """Compute or load climatological thresholds and statistics."""
    threshold_file = str(THRESHOLD_FILE)
    
    required_vars = [
        "HHW_Twbmax_P95", "DHW_Tmax_P95", "DHW_VPDmean_P75", "HW_Tmean_P90",
        "CHW_Tmax_P95", "CHW_VPDmean_P90", "CHW_SM1mean_P10", "CHW_WS10mean_P50",
        "CLIM_Tmax_mean", "CLIM_Tmax_std", "CLIM_VPDmean_mean", "CLIM_VPDmean_std",
        "CLIM_Tmean_mean", "CLIM_Tmean_std", "CLIM_Twbmax_mean", "CLIM_Twbmax_std",
        "CLIM_SM1mean_mean", "CLIM_SM1mean_std", "CLIM_WS10mean_mean", "CLIM_WS10mean_std",
    ]
    
    # Load existing file if complete
    if os.path.exists(threshold_file) and not FORCE_RECOMPUTE_THRESHOLDS:
        ds_thr = xr.open_dataset(threshold_file)
        if all(v in ds_thr.data_vars for v in required_vars):
            print(f"[INFO] Loaded thresholds: {os.path.basename(threshold_file)}")
            return ds_thr
        print("[WARN] Incomplete threshold file; recomputing.")
        ds_thr.close()
    
    # Compute thresholds from baseline period
    base = ds_daily.sel(time=slice(f"{BASE_START}-01-01", f"{BASE_END}-12-31"))
    ds_thr = xr.Dataset()
    
    # Heatwave-type percentile thresholds
    ds_thr["HHW_Twbmax_P95"] = compute_doy_quantile(base, "Twbmax", HHW_TWBMAX_P, mask_brazil, "HHW_Twbmax_P95")
    ds_thr["DHW_Tmax_P95"] = compute_doy_quantile(base, "Tmax", DHW_TMAX_P, mask_brazil, "DHW_Tmax_P95")
    ds_thr["DHW_VPDmean_P75"] = compute_doy_quantile(base, "VPDmean", DHW_VPD_P, mask_brazil, "DHW_VPDmean_P75")
    ds_thr["HW_Tmean_P90"] = compute_doy_quantile(base, "Tmean", HW_TMEAN_P, mask_brazil, "HW_Tmean_P90")
    ds_thr["CHW_Tmax_P95"] = compute_doy_quantile(base, "Tmax", CHW_TMAX_P, mask_brazil, "CHW_Tmax_P95")
    ds_thr["CHW_VPDmean_P90"] = compute_doy_quantile(base, "VPDmean", CHW_VPD_P, mask_brazil, "CHW_VPDmean_P90")
    ds_thr["CHW_SM1mean_P10"] = compute_doy_quantile(base, "SM1mean", CHW_SM_P, mask_brazil, "CHW_SM1mean_P10")
    ds_thr["CHW_WS10mean_P50"] = compute_doy_quantile(base, "WS10mean", CHW_WS_P, mask_brazil, "CHW_WS10mean_P50")
    
    # Climatological mean/std for standardized intensity metrics
    for varname, prefix in [
        ("Tmax", "CLIM_Tmax"), ("VPDmean", "CLIM_VPDmean"), ("Tmean", "CLIM_Tmean"),
        ("Twbmax", "CLIM_Twbmax"), ("SM1mean", "CLIM_SM1mean"), ("WS10mean", "CLIM_WS10mean"),
    ]:
        mn, sd = compute_doy_mean_std(base, varname, mask_brazil, f"{prefix}_mean", f"{prefix}_std")
        ds_thr[f"{prefix}_mean"], ds_thr[f"{prefix}_std"] = mn, sd
    
    # Save with compression
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_thr.data_vars}
    ds_thr.to_netcdf(threshold_file, encoding=encoding)
    print(f"[OK] Saved thresholds: {os.path.basename(threshold_file)}")
    return ds_thr




# ============================================================
# Event detection and trend estimation
# ============================================================

def detect_runs_1d(mask, intensity_excess, years, min_len=3):
    """Detect heatwave events and aggregate annual metrics for a single grid cell."""
    annual_freq = {y: 0.0 for y in range(START_YEAR, END_YEAR + 1)}
    annual_days = {y: 0.0 for y in range(START_YEAR, END_YEAR + 1)}
    annual_int = {y: 0.0 for y in range(START_YEAR, END_YEAR + 1)}
    
    n, i = len(mask), 0
    while i < n:
        if mask[i]:
            # Find event end
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            dur = j - i + 1
            
            if dur >= min_len:
                # Count event frequency by start year
                event_year = int(years[i])
                if START_YEAR <= event_year <= END_YEAR:
                    annual_freq[event_year] += 1.0
                
                # Accumulate duration and intensity by day
                for k in range(i, j + 1):
                    yy = int(years[k])
                    if START_YEAR <= yy <= END_YEAR:
                        annual_days[yy] += 1.0
                        ex = intensity_excess[k]
                        if np.isfinite(ex) and ex > 0:
                            annual_int[yy] += float(ex)
            i = j + 1
        else:
            i += 1
    
    return annual_freq, annual_days, annual_int


def mann_kendall_pvalue(y):
    """
    Two-sided Mann-Kendall test p-value with tie correction.
    
    Uses pymannkendall if available (10-30× faster), otherwise falls back
    to pure Python implementation with identical statistical logic.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    
    if n < MIN_VALID_YEARS_FOR_TREND:
        return np.nan
    
    # Optimization: use pymannkendall when available
    if _HAS_PYMK:
        try:
            result = mk.original_test(y)
            return float(result.p)  # two-sided p-value from optimized C/Fortran backend
        except Exception:
            # Fallback to pure Python if pymannkendall fails for any reason
            pass
    
    # Original pure-Python implementation (safe fallback)
    # Compute S statistic
    s = sum(np.sign(y[j] - y[i]) for i in range(n-1) for j in range(i+1, n))
    
    # Tie correction
    _, counts = np.unique(y, return_counts=True)
    tie_term = sum(c * (c-1) * (2*c + 5) for c in counts)
    var_s = (n * (n-1) * (2*n + 5) - tie_term) / 18.0
    
    if var_s <= 0:
        return np.nan
    
    # Standardized test statistic
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def theil_sen_slope_per_decade(y, years):
    """
    Theil-Sen median slope estimator, scaled to per-decade units.
    
    Optimization: uses scipy.stats.theilslopes instead of a pure-Python
    O(n²) loop, while preserving the same mathematical estimator.
    """
    y, x = np.asarray(y, float), np.asarray(years, float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    
    if len(y) < MIN_VALID_YEARS_FOR_TREND:
        return np.nan
    
    # The SciPy implementation is much faster than pure Python pairwise loops
    # theilslopes returns (slope, intercept, lower_slope, upper_slope)
    result = theilslopes(y, x, alpha=0.05)  # alpha unused but required by signature
    return float(result[0] * 10.0)  # scale from per-year to per-decade


def robust_trend_per_decade(y, years):
    """
    Primary trend estimation: Theil-Sen slope + Mann-Kendall p-value.
    OLS metrics returned for diagnostic purposes only.
    """
    sen_slope = theil_sen_slope_per_decade(y, years)
    mk_p = mann_kendall_pvalue(y)
    
    # OLS diagnostic (not used in figure)
    y_arr = np.asarray(y, float)
    x_arr = np.asarray(years, float)
    valid = np.isfinite(y_arr) & np.isfinite(x_arr)
    if valid.sum() >= MIN_VALID_YEARS_FOR_TREND:
        lr = linregress(x_arr[valid], y_arr[valid])
        ols_slope, ols_p = float(lr.slope * 10.0), float(lr.pvalue)
    else:
        ols_slope, ols_p = np.nan, np.nan
        
    return sen_slope, mk_p, ols_slope, ols_p


def has_real_soil_moisture(ds_daily):
    """Check if soil moisture data is present and non-trivial."""
    if "SM1mean" not in ds_daily.data_vars:
        return False
    sample = ds_daily["SM1mean"].isel(time=slice(0, min(365, ds_daily.sizes["time"])))
    return int(np.isfinite(sample.values).sum()) > 0


def compute_annual_metrics_and_trends(ds_daily, ds_thr, mask_brazil):
    """
    Compute annual heatwave metrics and robust trends for all types.

    MEMORY-SAFE VERSION
    -------------------
    The previous implementation created full-domain 3D numpy arrays for every
    heatwave flag and intensity field:

        flag_np = {k: flags[k].values ...}
        exc_np  = {k: intensity_excess[k].values ...}

    For 1990–2024 over Brazil, this can exceed the available RAM and the Linux
    kernel kills the process ("Killed") before Python can raise MemoryError.

    This version processes one latitude row at a time. It loads only arrays with
    shape (time, lon) for each variable/threshold, computes the flags/intensity
    locally, then immediately discards temporary arrays. Scientific definitions
    and output variables are unchanged.
    """
    import gc

    print("[INFO] Computing annual metrics and Theil-Sen trends (1990–2024)")
    print("[INFO] Memory-safe row-wise mode enabled; no full-domain flag/intensity arrays will be materialized.")

    use_soil = has_real_soil_moisture(ds_daily)

    if use_soil:
        print("[INFO] CHW: soil-moisture-based definition (Tmax≥P95 & VPD≥P90 & SM1≤P10)")
    else:
        print("[WARN] CHW: fallback definition (Tmax≥P95 & VPD≥P90 & WS10≤P50); "
              "consider downloading ERA5 soil moisture for enhanced analysis")

    # Time coordinates
    time = pd.to_datetime(ds_daily.time.values)
    years = np.asarray([t.year for t in time], dtype=np.int16)
    year_axis = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int16)
    year_indices = {int(y): np.where(years == y)[0] for y in year_axis}

    # Spatial coordinates
    lats, lons = ds_daily.lat.values, ds_daily.lon.values
    ny, nx = len(lats), len(lons)
    nyr = len(year_axis)

    metric_names = [
        f"{kind}_{metric}"
        for kind in ["HW", "HHW", "DHW", "CHW"]
        for metric in ["frequency", "duration", "intensity"]
    ]

    annual_data = {
        name: np.full((nyr, ny, nx), np.nan, dtype=np.float32)
        for name in metric_names
    }

    trend_data = {}
    pval_data = {}
    ols_trend_data = {}
    ols_pval_data = {}

    for name in metric_names:
        trend_data[f"{name}_trend_decade"] = np.full((ny, nx), np.nan, dtype=np.float32)
        pval_data[f"{name}_pvalue"] = np.full((ny, nx), np.nan, dtype=np.float32)
        ols_trend_data[f"{name}_OLS_trend_decade"] = np.full((ny, nx), np.nan, dtype=np.float32)
        ols_pval_data[f"{name}_OLS_pvalue"] = np.full((ny, nx), np.nan, dtype=np.float32)

    mask_np = mask_brazil.values.astype(bool)

    def _thr_row(varname, iy):
        """Return day-of-year threshold/statistic mapped to time for one latitude row."""
        return (
            ds_thr[varname]
            .sel(doy_noleap=ds_daily["doy_noleap"])
            .isel(lat=iy)
            .transpose("time", "lon")
            .values
            .astype(np.float32, copy=False)
        )

    def _var_row(varname, iy):
        """Return daily variable for one latitude row as (time, lon)."""
        return (
            ds_daily[varname]
            .isel(lat=iy)
            .transpose("time", "lon")
            .values
            .astype(np.float32, copy=False)
        )

    def _z_pos_row(value, clim_mean, clim_std):
        """Positive clipped local z-score for row arrays."""
        std = np.where(clim_std <= 1e-6, np.nan, clim_std)
        z = (value - clim_mean) / std
        z = np.clip(z, Z_CLIP_MIN, Z_CLIP_MAX)
        return np.where(z > 0, z, 0.0).astype(np.float32, copy=False)

    for iy in range(ny):
        print(f"       Processing latitude {iy + 1}/{ny}")

        valid_lon_idx = np.where(mask_np[iy, :])[0]
        if valid_lon_idx.size == 0:
            continue

        # Load only this latitude row into memory.
        Tmean_row = _var_row("Tmean", iy)
        Tmax_row = _var_row("Tmax", iy)
        Twbmax_row = _var_row("Twbmax", iy)
        VPD_row = _var_row("VPDmean", iy)
        WS_row = _var_row("WS10mean", iy)
        SM_row = _var_row("SM1mean", iy)

        # Threshold rows
        HW_Tmean_thr = _thr_row("HW_Tmean_P90", iy)
        HHW_Twb_thr = _thr_row("HHW_Twbmax_P95", iy)
        DHW_Tmax_thr = _thr_row("DHW_Tmax_P95", iy)
        DHW_VPD_thr = _thr_row("DHW_VPDmean_P75", iy)
        CHW_Tmax_thr = _thr_row("CHW_Tmax_P95", iy)
        CHW_VPD_thr = _thr_row("CHW_VPDmean_P90", iy)
        CHW_SM_thr = _thr_row("CHW_SM1mean_P10", iy)
        CHW_WS_thr = _thr_row("CHW_WS10mean_P50", iy)

        # Climatological statistic rows for standardized intensity.
        CLIM_Tmax_mean = _thr_row("CLIM_Tmax_mean", iy)
        CLIM_Tmax_std = _thr_row("CLIM_Tmax_std", iy)
        CLIM_VPD_mean = _thr_row("CLIM_VPDmean_mean", iy)
        CLIM_VPD_std = _thr_row("CLIM_VPDmean_std", iy)
        CLIM_SM_mean = _thr_row("CLIM_SM1mean_mean", iy)
        CLIM_SM_std = _thr_row("CLIM_SM1mean_std", iy)
        CLIM_WS_mean = _thr_row("CLIM_WS10mean_mean", iy)
        CLIM_WS_std = _thr_row("CLIM_WS10mean_std", iy)

        # Row-wise event masks. NaNs naturally evaluate to False in comparisons.
        flags_row = {
            "HW": (Tmean_row >= HW_Tmean_thr),
            "HHW": (Twbmax_row >= HHW_Twb_thr),
            "DHW": (Tmax_row >= DHW_Tmax_thr) & (VPD_row >= DHW_VPD_thr),
        }

        if use_soil:
            flags_row["CHW"] = (
                (Tmax_row >= CHW_Tmax_thr)
                & (VPD_row >= CHW_VPD_thr)
                & (SM_row <= CHW_SM_thr)
            )
        else:
            flags_row["CHW"] = (
                (Tmax_row >= CHW_Tmax_thr)
                & (VPD_row >= CHW_VPD_thr)
                & (WS_row <= CHW_WS_thr)
            )

        # Row-wise intensity metrics.
        hw_excess_row = np.where(Tmean_row - HW_Tmean_thr > 0, Tmean_row - HW_Tmean_thr, 0.0).astype(np.float32)
        hhw_excess_row = np.where(Twbmax_row - HHW_Twb_thr > 0, Twbmax_row - HHW_Twb_thr, 0.0).astype(np.float32)

        z_tmax_pos = _z_pos_row(Tmax_row, CLIM_Tmax_mean, CLIM_Tmax_std)
        z_vpd_pos = _z_pos_row(VPD_row, CLIM_VPD_mean, CLIM_VPD_std)

        dhw_excess_row = (
            DHW_WEIGHT_TMAX * z_tmax_pos
            + DHW_WEIGHT_VPD * z_vpd_pos
        ).astype(np.float32)

        if use_soil:
            # Soil dryness: positive anomaly of climatological mean minus actual soil moisture.
            z_sm_dry_pos = _z_pos_row(CLIM_SM_mean, SM_row, CLIM_SM_std)
            chw_excess_row = (
                CHW_WEIGHT_TMAX * z_tmax_pos
                + CHW_WEIGHT_VPD * z_vpd_pos
                + CHW_WEIGHT_THIRD * z_sm_dry_pos
            ).astype(np.float32)
        else:
            # Low-wind stagnation: positive anomaly of climatological mean minus actual wind speed.
            z_ws_low_pos = _z_pos_row(CLIM_WS_mean, WS_row, CLIM_WS_std)
            chw_excess_row = (
                CHW_WEIGHT_TMAX * z_tmax_pos
                + CHW_WEIGHT_VPD * z_vpd_pos
                + CHW_WEIGHT_THIRD * z_ws_low_pos
            ).astype(np.float32)

        intensity_row = {
            "HW": hw_excess_row,
            "HHW": hhw_excess_row,
            "DHW": dhw_excess_row,
            "CHW": chw_excess_row,
        }

        valid_row = np.isfinite(Tmean_row)

        for ix in valid_lon_idx:
            valid_cell = valid_row[:, ix]

            valid_by_year = {
                int(yy): int(np.sum(valid_cell[year_indices[int(yy)]]))
                for yy in year_axis
            }

            if sum(v >= MIN_VALID_DAYS_PER_YEAR for v in valid_by_year.values()) < MIN_VALID_YEARS_FOR_TREND:
                continue

            for kind in ["HW", "HHW", "DHW", "CHW"]:
                mask_cell = np.where(valid_cell, flags_row[kind][:, ix], False)
                exc_cell = intensity_row[kind][:, ix]

                annual_freq, annual_days, annual_int = detect_runs_1d(
                    mask_cell, exc_cell, years, min_len=MIN_LEN
                )

                for metric, arr in zip(
                    ["frequency", "duration", "intensity"],
                    [annual_freq, annual_days, annual_int],
                ):
                    annual_name = f"{kind}_{metric}"

                    series = np.array(
                        [
                            arr[int(y)] if valid_by_year[int(y)] >= MIN_VALID_DAYS_PER_YEAR else np.nan
                            for y in year_axis
                        ],
                        dtype=np.float32,
                    )

                    annual_data[annual_name][:, iy, ix] = series

                    sen_tr, mk_p, ols_tr, ols_p = robust_trend_per_decade(series, year_axis)
                    trend_data[f"{annual_name}_trend_decade"][iy, ix] = sen_tr
                    pval_data[f"{annual_name}_pvalue"][iy, ix] = mk_p
                    ols_trend_data[f"{annual_name}_OLS_trend_decade"][iy, ix] = ols_tr
                    ols_pval_data[f"{annual_name}_OLS_pvalue"][iy, ix] = ols_p

        # Explicitly release row-level temporaries before the next latitude.
        del (
            Tmean_row, Tmax_row, Twbmax_row, VPD_row, WS_row, SM_row,
            HW_Tmean_thr, HHW_Twb_thr, DHW_Tmax_thr, DHW_VPD_thr,
            CHW_Tmax_thr, CHW_VPD_thr, CHW_SM_thr, CHW_WS_thr,
            CLIM_Tmax_mean, CLIM_Tmax_std, CLIM_VPD_mean, CLIM_VPD_std,
            CLIM_SM_mean, CLIM_SM_std, CLIM_WS_mean, CLIM_WS_std,
            flags_row, intensity_row, valid_row,
            hw_excess_row, hhw_excess_row, z_tmax_pos, z_vpd_pos,
            dhw_excess_row, chw_excess_row
        )
        gc.collect()

    annual = xr.Dataset(
        {name: (("year", "lat", "lon"), data) for name, data in annual_data.items()},
        coords={"year": year_axis, "lat": lats, "lon": lons},
        attrs={
            "CHW_definition": (
                "Tmax≥P95 & VPD≥P90 & SM1≤P10"
                if use_soil else
                "Tmax≥P95 & VPD≥P90 & WS10≤P50"
            ),
            "DHW_definition": "Tmax≥P95 & VPD≥P75",
            "baseline": f"{BASE_START}–{BASE_END}",
            "minimum_event_length_days": MIN_LEN,
            "memory_mode": "row-wise latitude processing",
            "intensity_metric_note": (
                "HW/HHW intensity are cumulative threshold exceedances in degree-Celsius days; "
                "DHW/CHW intensity are dimensionless standardized severities. These metrics are "
                "not directly commensurate without an explicit harmonization step."
            ),
        },
    )

    trends = xr.Dataset(
        {
            **{name: (("lat", "lon"), data) for name, data in trend_data.items()},
            **{name: (("lat", "lon"), data) for name, data in pval_data.items()},
            **{name: (("lat", "lon"), data) for name, data in ols_trend_data.items()},
            **{name: (("lat", "lon"), data) for name, data in ols_pval_data.items()},
        },
        coords={"lat": lats, "lon": lons},
        attrs={
            "CHW_definition": annual.attrs["CHW_definition"],
            "DHW_definition": annual.attrs["DHW_definition"],
            "baseline": f"{BASE_START}–{BASE_END}",
            "minimum_event_length_days": MIN_LEN,
            "trend_units": "per decade",
            "main_trend_method": "Theil-Sen median slope",
            "main_significance_test": "Mann-Kendall two-sided p-value",
            "zscore_clip_range": f"{Z_CLIP_MIN} to {Z_CLIP_MAX}",
            "memory_mode": "row-wise latitude processing",
            "HW_intensity": "Σ(Tmean − P90)⁺ [°C·day]",
            "HHW_intensity": "Σ(Twbmax − P95)⁺ [°C·day]",
            "DHW_intensity": "0.5·z⁺(Tmax) + 0.5·z⁺(VPD) [dimensionless]",
            "CHW_intensity": "⅓·z⁺(Tmax) + ⅓·z⁺(VPD) + ⅓·z⁺(compound stressor) [dimensionless]",
            "intensity_metric_note": (
                "HW/HHW and DHW/CHW intensity trends use different native metrics/units. "
                "Do not directly subtract or compare their magnitudes without an explicit "
                "unit-consistent transformation."
            ),
            "VPD_internal_units": "hPa",
        },
    )

    return annual, trends


# ============================================================
# Figure generation
# ============================================================

def plot_figure1(trends, uf, biomes_gdf=None, sa_countries=None):
    """Generate the publication figure from the gridded trend dataset."""
    print("[INFO] Generating publication-ready Figure 1")
    
    # Figure setup
    fig, axes = plt.subplots(4, 3, figsize=(19, 21), constrained_layout=False)
    
    # Optimized spacing: increased gaps between subplots as requested
    fig.subplots_adjust(
        wspace=0.15,
        hspace=0.15,
        left=0.045,
        right=0.983,
        top=0.955,
        bottom=0.085
    )
    
    # Row definitions: ORDER MATCHES MANUSCRIPT SPECIFICATION
    rows = [
        ("HW",  "General heatwaves (HW)\nTmean ≥ P90"),
        ("HHW", "Humid heatwaves (HHW)\nTwbmax ≥ P95"), 
        ("DHW", "Dry heatwaves (DHW)\nTmax ≥ P95 & VPD ≥ P75"),
    ]
    
    chw_def = trends.attrs.get("CHW_definition", "")
    chw_label = ("Compound heatwaves (CHW)\nTmax ≥ P95 & VPD ≥ P90 & SM1 ≤ P10" 
                if "SM1mean" in chw_def else
                "Compound heatwaves (CHW)\nTmax ≥ P95 & VPD ≥ P90 & WS10 ≤ P50")
    rows.append(("CHW", chw_label))
    
    # Column definitions with precise units
    cols = [
        ("frequency", "Frequency", "events decade⁻¹", FREQ_VMIN, FREQ_VMAX),
        ("duration",  "Duration",  "days decade⁻¹",   DUR_VMIN, DUR_VMAX), 
        ("intensity", "Intensity", "severity decade⁻¹", INT_VMIN, INT_VMAX),
    ]
    
    # Colorblind-accessible diverging colormap
    cmap = plt.cm.RdBu_r
    
    # Panel labels: one letter per row only.
    panel_labels = ['a', 'b', 'c', 'd']
    
    # Define biome colors and linewidths (stronger colors: 0.5, weaker: 1.0)
    # Using a dictionary for biome-specific styling
    biome_styles = {
        'Amazon': {'color': '#006400', 'linewidth': 0.5, 'alpha': 1.0},       # Dark green - stronger
        'Cerrado': {'color': '#FF8C00', 'linewidth': 0.7, 'alpha': 0.7},        # Dark orange
        'Caatinga': {'color': '#8B4513', 'linewidth': 0.9, 'alpha': 0.8},       # Saddle brown - weaker
        'Atlantic Forest': {'color': '#228B22', 'linewidth': 0.6, 'alpha': 0.95}, # Forest green
        'Pampa': {'color': '#DAA520', 'linewidth': 1.0, 'alpha': 0.7},          # Goldenrod - weaker
        'Pantanal': {'color': '#1E90FF', 'linewidth': 0.8, 'alpha': 0.85},       # Dodger blue
    }
    
    # Default style for any other biome
    default_style = {'color': '#666666', 'linewidth': 0.7, 'alpha': 0.7}

    # Translate biome names from Brazilian shapefiles to English for styling and consistency.
    biome_name_map = {
        'amazonia': 'Amazon',
        'amazônia': 'Amazon',
        'amazon': 'Amazon',
        'cerrado': 'Cerrado',
        'caatinga': 'Caatinga',
        'mata atlantica': 'Atlantic Forest',
        'mata atlântica': 'Atlantic Forest',
        'atlantic forest': 'Atlantic Forest',
        'pampa': 'Pampa',
        'pantanal': 'Pantanal',
    }

    def translate_biome_name(raw_name):
        """Return the correct English biome name used in the figure."""
        key = str(raw_name).strip().lower()
        return biome_name_map.get(key, str(raw_name).strip())
    
    for i, (kind, row_label) in enumerate(rows):
        for j, (metric, col_label, unit, vmin, vmax) in enumerate(cols):
            ax = axes[i, j]
            panel_idx = i * 3 + j
            
            var = f"{kind}_{metric}_trend_decade"
            pvar = f"{kind}_{metric}_pvalue"
            
            # Diverging normalization centered at zero
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            
            # Main trend visualization
            im = ax.pcolormesh(
                trends.lon, trends.lat, trends[var],
                cmap=cmap, norm=norm, shading="auto", zorder=1,
                rasterized=True
            )
            
            # Significance stippling (Mann-Kendall p ≤ 0.05)
            sig = (trends[pvar] <= P_THRESHOLD).values
            lon2d, lat2d = np.meshgrid(trends.lon.values, trends.lat.values)
            
            # Adaptive subsampling for clear stippling
            step_y = max(1, sig.shape[0] // 50)
            step_x = max(1, sig.shape[1] // 50)
            sig_sub = sig[::step_y, ::step_x]
            
            ax.scatter(
                lon2d[::step_y, ::step_x][sig_sub],
                lat2d[::step_y, ::step_x][sig_sub], 
                s=7.0, c="black", marker=".", linewidths=0,
                alpha=0.85, zorder=3, rasterized=True
            )
            
            # ADD BIOME BOUNDARIES (just lines, no fill, with varying linewidths)
            if biomes_gdf is not None:
                for idx, biome in biomes_gdf.iterrows():
                    geometry = biome.geometry
                    if geometry is not None and not geometry.is_empty:
                        # Get biome name (try different possible column names)
                        biome_name = None
                        for col in ['nome', 'BIOMA', 'bioma', 'NAME', 'Bioma']:
                            if col in biome:
                                biome_name = biome[col]
                                break
                        
                        if biome_name is None:
                            biome_name = f"Biome_{idx}"
                        
                        # Select style based on the translated English biome name
                        biome_name_en = translate_biome_name(biome_name)
                        style = biome_styles.get(biome_name_en, default_style)
                        
                        # Convert to GeoSeries for easy plotting of boundaries
                        if geometry.geom_type == 'Polygon':
                            gs = gpd.GeoSeries([geometry], crs=biomes_gdf.crs)
                            gs.boundary.plot(ax=ax, color=style['color'], 
                                           linewidth=style['linewidth'],
                                           alpha=style['alpha'], zorder=2)
                        elif geometry.geom_type == 'MultiPolygon':
                            for poly in geometry.geoms:
                                gs = gpd.GeoSeries([poly], crs=biomes_gdf.crs)
                                gs.boundary.plot(ax=ax, color=style['color'],
                                               linewidth=style['linewidth'],
                                               alpha=style['alpha'], zorder=2)
            
            # ADD SOUTH AMERICAN COUNTRY BORDERS (if shapefile available)
            if sa_countries is not None:
                sa_countries.boundary.plot(ax=ax, color="#4a4a4a", linewidth=0.4, 
                                          linestyle='-', alpha=0.5, zorder=2)
            
            # Brazil boundary overlay (thicker, more prominent)
            uf.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.9, zorder=4)
            
            # Axis formatting with larger labels (font size 14 as requested)
            ax.set_xlim(LON_MIN, LON_MAX)
            ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.set_xlabel("Longitude", fontsize=FIG_FONT_SIZE, labelpad=3)
            ax.set_ylabel("Latitude", fontsize=FIG_FONT_SIZE, labelpad=3)
            ax.tick_params(labelsize=FIG_FONT_SIZE, direction="in", length=5, width=0.7)
            ax.grid(True, linestyle="--", alpha=0.12, linewidth=0.3)
            
            # Format longitude and latitude ticks
            lon_ticks = np.arange(-75, -30, 15)
            lat_ticks = np.arange(-35, 10, 10)
            ax.set_xticks(lon_ticks)
            ax.set_yticks(lat_ticks)
            ax.set_xticklabels([f"{int(x)}°" for x in lon_ticks])
            ax.set_yticklabels([f"{int(y)}°" for y in lat_ticks])
            
            # Column headers (top row only) - larger font
            if i == 0:
                ax.set_title(col_label, fontsize=FIG_FONT_SIZE, fontweight="semibold", pad=8)
            
            # Row labels (left column, rotated) - larger font
            if j == 0:
                ax.text(-0.27, 0.5, row_label, transform=ax.transAxes,
                       rotation=90, va="center", ha="center",
                       fontsize=FIG_FONT_SIZE, fontweight="semibold", linespacing=1.2)
            
            # Colorbar: compact, precise tick formatting
            cb = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.78, pad=0.012)
            cb.set_label(unit, fontsize=FIG_FONT_SIZE, labelpad=5)
            cb.ax.tick_params(labelsize=FIG_FONT_SIZE, length=4, width=0.5)
            
            # Format colorbar ticks
            if metric == "intensity":
                cb.locator.set_params(nbins=5)
            else:
                cb.locator.set_params(integer=True, nbins=5)
            cb.update_ticks()
            
            # Panel label: one letter per row only (a–d).
            if j == 0:
                ax.text(0.02, 0.978, panel_labels[i],
                        transform=ax.transAxes, fontsize=FIG_FONT_SIZE, fontweight="bold",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="none", alpha=0.9),
                        zorder=5)
    
    # Biome colour key shown in a framed box at the bottom of the figure.
    biome_legend_handles = [
        Line2D([0], [0], color=biome_styles['Amazon']['color'], lw=2.0, label='Amazon'),
        Line2D([0], [0], color=biome_styles['Cerrado']['color'], lw=2.0, label='Cerrado'),
        Line2D([0], [0], color=biome_styles['Caatinga']['color'], lw=2.0, label='Caatinga'),
        Line2D([0], [0], color=biome_styles['Atlantic Forest']['color'], lw=2.0, label='Atlantic Forest'),
        Line2D([0], [0], color=biome_styles['Pampa']['color'], lw=2.0, label='Pampa'),
        Line2D([0], [0], color=biome_styles['Pantanal']['color'], lw=2.0, label='Pantanal'),
    ]
    legend = fig.legend(
        handles=biome_legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.018),
        ncol=6,
        frameon=True,
        title='Biome boundary colors',
        fontsize=FIG_FONT_SIZE,
        title_fontsize=FIG_FONT_SIZE,
        handlelength=2.2,
        columnspacing=1.4,
        borderpad=0.6
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('0.35')
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(0.95)
    
    # Save outputs: vector PDF (primary) + high-res JPEG (preview)
    out_base = os.path.join(str(FIGURE_DIR), "figure_01_heatwave_trends")
    
    # PDF: vector format for manuscript submission
    fig.savefig(out_base + ".pdf", dpi=600, bbox_inches="tight", 
               facecolor="white", metadata={"Creator": "figure_01_heatwave_trends.py"})
    
    # JPEG: high-res raster for preview
    fig.savefig(out_base + ".jpeg", dpi=600, bbox_inches="tight", 
               facecolor="white", format="jpeg", pil_kwargs={"quality": 95})
    
    plt.close(fig)
    print(f"[OK] Saved: {os.path.basename(out_base)}.pdf (vector, publication-ready)")
    print(f"[OK] Saved: {os.path.basename(out_base)}.jpeg (350 dpi preview)")


# ============================================================
# Main execution workflow
# ============================================================

def parse_args():
    """Parse repository-facing command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Figure 01: Brazilian heatwave-regime trends from hourly ERA5."
        )
    )
    parser.add_argument(
        "--era5-dir",
        required=True,
        help="Directory containing hourly ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--brazil-shapefile",
        required=True,
        help="Brazil/UF boundary shapefile used to construct the analysis mask.",
    )
    parser.add_argument(
        "--biomes-shapefile",
        default=None,
        help="Optional Brazilian-biomes shapefile used for figure overlays.",
    )
    parser.add_argument(
        "--south-america-shapefile",
        default=None,
        help="Optional South America country-boundary shapefile used for figure overlays.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for cache, data, and figure outputs.",
    )
    parser.add_argument(
        "--overwrite-daily",
        action="store_true",
        help="Recompute yearly daily ERA5 cache files even when they already exist.",
    )
    parser.add_argument(
        "--recompute-thresholds",
        action="store_true",
        help="Recompute 1991–2020 climatological thresholds.",
    )
    parser.add_argument(
        "--recompute-annual-trends",
        action="store_true",
        help="Recompute annual heatwave metrics and trend products.",
    )
    return parser.parse_args()


def validate_runtime_inputs():
    """Fail early with clear messages when required inputs are unavailable."""
    if not os.path.isdir(ERA5_DIR):
        raise FileNotFoundError(f"ERA5 directory not found: {ERA5_DIR}")
    if not os.path.isfile(BRAZIL_SHP):
        raise FileNotFoundError(f"Brazil shapefile not found: {BRAZIL_SHP}")

    hourly = []
    for year in range(START_YEAR, END_YEAR + 1):
        hourly.extend(list_era5_files_for_year(year))
    if not hourly:
        raise FileNotFoundError(
            "No hourly ERA5 files matching "
            "'ERA5_single_levels_LatinAmerica_hourly_YYYY*.nc' were found in "
            f"{ERA5_DIR}"
        )


def main():
    args = parse_args()
    configure_runtime_paths(args)
    validate_runtime_inputs()

    print("=" * 78)
    print("FIGURE 01 — BRAZILIAN HEATWAVE-REGIME TRENDS")
    print("=" * 78)
    print(f"ERA5 directory : {ERA5_DIR}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print(f"Period         : {START_YEAR}–{END_YEAR}")
    print(f"Baseline       : {BASE_START}–{BASE_END}")
    print(f"Event duration : >= {MIN_LEN} consecutive days")
    print("=" * 78)
    print(
        "[METHOD NOTE] The calculations are preserved from the analysis workflow: "
        "HW/HHW intensity use cumulative °C·day exceedance, while DHW/CHW intensity "
        "use dimensionless standardized severity. Cross-regime intensity differences "
        "require an explicit harmonization step."
    )

    shp_path = find_brazil_shapefile()
    print(f"[INFO] Brazil shapefile: {os.path.basename(shp_path)}")

    biomes_gdf = load_biomes_shapefile()

    sa_shp_path = find_south_america_shapefile()
    sa_countries = None
    if sa_shp_path:
        try:
            sa_countries = gpd.read_file(sa_shp_path).to_crs(epsg=4326)
            if "CONTINENT" in sa_countries.columns:
                sa_countries = sa_countries[
                    sa_countries["CONTINENT"] == "South America"
                ]
            print(
                "[INFO] Loaded South American countries: "
                f"{os.path.basename(sa_shp_path)}"
            )
        except Exception as exc:
            print(f"[WARN] Could not load South American countries: {exc}")

    # ------------------------------------------------------------------
    # Stage 1: hourly ERA5 -> daily cache
    # ------------------------------------------------------------------
    daily_files = []
    for year in range(START_YEAR, END_YEAR + 1):
        out = process_daily_year(year, overwrite=args.overwrite_daily)
        if out:
            daily_files.append(out)

    if not daily_files:
        raise RuntimeError(
            "No daily ERA5 files were generated. Check --era5-dir and input names."
        )

    available_years = sorted(
        int(match.group(1))
        for f in daily_files
        for match in [re.search(r"ERA5_daily_Brazil_(\d{4})", os.path.basename(f))]
        if match
    )
    missing_years = [
        year for year in range(START_YEAR, END_YEAR + 1)
        if year not in available_years
    ]
    if missing_years:
        print(f"[WARN] Missing years: {missing_years}")

    # ------------------------------------------------------------------
    # Stage 2: thresholds, annual metrics, trends
    # ------------------------------------------------------------------
    ds_daily = open_daily_files_no_dask(daily_files)
    ds_daily = ds_daily.sel(
        time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    )
    ds_daily = remove_feb29(ds_daily)
    ds_daily = add_noleap_doy(ds_daily)

    mask_brazil, uf = make_brazil_mask(ds_daily, shp_path)
    print(f"[INFO] Brazil mask: {mask_brazil.sum().item():,.0f} grid cells")

    ds_thr = compute_or_load_thresholds(ds_daily, mask_brazil)

    annual_file = str(ANNUAL_FILE)
    trends_file = str(TRENDS_FILE)

    recompute = FORCE_RECOMPUTE_ANNUAL_TRENDS
    if os.path.exists(trends_file) and not recompute:
        try:
            with xr.open_dataset(trends_file) as old:
                if old.attrs.get("main_trend_method") != "Theil-Sen median slope":
                    recompute = True
        except Exception:
            recompute = True

    if os.path.exists(annual_file) and os.path.exists(trends_file) and not recompute:
        print("[INFO] Loading cached annual metrics and trends")
        annual = xr.open_dataset(annual_file)
        trends = xr.open_dataset(trends_file)
    else:
        annual, trends = compute_annual_metrics_and_trends(
            ds_daily, ds_thr, mask_brazil
        )

        for path in (annual_file, trends_file):
            if os.path.exists(path):
                os.remove(path)

        annual_encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for var in annual.data_vars
        }
        trend_encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for var in trends.data_vars
        }
        annual.to_netcdf(annual_file, encoding=annual_encoding)
        trends.to_netcdf(trends_file, encoding=trend_encoding)

        print(f"[OK] Saved: {annual_file}")
        print(f"[OK] Saved: {trends_file}")

    # ------------------------------------------------------------------
    # Stage 3: Figure 01
    # ------------------------------------------------------------------
    plot_figure1(trends, uf, biomes_gdf, sa_countries)

    ds_daily.close()
    ds_thr.close()
    annual.close()
    trends.close()

    print("[DONE] Figure 01 workflow completed.")
    print(f"[DONE] Figure: {FIGURE_DIR / 'figure_01_heatwave_trends.pdf'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

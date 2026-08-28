#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Station-level ERA5–INMET daily evaluation with Taylor diagrams.

Purpose
-------
Evaluate ERA5 against INMET station observations using paired daily values at
individual stations. Temperature, relative humidity, and 10-m wind speed are
assessed using Pearson correlation, bias, RMSE, MAE, standard-deviation ratio,
and Taylor diagrams.

Quality control
---------------
A station-day is retained for a variable only when the required hourly
completeness criterion is satisfied. Station-years are retained only when the
paired daily coverage exceeds the specified valid-day fraction, and each
station-variable must contain a minimum number of retained years and paired
days.

An optional conservative pairwise residual QC removes only isolated
station-variable ERA5–INMET differences that are extreme relative to each
station's own residual distribution and exceed a variable-specific absolute
threshold. Removal is capped per station-variable.

Time alignment
--------------
INMET automatic-station files generally report UTC and ERA5 valid times are
also UTC. UTC-day aggregation is therefore the default. Optional fixed
state-level offsets may be applied to both datasets with ``local_by_uf``.
Historical daylight-saving-time transitions are not reconstructed.

ERA5 input modes
----------------
``cache``
    Reuse an explicitly supplied station-daily ERA5 CSV.

``daily``
    Extract station values from preprocessed daily ERA5 files.

``hourly``
    Extract station values from hourly ERA5 files and aggregate them to daily
    values using the selected time basis.

Station coordinates
-------------------
Official INMET climatological-normal metadata may be supplied to control map
coordinates. An optional trusted station-selection table may also be supplied
to align the station population with another validation workflow. These inputs
affect station selection and map locations, not the paired-value calculations.

Usage
-----
python era5_inmet_station_daily_taylor_validation.py \
    --inmet-dir /path/to/inmet_hourly \
    --era5-source cache \
    --era5-station-cache /path/to/era5_station_daily.csv \
    --station-metadata-dir /path/to/inmet_normals_1991_2020 \
    --trusted-station-table /path/to/station_selection.csv \
    --states-shapefile /path/to/brazil_states.shp \
    --output-dir ./outputs/era5_inmet_daily_validation \
    --start 2002-01-01 \
    --end 2024-12-31 \
    --min-hours-day 18 \
    --min-valid-day-fraction 0.70 \
    --min-valid-years-variable 5 \
    --min-pairs 365 \
    --daily-time-basis utc

For ``daily`` mode, provide ``--era5-daily-dir``.
For ``hourly`` mode, provide ``--era5-hourly-glob``.
"""

import os
import re
import glob
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr


# ============================================================
# Defaults
# ============================================================
START_DATE = "2002-01-01"
END_DATE = "2024-12-31"
MIN_HOURS_PER_DAY = 18
MIN_PAIRED_DAYS = 365
MIN_VALID_DAY_FRACTION = 0.70
MIN_VALID_YEARS_VARIABLE = 5
INMET_FILE_PATTERNS = ["**/*.csv", "**/*.CSV"]
SOFTWARE_VERSION = "1.0.0"

# Fixed Brazilian standard-time offsets relative to UTC by state.
# INMET automatic hourly files generally use UTC ("Hora UTC"), and ERA5 is UTC.
# These offsets are used only when --daily_time_basis local_by_uf is selected.
# They intentionally do not include historical daylight-saving-time transitions.
UF_UTC_OFFSET_HOURS = {
    "AC": -5,
    "AM": -4, "RO": -4, "RR": -4, "MT": -4, "MS": -4,
    "AL": -3, "AP": -3, "BA": -3, "CE": -3, "DF": -3, "ES": -3,
    "GO": -3, "MA": -3, "MG": -3, "PA": -3, "PB": -3, "PE": -3,
    "PI": -3, "PR": -3, "RJ": -3, "RN": -3, "RS": -3, "SC": -3,
    "SE": -3, "SP": -3, "TO": -3,
}

TIME_BASIS_OPTIONS = {"utc", "fixed_shift", "local_by_uf"}

BRAZIL_EXTENT = (-75.0, -32.0, -35.0, 6.0)

UF_TO_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}

VARIABLES = {
    "temperature": {
        "label": "Temperature",
        "short": "TEMP",
        "unit": "°C",
        "obs_candidates": [
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA",
            "temperature", "temperatura", "temp", "tmed", "tmean",
            "temperatura_do_ar_bulbo_seco_horaria",
        ],
        "era5_candidates": ["t2m", "2t", "T2M", "T2", "temperature", "2m_temperature", "var167"],
        "bias_norm": TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5),
        "rmse_norm": Normalize(0, 6),
    },
    "relative_humidity": {
        "label": "Relative humidity",
        "short": "RH",
        "unit": "%",
        "obs_candidates": [
            "UMIDADE RELATIVA DO AR, HORARIA (%)",
            "UMIDADE RELATIVA DO AR, HORARIA",
            "relative humidity", "humidity", "umidade", "umidade relativa",
            "RH", "rh", "UR", "umid", "rh_pct",
        ],
        "era5_candidates": ["rh", "RH", "relative_humidity", "relative_humidity_2m"],
        "dewpoint_candidates": ["d2m", "2d", "D2M", "dewpoint", "2m_dewpoint_temperature", "var168"],
        "bias_norm": TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25),
        "rmse_norm": Normalize(0, 35),
    },
    "wind_speed": {
        "label": "Wind speed",
        "short": "WS",
        "unit": "m s$^{-1}$",
        "obs_candidates": [
            "VENTO, VELOCIDADE HORARIA (m/s)",
            "VENTO, VELOCIDADE HORARIA",
            "wind speed", "vento", "velocidade do vento", "vento velocidade",
            "ws", "ws10", "wind_ms",
        ],
        "era5_candidates": ["si10", "10si", "wind_speed", "10m_wind_speed"],
        "u_candidates": ["u10", "10u", "U10", "10m_u_component_of_wind", "var165"],
        "v_candidates": ["v10", "10v", "V10", "10m_v_component_of_wind", "var166"],
        "bias_norm": TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4),
        "rmse_norm": Normalize(0, 6),
    },
}

VAR_ORDER = ["temperature", "relative_humidity", "wind_speed"]


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_to_csv(df: pd.DataFrame, path: str, **kwargs):
    """Write CSV atomically to avoid corrupted caches if the job is interrupted."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def valid_csv_cache(path: str, min_bytes: int = 100) -> bool:
    """Return True only if a cache file exists and is not empty/corrupted."""
    return os.path.exists(path) and os.path.getsize(path) > min_bytes


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    accents = {
        "á": "a", "à": "a", "â": "a", "ã": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c",
    }
    for a, b in accents.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def clean_numeric_series(x: pd.Series) -> pd.Series:
    if x.dtype.kind in "biufc":
        return pd.to_numeric(x, errors="coerce")
    y = x.astype(str).str.strip()
    y = y.str.replace(",", ".", regex=False)
    y = y.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(y, errors="coerce")


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    norm_map = {normalize_name(c): c for c in df.columns}
    cand_norm = [normalize_name(c) for c in candidates]
    for c in cand_norm:
        if c in norm_map:
            return norm_map[c]
    for c in cand_norm:
        for norm_col, original in norm_map.items():
            if c and (c in norm_col or norm_col in c):
                return original
    if required:
        raise KeyError(f"Could not find required column among {candidates}. Available: {list(df.columns)}")
    return None


def find_variable(ds: xr.Dataset, candidates: List[str], required: bool = False) -> Optional[str]:
    lower_map = {v.lower(): v for v in ds.data_vars}
    norm_map = {normalize_name(v): v for v in ds.data_vars}
    for c in candidates:
        if c in ds.data_vars:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
        nc = normalize_name(c)
        if nc in norm_map:
            return norm_map[nc]
    for c in candidates:
        nc = normalize_name(c)
        for nv, original in norm_map.items():
            if nc and (nc in nv or nv in nc):
                return original
    if required:
        raise KeyError(f"Could not find variable among {candidates}. Available: {list(ds.data_vars)}")
    return None


def open_dataset_robust(path: str) -> xr.Dataset:
    """Open NetCDF files while avoiding fragile netCDF4/time-decoding paths.

    The function tries multiple xarray backends and decodes the time coordinate explicitly when required.
    """
    attempts = [
        ("h5netcdf", {"phony_dims": "sort"}, {"decode_times": False, "mask_and_scale": True}),
        ("h5netcdf", {"phony_dims": "sort"}, {"decode_times": False, "mask_and_scale": False}),
        ("scipy", {}, {"decode_times": False, "mask_and_scale": True}),
        ("netcdf4", {}, {"decode_times": False, "mask_and_scale": True}),
    ]
    errors = []
    for engine, backend_kwargs, open_kwargs in attempts:
        try:
            ds = xr.open_dataset(path, engine=engine, backend_kwargs=backend_kwargs, **open_kwargs)
            print(
                f"[INFO] Opened NetCDF with xarray engine='{engine}' "
                f"(decode_times=False): {Path(path).name}"
            )
            return ds
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
            continue
    msg = "\n".join(errors)
    raise RuntimeError(
        "Could not open NetCDF file with h5netcdf, scipy, or netcdf4 backends. "
        "This is an I/O/backend problem, not a evaluation-method problem.\n"
        f"File: {path}\nAttempts:\n{msg}"
    )


def decode_cf_time_to_pandas(time_values, attrs=None) -> pd.DatetimeIndex:
    """Decode common CF time coordinates without requiring cftime/netCDF4.

    Supports the ERA5-style units typically used in NetCDF files, such as
    'hours since 1900-01-01 00:00:00' or 'seconds since 1970-01-01'. If the
    coordinate is already datetime-like, it is returned directly.
    """
    vals = np.asarray(time_values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(vals))

    attrs = attrs or {}
    units = str(attrs.get("units", "")).strip()
    calendar = str(attrs.get("calendar", "standard")).lower()

    # Common fallback for already parseable strings.
    if vals.dtype.kind in {"U", "S", "O"}:
        dt = pd.to_datetime(vals, errors="coerce")
        if pd.Series(dt).notna().mean() > 0.8:
            return pd.DatetimeIndex(dt)

    m = re.match(
        r"^(seconds|second|minutes|minute|hours|hour|days|day)\s+since\s+(.+)$",
        units,
        flags=re.IGNORECASE,
    )
    if not m:
        # Last-resort: numeric YYYYMMDD or pandas numeric parsing.
        vals_float = pd.to_numeric(pd.Series(vals.ravel()), errors="coerce")
        if vals_float.notna().any():
            as_int = vals_float.round().astype("Int64").astype(str)
            dt = pd.to_datetime(as_int, format="%Y%m%d", errors="coerce")
            if dt.notna().mean() > 0.8:
                return pd.DatetimeIndex(dt)
        dt = pd.to_datetime(vals, errors="coerce")
        if pd.Series(dt).notna().mean() > 0.8:
            return pd.DatetimeIndex(dt)
        raise ValueError(f"Could not decode time coordinate. units={units!r}, calendar={calendar!r}")

    unit, origin = m.group(1).lower(), m.group(2).strip()
    # Remove timezone markers that pandas may interpret inconsistently.
    origin = origin.replace("UTC", "").replace("utc", "").strip()
    origin = re.sub(r"\s+0:00$", " 00:00:00", origin)
    base = pd.to_datetime(origin, errors="coerce")
    if pd.isna(base):
        # Some files use dates like '1900-01-01 00:00:0.0'.
        origin2 = re.sub(r"(\d{2}:\d{2}:\d{2})\.0+$", r"\1", origin)
        base = pd.to_datetime(origin2, errors="coerce")
    if pd.isna(base):
        raise ValueError(f"Could not parse time origin from units={units!r}")

    vals_num = pd.to_numeric(pd.Series(vals.ravel()), errors="coerce").to_numpy(float)
    if unit.startswith("second"):
        delta = pd.to_timedelta(vals_num, unit="s")
    elif unit.startswith("minute"):
        delta = pd.to_timedelta(vals_num, unit="m")
    elif unit.startswith("hour"):
        delta = pd.to_timedelta(vals_num, unit="h")
    elif unit.startswith("day"):
        delta = pd.to_timedelta(vals_num, unit="D")
    else:
        raise ValueError(f"Unsupported time unit in {units!r}")
    return pd.DatetimeIndex(base + delta)

def parse_inmet_datetime(date_series: pd.Series, hour_series: Optional[pd.Series] = None) -> pd.Series:
    date_str = date_series.astype(str).str.strip()
    if hour_series is not None:
        hour_str = hour_series.astype(str).str.strip()
        hour_clean = hour_str.str.extract(r"(\d{1,4})")[0].fillna("0")

        def fmt_hour(h):
            h = str(h)
            try:
                if len(h) <= 2:
                    return f"{int(h):02d}:00"
                return f"{int(h[:2]):02d}:{int(h[2:4] or 0):02d}"
            except Exception:
                return "00:00"

        combined = date_str + " " + hour_clean.map(fmt_hour)
        formats = ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M"]
    else:
        combined = date_str
        formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]

    dt_best = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")
    for fmt in formats:
        dt_try = pd.to_datetime(combined, format=fmt, errors="coerce")
        fill = dt_best.isna() & dt_try.notna()
        dt_best.loc[fill] = dt_try.loc[fill]
        if dt_best.notna().mean() > 0.99:
            break
    if dt_best.notna().mean() < 0.5:
        dt_try = pd.to_datetime(combined, errors="coerce", dayfirst=True)
        fill = dt_best.isna() & dt_try.notna()
        dt_best.loc[fill] = dt_try.loc[fill]
    return dt_best


def standardize_lat_lon_time(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for time_name in ["time", "valid_time", "datetime", "date"]:
        if time_name in ds.coords or time_name in ds.dims:
            if time_name != "time":
                rename[time_name] = "time"
            break
    for lat_name in ["latitude", "Latitude", "LAT", "lat", "y"]:
        if lat_name in ds.coords or lat_name in ds.dims:
            if lat_name != "lat":
                rename[lat_name] = "lat"
            break
    for lon_name in ["longitude", "Longitude", "LON", "lon", "x"]:
        if lon_name in ds.coords or lon_name in ds.dims:
            if lon_name != "lon":
                rename[lon_name] = "lon"
            break
    if rename:
        ds = ds.rename(rename)
    if "time" not in ds.coords or "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("ERA5 dataset must contain time, latitude and longitude coordinates.")
    time_attrs = dict(ds["time"].attrs) if hasattr(ds["time"], "attrs") else {}
    ds = ds.assign_coords(time=decode_cf_time_to_pandas(ds["time"].values, time_attrs))
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds


def reduce_to_time_lat_lon(da: xr.DataArray) -> xr.DataArray:
    keep = {"time", "lat", "lon"}
    for dim in list(da.dims):
        if dim not in keep:
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0})
            else:
                da = da.mean(dim=dim, skipna=True)
    return da


def maybe_kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    med = float(np.nanmedian(da.values))
    if np.isfinite(med) and med > 100:
        return da - 273.15
    return da


def saturation_vapour_pressure_hpa(temp_c):
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def compute_rh_from_t_td(t_k_or_c: xr.DataArray, td_k_or_c: xr.DataArray) -> xr.DataArray:
    t_c = maybe_kelvin_to_celsius(t_k_or_c)
    td_c = maybe_kelvin_to_celsius(td_k_or_c)
    es = saturation_vapour_pressure_hpa(t_c)
    ea = saturation_vapour_pressure_hpa(td_c)
    return (100.0 * ea / es).clip(min=0, max=100)


def validate_time_basis(daily_time_basis: str):
    if daily_time_basis not in TIME_BASIS_OPTIONS:
        raise ValueError(f"Invalid daily_time_basis={daily_time_basis}. Choose one of {sorted(TIME_BASIS_OPTIONS)}")


def get_time_shift_hours_for_uf(uf: str, daily_time_basis: str, fixed_time_shift_hours: int) -> int:
    """Return the time shift applied before daily aggregation."""
    validate_time_basis(daily_time_basis)
    if daily_time_basis == "utc":
        return 0
    if daily_time_basis == "fixed_shift":
        return int(fixed_time_shift_hours)
    return int(UF_UTC_OFFSET_HOURS.get(str(uf).upper(), -3))


def time_basis_tag(daily_time_basis: str, fixed_time_shift_hours: int) -> str:
    if daily_time_basis == "utc":
        return "UTCday"
    if daily_time_basis == "fixed_shift":
        sign = "p" if fixed_time_shift_hours >= 0 else "m"
        return f"FixedShift_{sign}{abs(int(fixed_time_shift_hours))}h"
    return "LocalUFday"


def apply_daily_time_shift_to_observations(obs: pd.DataFrame, daily_time_basis: str, fixed_time_shift_hours: int) -> pd.Series:
    """Return the timestamp used to assign INMET hourly observations to daily bins."""
    validate_time_basis(daily_time_basis)
    dt = pd.to_datetime(obs["datetime"], errors="coerce")
    if daily_time_basis == "utc":
        return dt
    if daily_time_basis == "fixed_shift":
        return dt + pd.to_timedelta(int(fixed_time_shift_hours), unit="h")
    shifts = obs["uf"].astype(str).str.upper().map(lambda u: UF_UTC_OFFSET_HOURS.get(u, -3))
    return dt + pd.to_timedelta(shifts, unit="h")


def resample_station_wide_to_daily(df_wide: pd.DataFrame, station_meta: pd.DataFrame, var: str,
                                   min_hours_day: int, daily_time_basis: str,
                                   fixed_time_shift_hours: int) -> pd.DataFrame:
    """Aggregate station-hour ERA5 values to daily station values using the selected day definition."""
    validate_time_basis(daily_time_basis)

    def _stack_daily(tmp: pd.DataFrame) -> pd.DataFrame:
        daily_mean = tmp.resample("D").mean()
        daily_count = tmp.resample("D").count()
        try:
            long_mean = daily_mean.stack(future_stack=True).rename(f"{var}_era5").reset_index()
            long_count = daily_count.stack(future_stack=True).rename(f"{var}_n_hours_era5").reset_index()
        except TypeError:
            long_mean = daily_mean.stack(dropna=False).rename(f"{var}_era5").reset_index()
            long_count = daily_count.stack(dropna=False).rename(f"{var}_n_hours_era5").reset_index()
        long_mean.columns = ["date", "station_id", f"{var}_era5"]
        long_count.columns = ["date", "station_id", f"{var}_n_hours_era5"]
        long = long_mean.merge(long_count, on=["date", "station_id"], how="left")
        long.loc[long[f"{var}_n_hours_era5"] < min_hours_day, f"{var}_era5"] = np.nan
        return long

    if daily_time_basis in ["utc", "fixed_shift"]:
        tmp = df_wide.copy()
        shift = get_time_shift_hours_for_uf("SP", daily_time_basis, fixed_time_shift_hours)
        if shift != 0:
            tmp.index = tmp.index + pd.Timedelta(hours=shift)
        return _stack_daily(tmp)

    parts = []
    meta = station_meta.copy()
    meta["station_id"] = meta["station_id"].astype(str)
    meta["time_shift_hours"] = meta["uf"].astype(str).str.upper().map(lambda u: UF_UTC_OFFSET_HOURS.get(u, -3))
    for shift, sub in meta.groupby("time_shift_hours"):
        cols = [c for c in sub["station_id"].astype(str).tolist() if c in df_wide.columns]
        if not cols:
            continue
        tmp = df_wide[cols].copy()
        tmp.index = tmp.index + pd.Timedelta(hours=int(shift))
        parts.append(_stack_daily(tmp))
    if not parts:
        return pd.DataFrame(columns=["date", "station_id", f"{var}_era5", f"{var}_n_hours_era5"])
    return pd.concat(parts, ignore_index=True)


# ============================================================
# INMET processing
# ============================================================
def read_inmet_csv(path: str) -> Optional[pd.DataFrame]:
    if Path(path).suffix.lower() != ".csv" or "readme" in Path(path).name.lower():
        return None

    header_idx = 0
    try:
        with open(path, "r", encoding="latin1", errors="ignore") as f:
            lines = [next(f) for _ in range(20)]
        for i, line in enumerate(lines):
            low = normalize_name(line)
            if ("data" in low and ("hora" in low or "time" in low)) or ("time_stamp" in low):
                header_idx = i
                break
    except Exception:
        header_idx = 0

    attempts = [
        dict(sep=";", decimal=",", encoding="latin1", skiprows=header_idx, engine="python"),
        dict(sep=",", decimal=".", encoding="utf-8", skiprows=header_idx, engine="python"),
        dict(sep=";", decimal=".", encoding="utf-8", skiprows=header_idx, engine="python"),
    ]
    df = None
    for kwargs in attempts:
        try:
            tmp = pd.read_csv(path, **kwargs).dropna(axis=1, how="all")
            if tmp.shape[1] >= 3:
                df = tmp
                break
        except Exception:
            continue
    if df is None or df.empty:
        return None

    df.columns = [str(c).strip() for c in df.columns]

    lat_col = find_column(df, ["latitude", "lat"], required=False)
    lon_col = find_column(df, ["longitude", "lon"], required=False)
    uf_col = find_column(df, ["uf", "estado", "state"], required=False)
    station_col = find_column(df, ["codigo", "cod_estacao", "station", "estacao", "wmo"], required=False)

    lat = clean_numeric_series(df[lat_col]).median() if lat_col else np.nan
    lon = clean_numeric_series(df[lon_col]).median() if lon_col else np.nan
    uf = None
    station = None

    if uf_col:
        vals = df[uf_col].dropna().astype(str).str.upper().str.extract(r"([A-Z]{2})")[0].dropna()
        uf = vals.iloc[0] if not vals.empty else None
    if station_col:
        vals = df[station_col].dropna().astype(str)
        station = vals.iloc[0] if not vals.empty else None

    try:
        with open(path, "r", encoding="latin1", errors="ignore") as f:
            meta_lines = []
            for _ in range(15):
                try:
                    meta_lines.append(next(f).strip())
                except StopIteration:
                    break

        def meta_value(keys):
            for line in meta_lines:
                parts = line.split(";")
                if len(parts) >= 2:
                    key = normalize_name(parts[0])
                    if any(normalize_name(k) in key for k in keys):
                        return parts[1].strip()
            return None

        if not np.isfinite(lat):
            v = meta_value(["latitude"])
            if v is not None:
                lat = float(v.replace(",", "."))
        if not np.isfinite(lon):
            v = meta_value(["longitude"])
            if v is not None:
                lon = float(v.replace(",", "."))
        if uf is None:
            v = meta_value(["uf"])
            if v is not None:
                m = re.search(r"([A-Z]{2})", v.upper())
                uf = m.group(1) if m else None
        code_v = meta_value(["codigo", "cod_estacao", "codigo_wmo"])
        if code_v is not None and str(code_v).strip():
            station = str(code_v).strip()
    except Exception:
        pass

    if station is None:
        station = Path(path).stem

    date_col = find_column(df, ["Data", "date", "data", "time_stamp", "datetime", "datetimeutc", "datetimelocal"], required=False)
    hour_col = find_column(df, ["Hora UTC", "hora", "hour", "hora_utc"], required=False)
    if date_col is None:
        return None

    dt = parse_inmet_datetime(df[date_col], df[hour_col] if hour_col and hour_col != date_col else None)

    out = pd.DataFrame({
        "datetime": dt,
        "station_id": str(station),
        "file": path,
        "lat": lat,
        "lon": lon,
        "uf": uf,
    })

    for var, meta in VARIABLES.items():
        c = find_column(df, meta["obs_candidates"], required=False)
        out[var] = clean_numeric_series(df[c]) if c else np.nan

    out = out.dropna(subset=["datetime"])
    if out.empty:
        return None

    # Physical range filtering only, not completeness filtering.
    out.loc[(out["temperature"] < -20) | (out["temperature"] > 55), "temperature"] = np.nan
    out.loc[(out["relative_humidity"] < 0) | (out["relative_humidity"] > 100), "relative_humidity"] = np.nan
    out.loc[(out["wind_speed"] < 0) | (out["wind_speed"] > 60), "wind_speed"] = np.nan
    return out


def read_all_inmet(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = []
    for pat in INMET_FILE_PATTERNS:
        files.extend(glob.glob(os.path.join(base_dir, pat), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No INMET CSV files found in {base_dir}")

    print(f"[INFO] Found {len(files)} INMET CSV files.")
    frames, inv_rows = [], []
    for i, path in enumerate(files, 1):
        if i % 500 == 0 or i == len(files):
            print(f"       INMET files processed: {i}/{len(files)}")
        df = read_inmet_csv(path)
        if df is None or df.empty:
            continue

        if df["uf"].isna().all() or df["uf"].iloc[0] is None:
            text = str(path).upper()
            for uf0 in UF_TO_REGION:
                if re.search(rf"[/_\-\s]{uf0}[/_\-\s]", text):
                    df["uf"] = uf0
                    break

        uf_vals = df["uf"].dropna().astype(str).str.upper().str.extract(r"([A-Z]{2})")[0].dropna()
        uf = uf_vals.iloc[0] if not uf_vals.empty else None
        lat = pd.to_numeric(df["lat"], errors="coerce").median()
        lon = pd.to_numeric(df["lon"], errors="coerce").median()
        if uf not in UF_TO_REGION or not np.isfinite(lat) or not np.isfinite(lon):
            continue

        df["uf"] = uf
        df["lat"] = lat
        df["lon"] = lon
        df["station_id"] = df["station_id"].astype(str)
        frames.append(df)
        inv_rows.append({
            "station_id": str(df["station_id"].iloc[0]),
            "uf": uf,
            "region": UF_TO_REGION.get(uf),
            "lat": lat,
            "lon": lon,
            "file": path,
            "first_date": df["datetime"].min(),
            "last_date": df["datetime"].max(),
            "n_records": len(df),
            **{f"has_{var}": int(df[var].notna().any()) for var in VARIABLES},
        })

    if not frames:
        raise RuntimeError("No valid INMET station files could be read.")

    obs = pd.concat(frames, ignore_index=True)
    obs = obs[(obs["datetime"] >= pd.Timestamp(START_DATE)) &
              (obs["datetime"] < pd.Timestamp(END_DATE) + pd.Timedelta(days=1))]
    obs = obs.sort_values(["station_id", "datetime"]).drop_duplicates(
        subset=["station_id", "uf", "datetime"], keep="last"
    )

    inv = pd.DataFrame(inv_rows)
    inv_unique = (
        inv.sort_values(["station_id", "first_date"])
        .drop_duplicates(subset=["station_id", "uf"], keep="first")
        .reset_index(drop=True)
    )
    return obs, inv_unique


def inmet_hourly_to_daily_station(obs: pd.DataFrame, min_hours_day: int,
                                  daily_time_basis: str = "utc",
                                  fixed_time_shift_hours: int = 0) -> pd.DataFrame:
    obs = obs.copy()
    obs["datetime_for_daily"] = apply_daily_time_shift_to_observations(
        obs, daily_time_basis=daily_time_basis, fixed_time_shift_hours=fixed_time_shift_hours
    )
    obs["date"] = pd.to_datetime(obs["datetime_for_daily"]).dt.floor("D")
    rows = []
    group_cols = ["station_id", "uf", "date"]
    for keys, g in obs.groupby(group_cols, dropna=False):
        station_id, uf, date = keys
        row = {"station_id": str(station_id), "uf": uf, "date": pd.Timestamp(date)}
        row["lat"] = pd.to_numeric(g["lat"], errors="coerce").median()
        row["lon"] = pd.to_numeric(g["lon"], errors="coerce").median()
        for var in VARIABLES:
            vals = pd.to_numeric(g[var], errors="coerce")
            n_valid = int(vals.notna().sum())
            row[f"{var}_obs"] = vals.mean() if n_valid >= min_hours_day else np.nan
            row[f"{var}_n_hours_obs"] = n_valid
        rows.append(row)
    daily = pd.DataFrame(rows)
    return daily.sort_values(["uf", "station_id", "date"])


# ============================================================
# ERA5 processing
# ============================================================
def resolve_era5_hourly_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ERA5 hourly files found with pattern:\n{pattern}")
    y0 = pd.Timestamp(START_DATE).year
    y1 = pd.Timestamp(END_DATE).year
    selected = []
    for f in files:
        m = re.search(r"(19|20)\d{2}", Path(f).name)
        if m:
            y = int(m.group(0))
            if y0 <= y <= y1:
                selected.append(f)
        else:
            selected.append(f)
    if not selected:
        raise FileNotFoundError("ERA5 files found, but none overlaps requested period.")
    return selected


def get_station_grid_indices(ds: xr.Dataset, inventory: pd.DataFrame) -> pd.DataFrame:
    lats = ds["lat"].values
    lons = ds["lon"].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    rows = []
    for _, st in inventory.iterrows():
        dist, flat_idx = tree.query([st["lat"], st["lon"]])
        iy, ix = np.unravel_index(flat_idx, lat2d.shape)
        rows.append({
            "station_id": str(st["station_id"]),
            "uf": st["uf"],
            "lat": st["lat"],
            "lon": st["lon"],
            "iy": int(iy),
            "ix": int(ix),
            "era5_lat": float(lats[iy]),
            "era5_lon": float(lons[ix]),
            "era5_grid_distance_deg": float(dist),
        })
    return pd.DataFrame(rows)


def build_era5_hourly_validation_dataset(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset(coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]})
    tvar = find_variable(ds, VARIABLES["temperature"]["era5_candidates"], required=True)
    print(f"[INFO] ERA5 temperature variable: {tvar}")
    t = reduce_to_time_lat_lon(ds[tvar])
    out["temperature"] = maybe_kelvin_to_celsius(t)

    rhvar = find_variable(ds, VARIABLES["relative_humidity"]["era5_candidates"], required=False)
    if rhvar is not None:
        rh = reduce_to_time_lat_lon(ds[rhvar])
        med = float(np.nanmedian(rh.values))
        if np.isfinite(med) and med <= 1.5:
            rh = rh * 100.0
        out["relative_humidity"] = rh.clip(min=0, max=100)
    else:
        tdvar = find_variable(ds, VARIABLES["relative_humidity"]["dewpoint_candidates"], required=True)
        print(f"[INFO] ERA5 dewpoint variable for RH: {tdvar}")
        td = reduce_to_time_lat_lon(ds[tdvar])
        out["relative_humidity"] = compute_rh_from_t_td(t, td)

    wsvar = find_variable(ds, VARIABLES["wind_speed"]["era5_candidates"], required=False)
    if wsvar is not None:
        out["wind_speed"] = reduce_to_time_lat_lon(ds[wsvar])
    else:
        uvar = find_variable(ds, VARIABLES["wind_speed"]["u_candidates"], required=True)
        vvar = find_variable(ds, VARIABLES["wind_speed"]["v_candidates"], required=True)
        print(f"[INFO] ERA5 wind variables: {uvar}, {vvar}")
        out["wind_speed"] = np.sqrt(reduce_to_time_lat_lon(ds[uvar]) ** 2 + reduce_to_time_lat_lon(ds[vvar]) ** 2)
    return out


def extract_era5_hourly_to_daily_station(file_path: str, inventory: pd.DataFrame,
                                          grid_index: Optional[pd.DataFrame], min_hours_day: int,
                                          daily_time_basis: str = "utc",
                                          fixed_time_shift_hours: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fast station-only ERA5 extraction.

    Nearest ERA5 grid cells are selected before computing station-level temperature, relative humidity, and wind speed.
    """
    print(f"[INFO] ERA5 hourly file: {Path(file_path).name}")
    ds_raw = open_dataset_robust(file_path)
    ds_raw = standardize_lat_lon_time(ds_raw)
    ds_raw = ds_raw.sel(time=slice(START_DATE, END_DATE))

    if ds_raw.sizes.get("time", 0) == 0:
        ds_raw.close()
        return pd.DataFrame(), grid_index

    if grid_index is None:
        grid_index = get_station_grid_indices(ds_raw, inventory)
        print(f"[INFO] ERA5 nearest grid cells prepared for {len(grid_index)} unique stations.")

    station_dim = "station"
    iy_da = xr.DataArray(grid_index["iy"].to_numpy(), dims=station_dim)
    ix_da = xr.DataArray(grid_index["ix"].to_numpy(), dims=station_dim)
    station_ids = grid_index["station_id"].astype(str).to_numpy()
    time_index = pd.to_datetime(ds_raw["time"].values)

    station_meta = grid_index[["station_id", "uf", "lat", "lon", "era5_lat", "era5_lon", "era5_grid_distance_deg"]].copy()
    station_meta["station_id"] = station_meta["station_id"].astype(str)

    # Resolve original ERA5 variables once.
    tvar = find_variable(ds_raw, VARIABLES["temperature"]["era5_candidates"], required=True)
    print(f"[INFO] ERA5 temperature variable: {tvar}")
    t_da = reduce_to_time_lat_lon(ds_raw[tvar]).isel(lat=iy_da, lon=ix_da).transpose("time", station_dim)
    t_c = maybe_kelvin_to_celsius(t_da).load()

    tdvar = None
    rh_da = None
    rhvar = find_variable(ds_raw, VARIABLES["relative_humidity"].get("era5_candidates", []), required=False)
    if rhvar is not None:
        print(f"[INFO] ERA5 relative humidity variable: {rhvar}")
        rh_da = reduce_to_time_lat_lon(ds_raw[rhvar]).isel(lat=iy_da, lon=ix_da).transpose("time", station_dim)
        rh_da = rh_da.load()
        med = float(np.nanmedian(rh_da.values))
        if np.isfinite(med) and med <= 1.5:
            rh_da = rh_da * 100.0
        rh_da = rh_da.clip(min=0, max=100)
    else:
        tdvar = find_variable(ds_raw, VARIABLES["relative_humidity"]["dewpoint_candidates"], required=True)
        print(f"[INFO] ERA5 dewpoint variable for RH: {tdvar}")
        td_da = reduce_to_time_lat_lon(ds_raw[tdvar]).isel(lat=iy_da, lon=ix_da).transpose("time", station_dim)
        td_da = td_da.load()
        rh_da = compute_rh_from_t_td(t_c, td_da)

    wsvar = find_variable(ds_raw, VARIABLES["wind_speed"].get("era5_candidates", []), required=False)
    if wsvar is not None:
        print(f"[INFO] ERA5 wind speed variable: {wsvar}")
        ws_da = reduce_to_time_lat_lon(ds_raw[wsvar]).isel(lat=iy_da, lon=ix_da).transpose("time", station_dim).load()
    else:
        uvar = find_variable(ds_raw, VARIABLES["wind_speed"]["u_candidates"], required=True)
        vvar = find_variable(ds_raw, VARIABLES["wind_speed"]["v_candidates"], required=True)
        print(f"[INFO] ERA5 wind variables: {uvar}, {vvar}")
        u_da = reduce_to_time_lat_lon(ds_raw[uvar]).isel(lat=iy_da, lon=ix_da).transpose("time", station_dim).load()
        v_da = reduce_to_time_lat_lon(ds_raw[vvar]).isel(lat=iy_da, lon=ix_da).transpose("time", station_dim).load()
        ws_da = np.sqrt(u_da ** 2 + v_da ** 2)

    daily_frames = []
    arrays = {
        "temperature": np.asarray(t_c.values, dtype="float32"),
        "relative_humidity": np.asarray(rh_da.values, dtype="float32"),
        "wind_speed": np.asarray(ws_da.values, dtype="float32"),
    }

    for var, arr in arrays.items():
        df_wide = pd.DataFrame(arr, index=time_index, columns=station_ids)
        df_wide.index.name = "datetime"
        long = resample_station_wide_to_daily(
            df_wide, station_meta, var, min_hours_day,
            daily_time_basis=daily_time_basis,
            fixed_time_shift_hours=fixed_time_shift_hours,
        )
        daily_frames.append(long)

    out = daily_frames[0]
    for dfv in daily_frames[1:]:
        out = out.merge(dfv, on=["date", "station_id"], how="outer")

    out = out.merge(station_meta, on="station_id", how="left")
    out["date"] = pd.to_datetime(out["date"])
    ds_raw.close()
    return out, grid_index

def build_era5_station_daily_all_years(era5_pattern: str, inventory: pd.DataFrame, out_dir: str,
                                        reuse_cache: bool, min_hours_day: int,
                                        daily_time_basis: str = "utc",
                                        fixed_time_shift_hours: int = 0,
                                        max_years_per_run: int = 0,
                                        only_finalize_from_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = resolve_era5_hourly_files(era5_pattern)
    print(f"[INFO] Found {len(files)} ERA5 hourly files overlapping requested period.")
    ensure_dir(out_dir)
    tb_tag = time_basis_tag(daily_time_basis, fixed_time_shift_hours)
    cache_grid = os.path.join(out_dir, "CACHE_ERA5_station_nearest_grid_index_QC.csv")
    cache_all = os.path.join(out_dir, f"CACHE_ERA5_station_daily_all_years_minH{min_hours_day}_{tb_tag}_QC.csv")

    if reuse_cache and valid_csv_cache(cache_all) and valid_csv_cache(cache_grid):
        print("[INFO] Reusing cached ERA5 station-daily table.")
        era_station_daily = pd.read_csv(cache_all, parse_dates=["date"])
        grid_index = pd.read_csv(cache_grid)
        era_station_daily["station_id"] = era_station_daily["station_id"].astype(str)
        grid_index["station_id"] = grid_index["station_id"].astype(str)
        return era_station_daily, grid_index

    all_daily = []
    grid_index = None
    if reuse_cache and valid_csv_cache(cache_grid):
        print("[INFO] Reusing cached ERA5 nearest-grid index.")
        grid_index = pd.read_csv(cache_grid)
        grid_index["station_id"] = grid_index["station_id"].astype(str)

    processed_this_run = 0
    for i, f in enumerate(files, 1):
        year_match = re.search(r"(19|20)\d{2}", Path(f).name)
        year_label = year_match.group(0) if year_match else f"file_{i:03d}"
        cache_year = os.path.join(out_dir, f"CACHE_ERA5_station_daily_{year_label}_minH{min_hours_day}_{tb_tag}_QC.csv")
        if reuse_cache and valid_csv_cache(cache_year):
            print(f"[SKIP] Reusing cached ERA5 year {i}/{len(files)}: {Path(cache_year).name}")
            df_year = pd.read_csv(cache_year, parse_dates=["date"])
            df_year["station_id"] = df_year["station_id"].astype(str)
            all_daily.append(df_year)
            continue

        if only_finalize_from_cache:
            print(f"[WAIT] Missing yearly cache, not processing because --only_finalize_from_cache is active: {Path(cache_year).name}")
            continue

        if max_years_per_run and processed_this_run >= max_years_per_run:
            print(f"[STOP] Reached --max_years_per_run={max_years_per_run}. Restart with --reuse_era5_cache to continue.")
            continue

        print(f"[INFO] Processing ERA5 file {i}/{len(files)}")
        processed_this_run += 1
        df_year, grid_index = extract_era5_hourly_to_daily_station(
            f, inventory, grid_index, min_hours_day,
            daily_time_basis=daily_time_basis,
            fixed_time_shift_hours=fixed_time_shift_hours,
        )
        if grid_index is not None and not valid_csv_cache(cache_grid):
            atomic_to_csv(grid_index, cache_grid)
            print(f"[OK] Cached ERA5 nearest-grid index: {cache_grid}")
        if not df_year.empty:
            atomic_to_csv(df_year, cache_year)
            print(f"[OK] Cached ERA5 station-daily year: {cache_year}")
            all_daily.append(df_year)

    if not all_daily:
        raise RuntimeError("No ERA5 station-daily data were produced.")
    era_station_daily = pd.concat(all_daily, ignore_index=True)
    era_station_daily = era_station_daily.sort_values(["station_id", "date"])
    era_station_daily = era_station_daily.drop_duplicates(subset=["station_id", "uf", "date"], keep="last")
    atomic_to_csv(era_station_daily, cache_all)
    if grid_index is not None:
        atomic_to_csv(grid_index, cache_grid)
    print(f"[OK] Cached combined ERA5 station-daily table: {cache_all}")
    return era_station_daily, grid_index


# ============================================================
# ERA5 daily-file processing (daily-file processing)
# ============================================================
def open_daily_dataset_robust(path: str) -> xr.Dataset:
    """Open preprocessed daily ERA5 files with conservative backend fallbacks.

    This path reads preprocessed daily ERA5 inputs directly.
    """
    errors = []
    attempts = [
        (None, {}, {"decode_times": True, "mask_and_scale": True}),
        ("scipy", {}, {"decode_times": True, "mask_and_scale": True}),
        ("h5netcdf", {"phony_dims": "sort"}, {"decode_times": False, "mask_and_scale": True}),
        ("netcdf4", {}, {"decode_times": False, "mask_and_scale": True}),
    ]
    for engine, backend_kwargs, open_kwargs in attempts:
        try:
            if engine is None:
                ds = xr.open_dataset(path, **open_kwargs)
                used = "default"
            else:
                ds = xr.open_dataset(path, engine=engine, backend_kwargs=backend_kwargs, **open_kwargs)
                used = engine
            # Harmonize coordinate names before time decoding.
            rename = {}
            for tname in ["time", "valid_time", "datetime", "date"]:
                if tname in ds.coords or tname in ds.dims:
                    if tname != "time":
                        rename[tname] = "time"
                    break
            for lname in ["latitude", "Latitude", "LAT", "lat", "y"]:
                if lname in ds.coords or lname in ds.dims:
                    if lname != "lat":
                        rename[lname] = "lat"
                    break
            for lname in ["longitude", "Longitude", "LON", "lon", "x"]:
                if lname in ds.coords or lname in ds.dims:
                    if lname != "lon":
                        rename[lname] = "lon"
                    break
            if rename:
                ds = ds.rename(rename)
            if "time" in ds.coords or "time" in ds.dims:
                if not np.issubdtype(ds["time"].dtype, np.datetime64):
                    ds = ds.assign_coords(time=decode_cf_time_to_pandas(ds["time"].values, ds["time"].attrs))
                else:
                    ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values))
            if "lat" not in ds.coords or "lon" not in ds.coords:
                raise ValueError(f"Missing lat/lon coordinates. Coords={list(ds.coords)}")
            if float(ds["lon"].max()) > 180:
                ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
            if ds["lat"].values[0] > ds["lat"].values[-1]:
                ds = ds.sortby("lat")
            print(f"[INFO] Opened daily ERA5 with xarray engine='{used}': {Path(path).name}")
            return ds
        except Exception as exc:
            errors.append(f"{engine or 'default'}: {type(exc).__name__}: {exc}")
            continue
    raise RuntimeError(
        "Could not open preprocessed daily ERA5 file with available xarray backends.\n"
        f"File: {path}\nAttempts:\n" + "\n".join(errors)
    )


def find_era5_daily_files(daily_dir: str) -> List[str]:
    patterns = [
        os.path.join(daily_dir, "ERA5_daily_Brazil_*.nc"),
        os.path.join(daily_dir, "*daily*Brazil*.nc"),
        os.path.join(daily_dir, "*.nc"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    y0 = pd.Timestamp(START_DATE).year
    y1 = pd.Timestamp(END_DATE).year
    selected = []
    for f in files:
        m = re.search(r"(19|20)\d{2}", Path(f).name)
        if m:
            y = int(m.group(0))
            if y0 <= y <= y1:
                selected.append(f)
    if not selected:
        raise FileNotFoundError(f"No daily ERA5 files overlapping {y0}-{y1} found in {daily_dir}")
    return selected


def select_daily_era5_var(ds: xr.Dataset, candidates: List[str], required: bool = True) -> Optional[xr.DataArray]:
    name = find_variable(ds, candidates, required=required)
    if name is None:
        return None
    return reduce_to_time_lat_lon(ds[name])


def extract_era5_daily_file_to_station(file_path: str, inventory: pd.DataFrame,
                                        grid_index: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract preprocessed daily ERA5 thermodynamic variables at station points.

    Variables are mapped to the same daily quantities used by the main heatwave
    workflow: temperature = Tmean, relative_humidity = RHmean, wind_speed =
    WS10mean. Since the input is already daily, ERA5 hourly counts are encoded as
    24 for finite values and 0 for missing values; INMET daily completeness is
    still controlled independently by --min_hours_day.
    """
    print(f"[INFO] ERA5 daily file: {Path(file_path).name}")
    ds = open_daily_dataset_robust(file_path)
    ds = ds.sel(time=slice(START_DATE, END_DATE))
    if ds.sizes.get("time", 0) == 0:
        ds.close()
        return pd.DataFrame(), grid_index

    if grid_index is None:
        grid_index = get_station_grid_indices(ds, inventory)
        print(f"[INFO] ERA5 nearest grid cells prepared for {len(grid_index)} unique stations.")

    station_dim = "station"
    iy_da = xr.DataArray(grid_index["iy"].to_numpy(), dims=station_dim)
    ix_da = xr.DataArray(grid_index["ix"].to_numpy(), dims=station_dim)
    station_ids = grid_index["station_id"].astype(str).to_numpy()
    time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))

    # Map main daily ERA5 workflow names to validation-variable names.
    tmean_da = select_daily_era5_var(ds, ["Tmean", "tmean", "T2mean", "t2m_mean", "temperature", "t2m"], True)
    rh_da = select_daily_era5_var(ds, ["RHmean", "rhmean", "RH", "relative_humidity", "relative_humidity_mean"], True)
    ws_da = select_daily_era5_var(ds, ["WS10mean", "ws10mean", "wind_speed", "WS10", "si10", "10m_wind_speed"], False)
    if ws_da is None:
        u_da = select_daily_era5_var(ds, ["U10mean", "u10mean", "u10", "U10", "var165"], True)
        v_da = select_daily_era5_var(ds, ["V10mean", "v10mean", "v10", "V10", "var166"], True)
        ws_da = np.sqrt(u_da ** 2 + v_da ** 2)

    arrs = {
        "temperature": maybe_kelvin_to_celsius(tmean_da).isel(lat=iy_da, lon=ix_da).values.astype("float32"),
        "relative_humidity": rh_da.isel(lat=iy_da, lon=ix_da).values.astype("float32"),
        "wind_speed": ws_da.isel(lat=iy_da, lon=ix_da).values.astype("float32"),
    }
    rows = []
    for var, arr in arrs.items():
        wide = pd.DataFrame(arr, index=time_index, columns=station_ids)
        try:
            long_val = wide.stack(future_stack=True).rename(f"{var}_era5").reset_index()
        except TypeError:
            long_val = wide.stack(dropna=False).rename(f"{var}_era5").reset_index()
        long_val.columns = ["date", "station_id", f"{var}_era5"]
        long_val[f"{var}_n_hours_era5"] = np.where(pd.to_numeric(long_val[f"{var}_era5"], errors="coerce").notna(), 24, 0)
        rows.append(long_val)
    out = rows[0]
    for dfv in rows[1:]:
        out = out.merge(dfv, on=["date", "station_id"], how="outer")
    meta = grid_index[["station_id", "uf", "lat", "lon", "era5_lat", "era5_lon", "era5_grid_distance_deg"]].copy()
    out = out.merge(meta, on="station_id", how="left")
    out["date"] = pd.to_datetime(out["date"])
    ds.close()
    return out, grid_index


def build_era5_station_daily_from_daily_files(daily_dir: str, inventory: pd.DataFrame, out_dir: str,
                                               reuse_cache: bool, max_years_per_run: int = 0,
                                               only_finalize_from_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = find_era5_daily_files(daily_dir)
    print(f"[INFO] Found {len(files)} preprocessed daily ERA5 files overlapping requested period.")
    ensure_dir(out_dir)
    cache_grid = os.path.join(out_dir, "CACHE_ERA5_station_nearest_grid_index_DAILY_QC.csv")
    cache_all = os.path.join(out_dir, "CACHE_ERA5_station_daily_all_years_DAILY_QC.csv")
    if reuse_cache and valid_csv_cache(cache_all) and valid_csv_cache(cache_grid):
        print("[INFO] Reusing cached ERA5 station-daily table from daily files.")
        era_station_daily = pd.read_csv(cache_all, parse_dates=["date"])
        grid_index = pd.read_csv(cache_grid)
        era_station_daily["station_id"] = era_station_daily["station_id"].astype(str)
        grid_index["station_id"] = grid_index["station_id"].astype(str)
        return era_station_daily, grid_index
    all_daily = []
    grid_index = None
    if reuse_cache and valid_csv_cache(cache_grid):
        grid_index = pd.read_csv(cache_grid)
        grid_index["station_id"] = grid_index["station_id"].astype(str)
        print("[INFO] Reusing cached ERA5 nearest-grid index for daily files.")
    processed_this_run = 0
    for i, f in enumerate(files, 1):
        year_match = re.search(r"(19|20)\d{2}", Path(f).name)
        year_label = year_match.group(0) if year_match else f"file_{i:03d}"
        cache_year = os.path.join(out_dir, f"CACHE_ERA5_station_daily_{year_label}_DAILY_QC.csv")
        if reuse_cache and valid_csv_cache(cache_year):
            print(f"[SKIP] Reusing cached ERA5 daily year {i}/{len(files)}: {Path(cache_year).name}")
            df_year = pd.read_csv(cache_year, parse_dates=["date"])
            df_year["station_id"] = df_year["station_id"].astype(str)
            all_daily.append(df_year)
            continue
        if only_finalize_from_cache:
            print(f"[WAIT] Missing daily-file cache, not processing because --only_finalize_from_cache is active: {Path(cache_year).name}")
            continue
        if max_years_per_run and processed_this_run >= max_years_per_run:
            print(f"[STOP] Reached --max_years_per_run={max_years_per_run}. Restart with --reuse_era5_cache to continue.")
            continue
        print(f"[INFO] Processing daily ERA5 file {i}/{len(files)}")
        processed_this_run += 1
        df_year, grid_index = extract_era5_daily_file_to_station(f, inventory, grid_index)
        if grid_index is not None and not valid_csv_cache(cache_grid):
            atomic_to_csv(grid_index, cache_grid)
            print(f"[OK] Cached ERA5 nearest-grid index: {cache_grid}")
        if not df_year.empty:
            atomic_to_csv(df_year, cache_year)
            print(f"[OK] Cached ERA5 station-daily year from daily file: {cache_year}")
            all_daily.append(df_year)
    if not all_daily:
        raise RuntimeError("No ERA5 station-daily data were produced from daily files.")
    era_station_daily = pd.concat(all_daily, ignore_index=True)
    era_station_daily = era_station_daily.sort_values(["station_id", "date"])
    era_station_daily = era_station_daily.drop_duplicates(subset=["station_id", "uf", "date"], keep="last")
    atomic_to_csv(era_station_daily, cache_all)
    if grid_index is not None:
        atomic_to_csv(grid_index, cache_grid)
    print(f"[OK] Cached combined ERA5 station-daily table from daily files: {cache_all}")
    return era_station_daily, grid_index


def candidate_era5_station_cache_paths(
    out_dir: str,
    explicit_cache: str = "",
) -> List[str]:
    """Return explicit and output-directory ERA5 station-cache candidates."""
    paths = []
    if explicit_cache:
        paths.append(explicit_cache)

    paths.append(
        os.path.join(out_dir, "CACHE_ERA5_daily_station_metrics_2002_2024.csv")
    )

    out = []
    seen = set()
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.path.expanduser(str(path)))
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out

def load_era5_station_daily_cache_for_basic_validation(cache_path: str) -> pd.DataFrame:
    """Load ERA5 station-daily cache produced by the heatwave QC workflow.

    Expected input columns include daily ERA5 variables such as Tmean, RHmean and
    WSmean at station points. The output is converted to the column names used by
    this Taylor-evaluation script: temperature_era5, relative_humidity_era5 and
    wind_speed_era5. Since the ERA5 values are already daily products, finite
    daily values are assigned an ERA5 hourly-count proxy of 24; INMET daily
    completeness remains controlled independently by --min_hours_day.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)
    df = pd.read_csv(cache_path, parse_dates=["date"])
    df["station_id"] = df["station_id"].astype(str)
    if "uf" not in df.columns:
        raise KeyError(f"ERA5 station cache lacks required 'uf' column: {cache_path}")
    df["uf"] = df["uf"].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]

    def first_existing(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    t_col = first_existing(["Tmean", "temperature", "tmean", "T2mean", "t2m_mean"])
    rh_col = first_existing(["RHmean", "relative_humidity", "rhmean", "RH"])
    ws_col = first_existing(["WSmean", "WS10mean", "wind_speed", "WS10", "ws10mean"])
    missing = []
    if t_col is None:
        missing.append("Tmean/temperature")
    if rh_col is None:
        missing.append("RHmean/relative_humidity")
    # Wind can be missing in some daily products; keep as NaN rather than failing.
    if missing:
        raise KeyError(f"ERA5 station cache lacks required columns {missing}. Available: {list(df.columns)}")

    out = df[["station_id", "uf", "date"]].copy()
    out["temperature_era5"] = pd.to_numeric(df[t_col], errors="coerce")
    out["relative_humidity_era5"] = pd.to_numeric(df[rh_col], errors="coerce")
    if ws_col is not None:
        out["wind_speed_era5"] = pd.to_numeric(df[ws_col], errors="coerce")
    else:
        out["wind_speed_era5"] = np.nan

    for var in VARIABLES:
        col = f"{var}_era5"
        out[f"{var}_n_hours_era5"] = np.where(np.isfinite(out[col]), 24, 0)

    # Apply the same broad physical limits used elsewhere in this script.
    out.loc[(out["temperature_era5"] < -50) | (out["temperature_era5"] > 60), "temperature_era5"] = np.nan
    out.loc[(out["relative_humidity_era5"] < 0) | (out["relative_humidity_era5"] > 100), "relative_humidity_era5"] = np.nan
    out.loc[(out["wind_speed_era5"] < 0) | (out["wind_speed_era5"] > 75), "wind_speed_era5"] = np.nan
    out = out.sort_values(["station_id", "date"]).drop_duplicates(["station_id", "uf", "date"], keep="last")
    return out


def try_load_precomputed_era5_station_cache(out_dir: str, explicit_cache: str = "") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Try loading a precomputed ERA5 station-daily cache before opening NetCDF files."""
    for cache in candidate_era5_station_cache_paths(out_dir, explicit_cache=explicit_cache):
        if os.path.exists(cache) and os.path.getsize(cache) > 100:
            print(f"[INFO] Reusing precomputed ERA5 station-daily cache: {cache}")
            era = load_era5_station_daily_cache_for_basic_validation(cache)
            era = era[(era["date"] >= pd.Timestamp(START_DATE)) & (era["date"] <= pd.Timestamp(END_DATE))].copy()
            print(f"[INFO] ERA5 station cache rows retained for requested period: {len(era)}")
            return era, None, cache
    return None, None, None

# ============================================================
# Pairing and metrics
# ============================================================
def build_station_daily_pairs(obs_daily: pd.DataFrame, era_daily: pd.DataFrame) -> pd.DataFrame:
    # Do not merge by lat/lon. Coordinates can differ slightly across yearly files.
    # Station identity, UF and date are sufficient and avoid dropping valid stations.
    keep_era = ["station_id", "uf", "date"] + [f"{v}_era5" for v in VARIABLES] + [f"{v}_n_hours_era5" for v in VARIABLES]
    era = era_daily[keep_era].copy()
    obs = obs_daily.copy()
    obs["station_id"] = obs["station_id"].astype(str)
    era["station_id"] = era["station_id"].astype(str)
    obs["date"] = pd.to_datetime(obs["date"])
    era["date"] = pd.to_datetime(era["date"])
    paired = obs.merge(era, on=["station_id", "uf", "date"], how="inner")
    return paired.sort_values(["uf", "station_id", "date"])


def apply_station_year_coverage_qc(
    paired: pd.DataFrame,
    min_valid_day_fraction: float = 0.70,
    min_valid_years_variable: int = 5,
) -> pd.DataFrame:
    """Apply variable-specific station-year coverage QC to paired daily data.

    A station-year is retained for a variable only if both INMET and ERA5 have
    paired daily values for at least `min_valid_day_fraction` of the calendar
    year. A station-variable is retained for metric calculation only if it has at
    least `min_valid_years_variable` retained station-years. Values failing these
    criteria are set to NaN, so downstream metrics use the same table structure
    but only quality-controlled daily pairs.
    """
    out = paired.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    days_in_year = pd.to_datetime(out["date"]).dt.is_leap_year.map({True: 366, False: 365}).to_numpy()
    out["days_in_year"] = days_in_year

    for var in VARIABLES:
        obs_col = f"{var}_obs"
        era_col = f"{var}_era5"
        if obs_col not in out.columns or era_col not in out.columns:
            continue
        paired_ok = out[obs_col].notna() & out[era_col].notna()
        tmp = out.loc[paired_ok, ["station_id", "uf", "year", "days_in_year"]].copy()
        if tmp.empty:
            out[[obs_col, era_col]] = np.nan
            continue
        annual = (
            tmp.groupby(["station_id", "uf", "year"], as_index=False)
            .agg(n_paired_days=("year", "size"), days_in_year=("days_in_year", "max"))
        )
        annual["coverage_fraction"] = annual["n_paired_days"] / annual["days_in_year"]
        annual["valid_station_year"] = annual["coverage_fraction"] >= float(min_valid_day_fraction)
        valid_years = annual.loc[annual["valid_station_year"], ["station_id", "uf", "year"]].copy()
        n_years = valid_years.groupby(["station_id", "uf"]).size().rename("n_valid_years").reset_index()
        valid_stations = n_years.loc[n_years["n_valid_years"] >= int(min_valid_years_variable), ["station_id", "uf"]]
        valid = valid_years.merge(valid_stations, on=["station_id", "uf"], how="inner")
        valid["__valid__"] = 1
        flag = out[["station_id", "uf", "year"]].merge(valid, on=["station_id", "uf", "year"], how="left")["__valid__"].fillna(0).astype(bool)
        out.loc[~flag, [obs_col, era_col]] = np.nan

    return out.drop(columns=["year", "days_in_year"], errors="ignore")

def _first_existing_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    norm = {normalize_name(c): c for c in df.columns}
    for c in candidates:
        nc = normalize_name(c)
        if nc in norm:
            return norm[nc]
    return None


def calculate_taylor_aggregate_metrics(paired: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows = []
    for var, meta in VARIABLES.items():
        obs = pd.to_numeric(paired[f"{var}_obs"], errors="coerce")
        mod = pd.to_numeric(paired[f"{var}_era5"], errors="coerce")
        ok = obs.notna() & mod.notna()
        n = int(ok.sum())
        if n >= min_pairs:
            xv = obs[ok].to_numpy(dtype=float)
            yv = mod[ok].to_numpy(dtype=float)
            if np.nanstd(xv) > 0 and np.nanstd(yv) > 0:
                corr = float(np.corrcoef(xv, yv)[0, 1])
            else:
                corr = np.nan
            obs_std = float(np.nanstd(xv, ddof=1))
            era5_std = float(np.nanstd(yv, ddof=1))
            std_ratio = era5_std / obs_std if obs_std > 0 else np.nan
            rmse_norm = float(np.sqrt(np.nanmean((yv - xv) ** 2)) / obs_std) if obs_std > 0 else np.nan
            bias = float(np.nanmean(yv - xv))
            rmse = float(np.sqrt(np.nanmean((yv - xv) ** 2)))
        else:
            corr = obs_std = era5_std = std_ratio = rmse_norm = bias = rmse = np.nan
        rows.append({
            "variable": var,
            "variable_label": meta["label"],
            "n_paired_station_days": n,
            "pearson_r": corr,
            "obs_std": obs_std,
            "era5_std": era5_std,
            "std_ratio": std_ratio,
            "rmse_norm": rmse_norm,
            "rmse": rmse,
            "bias_era5_minus_inmet": bias,
        })
    return pd.DataFrame(rows)


# ============================================================
# Shapefile
# ============================================================
def read_states_shapefile(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"State shapefile not found: {path}")
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf.set_crs("EPSG:4326")

    state_name_to_uf = {
        "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAPÁ": "AP", "AMAZONAS": "AM",
        "BAHIA": "BA", "CEARA": "CE", "CEARÁ": "CE", "DISTRITO FEDERAL": "DF",
        "ESPIRITO SANTO": "ES", "ESPÍRITO SANTO": "ES", "GOIAS": "GO", "GOIÁS": "GO",
        "MARANHAO": "MA", "MARANHÃO": "MA", "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS",
        "MINAS GERAIS": "MG", "PARA": "PA", "PARÁ": "PA", "PARAIBA": "PB", "PARAÍBA": "PB",
        "PARANA": "PR", "PARANÁ": "PR", "PERNAMBUCO": "PE", "PIAUI": "PI", "PIAUÍ": "PI",
        "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN", "RIO GRANDE DO SUL": "RS",
        "RONDONIA": "RO", "RONDÔNIA": "RO", "RORAIMA": "RR", "SANTA CATARINA": "SC",
        "SAO PAULO": "SP", "SÃO PAULO": "SP", "SERGIPE": "SE", "TOCANTINS": "TO",
    }
    state_name_to_uf_norm = {normalize_name(k): v for k, v in state_name_to_uf.items()}

    uf_col = None
    for c in gdf.columns:
        cl = normalize_name(c)
        if cl in ["uf", "sigla", "sigla_uf", "uf_05", "sigla_uf_05"]:
            vals = gdf[c].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]
            if vals.isin(list(UF_TO_REGION.keys())).sum() >= 10:
                uf_col = c
                gdf["uf"] = vals
                break
    if uf_col is None:
        name_col = None
        for c in gdf.columns:
            cl = normalize_name(c)
            if cl in ["nm_estado", "nome_estado", "estado_nome", "estado", "name", "nome"]:
                name_col = c
                break
        if name_col is None:
            for c in gdf.columns:
                if c == "geometry":
                    continue
                vals_norm = gdf[c].astype(str).map(normalize_name)
                if vals_norm.isin(state_name_to_uf_norm.keys()).sum() >= 10:
                    name_col = c
                    break
        if name_col is not None:
            gdf["uf"] = gdf[name_col].astype(str).map(lambda x: state_name_to_uf_norm.get(normalize_name(x), np.nan))
            uf_col = name_col
    if uf_col is None or "uf" not in gdf.columns:
        raise ValueError(f"Could not identify UF or state-name column. Columns: {list(gdf.columns)}")
    gdf = gdf[gdf["uf"].isin(UF_TO_REGION)].copy()
    gdf = gdf[["uf", "geometry"]].dissolve(by="uf", as_index=False)
    gdf["region"] = gdf["uf"].map(UF_TO_REGION)
    print(f"[INFO] State shapefile loaded using column '{uf_col}'. Valid UFs: {len(gdf)}")
    return gdf[["uf", "region", "geometry"]]


# ============================================================
# Plotting
# ============================================================
def station_size(n_values: pd.Series, min_size=16, max_size=54) -> np.ndarray:
    vals = pd.to_numeric(n_values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return np.full_like(vals, min_size, dtype=float)
    vmin, vmax = np.nanpercentile(vals, [5, 95])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full_like(vals, (min_size + max_size) / 2.0, dtype=float)
    scaled = (np.clip(vals, vmin, vmax) - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)


def setup_map_axis(ax, states: gpd.GeoDataFrame, row_idx: int, col_idx: int):
    fs = 13
    states.boundary.plot(ax=ax, color="0.35", linewidth=0.38, zorder=2)
    ax.set_xlim(BRAZIL_EXTENT[0], BRAZIL_EXTENT[1])
    ax.set_ylim(BRAZIL_EXTENT[2], BRAZIL_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.22, alpha=0.20)
    ax.set_xlabel("Longitude" if row_idx == 2 else "", fontsize=fs, labelpad=2.5)
    ax.set_ylabel("Latitude" if col_idx == 0 else "", fontsize=fs, labelpad=3.0)
    ax.tick_params(labelsize=fs, direction="in", length=2.8, width=0.45)
    if row_idx < 2:
        ax.set_xticklabels([])
    if col_idx > 0:
        ax.set_yticklabels([])


def setup_taylor_axis(ax, title: str):
    fs = 13
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlim(0, 2.0)
    corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    theta_ticks = np.degrees(np.arccos(corr_ticks))
    ax.set_thetagrids(theta_ticks, labels=[f"{c:g}" for c in corr_ticks], fontsize=fs)
    ax.set_rgrids([0.5, 1.0, 1.5, 2.0], angle=135, fontsize=fs)
    ax.grid(True, linestyle="--", alpha=0.40, linewidth=0.45)
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(theta, np.ones_like(theta), "k--", lw=0.9, alpha=0.65)
    ax.plot(0, 1, marker="*", color="black", markersize=8, linestyle="None", label="INMET reference")
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=10)


def plot_taylor_station_points(ax, metrics_var: pd.DataFrame, agg_row: pd.Series):
    d = metrics_var.copy()
    d = d[np.isfinite(pd.to_numeric(d["pearson_r"], errors="coerce")) & np.isfinite(pd.to_numeric(d["std_ratio"], errors="coerce"))].copy()
    if not d.empty:
        corr = np.clip(pd.to_numeric(d["pearson_r"], errors="coerce").to_numpy(dtype=float), 0.0, 1.0)
        theta = np.arccos(corr)
        r = np.clip(pd.to_numeric(d["std_ratio"], errors="coerce").to_numpy(dtype=float), 0.0, 2.0)
        sizes = station_size(d["n_paired_days"], min_size=10, max_size=28)
        ax.scatter(theta, r, s=sizes, color="0.55", edgecolor="white", linewidth=0.20, alpha=0.62, label="Station-level ERA5", zorder=5)
    if agg_row is not None and np.isfinite(agg_row.get("pearson_r", np.nan)) and np.isfinite(agg_row.get("std_ratio", np.nan)):
        corr = float(np.clip(agg_row["pearson_r"], 0.0, 1.0))
        stdr = float(np.clip(agg_row["std_ratio"], 0.0, 2.0))
        ax.plot(np.arccos(corr), stdr, marker="D", color="#D62728", markeredgecolor="black", markersize=7,
                linestyle="None", label="All-pairs ERA5 aggregate", zorder=8)


def _clean_plot_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["pearson_r"] = pd.to_numeric(out["pearson_r"], errors="coerce")
    out["bias_era5_minus_inmet"] = pd.to_numeric(out["bias_era5_minus_inmet"], errors="coerce")
    out["rmse"] = pd.to_numeric(out["rmse"], errors="coerce")
    # Last safety coordinate normalization before plotting.
    fixed = out.apply(lambda r: normalize_brazil_coordinates(r["lat"], r["lon"]), axis=1)
    out["lat"] = [x[0] for x in fixed]
    out["lon"] = [x[1] for x in fixed]
    out["inside_plot_domain"] = (
        out["lat"].between(BRAZIL_EXTENT[2], BRAZIL_EXTENT[3]) &
        out["lon"].between(BRAZIL_EXTENT[0], BRAZIL_EXTENT[1])
    )
    return out


def plot_validation_figure(metrics: pd.DataFrame, taylor_agg: pd.DataFrame, states: gpd.GeoDataFrame,
                           out_dir: str, dpi: int):
    print("[INFO] Plotting clean station-level validation maps and Taylor diagrams.")
    fs = 13
    metrics = _clean_plot_metrics(metrics)
    map_counts = {}
    for var in VAR_ORDER:
        sub = metrics[(metrics["variable"] == var) & metrics["inside_plot_domain"] & np.isfinite(metrics["pearson_r"])]
        map_counts[var] = int(len(sub))
    print(f"[INFO] Stations plotted on maps by variable: {map_counts}")

    fig = plt.figure(figsize=(14.4, 16.6), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        height_ratios=[1.0, 1.0, 1.0, 0.96],
        hspace=0.40, wspace=0.34,
        left=0.11, right=0.965, top=0.945, bottom=0.10,
    )

    metric_rows = [
        ("pearson_r", "Pearson\ncorrelation", "Pearson r", plt.cm.RdYlBu_r, Normalize(vmin=-0.2, vmax=1.0)),
        ("bias_era5_minus_inmet", "Bias\nERA5 − INMET", "Bias", plt.cm.RdBu_r, None),
        ("rmse", "RMSE", "RMSE", plt.cm.viridis, None),
    ]
    panel_letters = list("abcdefghijkl")
    letter_i = 0
    row_first_axes = {}
    taylor_axes = []

    for irow, (metric, row_title, cbar_base, cmap, norm_default) in enumerate(metric_rows):
        for jcol, var in enumerate(VAR_ORDER):
            ax = fig.add_subplot(gs[irow, jcol])
            if jcol == 0:
                row_first_axes[irow] = (ax, row_title)
            setup_map_axis(ax, states, row_idx=irow, col_idx=jcol)
            meta = VARIABLES[var]
            d = metrics[(metrics["variable"] == var) & metrics["inside_plot_domain"] & np.isfinite(metrics[metric])].copy()

            if metric == "pearson_r":
                norm = norm_default
                cbar_label = cbar_base
            elif metric == "bias_era5_minus_inmet":
                norm = meta["bias_norm"]
                cbar_label = f"Bias ({meta['unit']})"
            else:
                v95 = float(np.nanpercentile(d[metric], 95)) if not d.empty else np.nan
                base_vmax = meta["rmse_norm"].vmax
                vmax = max(base_vmax, v95) if np.isfinite(v95) else base_vmax
                norm = Normalize(0, vmax)
                cbar_label = f"RMSE ({meta['unit']})"

            sc = None
            if not d.empty:
                sc = ax.scatter(
                    d["lon"], d["lat"],
                    c=d[metric], cmap=cmap, norm=norm,
                    s=station_size(d["n_paired_days"], min_size=20, max_size=54),
                    edgecolor="black", linewidth=0.28, alpha=0.94, zorder=10,
                )
            else:
                ax.text(0.5, 0.5, "No stations\nafter QC", ha="center", va="center",
                        transform=ax.transAxes, fontsize=fs, color="0.35")

            ax.text(0.018, 0.982, f"({panel_letters[letter_i]})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=fs,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0), zorder=20)
            letter_i += 1

            if irow == 0:
                ax.set_title(meta["label"], fontsize=fs, fontweight="bold", pad=7)

            if sc is not None:
                cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.040, pad=0.012)
                cb.set_label(cbar_label, fontsize=fs, labelpad=3)
                cb.ax.tick_params(labelsize=fs, length=2.5, width=0.45)

    for jcol, var in enumerate(VAR_ORDER):
        ax = fig.add_subplot(gs[3, jcol], projection="polar")
        taylor_axes.append(ax)
        setup_taylor_axis(ax, VARIABLES[var]["label"])
        metrics_var = metrics[metrics["variable"] == var].copy()
        agg_sub = taylor_agg[taylor_agg["variable"] == var]
        agg_row = agg_sub.iloc[0] if not agg_sub.empty else None
        plot_taylor_station_points(ax, metrics_var, agg_row)
        ax.text(-0.10, 1.05, f"({panel_letters[letter_i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=fs,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0), zorder=20)
        letter_i += 1

    # Figure-level row labels to avoid overlap with axis labels.
    for _, (ax0, row_title) in row_first_axes.items():
        bbox = ax0.get_position()
        y = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.042, y, row_title, rotation=90, ha="center", va="center",
                 fontsize=fs, fontweight="bold")

    if taylor_axes:
        bbox = taylor_axes[0].get_position()
        y = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(0.042, y, "Taylor diagrams", rotation=90, ha="center", va="center",
                 fontsize=fs, fontweight="bold")

    # Compact legends only. No methodological paragraph is drawn inside the figure.
    nvals = metrics.loc[np.isfinite(metrics["n_paired_days"]), "n_paired_days"]
    if not nvals.empty:
        qvals = np.unique(np.nanpercentile(nvals, [25, 50, 75]).astype(int))
        handles_size = [plt.scatter([], [], s=station_size(pd.Series([q]), min_size=20, max_size=54)[0],
                                    edgecolor="black", facecolor="0.72", linewidth=0.25) for q in qvals]
        labels_size = [f"{q:,} d" for q in qvals]
        fig.legend(handles_size, labels_size, title="Paired days", loc="lower left",
                   bbox_to_anchor=(0.075, 0.028), frameon=False, fontsize=fs, title_fontsize=fs)

    handles_taylor = [
        Line2D([0], [0], marker="o", color="0.55", linestyle="None", markersize=6.5, label="Station-level ERA5"),
        Line2D([0], [0], marker="D", color="#D62728", markeredgecolor="black", linestyle="None", markersize=7.5, label="All-pairs ERA5"),
        Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=8.5, label="INMET reference"),
    ]
    fig.legend(handles_taylor, [h.get_label() for h in handles_taylor], loc="lower center",
               bbox_to_anchor=(0.58, 0.028), ncol=3, frameon=False, fontsize=fs)

    fig.suptitle("Station-level ERA5 daily evaluation against INMET observations after QC",
                 x=0.11, y=0.982, ha="left", fontsize=fs, fontweight="bold")

    out_base = os.path.join(out_dir, "Supplementary_Figure_ERA5_INMET_station_evaluation_Taylor_T_RH_WS")
    for ext in [".jpeg", ".pdf"]:
        try:
            os.remove(out_base + ext)
        except FileNotFoundError:
            pass
    fig.savefig(out_base + ".jpeg", dpi=dpi, bbox_inches="tight", pad_inches=0.04,
                facecolor="white", format="jpeg", pil_kwargs={"quality": 95})
    fig.savefig(out_base + ".pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved clean figure: {out_base}.jpeg")
    print(f"[OK] Saved clean figure: {out_base}.pdf")

# ============================================================
# Outputs
# ============================================================
def save_outputs(metrics: pd.DataFrame, paired: pd.DataFrame, inventory: pd.DataFrame,
                 grid_index: Optional[pd.DataFrame], taylor_agg: pd.DataFrame, out_dir: str,
                 daily_time_basis: str = "utc", fixed_time_shift_hours: int = 0,
                 min_valid_day_fraction: float = MIN_VALID_DAY_FRACTION,
                 min_valid_years_variable: int = MIN_VALID_YEARS_VARIABLE):
    ensure_dir(out_dir)
    paths = {
        "metrics_csv": os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_station_metrics_T_RH_WS.csv"),
        "metrics_xlsx": os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_station_metrics_T_RH_WS.xlsx"),
        "pairs_csv": os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_daily_station_pairs_T_RH_WS.csv"),
        "inventory_csv": os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_station_inventory_unique.csv"),
        "grid_csv": os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_station_nearest_grid_index.csv"),
        "taylor_csv": os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_Taylor_aggregate_metrics.csv"),
    }
    metrics.to_csv(paths["metrics_csv"], index=False)
    paired.to_csv(paths["pairs_csv"], index=False)
    inventory.to_csv(paths["inventory_csv"], index=False)
    taylor_agg.to_csv(paths["taylor_csv"], index=False)
    if grid_index is not None:
        grid_index.to_csv(paths["grid_csv"], index=False)

    with pd.ExcelWriter(paths["metrics_xlsx"]) as writer:
        metrics.to_excel(writer, sheet_name="station_metrics", index=False)
        taylor_agg.to_excel(writer, sheet_name="taylor_aggregate", index=False)
        inventory.to_excel(writer, sheet_name="station_inventory", index=False)
        if grid_index is not None:
            grid_index.to_excel(writer, sheet_name="nearest_grid_index", index=False)

    meta = {
        "software_version": SOFTWARE_VERSION,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "minimum_hourly_records_per_station_day": MIN_HOURS_PER_DAY,
        "minimum_paired_days_for_station_metrics": MIN_PAIRED_DAYS,
        "minimum_valid_day_fraction_station_year": min_valid_day_fraction,
        "minimum_valid_years_per_station_variable": min_valid_years_variable,
        "daily_time_basis": daily_time_basis,
        "fixed_time_shift_hours": fixed_time_shift_hours,
        "time_alignment_note": (
            "Default UTC day is appropriate when INMET uses Hora UTC and ERA5 valid_time is UTC. "
            "local_by_uf shifts both INMET and ERA5 to fixed Brazilian standard-time days before daily aggregation."
        ),
        "unit_consistency": {
            "temperature": "ERA5 K converted to °C; compared with INMET air temperature in °C",
            "relative_humidity": "ERA5 RH derived from T2m and dew point and expressed in %; compared with INMET RH in %",
            "wind_speed": "ERA5 wind speed computed as sqrt(u10^2+v10^2) in m s^-1; compared with INMET wind speed in m s^-1",
        },
        "aggregation": "station-level daily means only; no state averaging",
        "metrics": ["Pearson r", "RMSE", "Bias ERA5 minus INMET", "MAE", "Taylor normalized standard deviation"],
        "variables": list(VARIABLES.keys()),
        "bias_definition": "ERA5 minus INMET",
        "era5_sampling": "nearest ERA5 grid cell to each INMET station",
        "interpretation": "Station-level daily evaluation of ERA5 as a grid-scale reanalysis product.",
    }
    with open(os.path.join(out_dir, "software_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for p in paths.values():
        if os.path.exists(p):
            print(f"[OK] Saved: {p}")


# ============================================================
# Official INMET station metadata for coordinate control
# ============================================================

def _clean_numeric_scalar(x):
    """Parse scalar numeric values, preserving minus signs and comma decimals."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    txt = str(x).strip().replace(",", ".")
    txt = re.sub(r"[^0-9eE+\-.]", "", txt)
    try:
        return float(txt)
    except Exception:
        return np.nan


def _normalize_station_code_value(x) -> Optional[str]:
    """Normalize INMET station codes while preserving automatic-station IDs.

    INMET automatic stations are usually coded as A001, A866, etc. Preserving
    the letter avoids ambiguous matches with purely numeric WMO-style codes.
    Numeric IDs are still normalized to integer strings when appropriate.
    """
    if pd.isna(x):
        return None
    txt = str(x).strip().upper()
    if not txt:
        return None
    m_auto = re.search(r"\b([A-Z]\d{3})\b", txt)
    if m_auto:
        return m_auto.group(1)
    try:
        val = float(txt.replace(",", "."))
        if np.isfinite(val) and abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
    except Exception:
        pass
    m = re.search(r"\d+", txt)
    return m.group(0) if m else txt


def _find_header_row_xlsx(path: str, required_terms: List[str], max_rows: int = 20) -> int:
    raw = pd.read_excel(path, header=None, nrows=max_rows)
    req = [normalize_name(t) for t in required_terms]
    for i in range(len(raw)):
        vals = [normalize_name(v) for v in raw.iloc[i].tolist()]
        row = " ".join(vals)
        if all(any(r in v for v in vals) or r in row for r in req):
            return i
    return 0


def read_official_inmet_station_metadata(station_metadata_dir: str) -> pd.DataFrame:
    """Read official INMET station metadata from the climatological-normal workbook.

    This is used only to control the station coordinates used in the validation
    maps. The daily metrics themselves are still calculated from the paired
    INMET–ERA5 daily values. The official metadata avoids a common plotting
    problem in which coordinates recovered from raw hourly CSV caches can be
    parsed or overwritten incorrectly.
    """
    if not station_metadata_dir:
        return pd.DataFrame()
    patterns = [
        os.path.join(station_metadata_dir, "Normal-Climatologica-ESTA*.xlsx"),
        os.path.join(station_metadata_dir, "*ESTA*.xlsx"),
        os.path.join(station_metadata_dir, "*.xlsx"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print(f"[WARN] Official station metadata workbook not found in: {station_metadata_dir}")
        return pd.DataFrame()
    path = files[0]
    try:
        header = _find_header_row_xlsx(path, ["Código", "Latitude", "Longitude"])
        df = pd.read_excel(path, header=header)
        df.columns = [str(c).strip() for c in df.columns]
        code_col = find_column(df, ["Código", "Codigo", "Cod", "CD_ESTACAO"], required=True)
        uf_col = find_column(df, ["UF", "Estado"], required=True)
        lat_col = find_column(df, ["Latitude", "Lat"], required=True)
        lon_col = find_column(df, ["Longitude", "Lon"], required=True)
        name_col = find_column(df, ["Nome da Estação", "Nome da Estacao", "Estação", "Estacao", "Nome"], required=False)
        out = pd.DataFrame({
            "station_id": df[code_col].map(_normalize_station_code_value),
            "uf": df[uf_col].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0],
            "lat_official": df[lat_col].map(_clean_numeric_scalar),
            "lon_official": df[lon_col].map(_clean_numeric_scalar),
            "station_name_official": df[name_col].astype(str).str.strip() if name_col else "",
        })
        out = out.dropna(subset=["station_id", "uf", "lat_official", "lon_official"])
        rows = []
        for _, r in out.iterrows():
            la, lo = normalize_brazil_coordinates(r["lat_official"], r["lon_official"])
            if np.isfinite(la) and np.isfinite(lo):
                rows.append({
                    "station_id": str(r["station_id"]),
                    "uf": str(r["uf"]),
                    "lat_official": la,
                    "lon_official": lo,
                    "station_name_official": r.get("station_name_official", ""),
                })
        out = pd.DataFrame(rows).drop_duplicates(subset=["station_id", "uf"], keep="first") if rows else pd.DataFrame()
        print(f"[INFO] Official INMET station metadata loaded for coordinate control: {len(out)} stations from {Path(path).name}")
        return out
    except Exception as exc:
        print(f"[WARN] Could not read official INMET station metadata from {path}: {exc}")
        return pd.DataFrame()


def overwrite_inventory_coordinates_with_official(inventory: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Overwrite inventory coordinates with official INMET metadata when available."""
    inv = inventory.copy()
    inv["station_id"] = inv["station_id"].astype(str)
    inv["uf"] = inv["uf"].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]
    if official is None or official.empty:
        return inv
    off = official.copy()
    off["station_id"] = off["station_id"].astype(str)
    off["uf"] = off["uf"].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]
    inv = inv.merge(off[["station_id", "uf", "lat_official", "lon_official", "station_name_official"]], on=["station_id", "uf"], how="left")
    use_off = inv["lat_official"].notna() & inv["lon_official"].notna()
    inv.loc[use_off, "lat"] = inv.loc[use_off, "lat_official"]
    inv.loc[use_off, "lon"] = inv.loc[use_off, "lon_official"]
    inv["coordinate_source"] = np.where(use_off, "INMET_climatological_normals_metadata", "raw_INMET_hourly_inventory")
    print(f"[INFO] Inventory coordinate overwrite from official metadata: {int(use_off.sum())}/{len(inv)} station rows.")
    return inv


# ============================================================
# Coordinate handling
# ============================================================

def normalize_brazil_coordinates(lat, lon):
    """Robustly normalize INMET station coordinates for Brazil maps.

    Handles: decimal-degree values, positive west longitudes, 0--360 longitudes,
    swapped latitude/longitude fields, and coordinates accidentally stored as
    scaled integers such as -2355 or -4778. Returns (nan, nan) only when no
    plausible Brazil coordinate can be recovered.
    """
    def to_float(x):
        try:
            return float(str(x).replace(',', '.'))
        except Exception:
            return np.nan

    lat0 = to_float(lat)
    lon0 = to_float(lon)

    def fix_lon(x):
        if not np.isfinite(x):
            return x
        if x > 180:
            x = ((x + 180) % 360) - 180
        if 30 <= x <= 80:
            x = -x
        return x

    def in_domain(la, lo, buffer=True):
        if buffer:
            return np.isfinite(la) and np.isfinite(lo) and (-36.5 <= la <= 7.5) and (-77.5 <= lo <= -30.0)
        return np.isfinite(la) and np.isfinite(lo) and (BRAZIL_EXTENT[2] <= la <= BRAZIL_EXTENT[3]) and (BRAZIL_EXTENT[0] <= lo <= BRAZIL_EXTENT[1])

    candidates = []
    scales = [1.0, 0.1, 0.01, 0.001]
    for a, b in [(lat0, lon0), (lon0, lat0)]:
        for slat in scales:
            for slon in scales:
                la = a * slat if np.isfinite(a) else np.nan
                lo = b * slon if np.isfinite(b) else np.nan
                lo = fix_lon(lo)
                # latitude sometimes stored as positive magnitude in southern Brazil
                if np.isfinite(la) and la > 7.5 and -36.5 <= -la <= 7.5:
                    la = -la
                if in_domain(la, lo, buffer=True):
                    # prefer values inside plotting extent and close to decimal-degree scale
                    score = (0 if in_domain(la, lo, buffer=False) else 1) + abs(slat-1.0)*0.01 + abs(slon-1.0)*0.01
                    candidates.append((score, la, lo))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return float(candidates[0][1]), float(candidates[0][2])
    return np.nan, np.nan


def _candidate_coord_columns(df: pd.DataFrame):
    lat_names = ["lat", "latitude", "LATITUDE", "Lat", "LAT", "lat_x", "lat_y", "station_lat", "era5_lat"]
    lon_names = ["lon", "longitude", "LONGITUDE", "Lon", "LON", "lon_x", "lon_y", "station_lon", "era5_lon"]
    lat_cols = [c for c in lat_names if c in df.columns]
    lon_cols = [c for c in lon_names if c in df.columns]
    # Also include normalized-name matches.
    for c in df.columns:
        nc = normalize_name(c)
        if nc in [normalize_name(x) for x in lat_names] and c not in lat_cols:
            lat_cols.append(c)
        if nc in [normalize_name(x) for x in lon_names] and c not in lon_cols:
            lon_cols.append(c)
    return lat_cols, lon_cols


def merge_coordinate_lookups(*lookups):
    out = {}
    for lk in lookups:
        if lk:
            out.update(lk)
    return out


# ============================================================
# Trusted coordinate control for map plotting
# ============================================================
# Some INMET raw hourly CSV inventories can contain malformed coordinates after
# cache reuse or station-code parsing. For plotting only, use a trusted station
# coordinate table generated by the companion validation workflow when
# available. The validation metrics themselves are unchanged.

EXTRA_COORD_LOOKUP = {}

def _station_id_keys_for_matching(x):
    """Return robust station-id variants for coordinate matching."""
    if pd.isna(x):
        return []
    s = str(x).strip().upper()
    if not s:
        return []
    s0 = re.sub(r"\s+", "", s)
    s_alnum = re.sub(r"[^A-Z0-9]", "", s0)
    keys = {s0, s_alnum}
    # Numeric version, preserving useful automatic-station forms such as A001.
    m = re.search(r"([A-Z]+)(0*\d+)$", s_alnum)
    if m:
        prefix, digits = m.group(1), m.group(2)
        keys.add(prefix + digits)
        keys.add(prefix + str(int(digits)))
        keys.add(str(int(digits)))
    digits_only = re.sub(r"\D", "", s_alnum)
    if digits_only:
        try:
            keys.add(str(int(digits_only)))
        except Exception:
            keys.add(digits_only)
    return [k for k in keys if k]


def _make_coord_key_variants(station_id, uf):
    uf0 = str(uf).strip().upper()
    m = re.search(r"([A-Z]{2})", uf0)
    uf0 = m.group(1) if m else uf0
    return [(k, uf0) for k in _station_id_keys_for_matching(station_id)]


def _add_coord_to_lookup(lookup, station_id, uf, lat, lon, source="unknown"):
    la, lo = normalize_brazil_coordinates(lat, lon)
    if not (np.isfinite(la) and np.isfinite(lo)):
        return 0
    n = 0
    for key in _make_coord_key_variants(station_id, uf):
        if key not in lookup:
            lookup[key] = (la, lo, source)
            n += 1
    return n


def _get_coord_from_lookup(lookup, station_id, uf):
    for key in _make_coord_key_variants(station_id, uf):
        if key in lookup:
            la, lo, src = lookup[key]
            return la, lo, src
    return np.nan, np.nan, "missing"


def read_trusted_station_coordinate_table(path: str) -> Dict[Tuple[str, str], Tuple[float, float, str]]:
    """Read a trusted station coordinate table from the heatwave QC workflow.

    The function is intentionally flexible about column names. It requires a
    station id, UF/state, latitude and longitude. Rows selected by the heatwave
    QC script are preferred when a selected/retain/pass column is present.
    """
    lookup = {}
    if not path or not os.path.exists(path):
        if path:
            print(f"[WARN] Trusted coordinate table not found: {path}")
        return lookup
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read trusted coordinate table {path}: {exc}")
        return lookup
    if df.empty:
        return lookup

    # Prefer retained/selected stations if such a column exists.
    for c in df.columns:
        nc = normalize_name(c)
        if any(tok in nc for tok in ["selected", "retained", "pass", "used", "valid_for_trend", "use_for_trend"]):
            vals = df[c]
            # Boolean-like or 0/1-like columns only.
            try:
                s = vals.astype(str).str.lower().str.strip()
                mask = s.isin(["1", "true", "yes", "y", "selected", "retained", "pass"])
                if mask.sum() > 0:
                    df = df[mask].copy()
                    print(f"[INFO] Trusted coordinate table filtered by column '{c}': {len(df)} rows retained.")
                    break
            except Exception:
                pass

    sid_col = find_column(df, ["station_id", "station", "codigo", "cod_estacao", "wmo", "id"], required=False)
    uf_col = find_column(df, ["uf", "state", "estado"], required=False)
    lat_col = find_column(df, ["lat", "latitude", "station_lat", "lat_station"], required=False)
    lon_col = find_column(df, ["lon", "longitude", "station_lon", "lon_station"], required=False)
    if sid_col is None or uf_col is None or lat_col is None or lon_col is None:
        print(f"[WARN] Trusted coordinate table lacks required columns. Found sid={sid_col}, uf={uf_col}, lat={lat_col}, lon={lon_col}. Columns={list(df.columns)}")
        return lookup
    added_rows = 0
    for _, r in df.iterrows():
        added = _add_coord_to_lookup(lookup, r[sid_col], r[uf_col], r[lat_col], r[lon_col], source="trusted_station_table")
        if added:
            added_rows += 1
    print(f"[INFO] Trusted coordinate lookup loaded: {added_rows} station rows, {len(lookup)} station-id variants from {Path(path).name}")
    return lookup


def coordinate_lookup_from_dataframe(df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float, str]]:
    """Extract station coordinates from a table using robust station-id variants."""
    lookup = {}
    if df is None or df.empty or "station_id" not in df.columns or "uf" not in df.columns:
        return lookup
    tmp = df.copy()
    tmp["station_id"] = tmp["station_id"].astype(str)
    tmp["uf"] = tmp["uf"].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]
    lat_cols, lon_cols = _candidate_coord_columns(tmp)
    if not lat_cols or not lon_cols:
        return lookup
    for (sid, uf), g in tmp.groupby(["station_id", "uf"], dropna=False):
        for lat_col in lat_cols:
            for lon_col in lon_cols:
                lat_med = pd.to_numeric(g[lat_col], errors="coerce").median()
                lon_med = pd.to_numeric(g[lon_col], errors="coerce").median()
                if _add_coord_to_lookup(lookup, sid, uf, lat_med, lon_med, source=f"{lat_col}/{lon_col}"):
                    break
            else:
                continue
            break
    return lookup


def parse_outlier_min_abs(spec: str) -> Dict[str, float]:
    """Parse variable-specific absolute daily-difference limits.

    Expected format: "temperature=5,relative_humidity=20,wind_speed=3".
    Values are interpreted in each variable's native units.
    """
    defaults = {"temperature": 5.0, "relative_humidity": 20.0, "wind_speed": 3.0}
    if spec is None or str(spec).strip() == "":
        return defaults
    out = defaults.copy()
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --outlier_min_abs item '{item}'. Use variable=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in defaults:
            raise ValueError(f"Invalid variable '{k}' in --outlier_min_abs. Valid variables: {list(defaults)}")
        out[k] = float(v)
    return out


def apply_pairwise_outlier_qc(
    paired: pd.DataFrame,
    out_dir: str,
    enabled: bool = True,
    mad_k: float = 6.0,
    max_fraction: float = 0.03,
    min_abs_spec: str = "temperature=5,relative_humidity=20,wind_speed=3",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Conservative station-variable pairwise outlier QC.

    The QC is applied after physical daily QC and annual coverage QC. It removes
    only station-variable daily pairs whose ERA5-INMET difference is extreme
    relative to that station's own paired-difference distribution AND exceeds a
    variable-specific absolute difference threshold. To avoid over-cleaning, no
    more than max_fraction of valid pairs is removed within each station-variable.

    Removal is variable-specific: for an outlier in one variable, only that
    variable's obs/ERA5 pair is set to NaN on that date. Other variables on the
    same station-day are retained.
    """
    min_abs = parse_outlier_min_abs(min_abs_spec)
    summary_rows = []
    out = paired.copy()
    if not enabled:
        print("[INFO] Pairwise outlier QC enabled: False")
        return out, pd.DataFrame(summary_rows)

    print("[INFO] Pairwise outlier QC enabled: True")
    print(f"[INFO] Pairwise outlier QC settings: MAD k={mad_k:g}; max fraction={max_fraction:.3f}; min abs={min_abs}")

    total_removed = 0
    total_valid = 0
    max_fraction = max(0.0, min(float(max_fraction), 1.0))
    mad_k = float(mad_k)

    for (station_id, uf), idx in out.groupby(["station_id", "uf"]).groups.items():
        idx = list(idx)
        for var in VAR_ORDER:
            obs_col = f"{var}_obs"
            era_col = f"{var}_era5"
            if obs_col not in out.columns or era_col not in out.columns:
                continue
            obs = pd.to_numeric(out.loc[idx, obs_col], errors="coerce")
            era = pd.to_numeric(out.loc[idx, era_col], errors="coerce")
            ok = obs.notna() & era.notna()
            n_valid = int(ok.sum())
            total_valid += n_valid
            n_removed = 0
            median_diff = np.nan
            mad = np.nan
            robust_sigma = np.nan
            threshold_robust = np.nan
            abs_limit = float(min_abs.get(var, np.nan))
            if n_valid >= 30 and np.isfinite(abs_limit):
                valid_index = obs.index[ok]
                diff = (era.loc[valid_index] - obs.loc[valid_index]).astype(float)
                median_diff = float(np.nanmedian(diff))
                abs_dev = np.abs(diff - median_diff)
                mad = float(np.nanmedian(abs_dev))
                robust_sigma = float(1.4826 * mad) if np.isfinite(mad) else np.nan
                # If MAD collapses to zero, use a conservative percentile fallback.
                if not np.isfinite(robust_sigma) or robust_sigma <= 0:
                    q75, q25 = np.nanpercentile(diff, [75, 25])
                    robust_sigma = float((q75 - q25) / 1.349) if np.isfinite(q75 - q25) and (q75 - q25) > 0 else np.nan
                if np.isfinite(robust_sigma) and robust_sigma > 0:
                    threshold_robust = mad_k * robust_sigma
                    candidate_mask = (abs_dev > threshold_robust) & (np.abs(diff) > abs_limit)
                    cand_index = list(valid_index[candidate_mask.to_numpy()])
                    if cand_index:
                        max_remove = max(1, int(np.floor(max_fraction * n_valid))) if max_fraction > 0 else 0
                        if max_remove > 0:
                            # Remove the most extreme candidates first.
                            cand_scores = abs_dev.loc[cand_index].sort_values(ascending=False)
                            remove_index = list(cand_scores.index[:max_remove])
                            out.loc[remove_index, [obs_col, era_col]] = np.nan
                            n_removed = len(remove_index)
                            total_removed += n_removed
            summary_rows.append({
                "station_id": str(station_id),
                "uf": uf,
                "region": UF_TO_REGION.get(uf),
                "variable": var,
                "n_valid_pairs_before_outlier_qc": n_valid,
                "n_pairs_removed_by_outlier_qc": n_removed,
                "fraction_removed_by_outlier_qc": (n_removed / n_valid) if n_valid > 0 else np.nan,
                "median_difference_era5_minus_inmet": median_diff,
                "mad_difference": mad,
                "robust_sigma_difference": robust_sigma,
                "robust_threshold_used": threshold_robust,
                "minimum_absolute_difference_threshold": abs_limit,
                "mad_k": mad_k,
                "max_fraction": max_fraction,
            })

    summary = pd.DataFrame(summary_rows)
    if total_valid > 0:
        print(f"[INFO] Pairwise outlier QC removed {total_removed:,} of {total_valid:,} valid station-variable daily pairs ({total_removed / total_valid:.3%}).")
    else:
        print("[INFO] Pairwise outlier QC found no valid station-variable daily pairs.")
    if not summary.empty:
        by_var = summary.groupby("variable").apply(
            lambda g: float(g["n_pairs_removed_by_outlier_qc"].sum() / max(g["n_valid_pairs_before_outlier_qc"].sum(), 1))
        ).to_dict()
        print(f"[INFO] Pairwise outlier QC removed fractions by variable: {by_var}")
        summary_path = os.path.join(out_dir, "Supplementary_Table_ERA5_INMET_pairwise_outlier_QC_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"[OK] Saved pairwise outlier QC summary: {summary_path}")
    return out, summary

def calculate_station_metrics(paired: pd.DataFrame, inventory: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    """Calculate station metrics with controlled map coordinates.

    Coordinate priority for plotting:
      1) trusted station-selection table, if provided;
      2) official INMET climatological-normal metadata, if matched in inventory;
      3) raw INMET inventory coordinates.

    ERA5/cache coordinates are deliberately not used for maps, because they can
    represent nearest grid-cell locations or inherit malformed coordinate fields.
    """
    inv_lookup = coordinate_lookup_from_dataframe(inventory)
    rows = []
    for (station_id, uf), g in paired.groupby(["station_id", "uf"]):
        lat, lon, coord_source = _get_coord_from_lookup(EXTRA_COORD_LOOKUP, station_id, uf)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            lat, lon, coord_source = _get_coord_from_lookup(inv_lookup, station_id, uf)
        for var, meta in VARIABLES.items():
            obs = pd.to_numeric(g[f"{var}_obs"], errors="coerce")
            mod = pd.to_numeric(g[f"{var}_era5"], errors="coerce")
            ok = obs.notna() & mod.notna()
            n = int(ok.sum())
            if n >= min_pairs:
                xv = obs[ok].to_numpy(dtype=float)
                yv = mod[ok].to_numpy(dtype=float)
                if np.nanstd(xv) > 0 and np.nanstd(yv) > 0 and n >= 3:
                    r, p = pearsonr(xv, yv)
                else:
                    r, p = np.nan, np.nan
                diff = yv - xv
                rmse = float(np.sqrt(np.nanmean(diff ** 2)))
                bias = float(np.nanmean(diff))
                mae = float(np.nanmean(np.abs(diff)))
                obs_std = float(np.nanstd(xv, ddof=1)) if n > 1 else np.nan
                era5_std = float(np.nanstd(yv, ddof=1)) if n > 1 else np.nan
                std_ratio = era5_std / obs_std if np.isfinite(obs_std) and obs_std > 0 else np.nan
                obs_mean = float(np.nanmean(xv))
                era5_mean = float(np.nanmean(yv))
                first_date = g.loc[ok, "date"].min()
                last_date = g.loc[ok, "date"].max()
            else:
                r = p = rmse = bias = mae = obs_std = era5_std = std_ratio = obs_mean = era5_mean = np.nan
                first_date = last_date = pd.NaT
            rows.append({
                "station_id": str(station_id),
                "uf": uf,
                "region": UF_TO_REGION.get(uf),
                "lat": lat,
                "lon": lon,
                "coordinate_source": coord_source,
                "variable": var,
                "variable_label": meta["label"],
                "unit": meta["unit"].replace("$^{-1}$", "-1"),
                "n_paired_days": n,
                "minimum_required_pairs": min_pairs,
                "start_date": first_date,
                "end_date": last_date,
                "obs_mean": obs_mean,
                "era5_mean": era5_mean,
                "obs_std": obs_std,
                "era5_std": era5_std,
                "std_ratio": std_ratio,
                "pearson_r": float(r) if np.isfinite(r) else np.nan,
                "pearson_p": float(p) if np.isfinite(p) else np.nan,
                "rmse": rmse,
                "bias_era5_minus_inmet": bias,
                "mae": mae,
            })
    out = pd.DataFrame(rows)
    try:
        print("[INFO] Coordinate source counts:", out.drop_duplicates(["station_id", "uf"])["coordinate_source"].value_counts(dropna=False).to_dict())
    except Exception:
        pass
    return out


def repair_metrics_coordinates(metrics: pd.DataFrame, inventory: pd.DataFrame, paired: Optional[pd.DataFrame] = None, era_station_daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Final coordinate validation. Do not use ERA5/cache coordinates as fallback."""
    out = metrics.copy()
    inv_lookup = coordinate_lookup_from_dataframe(inventory)
    fixed_lat, fixed_lon, sources = [], [], []
    for _, r in out.iterrows():
        lat0, lon0 = normalize_brazil_coordinates(
            r.get("lat", np.nan),
            r.get("lon", np.nan),
        )
        if np.isfinite(lat0) and np.isfinite(lon0):
            lat, lon, src = lat0, lon0, r.get("coordinate_source", "existing")
        else:
            lat, lon, src = _get_coord_from_lookup(EXTRA_COORD_LOOKUP, r.get("station_id"), r.get("uf"))
            if not (np.isfinite(lat) and np.isfinite(lon)):
                lat, lon, src = _get_coord_from_lookup(inv_lookup, r.get("station_id"), r.get("uf"))
        fixed_lat.append(lat); fixed_lon.append(lon); sources.append(src)
    out["lat"] = fixed_lat
    out["lon"] = fixed_lon
    out["coordinate_source"] = sources
    out["inside_plot_domain"] = (
        pd.to_numeric(out["lat"], errors="coerce").between(BRAZIL_EXTENT[2], BRAZIL_EXTENT[3]) &
        pd.to_numeric(out["lon"], errors="coerce").between(BRAZIL_EXTENT[0], BRAZIL_EXTENT[1])
    )
    n_ok = int((out["inside_plot_domain"] & np.isfinite(pd.to_numeric(out["pearson_r"], errors="coerce"))).sum())
    print(f"[INFO] Coordinate repair: {n_ok} station-variable rows have valid metric and plot-ready coordinates.")
    try:
        print("[INFO] Final coordinate source counts:", out.drop_duplicates(["station_id", "uf"])["coordinate_source"].value_counts(dropna=False).to_dict())
        print("[INFO] Final plotted-station region counts:", out[(out["variable"] == VAR_ORDER[0]) & out["inside_plot_domain"]].groupby("region").size().to_dict())
    except Exception:
        pass
    return out


def trusted_selected_station_keys(path: str) -> set:
    """Return selected station keys from the heatwave-metric station-selection table.

    This keeps the daily T/RH/WS evaluation aligned with the station population
    used in the companion ERA5–INMET evaluation. The function accepts
    station_id variants such as A001 and 1 to avoid accidental losses caused by
    station-code formatting.
    """
    keys = set()
    if not path or not os.path.exists(path):
        return keys
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read trusted selected-station table {path}: {exc}")
        return keys
    if df.empty:
        return keys

    # Prefer rows selected by the heatwave evaluation if such a column exists.
    selected_col = None
    for c in df.columns:
        nc = normalize_name(c)
        if nc in ["selected", "retained", "valid_for_trend", "use_for_trend"] or any(tok in nc for tok in ["selected", "retained", "valid_for_trend", "use_for_trend"]):
            selected_col = c
            break
    if selected_col is not None:
        s = df[selected_col].astype(str).str.lower().str.strip()
        mask = s.isin(["1", "true", "yes", "y", "selected", "retained", "pass"])
        if mask.sum() > 0:
            df = df[mask].copy()

    sid_col = find_column(df, ["station_id", "station", "codigo", "cod_estacao", "wmo", "id"], required=False)
    uf_col = find_column(df, ["uf", "state", "estado"], required=False)
    if sid_col is None or uf_col is None:
        print(f"[WARN] Trusted selected-station table lacks station_id/uf columns. Columns={list(df.columns)}")
        return keys

    for _, r in df.iterrows():
        uf = str(r[uf_col]).upper()
        m = re.search(r"([A-Z]{2})", uf)
        if not m:
            continue
        uf = m.group(1)
        for sid_key, uf_key in _make_coord_key_variants(r[sid_col], uf):
            keys.add((sid_key, uf_key))
    print(f"[INFO] Trusted selected-station filter loaded: {len(df)} selected rows, {len(keys)} station-id variants.")
    return keys


def restrict_table_to_trusted_stations(df: pd.DataFrame, trusted_keys: set, label: str) -> pd.DataFrame:
    """Restrict any station table to trusted selected stations using robust station-id variants."""
    if df is None or df.empty or not trusted_keys:
        return df
    if "station_id" not in df.columns or "uf" not in df.columns:
        print(f"[WARN] Cannot apply trusted station filter to {label}: station_id/uf columns are missing.")
        return df
    out = df.copy()
    before = len(out)

    def keep_row(row):
        for key in _make_coord_key_variants(row["station_id"], row["uf"]):
            if key in trusted_keys:
                return True
        return False

    mask = out[["station_id", "uf"]].apply(keep_row, axis=1)
    out = out[mask].copy()
    nstations = out[["station_id", "uf"]].drop_duplicates().shape[0] if not out.empty else 0
    print(f"[INFO] Trusted selected-station filter for {label}: retained {len(out)}/{before} rows across {nstations} station-UF pairs.")
    return out


# ============================================================
# Main
# ============================================================
def main():
    global START_DATE, END_DATE, MIN_HOURS_PER_DAY, MIN_PAIRED_DAYS
    import argparse

    parser = argparse.ArgumentParser(
        description="Station-level ERA5–INMET daily evaluation with Taylor diagrams."
    )
    parser.add_argument(
        "--inmet-dir", "--inmet_dir",
        dest="inmet_dir",
        required=True,
        help="Directory containing INMET hourly CSV files.",
    )
    parser.add_argument(
        "--era5-source", "--era5_source",
        dest="era5_source",
        choices=["cache", "daily", "hourly"],
        default="cache",
        help="ERA5 input mode: station cache, preprocessed daily files, or hourly files.",
    )
    parser.add_argument(
        "--era5-station-cache", "--era5_station_cache",
        dest="era5_station_cache",
        default="",
        help="Station-daily ERA5 CSV used in cache mode.",
    )
    parser.add_argument(
        "--era5-daily-dir", "--era5_daily_dir",
        dest="era5_daily_dir",
        default=None,
        help="Directory containing preprocessed daily ERA5 files.",
    )
    parser.add_argument(
        "--era5-hourly-glob", "--era5_hourly_glob",
        dest="era5_hourly_glob",
        default=None,
        help="Glob pattern for hourly ERA5 files.",
    )
    parser.add_argument(
        "--station-metadata-dir", "--station_metadata_dir",
        dest="station_metadata_dir",
        default=None,
        help="Optional directory containing official INMET station metadata.",
    )
    parser.add_argument(
        "--trusted-station-table", "--trusted_station_coord_table",
        dest="trusted_station_coord_table",
        default=None,
        help="Optional station-selection/coordinate table used for station alignment and mapping.",
    )
    parser.add_argument(
        "--do-not-restrict-to-trusted-selected-stations",
        "--do_not_restrict_to_trusted_selected_stations",
        dest="do_not_restrict_to_trusted_selected_stations",
        action="store_true",
        help="Do not restrict the evaluation to stations selected in the trusted station table.",
    )
    parser.add_argument(
        "--states-shapefile", "--shp_uf",
        dest="shp_uf",
        required=True,
        help="Brazilian state shapefile used for plotting.",
    )
    parser.add_argument(
        "--output-dir", "--out_dir",
        dest="out_dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument(
        "--min-hours-day", "--min_hours_day",
        dest="min_hours_day",
        type=int,
        default=MIN_HOURS_PER_DAY,
    )
    parser.add_argument(
        "--min-pairs", "--min_pairs",
        dest="min_pairs",
        type=int,
        default=MIN_PAIRED_DAYS,
    )
    parser.add_argument(
        "--min-valid-day-fraction", "--min_valid_day_fraction",
        dest="min_valid_day_fraction",
        type=float,
        default=MIN_VALID_DAY_FRACTION,
    )
    parser.add_argument(
        "--min-valid-years-variable", "--min_valid_years_variable",
        dest="min_valid_years_variable",
        type=int,
        default=MIN_VALID_YEARS_VARIABLE,
    )
    parser.add_argument(
        "--daily-time-basis", "--daily_time_basis",
        dest="daily_time_basis",
        choices=sorted(TIME_BASIS_OPTIONS),
        default="utc",
    )
    parser.add_argument(
        "--fixed-time-shift-hours", "--fixed_time_shift_hours",
        dest="fixed_time_shift_hours",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--reuse-inmet-cache", "--reuse_inmet_cache",
        dest="reuse_inmet_cache",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-era5-cache", "--reuse_era5_cache",
        dest="reuse_era5_cache",
        action="store_true",
    )
    parser.add_argument(
        "--max-years-per-run", "--max_years_per_run",
        dest="max_years_per_run",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--only-finalize-from-cache", "--only_finalize_from_cache",
        dest="only_finalize_from_cache",
        action="store_true",
    )
    parser.add_argument(
        "--build-era5-cache-only", "--build_era5_cache_only",
        dest="build_era5_cache_only",
        action="store_true",
    )
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument(
        "--disable-outlier-qc", "--disable_outlier_qc",
        dest="disable_outlier_qc",
        action="store_true",
    )
    parser.add_argument(
        "--outlier-mad-k", "--outlier_mad_k",
        dest="outlier_mad_k",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--outlier-max-fraction", "--outlier_max_fraction",
        dest="outlier_max_fraction",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--outlier-min-abs", "--outlier_min_abs",
        dest="outlier_min_abs",
        default="temperature=5,relative_humidity=20,wind_speed=3",
    )
    args = parser.parse_args()

    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error("--start must be earlier than or equal to --end.")
    if args.era5_source == "cache" and not args.era5_station_cache:
        parser.error("--era5-station-cache is required when --era5-source cache is used.")
    if args.era5_source == "daily" and not args.era5_daily_dir:
        parser.error("--era5-daily-dir is required when --era5-source daily is used.")
    if args.era5_source == "hourly" and not args.era5_hourly_glob:
        parser.error("--era5-hourly-glob is required when --era5-source hourly is used.")

    START_DATE = args.start
    END_DATE = args.end
    MIN_HOURS_PER_DAY = args.min_hours_day
    MIN_PAIRED_DAYS = args.min_pairs
    ensure_dir(args.out_dir)

    print("[START] Station-level ERA5 vs INMET evaluation with Taylor diagrams")
    print(f"[INFO] Period: {START_DATE} to {END_DATE}")
    print(f"[INFO] INMET directory: {args.inmet_dir}")
    print(f"[INFO] Official station metadata directory: {args.station_metadata_dir or 'not supplied'}")
    print(f"[INFO] ERA5 hourly glob: {args.era5_hourly_glob or 'not supplied'}")
    print(f"[INFO] ERA5 daily directory: {args.era5_daily_dir or 'not supplied'}")
    print(f"[INFO] ERA5 source mode: {args.era5_source}")
    print(f"[INFO] Minimum hourly records per station-day: {MIN_HOURS_PER_DAY}")
    print(f"[INFO] Minimum paired days for station metrics: {MIN_PAIRED_DAYS}")
    print(f"[INFO] Station-year coverage QC: >= {args.min_valid_day_fraction:.0%} paired days year-1")
    print(f"[INFO] Minimum retained station-years per variable: {args.min_valid_years_variable}")
    print(f"[INFO] Daily time basis: {args.daily_time_basis}; fixed shift: {args.fixed_time_shift_hours} h")
    if args.daily_time_basis == "utc":
        print("[INFO] Time alignment: INMET Hora UTC and ERA5 valid_time are compared on UTC calendar days.")
    elif args.daily_time_basis == "local_by_uf":
        print("[INFO] Time alignment: both INMET and ERA5 are shifted to fixed station-local days by UF before daily aggregation.")
    if args.era5_source == "cache" and args.daily_time_basis != "utc":
        raise ValueError(
            "When --era5_source cache is used, --daily_time_basis must be utc because the precomputed ERA5 station cache is already daily and cannot be shifted consistently. "
            "Use --daily_time_basis utc for alignment with the companion validation workflow, or rerun with --era5_source daily/hourly if local-day aggregation is required."
        )
    print("[INFO] Unit check: temperature in °C, relative humidity in %, wind speed in m s^-1 for both ERA5 and INMET.")
    print("[INFO] No state averaging will be used; daily and annual coverage QC will be applied.")
    print(f"[INFO] Pairwise outlier QC enabled: {not args.disable_outlier_qc}")
    if not args.disable_outlier_qc:
        print(f"[INFO] Pairwise outlier QC parameters: MAD k={args.outlier_mad_k:g}; max fraction={args.outlier_max_fraction:.3f}; min abs='{args.outlier_min_abs}'")

    cache_daily = os.path.join(args.out_dir, f"CACHE_INMET_daily_station_minH{MIN_HOURS_PER_DAY}_{time_basis_tag(args.daily_time_basis, args.fixed_time_shift_hours)}_QC.csv")
    cache_inventory = os.path.join(args.out_dir, "CACHE_INMET_station_inventory_unique_QC.csv")

    if args.reuse_inmet_cache and os.path.exists(cache_daily) and os.path.exists(cache_inventory):
        print("[INFO] Reusing cached INMET daily station and inventory tables.")
        obs_daily_station = pd.read_csv(cache_daily, parse_dates=["date"])
        inventory = pd.read_csv(cache_inventory)
    else:
        obs_hourly, inventory = read_all_inmet(args.inmet_dir)
        print(f"[INFO] Unique valid INMET stations after metadata deduplication: {len(inventory)}")
        print(f"[INFO] INMET hourly records after date filter: {len(obs_hourly)}")
        obs_daily_station = inmet_hourly_to_daily_station(
            obs_hourly,
            min_hours_day=MIN_HOURS_PER_DAY,
            daily_time_basis=args.daily_time_basis,
            fixed_time_shift_hours=args.fixed_time_shift_hours,
        )
        print(f"[INFO] INMET daily station rows: {len(obs_daily_station)}")
        obs_daily_station.to_csv(cache_daily, index=False)
        inventory.to_csv(cache_inventory, index=False)
        print(f"[OK] Cached INMET daily station table: {cache_daily}")
        print(f"[OK] Cached INMET station inventory: {cache_inventory}")

    inventory["station_id"] = inventory["station_id"].astype(str)
    obs_daily_station["station_id"] = obs_daily_station["station_id"].astype(str)

    official_station_meta = read_official_inmet_station_metadata(args.station_metadata_dir)
    inventory = overwrite_inventory_coordinates_with_official(inventory, official_station_meta)
    global EXTRA_COORD_LOOKUP
    EXTRA_COORD_LOOKUP = read_trusted_station_coordinate_table(args.trusted_station_coord_table)
    if not EXTRA_COORD_LOOKUP and not official_station_meta.empty:
        EXTRA_COORD_LOOKUP = coordinate_lookup_from_dataframe(official_station_meta.rename(columns={"lat_official":"lat", "lon_official":"lon"}))
    # Also repair the daily observation table coordinates for downstream paired-table diagnostics.
    if not official_station_meta.empty:
        _off = official_station_meta.copy()
        _off["station_id"] = _off["station_id"].astype(str)
        _off["uf"] = _off["uf"].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]
        obs_daily_station = obs_daily_station.merge(_off[["station_id", "uf", "lat_official", "lon_official"]], on=["station_id", "uf"], how="left")
        _use = obs_daily_station["lat_official"].notna() & obs_daily_station["lon_official"].notna()
        obs_daily_station.loc[_use, "lat"] = obs_daily_station.loc[_use, "lat_official"]
        obs_daily_station.loc[_use, "lon"] = obs_daily_station.loc[_use, "lon_official"]
        obs_daily_station = obs_daily_station.drop(columns=["lat_official", "lon_official"], errors="ignore")

    trusted_keys_for_filter = set()
    if not args.do_not_restrict_to_trusted_selected_stations:
        trusted_keys_for_filter = trusted_selected_station_keys(args.trusted_station_coord_table)
        if trusted_keys_for_filter:
            obs_daily_station = restrict_table_to_trusted_stations(obs_daily_station, trusted_keys_for_filter, "INMET daily table")
            inventory = restrict_table_to_trusted_stations(inventory, trusted_keys_for_filter, "INMET inventory")
        else:
            print("[WARN] Trusted selected-station filter was requested but no trusted keys were available; proceeding without station-population restriction.")

    era_station_daily = None
    grid_index = None
    cache_used = None

    if args.era5_source == "cache":
        era_station_daily, grid_index, cache_used = try_load_precomputed_era5_station_cache(
            args.out_dir,
            explicit_cache=args.era5_station_cache,
        )
        if era_station_daily is not None:
            print("[INFO] Using the supplied precomputed ERA5 station cache.")

    if era_station_daily is None and args.era5_source == "cache":
        raise RuntimeError(
            "ERA5 source was set to 'cache', but the supplied station cache "
            "could not be loaded."
        )

    if era_station_daily is None and args.era5_source == "daily":
        print("[INFO] Using preprocessed daily ERA5 files. ERA5 hourly NetCDF files will not be opened.")
        print("[INFO] A station-level cache can be used instead with --era5-source cache.")
        era_station_daily, grid_index = build_era5_station_daily_from_daily_files(
            args.era5_daily_dir,
            inventory,
            out_dir=args.out_dir,
            reuse_cache=args.reuse_era5_cache,
            max_years_per_run=args.max_years_per_run,
            only_finalize_from_cache=args.only_finalize_from_cache,
        )
    elif era_station_daily is None:
        era_station_daily, grid_index = build_era5_station_daily_all_years(
            args.era5_hourly_glob,
            inventory,
            out_dir=args.out_dir,
            reuse_cache=args.reuse_era5_cache,
            min_hours_day=MIN_HOURS_PER_DAY,
            daily_time_basis=args.daily_time_basis,
            fixed_time_shift_hours=args.fixed_time_shift_hours,
            max_years_per_run=args.max_years_per_run,
            only_finalize_from_cache=args.only_finalize_from_cache,
        )
    print(f"[INFO] ERA5 station daily rows: {len(era_station_daily)}")

    if args.build_era5_cache_only:
        print("[DONE] ERA5 station-daily cache preparation completed.")
        return
    if trusted_keys_for_filter:
        era_station_daily = restrict_table_to_trusted_stations(era_station_daily, trusted_keys_for_filter, "ERA5 station-daily table")
        print(f"[INFO] ERA5 station daily rows after trusted station restriction: {len(era_station_daily)}")

    paired = build_station_daily_pairs(obs_daily_station, era_station_daily)
    print(f"[INFO] Station-level paired daily rows before annual coverage QC: {len(paired)}")
    paired = apply_station_year_coverage_qc(
        paired,
        min_valid_day_fraction=args.min_valid_day_fraction,
        min_valid_years_variable=args.min_valid_years_variable,
    )
    print(f"[INFO] Station-level paired daily rows after annual coverage QC: {len(paired)}")

    paired, outlier_qc_summary = apply_pairwise_outlier_qc(
        paired,
        out_dir=args.out_dir,
        enabled=not args.disable_outlier_qc,
        mad_k=args.outlier_mad_k,
        max_fraction=args.outlier_max_fraction,
        min_abs_spec=args.outlier_min_abs,
    )

    metrics = calculate_station_metrics(paired, inventory, min_pairs=MIN_PAIRED_DAYS)
    metrics = repair_metrics_coordinates(metrics, inventory, paired=paired, era_station_daily=era_station_daily)
    valid_counts = metrics.groupby("variable")["pearson_r"].apply(lambda x: int(np.isfinite(x).sum())).to_dict()
    coord_counts = metrics.groupby("variable").apply(
        lambda g: int((
            np.isfinite(pd.to_numeric(g["lat"], errors="coerce")) &
            np.isfinite(pd.to_numeric(g["lon"], errors="coerce")) &
            pd.to_numeric(g["lat"], errors="coerce").between(BRAZIL_EXTENT[2], BRAZIL_EXTENT[3]) &
            pd.to_numeric(g["lon"], errors="coerce").between(BRAZIL_EXTENT[0], BRAZIL_EXTENT[1]) &
            np.isfinite(pd.to_numeric(g["pearson_r"], errors="coerce"))
        ).sum())
    ).to_dict()
    print(f"[INFO] Stations with valid Pearson r by variable: {valid_counts}")
    print(f"[INFO] Stations with valid Pearson r and plot-ready coordinates by variable: {coord_counts}")

    taylor_agg = calculate_taylor_aggregate_metrics(paired, min_pairs=MIN_PAIRED_DAYS)
    states = read_states_shapefile(args.shp_uf)

    save_outputs(metrics, paired, inventory, grid_index, taylor_agg, args.out_dir, daily_time_basis=args.daily_time_basis, fixed_time_shift_hours=args.fixed_time_shift_hours, min_valid_day_fraction=args.min_valid_day_fraction, min_valid_years_variable=args.min_valid_years_variable)
    plot_validation_figure(metrics, taylor_agg, states, args.out_dir, dpi=args.dpi)

    print("[DONE] Station-level ERA5 vs INMET evaluation completed.")
    print(f"[INFO] Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()


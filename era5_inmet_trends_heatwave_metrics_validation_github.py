#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5–INMET evaluation of thermodynamic trends and heatwave metrics, 2002–2024.

Purpose
-------
Compare quality-controlled INMET station observations with ERA5 nearest-grid
diagnostics over their common validation period. Annual thermodynamic trends and
HW, HHW, and DHW frequency, duration, and accumulated intensity are evaluated
using source-specific percentile thresholds and identical persistence logic.

Quality control
---------------
INMET hourly observations are screened with broad physical limits. Daily
metrics require a minimum number of hourly observations; station-years require
a minimum valid-day fraction; retained stations must satisfy minimum baseline
and trend-period coverage requirements.

An optional conservative pairwise residual QC removes isolated
station-variable ERA5–INMET mismatches only when residuals are extreme relative
to each station's own residual distribution and exceed a variable-specific
absolute threshold. Removal is capped per station-variable so persistent
model–observation differences are not filtered out.

Heatwave definitions
--------------------
Thresholds are calculated separately for INMET and ERA5 using the same
validation baseline and a centred 31-day no-leap day-of-year climatology.

HW
    Tmean >= source-specific P90 for at least the minimum event duration.

HHW
    Twbmax >= source-specific P95 for at least the minimum event duration.

DHW
    Tmax >= source-specific P95 and VPDmean >= source-specific P75 for at
    least the minimum event duration.

Main annual intensity is accumulated event-day severity. DHW intensity uses
0.5*z+(Tmax) + 0.5*z+(VPDmean). Mean event-day severity is retained only as an
auxiliary diagnostic.

Interpretation
--------------
ERA5 is evaluated as a grid-scale regional thermodynamic diagnostic rather than
as an exact station-scale reconstruction of individual events.

Usage
-----
python era5_inmet_trends_heatwave_metrics_validation.py \
    --inmet-hourly-dir /path/to/inmet_hourly \
    --era5-daily-dir /path/to/figure_01_outputs/cache/era5_daily \
    --station-metadata-dir /path/to/inmet_normals_1991_2020 \
    --states-shapefile /path/to/brazil_states.shp \
    --output-dir ./outputs/era5_inmet_trend_validation \
    --start-year 2002 \
    --end-year 2024 \
    --baseline-start 2002 \
    --baseline-end 2020 \
    --min-valid-years-threshold 15 \
    --min-valid-years-trend 18 \
    --reuse-cache

A precomputed station-level ERA5 CSV may be supplied with
``--era5-station-cache``. If no reusable cache is available, the daily NetCDF
files are read directly.
"""

import os
import re
import glob
import json
import argparse
import warnings
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.spatial import cKDTree
from scipy.stats import theilslopes, kendalltau, pearsonr, spearmanr

try:
    import geopandas as gpd
except Exception:
    gpd = None


# ============================================================
# Constants and defaults
# ============================================================

UF_TO_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}

UF_UTC_OFFSET_HOURS = {
    "AC": -5, "AM": -4, "RO": -4, "RR": -4, "MT": -4, "MS": -4,
    "AP": -3, "PA": -3, "TO": -3, "MA": -3, "PI": -3, "CE": -3, "RN": -3,
    "PB": -3, "PE": -3, "AL": -3, "SE": -3, "BA": -3, "GO": -3, "DF": -3,
    "MG": -3, "ES": -3, "RJ": -3, "SP": -3, "PR": -3, "SC": -3, "RS": -3,
}

THERMO_VARS = ["Tmax", "Tmin", "Tmean", "Tdmean", "RHmean", "VPDmean", "Twbmax", "WSmean"]
REGIMES = ["HW", "HHW", "DHW"]

# Main heatwave metrics used in the manuscript and validation figures.
# "intensity" is defined as annual accumulated event-day severity, matching
# the primary Figure 01 ERA5 workflow.
METRICS = ["frequency", "duration", "intensity"]

# Auxiliary diagnostic retained in output tables only. It is not used as the
# manuscript's main intensity metric, because it measures average severity
# during persistent event days rather than annual accumulated severity.
AUX_METRICS = ["mean_event_day_severity"]
ALL_HEATWAVE_METRICS = METRICS + AUX_METRICS

def heatwave_metric_columns(include_aux: bool = True) -> List[str]:
    metrics = ALL_HEATWAVE_METRICS if include_aux else METRICS
    return [f"{r}_{m}" for r in REGIMES for m in metrics]

BRAZIL_EXTENT = [-75.5, -32.0, -35.5, 6.5]
PLOT_FONT = 13

# Figure 01-aligned threshold and intensity settings for station/source evaluation.
ROLLING_WINDOW_DAYS = 31
Z_CLIP_MIN = -5.0
Z_CLIP_MAX = 5.0
DHW_WEIGHT_TMAX = 0.5
DHW_WEIGHT_VPD = 0.5


# ============================================================
# Generic helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_name(s) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    repl = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "ä": "a",
        "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o", "õ": "o",
        "ú": "u", "ç": "c", "º": "", "ª": "",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def normalize_station_code(x) -> Optional[str]:
    """Normalize station codes while preserving INMET automatic-station IDs.

    INMET automatic stations are coded as A001, A866, etc. The previous
    version retained only digits, which could merge A001 with 001 and made
    station tracking less transparent. Numeric WMO codes remain numeric.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    # Keep automatic-station codes such as A001 or B803.
    m_auto = re.search(r"\b([A-Z]\d{3})\b", s)
    if m_auto:
        return m_auto.group(1)
    try:
        f = float(s.replace(",", "."))
        if np.isfinite(f) and abs(f - round(f)) < 1e-6:
            return str(int(round(f)))
    except Exception:
        pass
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in ["", "-", "–", "--", "nan", "NaN", "null", "None"]:
        return np.nan
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9eE+\-.]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def apply_physical_qc_to_hourly(out: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative physical-range QC to official INMET hourly values.

    INMET automatic-station CSVs may encode missing or invalid values as large
    negative numbers (e.g., -9999). If these values are not removed before
    aggregation, annual means and trend diagnostics become physically impossible
    and can dominate ERA5-INMET validation statistics. The ranges below are
    intentionally broad for Brazil and are used only to remove impossible values.
    """
    out = out.copy()
    ranges = {
        "Tair_C": (-50.0, 60.0),
        "Td_C": (-60.0, 40.0),
        "RH_pct": (0.0, 100.0),
        "WS_ms": (0.0, 75.0),
    }
    for col, (lo, hi) in ranges.items():
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            vals = vals.where((vals >= lo) & (vals <= hi), np.nan)
            out[col] = vals

    # Dew point cannot physically exceed dry-bulb temperature by more than a very
    # small tolerance. Values above this usually indicate bad humidity records.
    if {"Tair_C", "Td_C"}.issubset(out.columns):
        bad_td = out["Td_C"].notna() & out["Tair_C"].notna() & (out["Td_C"] > out["Tair_C"] + 0.5)
        out.loc[bad_td, "Td_C"] = np.nan
    return out


def apply_physical_qc_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Apply broad physical QC to daily INMET/ERA5 station metrics."""
    df = df.copy()
    ranges = {
        "Tmax": (-50.0, 60.0),
        "Tmin": (-60.0, 50.0),
        "Tmean": (-55.0, 55.0),
        "Tdmean": (-60.0, 40.0),
        "RHmean": (0.0, 100.0),
        "VPDmean": (0.0, 10.0),
        "Twbmax": (-50.0, 45.0),
        "WSmean": (0.0, 75.0),
    }
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[col] = vals.where((vals >= lo) & (vals <= hi), np.nan)
    # Thermodynamic consistency.
    if {"Tmax", "Tmin"}.issubset(df.columns):
        bad = df["Tmax"].notna() & df["Tmin"].notna() & (df["Tmax"] < df["Tmin"])
        df.loc[bad, ["Tmax", "Tmin", "Tmean", "Twbmax", "VPDmean"]] = np.nan
    return df

def find_col(columns, candidates: List[str], required: bool = True) -> Optional[str]:
    norm = {normalize_name(c): c for c in columns}
    cands = [normalize_name(c) for c in candidates]
    for c in cands:
        if c in norm:
            return norm[c]
    for original in columns:
        no = normalize_name(original)
        for c in cands:
            if c and (c in no or no in c):
                return original
    if required:
        raise ValueError(f"Could not find column among {candidates}. Available: {list(columns)}")
    return None


def year_days(year: int) -> int:
    return 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365

def noleap_doy_from_dates(dates) -> np.ndarray:
    """Return no-leap day of year (1-365), matching the Figure 01 workflow.

    February 29 should be removed before this function is used. For leap years,
    dates after February have their ordinary day-of-year reduced by one so that
    all years share a 365-day climatological calendar.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    doy = idx.dayofyear.to_numpy(dtype=np.int16)
    after_feb = (idx.is_leap_year) & ((idx.month > 2))
    doy = doy - after_feb.astype(np.int16)
    return doy.astype(np.int16)


def remove_feb29_daily_df(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Remove February 29 records, matching the no-leap Figure 01 workflow."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    mask = ~((out[date_col].dt.month == 2) & (out[date_col].dt.day == 29))
    return out.loc[mask].copy()


def finite_percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.nanpercentile(values, q)) if values.size else np.nan


def finite_mean_std(values) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan


def window_doys(doy: int, window_days: int = ROLLING_WINDOW_DAYS) -> List[int]:
    """Centered no-leap DOY window with year-end wrap-around."""
    half = window_days // 2
    return [((int(doy) + off - 1) % 365) + 1 for off in range(-half, half + 1)]


def lookup_by_doy(table: Dict[str, Dict[int, float]], key: str, doy_values: np.ndarray) -> np.ndarray:
    """Map a {doy: value} threshold/climatology dictionary to a daily vector."""
    mapper = table.get(key, {})
    return np.asarray([mapper.get(int(d), np.nan) for d in doy_values], dtype=float)


def saturation_vapour_pressure_kpa(temp_c):
    temp_c = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def vpd_from_t_rh_kpa(t_c, rh_pct):
    es = saturation_vapour_pressure_kpa(t_c)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    ea = es * rh / 100.0
    return np.maximum(es - ea, 0.0)


def dewpoint_from_t_rh(t_c, rh_pct):
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1e-3, 100.0)
    t = np.asarray(t_c, dtype=float)
    gamma = np.log(rh / 100.0) + (17.27 * t) / (237.3 + t)
    return (237.3 * gamma) / (17.27 - gamma)


def rh_from_t_td(t_c, td_c):
    es = saturation_vapour_pressure_kpa(t_c)
    ea = saturation_vapour_pressure_kpa(td_c)
    return np.clip(100.0 * ea / es, 0.0, 100.0)


def wetbulb_stull(t_c, rh_pct):
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    t = np.asarray(t_c, dtype=float)
    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw


def theil_sen_decade(years, values) -> Tuple[float, float]:
    y = np.asarray(values, dtype=float)
    x = np.asarray(years, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan
    try:
        slope = theilslopes(y[ok], x[ok])[0] * 10.0
    except Exception:
        slope = np.nan
    try:
        p = kendalltau(x[ok], y[ok], nan_policy="omit").pvalue
    except Exception:
        p = np.nan
    return float(slope), float(p)


def safe_corr(x, y, kind="pearson"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    try:
        if kind == "spearman":
            return float(spearmanr(x[ok], y[ok]).correlation)
        return float(pearsonr(x[ok], y[ok]).statistic)
    except Exception:
        return np.nan


def robust_read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    # Try common INMET separators and decimal formats.
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python", comment=None)
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="latin1", engine="python", comment=None)
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
    raise ValueError(f"Could not read table: {path}")


# ============================================================
# Station metadata from climatological-normal directory
# ============================================================

def find_header_row(path: str, required_terms: List[str], max_rows: int = 20) -> int:
    raw = pd.read_excel(path, header=None, nrows=max_rows)
    req = [normalize_name(t) for t in required_terms]
    for i in range(len(raw)):
        vals = [normalize_name(v) for v in raw.iloc[i].tolist()]
        row = " ".join(vals)
        if all(any(r in v for v in vals) or r in row for r in req):
            return i
    return 0


def find_station_file(normal_dir: str) -> Optional[str]:
    if not normal_dir:
        return None
    candidates = []
    candidates.extend(glob.glob(os.path.join(normal_dir, "Normal-Climatologica-ESTA*.xlsx")))
    candidates.extend(glob.glob(os.path.join(normal_dir, "*ESTA*.xlsx")))
    return sorted(candidates)[0] if candidates else None


def read_station_metadata_from_normals(normal_dir: Optional[str]) -> pd.DataFrame:
    path = find_station_file(normal_dir) if normal_dir else None
    if path is None:
        return pd.DataFrame(columns=["station_id", "station_name", "uf", "lat", "lon", "altitude_m", "region"])
    header = find_header_row(path, ["Código", "Latitude", "Longitude"])
    df = pd.read_excel(path, header=header)
    df.columns = [str(c).strip() for c in df.columns]
    code_col = find_col(df.columns, ["Código", "Codigo", "Cod"], True)
    name_col = find_col(df.columns, ["Nome da Estação", "Nome da Estacao", "Estação", "Estacao"], False)
    uf_col = find_col(df.columns, ["UF"], True)
    lat_col = find_col(df.columns, ["Latitude"], True)
    lon_col = find_col(df.columns, ["Longitude"], True)
    alt_col = find_col(df.columns, ["Altitude"], False)

    out = pd.DataFrame({
        "station_id": df[code_col].map(normalize_station_code),
        "station_name": df[name_col].astype(str).str.strip() if name_col else "",
        "uf": df[uf_col].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0],
        "lat": df[lat_col].map(clean_numeric),
        "lon": df[lon_col].map(clean_numeric),
        "altitude_m": df[alt_col].map(clean_numeric) if alt_col else np.nan,
    })
    out = out.dropna(subset=["station_id", "uf", "lat", "lon"])
    out = out[out["uf"].isin(UF_TO_REGION)].copy()
    out["region"] = out["uf"].map(UF_TO_REGION)
    out["station_id"] = out["station_id"].astype(str)
    out = out.drop_duplicates("station_id")
    print(f"[INFO] Station metadata from normals: {len(out)} stations")
    return out


# ============================================================
# INMET hourly parsing and daily aggregation
# ============================================================

def infer_metadata_from_inmet_file(path: str) -> Dict:
    """Read metadata from the official INMET automatic-station CSV header.

    Official files usually begin with lines such as:
      REGIAO:;S
      UF:;SC
      ESTACAO:;Laguna - Farol de Santa Marta
      CODIGO (WMO):;A866
      LATITUDE:;-28,60444444
      LONGITUDE:;-48,81333333

    This function reads a limited number of initial lines safely, without
    failing when the file is short, and accepts accented/non-accented labels.
    """
    meta = {}
    try:
        with open(path, "r", encoding="latin1", errors="ignore") as f:
            lines = list(islice(f, 25))
    except Exception:
        return meta
    head = "".join(lines)
    # Work line-by-line first; this is more robust than a single regex over all text.
    for line in lines:
        if ";" in line:
            key, val = line.split(";", 1)
        elif ":" in line:
            key, val = line.split(":", 1)
        else:
            continue
        nk = normalize_name(key)
        val = val.strip().strip(";")
        if nk in ["regiao", "regiao_"]:
            meta["region_inmet"] = val
        elif nk == "uf":
            meta["uf"] = val.upper()[:2]
        elif nk in ["estacao", "estacao_"]:
            meta["station_name"] = val
        elif "codigo" in nk and "wmo" in nk:
            meta["station_id"] = normalize_station_code(val)
        elif nk.startswith("latitude"):
            meta["lat"] = clean_numeric(val)
        elif nk.startswith("longitude"):
            meta["lon"] = clean_numeric(val)
        elif nk.startswith("altitude"):
            meta["altitude_m"] = clean_numeric(val)
    # Fallback regexes for non-standard headers.
    patterns = {
        "station_id": [r"C[OÓ]DIGO\s*\(WMO\)\s*[:;]\s*([A-Z0-9]+)", r"CODIGO\s*[:;]\s*([A-Z0-9]+)", r"Cod.*?[:;]\s*([A-Z0-9]+)"],
        "station_name": [r"ESTA[CÇ][AÃ]O\s*[:;]\s*([^;\n\r]+)", r"Estacao\s*[:;]\s*([^;\n\r]+)"],
        "uf": [r"UF\s*[:;]\s*([A-Z]{2})"],
        "lat": [r"LATITUDE\s*[:;]\s*([\-0-9\.,]+)", r"Latitude\s*[:;]\s*([\-0-9\.,]+)"],
        "lon": [r"LONGITUDE\s*[:;]\s*([\-0-9\.,]+)", r"Longitude\s*[:;]\s*([\-0-9\.,]+)"],
    }
    for key, pats in patterns.items():
        if key in meta and meta[key] not in [None, "", np.nan]:
            continue
        for pat in pats:
            m = re.search(pat, head, re.I)
            if m:
                meta[key] = m.group(1).strip()
                break
    if "lat" in meta:
        meta["lat"] = clean_numeric(meta["lat"])
    if "lon" in meta:
        meta["lon"] = clean_numeric(meta["lon"])
    if "station_id" in meta:
        meta["station_id"] = normalize_station_code(meta["station_id"])
    return meta

def find_data_header_for_csv(path: str) -> int:
    """Return the line index of the tabular header in official INMET CSV files."""
    try:
        with open(path, "r", encoding="latin1", errors="ignore") as f:
            lines = list(islice(f, 60))
    except Exception:
        return 0
    for i, line in enumerate(lines):
        n = normalize_name(line)
        # Official automatic-station files usually have: Data;Hora UTC;...
        parts = [normalize_name(p) for p in line.split(";")[:4]]
        if len(parts) >= 2 and parts[0] in ["data", "data_medicao", "date"] and "hora" in parts[1]:
            return i
        if "data_hora" in n or "data_medicao" in n or ("data" in n and "hora" in n and "temperatura" in n):
            return i
    return 0


def read_raw_inmet_file(path: str) -> pd.DataFrame:
    """Read official INMET automatic-station CSV files robustly.

    INMET files have metadata lines before the table, for example:
    REGIAO:;S
    UF:;SC
    ...
    Data;Hora UTC;PRECIPITACAO...;

    Some pandas engines fail on these files because of the metadata header,
    accents, decimal commas, duplicated/empty trailing columns, or irregular
    semicolon counts. This manual parser is intentionally conservative: it
    reads only the table beginning at the Data/Hora UTC header, drops the
    trailing empty column created by the final semicolon, pads/truncates rows to
    the header length, and leaves numeric conversion to clean_numeric().
    """
    suffix = Path(path).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    text = None
    for enc in ["latin1", "cp1252", "utf-8", "utf-8-sig"]:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                text = f.read()
            break
        except Exception:
            text = None
    if text is None:
        raise ValueError(f"Could not read table: {path}")

    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:100]):
        parts = [normalize_name(p) for p in line.split(";")]
        if len(parts) >= 2 and parts[0] in ["data", "data_medicao", "date"] and "hora" in parts[1]:
            header_idx = i
            break
        n = normalize_name(line)
        if "data" in n and "hora" in n and "temperatura" in n:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find INMET tabular header: {path}")

    cols = [c.strip().replace("\ufeff", "") for c in lines[header_idx].split(";")]
    while cols and cols[-1] == "":
        cols.pop()
    if len(cols) < 4:
        raise ValueError(f"INMET header has too few columns: {path}")

    safe_cols = []
    seen = {}
    for j, c in enumerate(cols):
        c = c.strip() or f"unnamed_{j}"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        safe_cols.append(c)

    rows = []
    ncols = len(safe_cols)
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(";")]
        while parts and parts[-1] == "":
            parts.pop()
        if len(parts) < 2:
            continue
        # Keep only data rows whose first field is a date.
        if not re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", parts[0]) and not re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", parts[0]):
            continue
        if len(parts) < ncols:
            parts = parts + [np.nan] * (ncols - len(parts))
        elif len(parts) > ncols:
            parts = parts[:ncols]
        rows.append(parts)

    if not rows:
        raise ValueError(f"No data rows found after INMET header: {path}")

    return pd.DataFrame(rows, columns=safe_cols)


def find_exact_metadata_col(columns, names: List[str]) -> Optional[str]:
    """Find metadata columns only by exact normalized name.

    This avoids false matches such as matching 'estacao' inside
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO' or 'lat' inside 'relativa'.
    """
    norm = {normalize_name(c): c for c in columns}
    for n in names:
        nn = normalize_name(n)
        if nn in norm:
            return norm[nn]
    return None

def standardize_inmet_hourly(df: pd.DataFrame, file_meta: Dict, station_meta_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Station metadata columns may exist in already standardized input.
    # For official INMET CSVs, metadata are in the file header, not in tabular
    # columns. Use exact matching here to avoid false matches with variables like
    # "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO" or "UMIDADE RELATIVA".
    sid_col = find_exact_metadata_col(df.columns, ["station_id", "codigo_wmo", "cod_wmo", "wmo"])
    name_col = find_exact_metadata_col(df.columns, ["station_name", "nome_estacao", "estacao"])
    uf_col = find_exact_metadata_col(df.columns, ["uf", "estado"])
    lat_col = find_exact_metadata_col(df.columns, ["lat", "latitude"])
    lon_col = find_exact_metadata_col(df.columns, ["lon", "longitude"])

    date_col = find_col(df.columns, ["datetime", "data_hora", "datahora", "data_medicao_hora_medicao"], required=False)
    if date_col is None:
        date_col = find_col(df.columns, ["data_medicao", "data", "date"], required=True)
        hour_col = find_col(df.columns, ["hora_medicao", "hora", "hour", "hr"], required=False)
        if hour_col is not None:
            date_raw = df[date_col].astype(str).str.strip()
            hour_raw = df[hour_col].astype(str).str.strip()
            hour_clean = hour_raw.str.extract(r"(\d{1,4})")[0].fillna("0000").str.zfill(4)
            # INMET uses UTC hour like 0000, 1200.
            dt = pd.to_datetime(date_raw + " " + hour_clean.str[:2] + ":" + hour_clean.str[2:], errors="coerce", dayfirst=False)
            bad = dt.isna()
            if bad.any():
                dt2 = pd.to_datetime(date_raw[bad] + " " + hour_clean[bad].str[:2] + ":" + hour_clean[bad].str[2:], errors="coerce", dayfirst=True)
                dt.loc[bad] = dt2
        else:
            dt = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dt = pd.to_datetime(df[date_col], errors="coerce")

    t_col = find_col(df.columns, [
        "Tair_C", "temperatura_do_ar_bulbo_seco_horaria_c", "temperatura_do_ar", "temp_inst", "temp", "tair"
    ], required=False)
    tmax_col = find_col(df.columns, ["Tmax", "temperatura_maxima", "temp_max"], required=False)
    tmin_col = find_col(df.columns, ["Tmin", "temperatura_minima", "temp_min"], required=False)
    td_col = find_col(df.columns, [
        "Td_C", "temperatura_do_ponto_de_orvalho_c", "temperatura_do_ponto_de_orvalho", "ponto_de_orvalho", "torv", "dewpoint"
    ], required=False)
    rh_col = find_col(df.columns, [
        "RH_pct", "umidade_relativa_do_ar_horaria", "umidade_relativa", "ur", "rh"
    ], required=False)
    ws_col = find_col(df.columns, [
        "WS_ms", "vento_velocidade_horaria_m_s", "vento_velocidade", "velocidade_do_vento", "wind_speed", "ventin"
    ], required=False)

    out = pd.DataFrame({"datetime": dt})
    if t_col is not None:
        out["Tair_C"] = df[t_col].map(clean_numeric)
    elif tmax_col is not None and tmin_col is not None:
        out["Tair_C"] = (df[tmax_col].map(clean_numeric) + df[tmin_col].map(clean_numeric)) / 2.0
    else:
        out["Tair_C"] = np.nan

    out["Td_C"] = df[td_col].map(clean_numeric) if td_col is not None else np.nan
    out["RH_pct"] = df[rh_col].map(clean_numeric) if rh_col is not None else np.nan
    out["WS_ms"] = df[ws_col].map(clean_numeric) if ws_col is not None else np.nan

    # Remove official missing-value flags and physically impossible values before
    # deriving humidity variables or aggregating to daily diagnostics.
    out = apply_physical_qc_to_hourly(out)

    # Fill missing RH or Td where possible.
    missing_rh = out["RH_pct"].isna() & out["Td_C"].notna() & out["Tair_C"].notna()
    out.loc[missing_rh, "RH_pct"] = rh_from_t_td(out.loc[missing_rh, "Tair_C"], out.loc[missing_rh, "Td_C"])
    missing_td = out["Td_C"].isna() & out["RH_pct"].notna() & out["Tair_C"].notna()
    out.loc[missing_td, "Td_C"] = dewpoint_from_t_rh(out.loc[missing_td, "Tair_C"], out.loc[missing_td, "RH_pct"])

    # Re-apply QC after derived RH/Td filling.
    out = apply_physical_qc_to_hourly(out)

    # Metadata: prefer file columns, then file header, then station metadata lookup.
    sid = None
    if sid_col is not None:
        vals = df[sid_col].dropna().map(normalize_station_code)
        sid = vals.iloc[0] if len(vals) else None
    if sid is None:
        sid = file_meta.get("station_id")
    if sid is None:
        # Try numeric code from filename.
        sid = normalize_station_code(Path(str(file_meta.get("path", ""))).stem)

    # Use lookup if available.
    meta_row = pd.DataFrame()
    if sid is not None and not station_meta_lookup.empty:
        meta_row = station_meta_lookup[station_meta_lookup["station_id"].astype(str) == str(sid)]

    def scalar_from_col(col, fallback=np.nan):
        if col is None:
            return fallback
        vals = df[col].dropna()
        return vals.iloc[0] if len(vals) else fallback

    station_name = scalar_from_col(name_col, file_meta.get("station_name", ""))
    uf = scalar_from_col(uf_col, file_meta.get("uf", None))
    lat = clean_numeric(scalar_from_col(lat_col, file_meta.get("lat", np.nan)))
    lon = clean_numeric(scalar_from_col(lon_col, file_meta.get("lon", np.nan)))

    if not meta_row.empty:
        station_name = station_name if station_name else meta_row.iloc[0].get("station_name", "")
        uf = uf if isinstance(uf, str) and re.search(r"[A-Z]{2}", uf.upper()) else meta_row.iloc[0].get("uf", None)
        if not np.isfinite(lat):
            lat = float(meta_row.iloc[0].get("lat", np.nan))
        if not np.isfinite(lon):
            lon = float(meta_row.iloc[0].get("lon", np.nan))

    if isinstance(uf, str):
        m = re.search(r"([A-Z]{2})", uf.upper())
        uf = m.group(1) if m else None

    out["station_id"] = str(sid)
    out["station_name"] = str(station_name)
    out["uf"] = uf
    out["lat"] = lat
    out["lon"] = lon
    out = out.dropna(subset=["datetime"])
    return out


def list_inmet_files(inmet_dir: str, start_year: int, end_year: int) -> List[str]:
    """List only official INMET hourly files for the requested period.

    The root INMET directory also contains climatological-normal spreadsheets,
    README files and sometimes 2025 files. These must not be parsed as hourly
    validation data.
    """
    pats = []
    for ext in ["*.csv", "*.CSV", "*.txt", "*.TXT"]:
        pats.append(os.path.join(inmet_dir, "**", ext))
    files = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    selected = []
    for f in sorted(set(files)):
        name = Path(f).name
        if not name.upper().startswith("INMET_"):
            continue
        # Prefer the year directory if present; otherwise use years in filename.
        years = [int(y) for y in re.findall(r"(19\d{2}|20\d{2})", f)]
        if not years:
            continue
        if any(start_year <= y <= end_year for y in years):
            selected.append(f)
    if not selected:
        raise FileNotFoundError(f"No INMET hourly files found for {start_year}-{end_year} in {inmet_dir}")
    return selected

def aggregate_hourly_to_daily(df: pd.DataFrame, min_hourly_per_day: int, local_day: bool) -> pd.DataFrame:
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str)
    if local_day:
        offsets = df["uf"].map(UF_UTC_OFFSET_HOURS).fillna(-3).astype(int)
        df["datetime_local"] = df["datetime"] + pd.to_timedelta(offsets, unit="h")
    else:
        df["datetime_local"] = df["datetime"]
    df["date"] = pd.to_datetime(df["datetime_local"]).dt.floor("D")

    rows = []
    for (sid, date), g in df.groupby(["station_id", "date"], sort=True):
        meta = g.iloc[0]
        t = pd.to_numeric(g["Tair_C"], errors="coerce").to_numpy(float)
        td = pd.to_numeric(g["Td_C"], errors="coerce").to_numpy(float)
        rh = pd.to_numeric(g["RH_pct"], errors="coerce").to_numpy(float)
        ws = pd.to_numeric(g["WS_ms"], errors="coerce").to_numpy(float)
        n_t = int(np.isfinite(t).sum())
        n_hum = int((np.isfinite(t) & np.isfinite(rh)).sum())
        if n_t < min_hourly_per_day:
            continue
        tmean = np.nanmean(t)
        rhmean = np.nanmean(rh) if np.isfinite(rh).sum() >= min_hourly_per_day else np.nan
        tdmean = np.nanmean(td) if np.isfinite(td).sum() >= min_hourly_per_day else np.nan
        if not np.isfinite(tdmean) and np.isfinite(tmean) and np.isfinite(rhmean):
            tdmean = float(dewpoint_from_t_rh(tmean, rhmean))
        vpdmean = np.nan
        twbmax = np.nan
        if np.isfinite(tmean) and np.isfinite(rhmean):
            vpdmean = float(vpd_from_t_rh_kpa(tmean, rhmean))
            # Hourly wet-bulb max if possible, otherwise daily proxy from mean.
            if n_hum >= min_hourly_per_day:
                tw_hourly = wetbulb_stull(t, rh)
                twbmax = float(np.nanmax(tw_hourly))
            else:
                twbmax = float(wetbulb_stull(tmean, rhmean))
        rows.append({
            "station_id": sid,
            "station_name": meta.get("station_name", ""),
            "uf": meta.get("uf", None),
            "region": UF_TO_REGION.get(str(meta.get("uf", "")), np.nan),
            "lat": float(meta.get("lat", np.nan)),
            "lon": float(meta.get("lon", np.nan)),
            "date": pd.Timestamp(date),
            "year": pd.Timestamp(date).year,
            "Tmax": float(np.nanmax(t)),
            "Tmin": float(np.nanmin(t)),
            "Tmean": float(tmean),
            "Tdmean": float(tdmean) if np.isfinite(tdmean) else np.nan,
            "RHmean": float(rhmean) if np.isfinite(rhmean) else np.nan,
            "VPDmean": float(vpdmean) if np.isfinite(vpdmean) else np.nan,
            "Twbmax": float(twbmax) if np.isfinite(twbmax) else np.nan,
            "WSmean": float(np.nanmean(ws)) if np.isfinite(ws).sum() >= min_hourly_per_day else np.nan,
            "n_hourly_T": n_t,
            "n_hourly_humidity": n_hum,
        })
    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily = apply_physical_qc_to_daily(daily)
        # Require core thermodynamic variables after QC. Humidity-dependent metrics
        # remain NaN when humidity observations are insufficient.
        daily = daily.dropna(subset=["Tmax", "Tmin", "Tmean"])
    return daily


def build_inmet_daily_cache(args, station_meta_lookup: pd.DataFrame) -> pd.DataFrame:
    out_cache = os.path.join(args.out_dir, f"CACHE_INMET_daily_station_metrics_{args.start_year}_{args.end_year}.csv")
    if args.reuse_cache and os.path.exists(out_cache):
        print(f"[INFO] Reusing INMET daily cache: {out_cache}")
        df = pd.read_csv(out_cache, parse_dates=["date"])
        df["station_id"] = df["station_id"].astype(str)
        df = apply_physical_qc_to_daily(df)
        # Cache-safety: reject old caches contaminated by INMET missing-value flags.
        core = [c for c in ["Tmax", "Tmin", "Tmean", "RHmean", "VPDmean", "Twbmax"] if c in df.columns]
        bad_cache = False
        for c in core:
            vals = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(vals).any():
                if c in ["Tmax", "Tmin", "Tmean", "Twbmax"] and (np.nanpercentile(np.abs(vals), 99) > 100):
                    bad_cache = True
                if c == "VPDmean" and np.nanpercentile(vals, 99) > 10:
                    bad_cache = True
        if not bad_cache and len(df) > 0:
            return df
        print("[WARN] Existing INMET daily cache failed physical sanity checks; rebuilding cache.")

    files = list_inmet_files(args.inmet_hourly_dir, args.start_year, args.end_year)
    print(f"[INFO] INMET files found: {len(files)}")
    daily_frames = []
    for i, f in enumerate(files, 1):
        try:
            print(f"[INFO] Reading INMET file {i}/{len(files)}: {Path(f).name}")
            meta = infer_metadata_from_inmet_file(f)
            meta["path"] = f
            raw = read_raw_inmet_file(f)
            hourly = standardize_inmet_hourly(raw, meta, station_meta_lookup)
            hourly = hourly[(hourly["datetime"].dt.year >= args.start_year) & (hourly["datetime"].dt.year <= args.end_year)]
            hourly = hourly.dropna(subset=["lat", "lon"])
            if hourly.empty:
                continue
            daily = aggregate_hourly_to_daily(hourly, args.min_hourly_per_day, local_day=not args.use_utc_day)
            if not daily.empty:
                daily_frames.append(daily)
        except Exception as exc:
            print(f"[WARN] Skipping file because parsing failed: {f} | {exc}")

    if not daily_frames:
        raise RuntimeError("No INMET daily records could be built. Check input format/paths.")
    daily = pd.concat(daily_frames, ignore_index=True)
    daily = daily.drop_duplicates(["station_id", "date"], keep="last")
    daily = daily[(daily["year"] >= args.start_year) & (daily["year"] <= args.end_year)].copy()
    daily.to_csv(out_cache, index=False)
    print(f"[OK] Saved INMET daily cache: {out_cache}")
    return daily


# ============================================================
# ERA5 station extraction from preprocessed daily files
# ============================================================

def standardize_lat_lon_time(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for t in ["time", "valid_time", "datetime", "date"]:
        if t in ds.coords or t in ds.dims:
            if t != "time":
                rename[t] = "time"
            break
    for la in ["latitude", "lat", "LAT", "Latitude", "y"]:
        if la in ds.coords or la in ds.dims:
            if la != "lat":
                rename[la] = "lat"
            break
    for lo in ["longitude", "lon", "LON", "Longitude", "x"]:
        if lo in ds.coords or lo in ds.dims:
            if lo != "lon":
                rename[lo] = "lon"
            break
    if rename:
        ds = ds.rename(rename)
    if "time" not in ds.coords or "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(f"Dataset missing required coordinates. Coords: {list(ds.coords)}")
    ds["time"] = pd.to_datetime(ds["time"].values)
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds


def find_variable_xr(ds: xr.Dataset, candidates: List[str], required: bool = True) -> Optional[str]:
    lower = {v.lower(): v for v in ds.data_vars}
    norm = {normalize_name(v): v for v in ds.data_vars}
    for c in candidates:
        if c in ds.data_vars:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
        nc = normalize_name(c)
        if nc in norm:
            return norm[nc]
    for c in candidates:
        nc = normalize_name(c)
        for nv, original in norm.items():
            if nc and (nc in nv or nv in nc):
                return original
    if required:
        raise ValueError(f"Could not find variable among {candidates}. Available: {list(ds.data_vars)}")
    return None


def reduce_to_time_lat_lon(da: xr.DataArray) -> xr.DataArray:
    keep = {"time", "lat", "lon"}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.isel({dim: 0}) if da.sizes[dim] == 1 else da.mean(dim=dim, skipna=True)
    return da


def select_xr(ds: xr.Dataset, candidates: List[str], required=True) -> xr.DataArray:
    name = find_variable_xr(ds, candidates, required=required)
    return reduce_to_time_lat_lon(ds[name])


def find_era5_daily_files(daily_dir: str, start_year: int, end_year: int) -> List[str]:
    files = []
    for pat in ["ERA5_daily_Brazil_*.nc", "*daily*Brazil*.nc"]:
        files.extend(glob.glob(os.path.join(daily_dir, pat)))
    selected = []
    for f in sorted(set(files)):
        m = re.search(r"(19|20)\d{2}", Path(f).name)
        if m and start_year <= int(m.group(0)) <= end_year:
            selected.append(f)
    if not selected:
        raise FileNotFoundError(f"No ERA5 daily files found in {daily_dir}")
    return selected


def enforce_common_validation_period(df: pd.DataFrame, args, source_label: str) -> pd.DataFrame:
    """Return only records inside the common ERA5-INMET validation period.

    This guard prevents accidental use of 1990-2024 ERA5 cache content when the
    observational validation is defined over 2002-2024. The main manuscript may
    analyze ERA5 over 1990-2024, but this validation compares ERA5 and INMET only
    over the common period requested by --start_year and --end_year.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["year"] = out["date"].dt.year
    elif "year" not in out.columns:
        raise ValueError(f"{source_label} table must contain either 'date' or 'year'.")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    before = len(out)
    out = out[(out["year"] >= int(args.start_year)) & (out["year"] <= int(args.end_year))].copy()
    after = len(out)
    if before != after:
        print(f"[INFO] {source_label}: retained {after}/{before} rows inside common validation period {args.start_year}-{args.end_year}.")
    else:
        print(f"[INFO] {source_label}: all {after} rows are inside common validation period {args.start_year}-{args.end_year}.")
    return out


def get_station_grid_indices(ds: xr.Dataset, stations: pd.DataFrame) -> pd.DataFrame:
    lats = ds["lat"].values
    lons = ds["lon"].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    rows = []
    for _, st in stations.iterrows():
        dist, flat_idx = tree.query([st["lat"], st["lon"]])
        iy, ix = np.unravel_index(flat_idx, lat2d.shape)
        rows.append({
            "station_id": str(st["station_id"]),
            "station_name": st.get("station_name", ""),
            "uf": st.get("uf", None),
            "region": st.get("region", UF_TO_REGION.get(st.get("uf", ""), None)),
            "lat": float(st["lat"]),
            "lon": float(st["lon"]),
            "iy": int(iy),
            "ix": int(ix),
            "era5_lat": float(lats[iy]),
            "era5_lon": float(lons[ix]),
            "era5_grid_distance_deg": float(dist),
        })
    return pd.DataFrame(rows)


def build_grid_index_from_era5_cache(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Build a metadata-only grid-index table when ERA5 station values are read from CSV cache."""
    st = stations.copy()
    st["station_id"] = st["station_id"].astype(str)

    if df is None or df.empty:
        gi = st[["station_id", "station_name", "uf", "region", "lat", "lon"]].drop_duplicates("station_id").copy()
    else:
        df = df.copy()
        df["station_id"] = df["station_id"].astype(str)
        preferred = ["station_id", "station_name", "uf", "region", "lat", "lon",
                     "iy", "ix", "era5_lat", "era5_lon", "era5_grid_distance_deg"]
        avail = [c for c in preferred if c in df.columns]
        gi = df[avail].drop_duplicates("station_id").copy()

        # Fill missing station metadata from selected-station table.
        for c in ["station_name", "uf", "region", "lat", "lon"]:
            fill = st[["station_id", c]].drop_duplicates("station_id").rename(columns={c: f"{c}_selected"})
            gi = gi.merge(fill, on="station_id", how="outer")
            if c not in gi.columns:
                gi[c] = gi[f"{c}_selected"]
            else:
                gi[c] = gi[c].where(gi[c].notna(), gi[f"{c}_selected"])
            gi = gi.drop(columns=[f"{c}_selected"], errors="ignore")

    for c in ["iy", "ix", "era5_lat", "era5_lon", "era5_grid_distance_deg"]:
        if c not in gi.columns:
            gi[c] = np.nan
    return gi[["station_id", "station_name", "uf", "region", "lat", "lon",
               "iy", "ix", "era5_lat", "era5_lon", "era5_grid_distance_deg"]]

def build_era5_daily_station_cache(args, stations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read/reuse ERA5 daily station values for the common evaluation period.

    Default behaviour is now CACHE-FIRST and NETCDF-SAFE. The function tries to
    use a precomputed CSV station cache and will not open NetCDF files unless
    --allow_netcdf_extract is explicitly supplied. This prevents the known
    xarray/netCDF4 GLIBCXX error in the current runtime while preserving the strict
    Figure 01-aligned heatwave metric calculations downstream.
    """
    period_cache = os.path.join(args.out_dir, f"CACHE_ERA5_daily_station_metrics_{args.start_year}_{args.end_year}.csv")
    cache_grid = os.path.join(args.out_dir, "CACHE_ERA5_station_grid_index_trend_validation.csv")

    cache_candidates = find_existing_era5_station_cache(args)
    print("[INFO] ERA5 station-cache candidates checked:")
    for cand in cache_candidates[:20]:
        print(f"       {cand} {'[FOUND]' if os.path.exists(cand) else '[missing]'}")
    if len(cache_candidates) > 20:
        print(f"       ... {len(cache_candidates) - 20} additional candidate path(s) omitted from log")

    cache_to_read = None
    if args.reuse_cache:
        for cand in cache_candidates:
            if cand and os.path.exists(cand):
                cache_to_read = cand
                break

    if cache_to_read is not None:
        print(f"[INFO] Reusing ERA5 daily station cache: {cache_to_read}")
        print("[INFO] Using the supplied ERA5 station cache; NetCDF extraction is not required.")
        df = pd.read_csv(cache_to_read, parse_dates=["date"])
        df["station_id"] = df["station_id"].astype(str)
        df = enforce_common_validation_period(df, args, "ERA5 station cache")

        if "VPDmean" in df.columns:
            vpd_vals = pd.to_numeric(df["VPDmean"], errors="coerce")
            if np.isfinite(vpd_vals).any() and np.nanmedian(vpd_vals) > 5.0:
                print("[INFO] ERA5 station cache VPDmean appears to be in hPa; converting to kPa before QC.")
                df["VPDmean"] = vpd_vals / 10.0

        df = apply_physical_qc_to_daily(df)

        keep_ids = set(stations["station_id"].astype(str))
        before = len(df)
        df = df[df["station_id"].astype(str).isin(keep_ids)].copy()
        print(f"[INFO] ERA5 station cache selected-station filter: retained {len(df)}/{before} rows for {len(keep_ids)} selected stations.")
        if df.empty:
            raise RuntimeError("ERA5 station cache was found, but none of its station_id values match the selected INMET stations.")

        gi = None
        # Prefer an existing grid-index table next to the cache, then the output directory.
        cache_dir = os.path.dirname(cache_to_read)
        candidate_gi = [
            os.path.join(cache_dir, "CACHE_ERA5_station_grid_index_trend_validation.csv"),
            os.path.join(cache_dir, "Supplementary_Table_ERA5_INMET_nearest_grid_index_trend_validation.csv"),
            cache_grid,
        ]
        for gip in candidate_gi:
            if os.path.exists(gip):
                try:
                    gi = pd.read_csv(gip)
                    gi["station_id"] = gi["station_id"].astype(str)
                    gi = gi[gi["station_id"].isin(keep_ids)].copy()
                    print(f"[INFO] Reusing ERA5 station grid-index table: {gip}")
                    break
                except Exception as exc:
                    print(f"[WARN] Could not read grid-index table {gip}: {exc}")
                    gi = None

        if gi is None or gi.empty:
            gi = build_grid_index_from_era5_cache(df, stations)
            print("[INFO] Built metadata-only ERA5 station grid-index table from cache and selected station metadata.")

        gi.to_csv(cache_grid, index=False)
        print(f"[OK] Saved ERA5 grid-index table: {cache_grid}")

        if os.path.abspath(cache_to_read) != os.path.abspath(period_cache):
            df.to_csv(period_cache, index=False)
            print(f"[OK] Saved period-specific ERA5 cache: {period_cache}")
        return df, gi

    print(
        "[INFO] No reusable ERA5 station cache found; "
        "extracting station values from daily NetCDF files."
    )

    files = find_era5_daily_files(args.era5_daily_dir, args.start_year, args.end_year)
    print(f"[INFO] ERA5 daily files found for common validation period {args.start_year}-{args.end_year}: {len(files)}")
    grid_index = None
    out_frames = []

    for i, f in enumerate(files, 1):
        print(f"[INFO] ERA5 daily file {i}/{len(files)}: {Path(f).name}")
        ds = xr.open_dataset(f)
        ds = standardize_lat_lon_time(ds)
        ds = ds.sel(time=slice(f"{args.start_year}-01-01", f"{args.end_year}-12-31"))
        if ds.sizes.get("time", 0) == 0:
            ds.close()
            continue
        if grid_index is None:
            grid_index = get_station_grid_indices(ds, stations)
            grid_index.to_csv(cache_grid, index=False)
            print(f"[OK] Saved ERA5 grid-index cache: {cache_grid}")

        sd = "station"
        iy = xr.DataArray(grid_index["iy"].values, dims=sd)
        ix = xr.DataArray(grid_index["ix"].values, dims=sd)

        tmean = select_xr(ds, ["Tmean", "tmean", "T2mean", "t2m_mean"], True).isel(lat=iy, lon=ix).values.astype(np.float32)
        tmax = select_xr(ds, ["Tmax", "tmax", "T2max", "t2m_max"], True).isel(lat=iy, lon=ix).values.astype(np.float32)
        tmin = select_xr(ds, ["Tmin", "tmin", "T2min", "t2m_min"], True).isel(lat=iy, lon=ix).values.astype(np.float32)
        rh = select_xr(ds, ["RHmean", "rhmean", "RH", "relative_humidity"], True).isel(lat=iy, lon=ix).values.astype(np.float32)

        vpd_name = find_variable_xr(ds, ["VPDmean", "vpdmean", "VPD", "vpd"], required=False)
        if vpd_name is not None:
            vpd = reduce_to_time_lat_lon(ds[vpd_name]).isel(lat=iy, lon=ix).values.astype(np.float32)
            if np.isfinite(vpd).any() and np.nanmedian(vpd) > 5.0:
                vpd = vpd / 10.0
        else:
            vpd = vpd_from_t_rh_kpa(tmean, rh).astype(np.float32)

        twb_name = find_variable_xr(ds, ["Twbmax", "twbmax", "Twbmean", "twbmean", "Twb", "wetbulb"], required=False)
        if twb_name is not None:
            twb = reduce_to_time_lat_lon(ds[twb_name]).isel(lat=iy, lon=ix).values.astype(np.float32)
        else:
            twb = wetbulb_stull(tmean, rh).astype(np.float32)

        td_name = find_variable_xr(ds, ["Tdmean", "tdmean", "D2mean", "d2m_mean", "dewpoint_mean"], required=False)
        if td_name is not None:
            td = reduce_to_time_lat_lon(ds[td_name]).isel(lat=iy, lon=ix).values.astype(np.float32)
            if np.nanmedian(td) > 100:
                td = td - 273.15
        else:
            td = dewpoint_from_t_rh(tmean, rh).astype(np.float32)

        ws_name = find_variable_xr(ds, ["WSmean", "WS10mean", "ws10mean", "wind_speed", "WS10"], required=False)
        if ws_name is not None:
            ws = reduce_to_time_lat_lon(ds[ws_name]).isel(lat=iy, lon=ix).values.astype(np.float32)
        else:
            ws = np.full_like(tmean, np.nan, dtype=np.float32)

        dates = pd.to_datetime(ds["time"].values)
        station_ids = grid_index["station_id"].astype(str).values
        year = pd.DatetimeIndex(dates).year
        base = pd.MultiIndex.from_product([dates, station_ids], names=["date", "station_id"]).to_frame(index=False)
        base["year"] = np.repeat(year, len(station_ids))
        for name, arr in {
            "Tmax": tmax, "Tmin": tmin, "Tmean": tmean, "Tdmean": td,
            "RHmean": rh, "VPDmean": vpd, "Twbmax": twb, "WSmean": ws
        }.items():
            base[name] = arr.reshape(-1)
        meta_cols = ["station_id", "station_name", "uf", "region", "lat", "lon"]
        base = base.merge(grid_index[meta_cols], on="station_id", how="left")
        out_frames.append(base)
        ds.close()

    if not out_frames:
        raise RuntimeError("No ERA5 station daily data was extracted.")
    out = pd.concat(out_frames, ignore_index=True)
    out = enforce_common_validation_period(out, args, "ERA5 station extraction")
    out = apply_physical_qc_to_daily(out)
    out.to_csv(period_cache, index=False)
    print(f"[OK] Saved ERA5 daily station cache for common validation period: {period_cache}")
    return out, grid_index


# ============================================================
# Conservative pairwise ERA5-INMET outlier QC
# ============================================================

def parse_pairwise_outlier_min_abs(spec: str) -> Dict[str, float]:
    """Parse variable-specific absolute residual thresholds for pairwise QC.

    Example:
        Tmax=5,Tmin=5,Tmean=5,Tdmean=5,RHmean=20,VPDmean=1.5,Twbmax=5,WSmean=3

    Threshold units follow the daily tables:
        temperature and wet-bulb variables: deg C
        RHmean: %
        VPDmean: kPa
        WSmean: m s-1

    A few aliases are accepted for convenience.
    """
    defaults = {
        "Tmax": 5.0,
        "Tmin": 5.0,
        "Tmean": 5.0,
        "Tdmean": 5.0,
        "Twbmax": 5.0,
        "RHmean": 20.0,
        "VPDmean": 1.5,
        "WSmean": 3.0,
    }
    aliases = {
        "temperature": ["Tmax", "Tmin", "Tmean", "Tdmean", "Twbmax"],
        "relative_humidity": ["RHmean"],
        "wind_speed": ["WSmean"],
        "vpd": ["VPDmean"],
        "rh": ["RHmean"],
        "ws": ["WSmean"],
    }
    if spec is None or str(spec).strip() == "":
        return defaults

    out = defaults.copy()
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            print(f"[WARN] Ignoring malformed --outlier_min_abs entry: {item}")
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        try:
            value = float(val)
        except Exception:
            print(f"[WARN] Ignoring non-numeric --outlier_min_abs value for {key}: {val}")
            continue

        if key in out:
            out[key] = value
        elif key in aliases:
            for v in aliases[key]:
                out[v] = value
        else:
            print(f"[WARN] Ignoring --outlier_min_abs for unknown variable/alias: {key}")
    return out


def _robust_sigma_from_residuals(diff: pd.Series) -> Tuple[float, float, float]:
    """Return median residual, MAD and robust sigma with conservative fallbacks."""
    arr = pd.to_numeric(diff, errors="coerce").astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return np.nan, np.nan, np.nan
    med = float(np.nanmedian(arr))
    abs_dev = np.abs(arr - med)
    mad = float(np.nanmedian(abs_dev))
    robust_sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nan
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        q25, q75 = np.nanpercentile(arr, [25, 75])
        iqr = q75 - q25
        robust_sigma = float(iqr / 1.349) if np.isfinite(iqr) and iqr > 0 else np.nan
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        sd = float(np.nanstd(arr, ddof=1))
        robust_sigma = sd if np.isfinite(sd) and sd > 0 else np.nan
    return med, mad, robust_sigma


def apply_pairwise_outlier_qc_to_daily_tables(
    inmet_daily: pd.DataFrame,
    era5_daily: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply conservative paired residual outlier QC to daily INMET and ERA5 tables.

    This QC is applied after station selection and common-period enforcement, and
    before percentile thresholds, event detection and annual trend metrics are
    calculated. It compares ERA5 and INMET for the same station-date and removes
    only isolated station-variable pairs whose ERA5-INMET residual is extreme
    relative to that station's own residual distribution AND exceeds a
    variable-specific absolute minimum threshold.

    Removal is variable-specific: if Tmax is flagged for a station-date, only
    Tmax is set to NaN in both INMET and ERA5 for that station-date. Other daily
    variables are retained. This avoids deleting entire heatwave days because of
    one bad variable while still preventing obvious pairwise mismatches from
    dominating thresholds, annual metrics, and trend validation.

    The procedure is deliberately conservative and capped by
    --outlier_max_fraction per station-variable so that systematic ERA5 biases
    or persistent model-observation differences are not removed.
    """
    enabled = not getattr(args, "disable_outlier_qc", False)
    if not enabled:
        print("[INFO] Pairwise residual outlier QC enabled: False")
        return inmet_daily.copy(), era5_daily.copy(), pd.DataFrame()

    mad_k = float(getattr(args, "outlier_mad_k", 6.0))
    max_fraction = max(0.0, min(float(getattr(args, "outlier_max_fraction", 0.03)), 1.0))
    min_pairs = int(getattr(args, "outlier_min_pairs", 30))
    min_abs = parse_pairwise_outlier_min_abs(getattr(args, "outlier_min_abs", ""))

    print("[INFO] Pairwise residual outlier QC enabled: True")
    print(f"[INFO] Pairwise residual outlier QC settings: MAD k={mad_k:g}; max fraction={max_fraction:.3f}; min pairs={min_pairs}; min abs={min_abs}")

    obs = inmet_daily.copy()
    mod = era5_daily.copy()
    for df in [obs, mod]:
        df["station_id"] = df["station_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "uf" in df.columns:
            df["uf"] = df["uf"].astype(str).str.upper().str.extract(r"([A-Z]{2})")[0]

    obs = obs.set_index(["station_id", "date"], drop=False).sort_index()
    mod = mod.set_index(["station_id", "date"], drop=False).sort_index()
    common_index = obs.index.intersection(mod.index)
    if len(common_index) == 0:
        print("[WARN] Pairwise residual outlier QC skipped: no common station-date pairs.")
        return obs.reset_index(drop=True), mod.reset_index(drop=True), pd.DataFrame()

    summary_rows = []
    total_removed = 0

    station_ids = pd.Index(common_index.get_level_values("station_id")).unique()
    for sid in station_ids:
        idx_sid = common_index[common_index.get_level_values("station_id") == sid]
        if len(idx_sid) == 0:
            continue
        uf_val = np.nan
        try:
            uf_val = obs.loc[idx_sid, "uf"].dropna().iloc[0]
        except Exception:
            pass

        for var in THERMO_VARS:
            if var not in obs.columns or var not in mod.columns:
                continue

            x = pd.to_numeric(obs.loc[idx_sid, var], errors="coerce")
            y = pd.to_numeric(mod.loc[idx_sid, var], errors="coerce")
            ok = x.notna() & y.notna()
            n_valid = int(ok.sum())
            n_removed = 0
            med = mad = robust_sigma = threshold = np.nan

            if n_valid >= min_pairs:
                valid_index = x.index[ok]
                diff = (y.loc[valid_index] - x.loc[valid_index]).astype(float)
                med, mad, robust_sigma = _robust_sigma_from_residuals(diff)
                abs_limit = float(min_abs.get(var, np.nan))

                max_remove = int(np.floor(max_fraction * n_valid))
                if max_fraction > 0 and n_valid > 0 and max_remove < 1:
                    max_remove = 1

                if np.isfinite(robust_sigma) and robust_sigma > 0 and np.isfinite(abs_limit) and max_remove > 0:
                    threshold = max(abs_limit, mad_k * robust_sigma)
                    abs_dev = np.abs(diff - med)
                    eligible = abs_dev > threshold
                    eligible_idx = list(valid_index[eligible.to_numpy()])
                    if eligible_idx:
                        # Remove the strongest residual outliers first, capped by max_fraction.
                        eligible_sorted = sorted(
                            eligible_idx,
                            key=lambda ii: float(abs_dev.loc[ii]) if np.isfinite(abs_dev.loc[ii]) else -np.inf,
                            reverse=True,
                        )
                        remove_idx = eligible_sorted[:max_remove]
                        obs.loc[remove_idx, var] = np.nan
                        mod.loc[remove_idx, var] = np.nan
                        n_removed = len(remove_idx)
                        total_removed += n_removed

            summary_rows.append({
                "station_id": sid,
                "uf": uf_val,
                "variable": var,
                "n_valid_before_outlier_qc": n_valid,
                "n_removed_outlier_pairs": n_removed,
                "fraction_removed": (n_removed / n_valid) if n_valid else np.nan,
                "median_residual_era5_minus_inmet": med,
                "mad_residual": mad,
                "robust_sigma_residual": robust_sigma,
                "threshold_used": threshold,
                "mad_k": mad_k,
                "max_fraction_allowed": max_fraction,
                "min_abs_threshold": min_abs.get(var, np.nan),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        by_var = (
            summary.groupby("variable", as_index=False)
            .agg(
                n_valid_before_outlier_qc=("n_valid_before_outlier_qc", "sum"),
                n_removed_outlier_pairs=("n_removed_outlier_pairs", "sum"),
            )
        )
        by_var["fraction_removed"] = by_var["n_removed_outlier_pairs"] / by_var["n_valid_before_outlier_qc"].replace(0, np.nan)
        print("[INFO] Pairwise residual outlier QC removed pairs by variable:",
              dict(zip(by_var["variable"], by_var["n_removed_outlier_pairs"].astype(int))))
        print("[INFO] Pairwise residual outlier QC removed fractions by variable:",
              {r["variable"]: round(float(r["fraction_removed"]), 5) for _, r in by_var.iterrows()})
        print(f"[INFO] Pairwise residual outlier QC total removed station-variable pairs: {total_removed}")

        out_dir = getattr(args, "out_dir", None)
        if out_dir:
            ensure_dir(out_dir)
            period_tag = f"{args.start_year}_{args.end_year}"
            summary_path = os.path.join(out_dir, f"Supplementary_Table_ERA5_INMET_pairwise_outlier_QC_summary_{period_tag}.csv")
            byvar_path = os.path.join(out_dir, f"Supplementary_Table_ERA5_INMET_pairwise_outlier_QC_by_variable_{period_tag}.csv")
            summary.to_csv(summary_path, index=False)
            by_var.to_csv(byvar_path, index=False)
            print(f"[OK] Saved pairwise residual outlier QC summary: {summary_path}")
            print(f"[OK] Saved pairwise residual outlier QC by-variable summary: {byvar_path}")

    return obs.reset_index(drop=True), mod.reset_index(drop=True), summary


# ============================================================
# Coverage, thresholds, events and annual metrics
# ============================================================

def yearly_coverage(daily: pd.DataFrame, min_valid_day_fraction: float) -> pd.DataFrame:
    rows = []
    for (sid, year), g in daily.groupby(["station_id", "year"]):
        n = int(g["Tmax"].notna().sum())
        valid = n >= int(np.ceil(min_valid_day_fraction * year_days(int(year))))
        meta = g.iloc[0]
        rows.append({
            "station_id": sid,
            "year": int(year),
            "n_valid_days": n,
            "valid_year": bool(valid),
            "station_name": meta.get("station_name", ""),
            "uf": meta.get("uf", None),
            "region": meta.get("region", None),
            "lat": meta.get("lat", np.nan),
            "lon": meta.get("lon", np.nan),
        })
    return pd.DataFrame(rows)


def station_selection(daily_inmet: pd.DataFrame, args) -> pd.DataFrame:
    cov = yearly_coverage(daily_inmet, args.min_valid_day_fraction)
    baseline = cov[(cov["year"] >= args.baseline_start) & (cov["year"] <= args.baseline_end)]
    full = cov[(cov["year"] >= args.start_year) & (cov["year"] <= args.end_year)]
    sel = (
        cov.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            uf=("uf", "first"),
            region=("region", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
    )
    b = baseline.groupby("station_id")["valid_year"].sum().rename("n_valid_baseline_years")
    f = full.groupby("station_id")["valid_year"].sum().rename("n_valid_full_years")
    sel = sel.merge(b, on="station_id", how="left").merge(f, on="station_id", how="left")
    sel[["n_valid_baseline_years", "n_valid_full_years"]] = sel[["n_valid_baseline_years", "n_valid_full_years"]].fillna(0).astype(int)
    sel["selected"] = (
        (sel["n_valid_baseline_years"] >= args.min_valid_years_threshold)
        & (sel["n_valid_full_years"] >= args.min_valid_years_trend)
        & np.isfinite(sel["lat"]) & np.isfinite(sel["lon"])
    )
    return sel


def compute_thresholds_for_station(g: pd.DataFrame, baseline_start: int, baseline_end: int) -> Dict:
    """Compute Figure 01-aligned station/source-specific DOY thresholds.

    Unlike a single all-season percentile, the primary Figure 01 workflow uses
    day-of-year-specific thresholds based on a 31-day centered moving window and
    a no-leap calendar. This function reproduces that logic for each INMET
    station and its nearest ERA5 grid cell over the common evaluation baseline.
    """
    gg = remove_feb29_daily_df(g, "date")
    gg["year"] = gg["date"].dt.year
    gg["doy_noleap"] = noleap_doy_from_dates(gg["date"])
    b = gg[(gg["year"] >= baseline_start) & (gg["year"] <= baseline_end)].copy()

    out = {
        "threshold_mode": "doy_31day_noleap",
        "rolling_window_days": ROLLING_WINDOW_DAYS,
        "baseline_start": int(baseline_start),
        "baseline_end": int(baseline_end),
    }

    keys = [
        "HW_Tmean_P90", "HHW_Twbmax_P95", "DHW_Tmax_P95", "DHW_VPDmean_P75",
        "Tmean_mean", "Tmean_std", "Twbmax_mean", "Twbmax_std",
        "Tmax_mean", "Tmax_std", "VPDmean_mean", "VPDmean_std",
    ]
    for key in keys:
        out[key] = {}

    for doy in range(1, 366):
        win = window_doys(doy, ROLLING_WINDOW_DAYS)
        sub = b[b["doy_noleap"].isin(win)]
        out["HW_Tmean_P90"][doy] = finite_percentile(sub["Tmean"], 90)
        out["HHW_Twbmax_P95"][doy] = finite_percentile(sub["Twbmax"], 95)
        out["DHW_Tmax_P95"][doy] = finite_percentile(sub["Tmax"], 95)
        out["DHW_VPDmean_P75"][doy] = finite_percentile(sub["VPDmean"], 75)

        for var, prefix in [("Tmean", "Tmean"), ("Twbmax", "Twbmax"),
                            ("Tmax", "Tmax"), ("VPDmean", "VPDmean")]:
            mn, sd = finite_mean_std(sub[var])
            out[f"{prefix}_mean"][doy] = mn
            out[f"{prefix}_std"][doy] = sd

    # Compact scalar summaries for the supplementary thresholds table only.
    # Event detection uses the full DOY-resolved dictionaries above.
    for key in ["HW_Tmean_P90", "HHW_Twbmax_P95", "DHW_Tmax_P95", "DHW_VPDmean_P75",
                "Tmean_mean", "Tmean_std", "Twbmax_mean", "Twbmax_std",
                "Tmax_mean", "Tmax_std", "VPDmean_mean", "VPDmean_std"]:
        vals = np.asarray(list(out[key].values()), dtype=float)
        out[f"{key}_annual_median"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan

    return out


def mark_persistent_events(mask: np.ndarray, min_duration: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    mask = np.asarray(mask, bool)
    persistent = np.zeros(mask.shape, dtype=bool)
    events = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_duration:
            persistent[i:j] = True
            events.append((i, j))
        i = j
    return persistent, events


def annual_metrics_for_station(
    g: pd.DataFrame,
    thresholds: Dict,
    min_event_duration: int,
    min_valid_day_fraction: float = 0.70,
) -> pd.DataFrame:
    """Compute annual heatwave metrics using Figure 01-aligned event logic.

    Main manuscript intensity is annual accumulated event-day severity:

        annual intensity = sum(daily severity during persistent event days)

    This function now matches the primary Figure 01 workflow in the key
    methodological components:
      * no-leap calendar, with February 29 removed;
      * day-of-year-specific 31-day moving-window thresholds;
      * HW condition: Tmean >= local/source-specific P90;
      * HHW condition: Twbmax >= local/source-specific P95;
      * DHW condition: Tmax >= local/source-specific P95 AND VPDmean >= P75;
      * event persistence: at least min_event_duration consecutive days;
      * frequency: counted by event start year;
      * duration and intensity: accumulated by calendar year for persistent
        event days;
      * DHW severity: 0.5*z+(Tmax) + 0.5*z+(VPDmean), clipped to [-5, +5]
        before retaining positive anomalies.

    A complementary metric, mean_event_day_severity, is also saved. It is not
    the main intensity metric used for DHW-HHW; it diagnoses average event-day
    magnitude only.
    """
    g = g.sort_values("date").copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = remove_feb29_daily_df(g, "date")

    dates = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    g = g.set_index("date").reindex(dates)
    g.index.name = "date"
    g["year"] = g.index.year
    g["doy_noleap"] = noleap_doy_from_dates(g.index)

    tmean = pd.to_numeric(g["Tmean"], errors="coerce").to_numpy(float)
    tmax = pd.to_numeric(g["Tmax"], errors="coerce").to_numpy(float)
    twbmax = pd.to_numeric(g["Twbmax"], errors="coerce").to_numpy(float)
    vpd = pd.to_numeric(g["VPDmean"], errors="coerce").to_numpy(float)
    doys = g["doy_noleap"].to_numpy(np.int16)

    hw_thr = lookup_by_doy(thresholds, "HW_Tmean_P90", doys)
    hhw_thr = lookup_by_doy(thresholds, "HHW_Twbmax_P95", doys)
    dhw_tmax_thr = lookup_by_doy(thresholds, "DHW_Tmax_P95", doys)
    dhw_vpd_thr = lookup_by_doy(thresholds, "DHW_VPDmean_P75", doys)

    tmean_mean = lookup_by_doy(thresholds, "Tmean_mean", doys)
    tmean_std = lookup_by_doy(thresholds, "Tmean_std", doys)
    twb_mean = lookup_by_doy(thresholds, "Twbmax_mean", doys)
    twb_std = lookup_by_doy(thresholds, "Twbmax_std", doys)
    tmax_mean = lookup_by_doy(thresholds, "Tmax_mean", doys)
    tmax_std = lookup_by_doy(thresholds, "Tmax_std", doys)
    vpd_mean = lookup_by_doy(thresholds, "VPDmean_mean", doys)
    vpd_std = lookup_by_doy(thresholds, "VPDmean_std", doys)

    conditions = {
        "HW": (tmean >= hw_thr),
        "HHW": (twbmax >= hhw_thr),
        "DHW": (tmax >= dhw_tmax_thr) & (vpd >= dhw_vpd_thr),
    }

    def _pos_exceedance(values, threshold):
        out = np.asarray(values, dtype=float) - np.asarray(threshold, dtype=float)
        return np.where(np.isfinite(out) & (out > 0), out, 0.0)

    def _z_pos(values, mean, std):
        values = np.asarray(values, dtype=float)
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        std = np.where(std <= 1e-6, np.nan, std)
        z = (values - mean) / std
        z = np.clip(z, Z_CLIP_MIN, Z_CLIP_MAX)
        return np.where(np.isfinite(z) & (z > 0), z, 0.0)

    severity = {
        "HW": _pos_exceedance(tmean, hw_thr),
        "HHW": _pos_exceedance(twbmax, hhw_thr),
        "DHW": DHW_WEIGHT_TMAX * _z_pos(tmax, tmax_mean, tmax_std)
               + DHW_WEIGHT_VPD * _z_pos(vpd, vpd_mean, vpd_std),
    }

    years_all = np.asarray(g["year"].values, dtype=int)
    unique_years = sorted(np.unique(years_all[np.isfinite(years_all)]))
    n_valid_tmean = np.isfinite(tmean)

    rows = []
    for regime in REGIMES:
        valid = np.isfinite(severity[regime])
        base_mask = conditions[regime] & valid
        persistent, events = mark_persistent_events(base_mask, min_event_duration)

        for year in unique_years:
            yr = years_all == int(year)
            # Match Figure 01 data-quality logic: years with insufficient daily
            # coverage are excluded from event-metric trend calculations.
            min_days = int(np.ceil(min_valid_day_fraction * year_days(int(year))))
            if int(np.nansum(n_valid_tmean & yr)) < min_days:
                rows.append({
                    "year": int(year),
                    "regime": regime,
                    "frequency": np.nan,
                    "duration": np.nan,
                    "intensity": np.nan,
                    "mean_event_day_severity": np.nan,
                })
                continue

            ev_count = 0
            for i, j in events:
                # Figure 01 counts frequency by event start year.
                if int(years_all[i]) == int(year):
                    ev_count += 1

            dur = int(np.nansum(persistent & yr))
            sev_vals = severity[regime][persistent & yr]
            if len(sev_vals) and np.isfinite(sev_vals).any():
                annual_intensity = float(np.nansum(sev_vals))
                mean_event_day_severity = float(np.nanmean(sev_vals))
            else:
                annual_intensity = 0.0
                mean_event_day_severity = 0.0

            rows.append({
                "year": int(year),
                "regime": regime,
                "frequency": int(ev_count),
                "duration": dur,
                "intensity": annual_intensity,
                "mean_event_day_severity": mean_event_day_severity,
            })
    return pd.DataFrame(rows)


def build_annual_validation_dataset(daily: pd.DataFrame, source: str, selected_stations: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected_stations.loc[selected_stations["selected"], "station_id"].astype(str))
    daily = daily[daily["station_id"].astype(str).isin(selected_ids)].copy()
    daily = daily[(daily["year"] >= args.start_year) & (daily["year"] <= args.end_year)].copy()

    annual_var = (
        daily.groupby(["station_id", "year"], as_index=False)
        .agg(
            Tmax=("Tmax", "mean"),
            Tmin=("Tmin", "mean"),
            Tmean=("Tmean", "mean"),
            Tdmean=("Tdmean", "mean"),
            RHmean=("RHmean", "mean"),
            VPDmean=("VPDmean", "mean"),
            Twbmax=("Twbmax", "mean"),
            WSmean=("WSmean", "mean"),
            n_valid_days=("Tmax", lambda x: int(np.isfinite(x).sum())),
        )
    )

    metric_frames = []
    threshold_rows = []
    for sid, g in daily.groupby("station_id"):
        try:
            thr = compute_thresholds_for_station(g, args.baseline_start, args.baseline_end)
            threshold_rows.append({
                "station_id": sid,
                "source": source,
                "threshold_mode": thr.get("threshold_mode", "unknown"),
                "rolling_window_days": thr.get("rolling_window_days", np.nan),
                **{k: v for k, v in thr.items() if k.endswith("_annual_median")}
            })
            annual_met = annual_metrics_for_station(g, thr, args.min_event_duration, args.min_valid_day_fraction)
            annual_met["station_id"] = sid
            metric_frames.append(annual_met)
        except Exception as exc:
            print(f"[WARN] Could not compute heatwave metrics for {source} station {sid}: {exc}")
    annual_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()

    # Wide format for heatwave metrics.
    if not annual_metrics.empty:
        wide = annual_metrics.pivot_table(index=["station_id", "year"], columns="regime", values=ALL_HEATWAVE_METRICS)
        wide.columns = [f"{regime}_{metric}" for metric, regime in wide.columns]
        wide = wide.reset_index()
    else:
        wide = pd.DataFrame(columns=["station_id", "year"])

    annual = annual_var.merge(wide, on=["station_id", "year"], how="left")
    for col in heatwave_metric_columns(include_aux=True):
        if col not in annual.columns:
            annual[col] = np.nan
    annual["source"] = source
    meta_cols = ["station_id", "station_name", "uf", "region", "lat", "lon"]
    annual = annual.merge(selected_stations[meta_cols], on="station_id", how="left", suffixes=("", "_meta"))
    thresholds = pd.DataFrame(threshold_rows)
    return annual, thresholds


# ============================================================
# Comparison metrics
# ============================================================

def compute_station_trends(annual: pd.DataFrame) -> pd.DataFrame:
    value_cols = THERMO_VARS + heatwave_metric_columns(include_aux=True)
    rows = []
    for (source, sid), g in annual.groupby(["source", "station_id"]):
        meta = g.iloc[0]
        years = g["year"].values
        for var in value_cols:
            slope, p = theil_sen_decade(years, g[var].values)
            rows.append({
                "source": source,
                "station_id": sid,
                "station_name": meta.get("station_name", ""),
                "uf": meta.get("uf", None),
                "region": meta.get("region", None),
                "lat": meta.get("lat", np.nan),
                "lon": meta.get("lon", np.nan),
                "diagnostic": var,
                "trend_decade": slope,
                "mk_pvalue": p,
                "n_years": int(np.isfinite(g[var].values).sum()),
            })
    return pd.DataFrame(rows)


def compare_trends(trends: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = trends[trends["source"] == "INMET"].copy()
    b = trends[trends["source"] == "ERA5"].copy()
    paired = a.merge(
        b,
        on=["station_id", "diagnostic"],
        suffixes=("_inmet", "_era5"),
        how="inner",
    )
    paired = paired[np.isfinite(paired["trend_decade_inmet"]) & np.isfinite(paired["trend_decade_era5"])].copy()
    rows = []
    for diag, g in paired.groupby("diagnostic"):
        obs = g["trend_decade_inmet"].values
        mod = g["trend_decade_era5"].values
        diff = mod - obs
        # Sign agreement excluding near-zero paired trends.
        sign_ok = np.sign(obs) == np.sign(mod)
        rows.append({
            "diagnostic": diag,
            "n_stations": int(len(g)),
            "pearson_r": safe_corr(obs, mod, "pearson"),
            "spearman_r": safe_corr(obs, mod, "spearman"),
            "sign_agreement_percent": float(100.0 * np.nanmean(sign_ok)) if len(g) else np.nan,
            "bias_era5_minus_inmet": float(np.nanmean(diff)),
            "mae": float(np.nanmean(np.abs(diff))),
            "rmse": float(np.sqrt(np.nanmean(diff**2))),
            "median_inmet_trend": float(np.nanmedian(obs)),
            "median_era5_trend": float(np.nanmedian(mod)),
        })
    return paired, pd.DataFrame(rows)


def compare_annual_metrics(annual: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs = annual[annual["source"] == "INMET"].copy()
    mod = annual[annual["source"] == "ERA5"].copy()
    value_cols = THERMO_VARS + heatwave_metric_columns(include_aux=True)
    paired = obs.merge(mod, on=["station_id", "year"], suffixes=("_inmet", "_era5"), how="inner")
    rows = []
    for diag in value_cols:
        x = paired[f"{diag}_inmet"].values
        y = paired[f"{diag}_era5"].values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        diff = y[ok] - x[ok]
        rows.append({
            "diagnostic": diag,
            "n_station_years": int(ok.sum()),
            "pearson_r": safe_corr(x[ok], y[ok], "pearson"),
            "spearman_r": safe_corr(x[ok], y[ok], "spearman"),
            "bias_era5_minus_inmet": float(np.nanmean(diff)),
            "mae": float(np.nanmean(np.abs(diff))),
            "rmse": float(np.sqrt(np.nanmean(diff**2))),
            "median_inmet": float(np.nanmedian(x[ok])),
            "median_era5": float(np.nanmedian(y[ok])),
        })
    return paired, pd.DataFrame(rows)


# ============================================================
# Regional aggregation validation
# ============================================================

REGION_ORDER = ["Brazil", "North", "Northeast", "Central-West", "Southeast", "South"]


def build_regional_annual_series(annual: pd.DataFrame, min_stations_per_region: int = 1) -> pd.DataFrame:
    """Build regional annual median series from selected station annual metrics.

    This is intentionally based on the same selected INMET stations and the
    nearest ERA5 grid cells used in the station-level validation. Aggregating by
    macro-region reduces station-scale noise and better matches the spatial
    interpretation of the ERA5 analysis as a grid-scale/regional thermodynamic
    diagnostic.

    The default min_stations_per_region is 1 so that sparsely sampled regions
    such as North are still displayed when at least one selected station has
    valid data. Regions based on one or very few stations should be interpreted
    cautiously and reported as limited-coverage diagnostics rather than robust
    macro-regional validation.
    """
    value_cols = THERMO_VARS + heatwave_metric_columns(include_aux=True)
    base = annual.copy()
    base["region"] = base["region"].fillna("Unknown")

    frames = []
    # Brazil-wide aggregate from all selected stations.
    tmp = base.copy()
    tmp["validation_region"] = "Brazil"
    frames.append(tmp)

    # Macro-region aggregates.
    tmp = base[base["region"].isin(REGION_ORDER)].copy()
    tmp["validation_region"] = tmp["region"]
    frames.append(tmp)

    dat = pd.concat(frames, ignore_index=True)
    rows = []
    for (source, reg, year), g in dat.groupby(["source", "validation_region", "year"]):
        row = {"source": source, "validation_region": reg, "year": int(year)}
        n_all = int(g["station_id"].nunique())
        row["n_stations_available"] = n_all
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce").values
            n_valid = int(np.isfinite(vals).sum())
            row[f"n_{col}"] = n_valid
            row[col] = float(np.nanmedian(vals)) if n_valid >= min_stations_per_region else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["validation_region"] = pd.Categorical(out["validation_region"], categories=REGION_ORDER, ordered=True)
        out = out.sort_values(["validation_region", "source", "year"]).reset_index(drop=True)
    return out


def compute_regional_trends(regional_annual: pd.DataFrame) -> pd.DataFrame:
    value_cols = THERMO_VARS + heatwave_metric_columns(include_aux=True)
    rows = []
    for (source, reg), g in regional_annual.groupby(["source", "validation_region"]):
        years = g["year"].values
        for diag in value_cols:
            slope, p = theil_sen_decade(years, g[diag].values)
            rows.append({
                "source": source,
                "validation_region": str(reg),
                "diagnostic": diag,
                "trend_decade": slope,
                "mk_pvalue": p,
                "n_years": int(np.isfinite(g[diag].values).sum()),
            })
    return pd.DataFrame(rows)


def compare_regional_annual_metrics(regional_annual: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs = regional_annual[regional_annual["source"] == "INMET"].copy()
    mod = regional_annual[regional_annual["source"] == "ERA5"].copy()
    value_cols = THERMO_VARS + heatwave_metric_columns(include_aux=True)
    paired = obs.merge(mod, on=["validation_region", "year"], suffixes=("_inmet", "_era5"), how="inner")
    rows = []
    for (reg), g in paired.groupby("validation_region"):
        for diag in value_cols:
            x = pd.to_numeric(g[f"{diag}_inmet"], errors="coerce").values
            y = pd.to_numeric(g[f"{diag}_era5"], errors="coerce").values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 8:
                continue
            diff = y[ok] - x[ok]
            rows.append({
                "validation_region": str(reg),
                "diagnostic": diag,
                "n_years": int(ok.sum()),
                "pearson_r": safe_corr(x[ok], y[ok], "pearson"),
                "spearman_r": safe_corr(x[ok], y[ok], "spearman"),
                "bias_era5_minus_inmet": float(np.nanmean(diff)),
                "mae": float(np.nanmean(np.abs(diff))),
                "rmse": float(np.sqrt(np.nanmean(diff**2))),
                "median_inmet": float(np.nanmedian(x[ok])),
                "median_era5": float(np.nanmedian(y[ok])),
            })
    return paired, pd.DataFrame(rows)


def compare_regional_trends(regional_trends: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs = regional_trends[regional_trends["source"] == "INMET"].copy()
    mod = regional_trends[regional_trends["source"] == "ERA5"].copy()
    paired = obs.merge(mod, on=["validation_region", "diagnostic"], suffixes=("_inmet", "_era5"), how="inner")
    paired = paired[np.isfinite(paired["trend_decade_inmet"]) & np.isfinite(paired["trend_decade_era5"])].copy()
    paired["same_sign"] = np.sign(paired["trend_decade_inmet"]) == np.sign(paired["trend_decade_era5"])
    paired["trend_bias_era5_minus_inmet"] = paired["trend_decade_era5"] - paired["trend_decade_inmet"]

    rows = []
    for diag, g in paired.groupby("diagnostic"):
        obs_v = g["trend_decade_inmet"].values
        mod_v = g["trend_decade_era5"].values
        diff = mod_v - obs_v
        rows.append({
            "diagnostic": diag,
            "n_regions": int(len(g)),
            "pearson_r_across_regions": safe_corr(obs_v, mod_v, "pearson"),
            "spearman_r_across_regions": safe_corr(obs_v, mod_v, "spearman"),
            "sign_agreement_percent": float(100.0 * np.nanmean(g["same_sign"].values)) if len(g) else np.nan,
            "bias_era5_minus_inmet": float(np.nanmean(diff)),
            "mae": float(np.nanmean(np.abs(diff))),
            "rmse": float(np.sqrt(np.nanmean(diff**2))),
            "median_inmet_trend": float(np.nanmedian(obs_v)),
            "median_era5_trend": float(np.nanmedian(mod_v)),
        })
    return paired, pd.DataFrame(rows)


def plot_regional_validation_summary(regional_annual_summary: pd.DataFrame,
                                     regional_trend_paired: pd.DataFrame,
                                     regional_trend_summary: pd.DataFrame,
                                     out_dir: str,
                                     start_year: int,
                                     end_year: int,
                                     baseline_start: int,
                                     baseline_end: int):
    """Plot regional aggregated validation summary."""
    key_diags = [
        "Tmax", "Tmean", "Tdmean", "RHmean", "VPDmean", "Twbmax",
        "HW_intensity", "HHW_intensity", "DHW_frequency", "DHW_duration", "DHW_intensity",
    ]
    regs = REGION_ORDER
    fig = plt.figure(figsize=(17.5, 12.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.975, bottom=0.08, top=0.94, wspace=0.28, hspace=0.34)

    # (a) Regional annual temporal correlation.
    ax = fig.add_subplot(gs[0, 0])
    mat = np.full((len(key_diags), len(regs)), np.nan)
    for i, diag in enumerate(key_diags):
        for j, reg in enumerate(regs):
            row = regional_annual_summary[(regional_annual_summary["diagnostic"] == diag) & (regional_annual_summary["validation_region"] == reg)]
            if not row.empty:
                mat[i, j] = row["pearson_r"].iloc[0]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(np.arange(len(key_diags)))
    ax.set_yticklabels([diagnostic_label(d) for d in key_diags], fontsize=PLOT_FONT)
    ax.set_xticks(np.arange(len(regs)))
    ax.set_xticklabels(regs, rotation=35, ha="right", fontsize=PLOT_FONT)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=PLOT_FONT, color=("white" if abs(v) > 0.65 else "black"))
    ax.set_title("(a) Regional annual-series correlation\nERA5 versus INMET", fontsize=PLOT_FONT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r", fontsize=PLOT_FONT)
    cb.ax.tick_params(labelsize=PLOT_FONT)

    # (b) Regional trend sign agreement.
    ax = fig.add_subplot(gs[0, 1])
    mat2 = np.full((len(key_diags), len(regs)), np.nan)
    for i, diag in enumerate(key_diags):
        for j, reg in enumerate(regs):
            row = regional_trend_paired[(regional_trend_paired["diagnostic"] == diag) & (regional_trend_paired["validation_region"] == reg)]
            if not row.empty:
                mat2[i, j] = 100.0 if bool(row["same_sign"].iloc[0]) else 0.0
    im = ax.imshow(mat2, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(key_diags)))
    ax.set_yticklabels([diagnostic_label(d) for d in key_diags], fontsize=PLOT_FONT)
    ax.set_xticks(np.arange(len(regs)))
    ax.set_xticklabels(regs, rotation=35, ha="right", fontsize=PLOT_FONT)
    for i in range(mat2.shape[0]):
        for j in range(mat2.shape[1]):
            v = mat2[i, j]
            if np.isfinite(v):
                ax.text(j, i, "✓" if v == 100 else "×", ha="center", va="center", fontsize=PLOT_FONT, color=("white" if v == 0 else "black"))
    ax.set_title("(b) Regional trend-sign agreement\nTheil–Sen trends", fontsize=PLOT_FONT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("same sign (%)", fontsize=PLOT_FONT)
    cb.ax.tick_params(labelsize=PLOT_FONT)

    # (c) Regional trend comparison for selected thermodynamic variables.
    ax = fig.add_subplot(gs[1, 0])
    colors = {"Tmax": "C3", "Tmean": "C4", "VPDmean": "C1", "RHmean": "C0", "Twbmax": "C2"}
    for diag, color in colors.items():
        g = regional_trend_paired[regional_trend_paired["diagnostic"] == diag]
        ax.scatter(g["trend_decade_inmet"], g["trend_decade_era5"], s=58, alpha=0.78,
                   label=diagnostic_label(diag), edgecolor="black", linewidth=0.35)
        # Label Brazil point when available.
        gb = g[g["validation_region"] == "Brazil"]
        if not gb.empty:
            ax.scatter(gb["trend_decade_inmet"], gb["trend_decade_era5"], s=115,
                       facecolor="none", edgecolor="black", linewidth=1.1, zorder=5)
    allv = np.r_[regional_trend_paired["trend_decade_inmet"].values, regional_trend_paired["trend_decade_era5"].values]
    finite = allv[np.isfinite(allv)]
    lim = np.nanpercentile(np.abs(finite), 97) if finite.size else 1.0
    lim = max(float(lim), 0.5)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.75)
    ax.axhline(0, color="0.55", lw=0.8)
    ax.axvline(0, color="0.55", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("INMET regional trend", fontsize=PLOT_FONT)
    ax.set_ylabel("ERA5 regional trend", fontsize=PLOT_FONT)
    ax.set_title("(c) Regional trend comparison\nselected thermodynamic diagnostics", fontsize=PLOT_FONT, fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=PLOT_FONT, ncol=2)

    # (d) Mean annual correlation by heatwave regime/metric across macro-regions.
    ax = fig.add_subplot(gs[1, 1])
    arr = np.full((len(REGIMES), len(METRICS)), np.nan)
    for i, r in enumerate(REGIMES):
        for j, m in enumerate(METRICS):
            diag = f"{r}_{m}"
            sub = regional_annual_summary[(regional_annual_summary["diagnostic"] == diag) & (regional_annual_summary["validation_region"].isin(regs))]
            arr[i, j] = np.nanmedian(sub["pearson_r"].values) if not sub.empty else np.nan
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(np.arange(len(REGIMES)))
    ax.set_yticklabels(REGIMES, fontsize=PLOT_FONT)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation=25, ha="right", fontsize=PLOT_FONT)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=PLOT_FONT, color=("white" if abs(v) > 0.65 else "black"))
    ax.set_title("(d) Median regional annual correlation\nheatwave-regime metrics", fontsize=PLOT_FONT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r", fontsize=PLOT_FONT)
    cb.ax.tick_params(labelsize=PLOT_FONT)

    out_base = os.path.join(out_dir, f"Supplementary_Figure_ERA5_INMET_regional_validation_{start_year}_{end_year}")
    fig.savefig(out_base + ".jpeg", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved regional validation figure: {out_base}.jpeg")
    print(f"[OK] Saved regional validation figure: {out_base}.pdf")


# ============================================================
# Plotting
# ============================================================

def diagnostic_label(x: str) -> str:
    mapping = {
        "Tmax": r"T$_{max}$", "Tmin": r"T$_{min}$", "Tmean": r"T$_{mean}$",
        "Tdmean": r"T$_d$", "RHmean": "RH", "VPDmean": "VPD", "Twbmax": r"T$_{wb,max}$", "WSmean": "WS",
    }
    if x in mapping:
        return mapping[x]
    for r in REGIMES:
        for m in METRICS:
            if x == f"{r}_{m}":
                return f"{r} {m}"
    return x


def read_states(path: str):
    if not path or gpd is None or not os.path.exists(path):
        return None
    try:
        gdf = gpd.read_file(path)
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf.set_crs("EPSG:4326")
        return gdf
    except Exception as exc:
        print(f"[WARN] Could not read shapefile for plotting: {exc}")
        return None


def plot_validation_summary(trend_summary: pd.DataFrame, trend_paired: pd.DataFrame, annual_summary: pd.DataFrame, selected: pd.DataFrame, shp: str, out_dir: str, start_year: int, end_year: int):
    fig = plt.figure(figsize=(19, 14), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, bottom=0.075, top=0.955, wspace=0.35, hspace=0.35)

    # Panel a: sign agreement heatmap.
    ax = fig.add_subplot(gs[0, 0])
    order = THERMO_VARS + [f"{r}_{m}" for r in REGIMES for m in METRICS]
    d = trend_summary.set_index("diagnostic").reindex(order)
    vals = d["sign_agreement_percent"].values.reshape(-1, 1)
    im = ax.imshow(vals, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([diagnostic_label(x) for x in order], fontsize=PLOT_FONT)
    ax.set_xticks([0])
    ax.set_xticklabels(["sign agreement"], fontsize=PLOT_FONT)
    for i, v in enumerate(vals[:, 0]):
        if np.isfinite(v):
            ax.text(0, i, f"{v:.0f}%", ha="center", va="center", fontsize=PLOT_FONT, color=("white" if v < 45 or v > 75 else "black"))
    ax.set_title("(a) ERA5–INMET trend sign agreement", fontsize=PLOT_FONT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% stations", fontsize=PLOT_FONT)
    cb.ax.tick_params(labelsize=PLOT_FONT)

    # Panel b: trend magnitude correlation heatmap.
    ax = fig.add_subplot(gs[0, 1])
    vals = d["pearson_r"].values.reshape(-1, 1)
    im = ax.imshow(vals, aspect="auto", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap="RdBu_r")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([diagnostic_label(x) for x in order], fontsize=PLOT_FONT)
    ax.set_xticks([0])
    ax.set_xticklabels(["Pearson r"], fontsize=PLOT_FONT)
    for i, v in enumerate(vals[:, 0]):
        if np.isfinite(v):
            ax.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=PLOT_FONT, color=("white" if abs(v) > 0.65 else "black"))
    ax.set_title("(b) ERA5–INMET trend-magnitude correlation", fontsize=PLOT_FONT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("r", fontsize=PLOT_FONT)
    cb.ax.tick_params(labelsize=PLOT_FONT)

    # Panel c: scatter selected thermodynamic trends.
    ax = fig.add_subplot(gs[1, 0])
    colors = {"Tmax": "C3", "VPDmean": "C1", "RHmean": "C0", "Twbmax": "C2"}
    for diag, color in colors.items():
        g = trend_paired[trend_paired["diagnostic"] == diag]
        ax.scatter(g["trend_decade_inmet"], g["trend_decade_era5"], s=24, alpha=0.65, label=diagnostic_label(diag), edgecolor="black", linewidth=0.2)
    allx = np.r_[trend_paired["trend_decade_inmet"].values, trend_paired["trend_decade_era5"].values]
    lim = np.nanpercentile(np.abs(allx[np.isfinite(allx)]), 97) if np.isfinite(allx).any() else 1
    lim = max(lim, 1e-6)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.75)
    ax.axhline(0, color="0.55", lw=0.8)
    ax.axvline(0, color="0.55", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("INMET trend", fontsize=PLOT_FONT)
    ax.set_ylabel("ERA5 trend", fontsize=PLOT_FONT)
    ax.set_title("(c) Station trend comparison for key thermodynamic variables", fontsize=PLOT_FONT, fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=PLOT_FONT, ncol=2)

    # Panel d: heatwave metric sign agreement by regime/metric.
    ax = fig.add_subplot(gs[1, 1])
    rows = []
    for r in REGIMES:
        for m in METRICS:
            diag = f"{r}_{m}"
            row = trend_summary[trend_summary["diagnostic"] == diag]
            rows.append(row["sign_agreement_percent"].iloc[0] if not row.empty else np.nan)
    arr = np.array(rows).reshape(len(REGIMES), len(METRICS))
    im = ax.imshow(arr, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(REGIMES)))
    ax.set_yticklabels(REGIMES, fontsize=PLOT_FONT)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation=25, ha="right", fontsize=PLOT_FONT)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.0f}%", ha="center", va="center", fontsize=PLOT_FONT, color=("white" if arr[i,j] < 45 or arr[i,j] > 75 else "black"))
    ax.set_title("(d) Heatwave-metric trend sign agreement", fontsize=PLOT_FONT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% stations", fontsize=PLOT_FONT)
    cb.ax.tick_params(labelsize=PLOT_FONT)

    out_base = os.path.join(out_dir, f"Supplementary_Figure_ERA5_INMET_trends_heatwave_metrics_validation_{start_year}_{end_year}")
    fig.savefig(out_base + ".jpeg", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_base}.jpeg")
    print(f"[OK] Saved figure: {out_base}.pdf")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="ERA5–INMET evaluation of trends and heatwave metrics."
    )
    ap.add_argument(
        "--inmet-hourly-dir", "--inmet_hourly_dir",
        dest="inmet_hourly_dir",
        required=True,
        help="Directory containing INMET hourly observation files.",
    )
    ap.add_argument(
        "--era5-daily-dir", "--era5_daily_dir",
        dest="era5_daily_dir",
        required=True,
        help="Directory containing preprocessed daily ERA5 heatwave-input NetCDF files.",
    )
    ap.add_argument(
        "--era5-station-cache", "--era5_station_cache",
        dest="era5_station_cache",
        default=None,
        help="Optional precomputed ERA5 station-daily CSV cache.",
    )
    ap.add_argument(
        "--station-metadata-dir", "--station_metadata_dir",
        dest="station_metadata_dir",
        default=None,
        help="Optional INMET climatological-normal directory containing station metadata.",
    )
    ap.add_argument(
        "--states-shapefile", "--shp_uf",
        dest="shp_uf",
        default=None,
        help="Optional Brazilian state shapefile used for plotting.",
    )
    ap.add_argument(
        "--output-dir", "--out_dir",
        dest="out_dir",
        required=True,
        help="Output directory.",
    )
    ap.add_argument("--start-year", "--start_year", dest="start_year", type=int, default=2002)
    ap.add_argument("--end-year", "--end_year", dest="end_year", type=int, default=2024)
    ap.add_argument("--baseline-start", "--baseline_start", dest="baseline_start", type=int, default=2002)
    ap.add_argument("--baseline-end", "--baseline_end", dest="baseline_end", type=int, default=2020)
    ap.add_argument("--min-hourly-per-day", "--min_hourly_per_day", dest="min_hourly_per_day", type=int, default=18)
    ap.add_argument("--min-valid-day-fraction", "--min_valid_day_fraction", dest="min_valid_day_fraction", type=float, default=0.70)
    ap.add_argument("--min-valid-years-threshold", "--min_valid_years_threshold", dest="min_valid_years_threshold", type=int, default=15)
    ap.add_argument("--min-valid-years-trend", "--min_valid_years_trend", dest="min_valid_years_trend", type=int, default=18)
    ap.add_argument("--min-event-duration", "--min_event_duration", dest="min_event_duration", type=int, default=3)
    ap.add_argument(
        "--use-utc-day", "--use_utc_day",
        dest="use_utc_day",
        action="store_true",
        help="Use UTC-day aggregation for INMET; default uses fixed state-level local time.",
    )
    ap.add_argument("--reuse-cache", "--reuse_cache", dest="reuse_cache", action="store_true")
    ap.add_argument(
        "--strict-coverage-80", "--strict_coverage_80",
        dest="strict_coverage_80",
        action="store_true",
        help="Generate an additional 80% valid-day/year station-selection summary.",
    )
    ap.add_argument(
        "--min-stations-per-region", "--min_stations_per_region",
        dest="min_stations_per_region",
        type=int,
        default=1,
        help="Minimum valid stations required for a regional annual median.",
    )
    ap.add_argument(
        "--disable-outlier-qc", "--disable_outlier_qc",
        dest="disable_outlier_qc",
        action="store_true",
        help="Disable conservative pairwise ERA5–INMET residual outlier QC.",
    )
    ap.add_argument(
        "--outlier-mad-k", "--outlier_mad_k",
        dest="outlier_mad_k",
        type=float,
        default=6.0,
        help="MAD multiplier for pairwise residual outlier QC.",
    )
    ap.add_argument(
        "--outlier-max-fraction", "--outlier_max_fraction",
        dest="outlier_max_fraction",
        type=float,
        default=0.03,
        help="Maximum fraction of daily pairs removed per station-variable.",
    )
    ap.add_argument(
        "--outlier-min-pairs", "--outlier_min_pairs",
        dest="outlier_min_pairs",
        type=int,
        default=30,
        help="Minimum paired daily observations required for residual outlier QC.",
    )
    ap.add_argument(
        "--outlier-min-abs", "--outlier_min_abs",
        dest="outlier_min_abs",
        default="Tmax=5,Tmin=5,Tmean=5,Tdmean=5,RHmean=20,VPDmean=1.5,Twbmax=5,WSmean=3",
        help="Comma-separated absolute residual thresholds for pairwise outlier QC.",
    )
    args = ap.parse_args()

    if args.start_year > args.end_year:
        ap.error("--start-year must be <= --end-year.")
    if args.baseline_start > args.baseline_end:
        ap.error("--baseline-start must be <= --baseline-end.")
    if args.baseline_start < args.start_year or args.baseline_end > args.end_year:
        ap.error("The validation baseline must lie within the common validation period.")

    ensure_dir(args.out_dir)
    print(f"[START] ERA5-INMET trends and heatwave metrics evaluation")
    print(f"[INFO] Period: {args.start_year}-{args.end_year}; baseline: {args.baseline_start}-{args.baseline_end}")
    print(f"[INFO] Daily QC: >= {args.min_hourly_per_day} hourly records day-1")
    print(f"[INFO] ERA5 station cache: {args.era5_station_cache or 'not supplied'}")
    print(f"[INFO] Year QC: >= {args.min_valid_day_fraction:.0%} valid days year-1")
    if args.disable_outlier_qc:
        print("[INFO] Pairwise residual outlier QC: disabled")
    else:
        print(f"[INFO] Pairwise residual outlier QC: enabled; MAD k={args.outlier_mad_k:g}; max fraction={args.outlier_max_fraction:.3f}; min abs={args.outlier_min_abs}")

    period_tag = f"{args.start_year}_{args.end_year}"
    baseline_tag = f"{args.baseline_start}_{args.baseline_end}"

    station_meta_lookup = read_station_metadata_from_normals(args.station_metadata_dir)
    inmet_daily = build_inmet_daily_cache(args, station_meta_lookup)
    inmet_daily = enforce_common_validation_period(inmet_daily, args, "INMET daily cache")
    selected = station_selection(inmet_daily, args)
    selected_path = os.path.join(args.out_dir, f"Supplementary_Table_INMET_station_coverage_selection_{period_tag}.csv")
    selected.to_csv(selected_path, index=False)
    print(f"[OK] Saved station coverage/selection table: {selected_path}")
    print(f"[INFO] Selected stations: {int(selected['selected'].sum())}/{len(selected)}")

    if args.strict_coverage_80:
        args80 = argparse.Namespace(**vars(args))
        args80.min_valid_day_fraction = 0.80
        selected80 = station_selection(inmet_daily, args80)
        selected80_path = os.path.join(
            args.out_dir,
            f"Supplementary_Table_INMET_station_coverage_selection_80pct_{period_tag}.csv",
        )
        selected80.to_csv(selected80_path, index=False)
        print(
            f"[INFO] 80% valid-day sensitivity: "
            f"{int(selected80['selected'].sum())}/{len(selected80)} stations selected."
        )
        print(f"[OK] Saved 80% coverage sensitivity table: {selected80_path}")

    if selected["selected"].sum() < 5:
        raise RuntimeError("Too few selected stations. Relax coverage thresholds or check INMET input files.")

    stations_for_era5 = selected[selected["selected"]].copy()
    era5_daily, grid_index = build_era5_daily_station_cache(args, stations_for_era5)

    # Keep exactly the selected stations and the common validation period.
    keep_ids = set(stations_for_era5["station_id"].astype(str))
    inmet_daily = inmet_daily[inmet_daily["station_id"].astype(str).isin(keep_ids)].copy()
    era5_daily = era5_daily[era5_daily["station_id"].astype(str).isin(keep_ids)].copy()
    inmet_daily = enforce_common_validation_period(inmet_daily, args, "INMET selected-station daily table")
    era5_daily = enforce_common_validation_period(era5_daily, args, "ERA5 selected-station daily table")
    print(f"[INFO] COMMON VALIDATION PERIOD USED FOR ERA5-INMET EVENT METRICS: {args.start_year}-{args.end_year}")
    print(f"[INFO] PERCENTILE BASELINE USED ONLY FOR THIS VALIDATION: {args.baseline_start}-{args.baseline_end}")

    inmet_daily, era5_daily, outlier_qc_summary = apply_pairwise_outlier_qc_to_daily_tables(
        inmet_daily,
        era5_daily,
        args,
    )

    annual_inmet, thresholds_inmet = build_annual_validation_dataset(inmet_daily, "INMET", selected, args)
    annual_era5, thresholds_era5 = build_annual_validation_dataset(era5_daily, "ERA5", selected, args)
    annual = pd.concat([annual_inmet, annual_era5], ignore_index=True)
    thresholds = pd.concat([thresholds_inmet, thresholds_era5], ignore_index=True)

    trends = compute_station_trends(annual)
    trend_paired, trend_summary = compare_trends(trends)
    annual_paired, annual_summary = compare_annual_metrics(annual)

    # Regional aggregated validation: annual regional medians from the same
    # selected stations and nearest ERA5 grid cells. This complements the
    # station-level validation and better matches the regional/national scale of
    # the ERA5 heatwave-regime analysis.
    regional_annual = build_regional_annual_series(annual, min_stations_per_region=args.min_stations_per_region)

    # Diagnostic printout to avoid silently blank regions in the regional figure.
    # When min_stations_per_region=1, North and other sparse regions will be plotted
    # if at least one selected station has valid annual values. Such regions should
    # be interpreted cautiously.
    if not regional_annual.empty:
        reg_counts = (
            regional_annual.groupby("validation_region", observed=False)["n_stations_available"]
            .max()
            .reset_index()
            .sort_values("validation_region")
        )
        print("[INFO] Maximum selected stations available by validation region:")
        for _, _r in reg_counts.iterrows():
            print(f"  {_r['validation_region']}: {int(_r['n_stations_available'])}")

    regional_trends = compute_regional_trends(regional_annual)
    regional_trend_paired, regional_trend_summary = compare_regional_trends(regional_trends)
    regional_annual_paired, regional_annual_summary = compare_regional_annual_metrics(regional_annual)

    paths = {
        "annual": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_annual_station_metrics_{period_tag}.csv"),
        "thresholds": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_station_thresholds_{baseline_tag}.csv"),
        "trends": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_station_trends_{period_tag}.csv"),
        "trend_paired": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_paired_station_trends_{period_tag}.csv"),
        "trend_summary": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_trend_validation_summary_{period_tag}.csv"),
        "annual_paired": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_paired_annual_metrics_{period_tag}.csv"),
        "annual_summary": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_annual_metric_validation_summary_{period_tag}.csv"),
        "regional_annual": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_regional_annual_median_series_{period_tag}.csv"),
        "regional_trends": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_regional_trends_{period_tag}.csv"),
        "regional_trend_paired": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_regional_paired_trends_{period_tag}.csv"),
        "regional_trend_summary": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_regional_trend_validation_summary_{period_tag}.csv"),
        "regional_annual_paired": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_regional_paired_annual_metrics_{period_tag}.csv"),
        "regional_annual_summary": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_regional_annual_metric_validation_summary_{period_tag}.csv"),
        "grid_index": os.path.join(args.out_dir, "Supplementary_Table_ERA5_INMET_nearest_grid_index_trend_validation.csv"),
        "xlsx": os.path.join(args.out_dir, f"Supplementary_Tables_ERA5_INMET_trends_heatwave_metrics_validation_{period_tag}.xlsx"),
        "metadata": os.path.join(args.out_dir, f"software_metadata_ERA5_INMET_trends_heatwave_metrics_validation_{period_tag}.json"),
        "outlier_qc_summary": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_pairwise_outlier_QC_summary_{period_tag}.csv"),
        "outlier_qc_by_variable": os.path.join(args.out_dir, f"Supplementary_Table_ERA5_INMET_pairwise_outlier_QC_by_variable_{period_tag}.csv"),
    }

    annual.to_csv(paths["annual"], index=False)
    thresholds.to_csv(paths["thresholds"], index=False)
    trends.to_csv(paths["trends"], index=False)
    trend_paired.to_csv(paths["trend_paired"], index=False)
    trend_summary.to_csv(paths["trend_summary"], index=False)
    annual_paired.to_csv(paths["annual_paired"], index=False)
    annual_summary.to_csv(paths["annual_summary"], index=False)
    regional_annual.to_csv(paths["regional_annual"], index=False)
    regional_trends.to_csv(paths["regional_trends"], index=False)
    regional_trend_paired.to_csv(paths["regional_trend_paired"], index=False)
    regional_trend_summary.to_csv(paths["regional_trend_summary"], index=False)
    regional_annual_paired.to_csv(paths["regional_annual_paired"], index=False)
    regional_annual_summary.to_csv(paths["regional_annual_summary"], index=False)
    grid_index.to_csv(paths["grid_index"], index=False)

    with pd.ExcelWriter(paths["xlsx"]) as writer:
        selected.to_excel(writer, sheet_name="station_selection", index=False)
        thresholds.to_excel(writer, sheet_name="thresholds", index=False)
        trend_summary.to_excel(writer, sheet_name="trend_summary", index=False)
        annual_summary.to_excel(writer, sheet_name="annual_summary", index=False)
        regional_trend_summary.to_excel(writer, sheet_name="regional_trend_sum", index=False)
        regional_annual_summary.to_excel(writer, sheet_name="regional_annual_sum", index=False)
        regional_trend_paired.to_excel(writer, sheet_name="regional_trends", index=False)
        regional_annual.to_excel(writer, sheet_name="regional_series", index=False)
        trend_paired.to_excel(writer, sheet_name="paired_trends", index=False)
        # Excel row limits may be exceeded for annual_paired, so write only if moderate.
        if len(annual_paired) < 900000:
            annual_paired.to_excel(writer, sheet_name="paired_annual", index=False)

    metadata = {
        "script": Path(__file__).name,
        "period": f"{args.start_year}-{args.end_year}",
        "baseline": f"{args.baseline_start}-{args.baseline_end}",
        "min_hourly_per_day": args.min_hourly_per_day,
        "min_valid_day_fraction": args.min_valid_day_fraction,
        "min_valid_years_threshold": args.min_valid_years_threshold,
        "min_valid_years_trend": args.min_valid_years_trend,
        "min_event_duration": args.min_event_duration,
        "min_stations_per_region": args.min_stations_per_region,
        "pairwise_outlier_qc": {
            "enabled": not args.disable_outlier_qc,
            "mad_k": args.outlier_mad_k,
            "max_fraction_per_station_variable": args.outlier_max_fraction,
            "min_pairs": args.outlier_min_pairs,
            "min_abs_thresholds": args.outlier_min_abs,
            "description": "Conservative station-date, station-variable ERA5–INMET residual QC applied after station selection and common-period enforcement and before threshold and event-metric calculation."
        },
        "regional_validation": "Annual regional medians were calculated from selected stations by Brazil and macro-region, using the same station set and nearest ERA5 grid cells. Sparse regions are plotted when at least min_stations_per_region stations are available, but one-station regional diagnostics should be interpreted cautiously.",
        "heatwave_definitions": {
            "HW": "Tmean >= station/source-specific P90 for at least min_event_duration consecutive days",
            "HHW": "Twbmax >= station/source-specific P95 for at least min_event_duration consecutive days",
            "DHW": "Tmax >= station/source-specific P95 and VPDmean >= station/source-specific P75 for at least min_event_duration consecutive days",
        },
        "intensity_definition": {
            "main_intensity_metric": "Annual accumulated event-day severity, calculated as the sum of daily severity during persistent event days. This is aligned with the primary Figure 01 ERA5 workflow.",
            "HW_intensity": "sum((Tmean - station/source-specific P90)+) during persistent HW days [deg C day]",
            "HHW_intensity": "sum((Twbmax - station/source-specific P95)+) during persistent HHW days [deg C day]",
            "DHW_intensity": "sum(0.5*z+(Tmax) + 0.5*z+(VPDmean)) during persistent DHW days [dimensionless accumulated severity]",
            "mean_event_day_severity": "Auxiliary diagnostic only: mean daily severity during persistent event days; not the main DHW-HHW intensity metric."
        },
        "interpretation": "ERA5–INMET evaluation of trends and annual heatwave metrics. ERA5 is interpreted as a grid-scale thermodynamic diagnostic. Main heatwave intensity is annual accumulated event-day severity; mean_event_day_severity is retained only as an auxiliary diagnostic.",
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    for p in paths.values():
        print(f"[OK] Saved: {p}")

    plot_validation_summary(trend_summary, trend_paired, annual_summary, selected, args.shp_uf, args.out_dir, args.start_year, args.end_year)
    plot_regional_validation_summary(
        regional_annual_summary=regional_annual_summary,
        regional_trend_paired=regional_trend_paired,
        regional_trend_summary=regional_trend_summary,
        out_dir=args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )

    print("\n[SUMMARY] Selected stations:", int(selected["selected"].sum()))
    key = trend_summary[trend_summary["diagnostic"].isin(["Tmax", "VPDmean", "RHmean", "Twbmax", "HW_frequency", "HHW_frequency", "DHW_frequency"])]
    for _, r in key.iterrows():
        print(
            f"  {r['diagnostic']:<16} n={int(r['n_stations']):3d} "
            f"r={r['pearson_r']:+.2f} sign={r['sign_agreement_percent']:.1f}% "
            f"bias={r['bias_era5_minus_inmet']:+.3f}"
        )

    print("\n[SUMMARY] Regional annual-series correlations, Brazil aggregate:")
    rb = regional_annual_summary[
        (regional_annual_summary["validation_region"] == "Brazil")
        & (regional_annual_summary["diagnostic"].isin(["Tmax", "Tmean", "VPDmean", "RHmean", "Twbmax", "DHW_frequency", "DHW_duration", "DHW_intensity"]))
    ]
    for _, r in rb.iterrows():
        print(
            f"  {r['diagnostic']:<16} n_years={int(r['n_years']):2d} "
            f"annual r={r['pearson_r']:+.2f} bias={r['bias_era5_minus_inmet']:+.3f}"
        )
    print("[DONE] ERA5-INMET trend and heatwave-metric evaluation completed.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controlled diagnostics for background warming and climatological aridity.

Purpose
-------
Evaluate whether the spatial association between cumulative transformed/
non-native land fraction and the common-scale DHW–HHW contrast remains after
accounting for background warming, climatological aridity, biome structure,
and broad spatial gradients.

Main models
-----------
M1: DHW–HHW   ~ transformed land + biome + latitude + longitude
M2: DHW–HHW   ~ transformed land + Tmean trend + biome + latitude + longitude
M3: DHW–HHW   ~ transformed land + VPD climatology + biome + latitude + longitude
M4: VPD trend ~ transformed land + Tmean trend + biome + latitude + longitude
M5: DHW–HHW   ~ transformed land + Tmean trend + VPD climatology
                + biome + latitude + longitude

The predictor of interest is standardised, and weighted least squares (WLS)
uses weights proportional to the square root of municipal area. HC3
heteroscedasticity-robust standard errors are used.

These models quantify conditional spatial associations and are not interpreted
as causal attribution or mediation analyses.

The DHW–HHW response must be the common-scale Tmax-only contrast propagated by
Figures 01–03; primary regime-specific DHW and HHW intensity metrics are not
subtracted.

Usage
-----
python controlled_background_warming_aridity.py \
    --figure3-municipality-table /path/to/figure_03_municipality_mechanism_table_1990_2024.csv \
    --figure3-annual-metrics /path/to/figure_03_ERA5_annual_mechanism_metrics_1990_2024.nc \
    --figure1-daily-dir /path/to/figure_01_outputs/cache/era5_daily \
    --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
    --output-dir ./outputs/controlled_background_warming_aridity

Use ``--overwrite`` to rebuild cached covariates and municipality tables.
"""

import os
import re
import json
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm


# ============================================================
# Runtime paths and period
# ============================================================

FIG3_MUNICIPALITY_TABLE = None
FIG3_ANNUAL_MECHANISM_NC = None
DAILY_DIR = None
MUNICIPALITY_SHP = None

YEAR0, YEAR1 = 1990, 2024
BASELINE0, BASELINE1 = 1991, 2020
MIN_YEARS_FOR_TREND = 20
SOFTWARE_VERSION = "1.2.0"

REGION_ORDER = [
    "Amazônia", "Cerrado", "MATOPIBA", "Semiárido",
    "Sudeste urbano", "Pantanal", "Mata Atlântica", "Pampas",
]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).upper()


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


def find_var(ds, candidates):
    for c in candidates:
        if c in ds.data_vars:
            return c
    lowered = {v.lower(): v for v in ds.data_vars}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    normalized = {normalize_text(v): v for v in ds.data_vars}
    for c in candidates:
        nc = normalize_text(c)
        if nc in normalized:
            return normalized[nc]
    return None


def standardize_lat_lon(ds):
    rename = {}
    for c in ["latitude", "Latitude", "LAT"]:
        if c in ds.coords or c in ds.dims:
            rename[c] = "lat"
            break
    for c in ["longitude", "Longitude", "LON"]:
        if c in ds.coords or c in ds.dims:
            rename[c] = "lon"
            break
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords and float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds


def maybe_kelvin_to_celsius(da):
    med = float(np.nanmedian(da.values))
    if np.isfinite(med) and med > 100:
        return da - 273.15
    return da


def reduce_to_lat_lon(da):
    keep = {"time", "lat", "lon"}
    for dim in list(da.dims):
        if dim not in keep:
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0})
            else:
                da = da.mean(dim=dim, skipna=True)
    return da


def theil_sen_slope_decade(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < MIN_YEARS_FOR_TREND:
        return np.nan
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-12:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return np.nan
    return float(np.nanmedian(slopes) * 10.0)


def grid_trend_decade(ds_ann, var):
    years = ds_ann["year"].values.astype(float)
    arr = ds_ann[var].values
    _, nlat, nlon = arr.shape
    slope = np.full((nlat, nlon), np.nan, dtype=np.float32)
    for i in range(nlat):
        if i % 20 == 0:
            print(f"       {var}: lat row {i+1}/{nlat}")
        for j in range(nlon):
            slope[i, j] = theil_sen_slope_decade(years, arr[:, i, j])
    return xr.DataArray(slope, coords={"lat": ds_ann["lat"], "lon": ds_ann["lon"]}, dims=("lat", "lon"))


def find_municipality_shapefile():
    """Return the configured municipality shapefile."""
    if MUNICIPALITY_SHP and os.path.isfile(MUNICIPALITY_SHP):
        return MUNICIPALITY_SHP
    raise FileNotFoundError(f"Municipality shapefile not found: {MUNICIPALITY_SHP}")


# ============================================================
# Build gridded covariates
# ============================================================

def process_daily_tmean(path, year):
    ds = xr.open_dataset(path)
    ds = standardize_lat_lon(ds)
    tvar = find_var(ds, [
        "Tmean", "tmean", "T2M", "t2m", "T2", "2t", "temperature", "temp", "var167"
    ])
    if tvar is None:
        raise ValueError(f"Could not identify Tmean/T2m in {path}. Available: {list(ds.data_vars)}")
    t = maybe_kelvin_to_celsius(reduce_to_lat_lon(ds[tvar]))
    if "time" in t.dims:
        tmean_annual = t.mean("time", skipna=True)
    else:
        tmean_annual = t
    out = xr.Dataset({"Tmean": tmean_annual}).assign_coords(year=year)
    ds.close()
    return out


def build_annual_tmean_dataset(out_dir, overwrite=False):
    out_nc = os.path.join(out_dir, "Supplementary_ERA5_annual_Tmean_1990_2024.nc")
    if os.path.exists(out_nc) and not overwrite:
        ds = xr.open_dataset(out_nc)
        if "Tmean" in ds.data_vars:
            print(f"[SKIP] Annual Tmean file exists: {out_nc}")
            return ds
        ds.close()

    annual = []
    for year in range(YEAR0, YEAR1 + 1):
        path = os.path.join(DAILY_DIR, f"ERA5_daily_Brazil_{year}.nc")
        if not os.path.exists(path):
            print(f"[WARN] Missing daily ERA5 file for Tmean: {path}")
            continue
        print(f"[INFO] Reading annual Tmean from {os.path.basename(path)}")
        annual.append(process_daily_tmean(path, year))

    if len(annual) < MIN_YEARS_FOR_TREND:
        raise RuntimeError(f"Only {len(annual)} annual Tmean files available; need at least {MIN_YEARS_FOR_TREND}.")

    ds_ann = xr.concat(annual, dim="year")
    ds_ann.to_netcdf(out_nc)
    print(f"[OK] Saved annual Tmean: {out_nc}")
    return ds_ann


def build_gridded_covariates(out_dir, overwrite=False):
    out_nc = os.path.join(out_dir, "Supplementary_BackgroundWarming_Aridity_gridded_covariates_1990_2024.nc")
    if os.path.exists(out_nc) and not overwrite:
        ds = xr.open_dataset(out_nc)
        if {"Tmean_trend_decade", "VPD_clim_1991_2020"}.issubset(ds.data_vars):
            print(f"[SKIP] Gridded covariates file exists: {out_nc}")
            return ds
        ds.close()

    if not os.path.exists(FIG3_ANNUAL_MECHANISM_NC):
        raise FileNotFoundError(
            f"Figure 3 annual mechanism file not found: {FIG3_ANNUAL_MECHANISM_NC}. Run Figure 3 first."
        )

    print(f"[INFO] Reading annual VPD from: {FIG3_ANNUAL_MECHANISM_NC}")
    ds_vpd = xr.open_dataset(FIG3_ANNUAL_MECHANISM_NC)
    ds_vpd = standardize_lat_lon(ds_vpd)
    if "VPD" not in ds_vpd.data_vars:
        raise ValueError(f"VPD not found in {FIG3_ANNUAL_MECHANISM_NC}. Available: {list(ds_vpd.data_vars)}")

    vpd_clim = ds_vpd["VPD"].sel(year=slice(BASELINE0, BASELINE1)).mean("year", skipna=True)
    vpd_clim.name = "VPD_clim_1991_2020"
    vpd_clim.attrs["units"] = "kPa"
    vpd_clim.attrs["description"] = "Mean annual VPD over 1991–2020 baseline"

    ds_tmean_ann = build_annual_tmean_dataset(out_dir, overwrite=overwrite)
    ds_tmean_ann = standardize_lat_lon(ds_tmean_ann)

    # Interpolate Tmean annual fields if needed.
    if not (np.array_equal(ds_tmean_ann["lat"], ds_vpd["lat"]) and np.array_equal(ds_tmean_ann["lon"], ds_vpd["lon"])):
        ds_tmean_ann = ds_tmean_ann.interp(lat=ds_vpd["lat"], lon=ds_vpd["lon"], method="nearest")

    print("[INFO] Computing Tmean Theil–Sen trend per decade")
    tmean_trend = grid_trend_decade(ds_tmean_ann, "Tmean")
    tmean_trend.name = "Tmean_trend_decade"
    tmean_trend.attrs["units"] = "°C decade-1"
    tmean_trend.attrs["description"] = "Theil–Sen trend in annual mean 2-m temperature, 1990–2024"

    out = xr.Dataset({
        "Tmean_trend_decade": tmean_trend.astype("float32"),
        "VPD_clim_1991_2020": vpd_clim.astype("float32"),
    })
    out.attrs.update({
        "software_version": SOFTWARE_VERSION,
        "purpose": "Controlled covariates for background warming and climatological aridity",
        "Tmean_trend_period": f"{YEAR0}-{YEAR1}",
        "VPD_climatology_period": f"{BASELINE0}-{BASELINE1}",
    })
    out.to_netcdf(out_nc)
    print(f"[OK] Saved gridded covariates: {out_nc}")
    return out


# ============================================================
# Municipality aggregation
# ============================================================

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
    gdf["rep_lon_geom"] = pts.x
    gdf["rep_lat_geom"] = pts.y
    return gdf


def aggregate_grid_to_municipalities(gdf_mun, ds):
    lon = ds["lon"].values
    lat = ds["lat"].values
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    lon2d, lat2d = np.meshgrid(lon, lat)
    data = {"era5_lat": lat2d.ravel(), "era5_lon": lon2d.ravel()}
    for v in ds.data_vars:
        data[v] = ds[v].values.ravel()

    pts = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["era5_lon"], data["era5_lat"]),
        crs="EPSG:4326",
    )
    keys = ["municipality_norm", "state_acronym_norm"]
    gsmall = gdf_mun[keys + ["rep_lon_geom", "rep_lat_geom", "geometry"]].copy()

    joined = gpd.sjoin(
        pts,
        gsmall[keys + ["geometry"]],
        how="inner",
        predicate="within",
    )
    joined["grid_weight"] = np.cos(
        np.deg2rad(pd.to_numeric(joined["era5_lat"], errors="coerce"))
    )

    rows = []
    for name, g in joined.groupby(keys, dropna=False):
        w = pd.to_numeric(g["grid_weight"], errors="coerce").to_numpy(float)
        row = {
            "municipality_norm": name[0],
            "state_acronym_norm": name[1],
            "supplementary_covariate_method": "zonal_gridpoint_coslat_weighted",
            "supplementary_covariate_n_gridpoints": len(g),
        }
        for v in ds.data_vars:
            values = pd.to_numeric(g[v], errors="coerce").to_numpy(float)
            ok = np.isfinite(values) & np.isfinite(w) & (w > 0)
            row[v] = (
                float(np.average(values[ok], weights=w[ok]))
                if np.any(ok) else np.nan
            )
        rows.append(row)
    zonal = pd.DataFrame(rows)

    missing = gsmall[keys + ["rep_lon_geom", "rep_lat_geom"]].merge(
        zonal[keys], on=keys, how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns="_merge")

    if len(missing) == 0:
        return zonal

    print(f"[INFO] Supplementary covariate fallback for {len(missing)} municipalities")
    tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    query = np.column_stack([missing["rep_lat_geom"].values, missing["rep_lon_geom"].values])
    _, idx = tree.query(query)

    nearest_rows = []
    for k, (_, r) in enumerate(missing.iterrows()):
        flat = idx[k]
        iy, ix = np.unravel_index(flat, lat2d.shape)
        row = {
            "municipality_norm": r["municipality_norm"],
            "state_acronym_norm": r["state_acronym_norm"],
            "supplementary_covariate_method": "nearest_gridpoint_fallback",
            "supplementary_covariate_n_gridpoints": 1,
        }
        for v in ds.data_vars:
            row[v] = float(ds[v].values[iy, ix])
        nearest_rows.append(row)

    return pd.concat([zonal, pd.DataFrame(nearest_rows)], ignore_index=True)


def build_augmented_municipality_table(out_dir, overwrite=False):
    out_csv = os.path.join(out_dir, "Supplementary_Table_municipality_with_background_warming_aridity_covariates.csv")
    if os.path.exists(out_csv) and not overwrite:
        d = pd.read_csv(out_csv)
        needed = {"Tmean_trend_decade", "VPD_clim_1991_2020"}
        if needed.issubset(d.columns):
            print(f"[SKIP] Augmented municipality table exists: {out_csv}")
            return d

    if not os.path.exists(FIG3_MUNICIPALITY_TABLE):
        raise FileNotFoundError(f"Figure 3 municipality table not found: {FIG3_MUNICIPALITY_TABLE}. Run Figure 3 first.")

    print(f"[INFO] Reading Figure 03 municipality table: {FIG3_MUNICIPALITY_TABLE}")
    mun = pd.read_csv(FIG3_MUNICIPALITY_TABLE)

    if "dry_minus_humid_intensity_trend" not in mun.columns:
        common = "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
        if common in mun.columns:
            mun["dry_minus_humid_intensity_trend"] = pd.to_numeric(
                mun[common],
                errors="coerce",
            )
        else:
            raise ValueError(
                "Figure 03 municipality table is missing the common-scale "
                "DHW–HHW contrast. Primary DHW and HHW regime-specific "
                "intensity trends are not subtracted."
            )

    gdf = load_municipality_geometry()
    cov_grid = build_gridded_covariates(out_dir, overwrite=overwrite)
    cov_mun = aggregate_grid_to_municipalities(gdf, cov_grid)

    out = mun.merge(cov_mun, on=["municipality_norm", "state_acronym_norm"], how="left")

    # Guarantee coordinates for spatial controls.
    coords = gdf[["municipality_norm", "state_acronym_norm", "rep_lat_geom", "rep_lon_geom"]].drop_duplicates()
    out = out.merge(coords, on=["municipality_norm", "state_acronym_norm"], how="left")
    if "rep_lat" not in out.columns:
        out["rep_lat"] = out["rep_lat_geom"]
    else:
        out["rep_lat"] = pd.to_numeric(out["rep_lat"], errors="coerce").fillna(out["rep_lat_geom"])
    if "rep_lon" not in out.columns:
        out["rep_lon"] = out["rep_lon_geom"]
    else:
        out["rep_lon"] = pd.to_numeric(out["rep_lon"], errors="coerce").fillna(out["rep_lon_geom"])

    if "transformed_non_native_pct_2024" not in out.columns and "native_pct_2024" in out.columns:
        out["transformed_non_native_pct_2024"] = 100.0 - pd.to_numeric(out["native_pct_2024"], errors="coerce")

    out.to_csv(out_csv, index=False)
    print(f"[OK] Saved augmented municipality table: {out_csv}")
    return out


# ============================================================
# Controlled regressions
# ============================================================

def zscore_weighted(d, col, weight_col="weight"):
    x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(d[weight_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.full(len(d), np.nan)
    mu = np.average(x[ok], weights=w[ok])
    sd = np.sqrt(np.average((x[ok] - mu) ** 2, weights=w[ok]))
    if not np.isfinite(sd) or sd <= 0:
        return np.full(len(d), np.nan)
    return (x - mu) / sd


def fit_model(df, model_id, response, predictor, controls, description):
    d = df.copy()
    required = [response, predictor, "total_area_2024_ha"] + controls
    missing = [c for c in required if c not in d.columns]
    if missing:
        return {
            "Model": model_id,
            "Response": response,
            "Predictor of interest": predictor,
            "Controls": " + ".join(controls),
            "Description": description,
            "n": 0,
            "status": f"missing columns: {missing}",
        }

    # Domain aligned with Figure 3/4 regional analyses.
    if "focus_region" in d.columns:
        d = d[d["focus_region"].isin(REGION_ORDER)].copy()

    # WLS weights are proportional to the square root of municipal area.
    d["weight"] = np.sqrt(pd.to_numeric(d["total_area_2024_ha"], errors="coerce"))
    d["weight"] = d["weight"] / np.nanmean(d["weight"])

    # Convert continuous fields.
    numeric_cols = [response, predictor, "weight", "rep_lat", "rep_lon", "Tmean_trend_decade", "VPD_clim_1991_2020"]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Z-score predictor of interest and continuous covariates.
    z_predictor = predictor + "_z"
    d[z_predictor] = zscore_weighted(d, predictor)

    formula_terms = [z_predictor]
    used_controls = []
    for c in controls:
        if c == "dominant_biome":
            formula_terms.append("C(dominant_biome)")
            used_controls.append("biome fixed effects")
        elif c == "focus_region":
            formula_terms.append("C(focus_region)")
            used_controls.append("focus-region fixed effects")
        elif c in ["rep_lat", "rep_lon", "Tmean_trend_decade", "VPD_clim_1991_2020"]:
            zc = c + "_z"
            d[zc] = zscore_weighted(d, c)
            formula_terms.append(zc)
            used_controls.append(c + " (z)")
        else:
            formula_terms.append(c)
            used_controls.append(c)

    subset_cols = [response, z_predictor, "weight"]
    for c in controls:
        if c == "dominant_biome":
            subset_cols.append("dominant_biome")
        elif c == "focus_region":
            subset_cols.append("focus_region")
        elif c in ["rep_lat", "rep_lon", "Tmean_trend_decade", "VPD_clim_1991_2020"]:
            subset_cols.append(c + "_z")
        else:
            subset_cols.append(c)
    d = d.dropna(subset=list(dict.fromkeys(subset_cols))).copy()
    d = d[d["weight"] > 0].copy()

    if len(d) < 80:
        return {
            "Model": model_id,
            "Response": response,
            "Predictor of interest": predictor,
            "Controls": " + ".join(used_controls),
            "Description": description,
            "n": len(d),
            "status": "not fitted: insufficient observations",
        }

    formula = f"{response} ~ " + " + ".join(formula_terms)

    try:
        fit = sm.WLS.from_formula(formula, data=d, weights=d["weight"]).fit(cov_type="HC3")
        ci = fit.conf_int()
        return {
            "Model": model_id,
            "Response": response,
            "Predictor of interest": predictor,
            "Controls": " + ".join(used_controls),
            "Description": description,
            "Formula": formula,
            "n": int(fit.nobs),
            "Beta_per_1sd_predictor": float(fit.params.get(z_predictor, np.nan)),
            "SE_HC3": float(fit.bse.get(z_predictor, np.nan)),
            "t_HC3": float(fit.tvalues.get(z_predictor, np.nan)),
            "p_HC3": float(fit.pvalues.get(z_predictor, np.nan)),
            "CI95_low": float(ci.loc[z_predictor, 0]) if z_predictor in ci.index else np.nan,
            "CI95_high": float(ci.loc[z_predictor, 1]) if z_predictor in ci.index else np.nan,
            "R2": float(fit.rsquared),
            "Adj_R2": float(fit.rsquared_adj),
            "AIC": float(fit.aic),
            "BIC": float(fit.bic),
            "status": "fitted",
            "Interpretation": "Conditional spatial association; not causal attribution or mediation.",
        }
    except Exception as exc:
        return {
            "Model": model_id,
            "Response": response,
            "Predictor of interest": predictor,
            "Controls": " + ".join(used_controls),
            "Description": description,
            "Formula": formula,
            "n": len(d),
            "status": f"fit failed: {exc}",
        }


def run_models(mun):
    model_specs = [
        (
            "M1",
            "dry_minus_humid_intensity_trend",
            "transformed_non_native_pct_2024",
            ["dominant_biome", "rep_lat", "rep_lon"],
            "Base spatial-biome control model",
        ),
        (
            "M2",
            "dry_minus_humid_intensity_trend",
            "transformed_non_native_pct_2024",
            ["dominant_biome", "rep_lat", "rep_lon", "Tmean_trend_decade"],
            "Controls for background warming",
        ),
        (
            "M3",
            "dry_minus_humid_intensity_trend",
            "transformed_non_native_pct_2024",
            ["dominant_biome", "rep_lat", "rep_lon", "VPD_clim_1991_2020"],
            "Controls for climatological aridity",
        ),
        (
            "M4",
            "VPD_trend_decade",
            "transformed_non_native_pct_2024",
            ["dominant_biome", "rep_lat", "rep_lon", "Tmean_trend_decade"],
            "Association between transformed land and VPD trend after controlling for background warming",
        ),
        (
            "M5",
            "dry_minus_humid_intensity_trend",
            "transformed_non_native_pct_2024",
            ["dominant_biome", "rep_lat", "rep_lon", "Tmean_trend_decade", "VPD_clim_1991_2020"],
            "Conservative model controlling for both warming and climatological aridity",
        ),
    ]
    rows = [fit_model(mun, *spec) for spec in model_specs]
    return pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

def _format_pvalue(p):
    try:
        p = float(p)
    except Exception:
        return "p=NA"
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-99:
        return "p<1e−99"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _model_control_label(model_id):
    mapping = {
        "M1": "M1  baseline controls",
        "M2": "M2  + background warming",
        "M3": "M3  + VPD climatology",
        "M5": "M5  + warming and VPD climatology",
        "M4": "M4  + background warming",
    }
    return mapping.get(str(model_id), str(model_id))


def _forest_panel(ax, data, panel_title, xlabel, xlim=None):
    """
    Forest-style coefficient plot for one response variable.

    This function is intentionally split by response variable because M1/M2/M3/M5
    have response units of severity decade-1, whereas M4 has response units of
    kPa decade-1. Plotting them on the same x-axis would be visually misleading.
    """
    data = data.copy()
    if data.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No fitted models available", ha="center", va="center", fontsize=10)
        return

    numeric_cols = ["Beta_per_1sd_predictor", "CI95_low", "CI95_high", "p_HC3"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Beta_per_1sd_predictor", "CI95_low", "CI95_high"])
    if data.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No valid coefficient intervals available", ha="center", va="center", fontsize=10)
        return

    y = np.arange(len(data))[::-1]
    x = data["Beta_per_1sd_predictor"].to_numpy(float)
    lo = data["CI95_low"].to_numpy(float)
    hi = data["CI95_high"].to_numpy(float)

    for yi, xi, xlo, xhi in zip(y, x, lo, hi):
        color = "#B2182B" if xi >= 0 else "#2166AC"
        ax.plot([xlo, xhi], [yi, yi], color=color, lw=4.0, alpha=0.24,
                solid_capstyle="round", zorder=2)
        ax.scatter(xi, yi, s=78, color=color, edgecolor="white",
                   linewidth=0.9, zorder=3)

    ax.axvline(0, color="0.25", linestyle="--", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_model_control_label(r.Model) for _, r in data.iterrows()], fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(panel_title, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)

    if xlim is None:
        xmin = float(np.nanmin(np.r_[lo, [0.0]]))
        xmax = float(np.nanmax(np.r_[hi, [0.0]]))
        xrng = xmax - xmin if np.isfinite(xmax - xmin) and (xmax - xmin) > 0 else 1.0
        xlim = (xmin - 0.08 * xrng, xmax + 0.30 * xrng)
    ax.set_xlim(*xlim)

    xmin, xmax = ax.get_xlim()
    xrng = xmax - xmin
    x_text = xmax - 0.015 * xrng

    for yi, (_, r) in zip(y, data.iterrows()):
        beta = float(r["Beta_per_1sd_predictor"])
        ptxt = _format_pvalue(r.get("p_HC3", np.nan))
        label = f"β={beta:.2f}; {ptxt}"
        ax.text(x_text, yi, label, ha="right", va="center",
                fontsize=8.7, color="0.22")


def plot_coefficients(models, out_dir):
    d = models[models["status"].eq("fitted")].copy()
    if d.empty:
        print("[WARN] No fitted models for coefficient plot.")
        return

    # Separate responses to avoid mixing coefficients with different units.
    dry_order = ["M1", "M2", "M3", "M5"]
    vpd_order = ["M4"]

    dry = d[d["Model"].isin(dry_order)].copy()
    vpd = d[d["Model"].isin(vpd_order)].copy()

    dry["plot_order"] = dry["Model"].map({m: i for i, m in enumerate(dry_order)})
    vpd["plot_order"] = vpd["Model"].map({m: i for i, m in enumerate(vpd_order)})
    dry = dry.sort_values("plot_order")
    vpd = vpd.sort_values("plot_order")

    # Vertical layout is more readable for a supplementary figure because panel b
    # contains a single model with a different response unit.
    fig = plt.figure(figsize=(9.4, 7.4))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[3.8, 1.35],
        hspace=0.55,
        left=0.22,
        right=0.97,
        top=0.88,
        bottom=0.20,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    _forest_panel(
        ax1,
        dry,
        "(a) DHW−HHW contrast",
        "β for transformed/non-native land fraction\n"
        "(severity decade$^{-1}$ per 1 s.d. increase in predictor)",
    )

    _forest_panel(
        ax2,
        vpd,
        "(b) Atmospheric drying",
        "β for transformed/non-native land fraction\n"
        "(kPa decade$^{-1}$ per 1 s.d. increase in predictor)",
    )

    fig.suptitle(
        "Supplementary Fig. X | Controlled associations accounting for warming and climatological aridity",
        fontsize=13.5,
        fontweight="bold",
        y=0.965,
    )

    foot = (
        "WLS models weighted by √municipality area with HC3 standard errors. "
        "All models include biome fixed effects and latitude/longitude controls. "
        "M2 and M4 additionally control for Tmean trend; M3 controls for 1991–2020 VPD climatology; "
        "M5 controls for both Tmean trend and VPD climatology. "
        "Coefficients quantify conditional spatial associations; they are not causal or mediation estimates."
    )
    fig.text(0.22, 0.065, foot, ha="left", va="bottom",
             fontsize=8.2, color="0.25", wrap=True)

    out_base = os.path.join(out_dir, "Supplementary_Figure_Controlled_BackgroundWarming_Aridity_coefficients")
    fig.savefig(out_base + ".jpeg", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved coefficient figure: {out_base}.jpeg/.pdf")


# ============================================================
# Output
# ============================================================

def save_outputs(models, mun, out_dir):
    csv_models = os.path.join(out_dir, "Supplementary_Table_Controlled_BackgroundWarming_Aridity_models.csv")
    xlsx = os.path.join(out_dir, "Supplementary_Tables_Controlled_BackgroundWarming_Aridity.xlsx")
    models.to_csv(csv_models, index=False)

    with pd.ExcelWriter(xlsx) as writer:
        models.to_excel(writer, index=False, sheet_name="controlled_models")
        keep_cols = [
            "municipality_norm", "state_acronym_norm", "municipality", "state_acronym",
            "dominant_biome", "focus_region", "total_area_2024_ha",
            "dry_minus_humid_intensity_trend", "VPD_trend_decade",
            "transformed_non_native_pct_2024", "Tmean_trend_decade", "VPD_clim_1991_2020",
            "rep_lat", "rep_lon", "supplementary_covariate_method", "supplementary_covariate_n_gridpoints",
        ]
        keep_cols = [c for c in keep_cols if c in mun.columns]
        mun[keep_cols].to_excel(writer, index=False, sheet_name="municipality_covariates")

    meta = {
        "software_version": SOFTWARE_VERSION,
        "purpose": "Controlled associations accounting for background warming and climatological aridity",
        "models": ["M1", "M2", "M3", "M4", "M5"],
        "interpretation": "Conditional spatial association; not causal attribution or mediation.",
        "weights": "proportional to sqrt(total_area_2024_ha), normalized by mean",
        "standard_errors": "HC3 heteroscedasticity-robust standard errors",
        "Tmean_trend_period": f"{YEAR0}-{YEAR1}",
        "VPD_climatology_period": f"{BASELINE0}-{BASELINE1}",
    }
    with open(os.path.join(out_dir, "software_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved model table: {csv_models}")
    print(f"[OK] Saved workbook: {xlsx}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build controlled association diagnostics for background warming "
            "and climatological aridity."
        )
    )
    parser.add_argument(
        "--figure3-municipality-table",
        required=True,
        help="Figure 03 municipality-level CSV.",
    )
    parser.add_argument(
        "--figure3-annual-metrics",
        required=True,
        help="Figure 03 annual ERA5 atmospheric-metrics NetCDF.",
    )
    parser.add_argument(
        "--figure1-daily-dir",
        required=True,
        help="Directory containing ERA5_daily_Brazil_<year>.nc files.",
    )
    parser.add_argument(
        "--municipality-shapefile",
        required=True,
        help="Brazilian municipality shapefile.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cached covariates and municipality tables.",
    )
    args = parser.parse_args()

    global FIG3_MUNICIPALITY_TABLE, FIG3_ANNUAL_MECHANISM_NC
    global DAILY_DIR, MUNICIPALITY_SHP

    FIG3_MUNICIPALITY_TABLE = str(
        Path(args.figure3_municipality_table).expanduser().resolve()
    )
    FIG3_ANNUAL_MECHANISM_NC = str(
        Path(args.figure3_annual_metrics).expanduser().resolve()
    )
    DAILY_DIR = str(Path(args.figure1_daily_dir).expanduser().resolve())
    MUNICIPALITY_SHP = str(
        Path(args.municipality_shapefile).expanduser().resolve()
    )
    out_dir = str(Path(args.output_dir).expanduser().resolve())

    required = [
        (FIG3_MUNICIPALITY_TABLE, "Figure 03 municipality table"),
        (FIG3_ANNUAL_MECHANISM_NC, "Figure 03 annual metrics"),
        (DAILY_DIR, "Figure 01 daily-cache directory"),
        (MUNICIPALITY_SHP, "municipality shapefile"),
    ]
    for path, label in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    ensure_dir(out_dir)

    print("[START] Supplementary controlled diagnostics: background warming and climatological aridity")
    mun = build_augmented_municipality_table(out_dir, overwrite=args.overwrite)
    models = run_models(mun)
    save_outputs(models, mun, out_dir)
    plot_coefficients(models, out_dir)

    print("[SUMMARY]")
    cols = ["Model", "Response", "Predictor of interest", "Controls", "Beta_per_1sd_predictor", "SE_HC3", "p_HC3", "n", "R2", "status"]
    cols = [c for c in cols if c in models.columns]
    print(models[cols].to_string(index=False))
    print("[DONE] Supplementary controlled diagnostics completed.")
    print(f"[INFO] Outputs saved in: {out_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary hotspot-threshold sensitivity for the common-scale DHW−HHW
Tmax-only intensity-trend contrast.

Purpose
-------
Evaluate the robustness of the diagnostic hotspot definition based on the
common-scale DHW−HHW contrast:

    DHW_minus_HHW_TmaxOnly_intensity_trend_decade

or, when the direct contrast is unavailable:

    DHW_TmaxOnly_intensity_trend_decade
    - HHW_TmaxOnly_intensity_trend_decade

Primary regime-specific DHW and HHW intensity metrics are not subtracted
because they are not directly commensurate.

Hotspot sensitivity
-------------------
The primary diagnostic cutoff is:

    DHW−HHW > 4 Tmax-standardized severity decade^-1

Sensitivity is evaluated using >3, >5, the upper quartile, and the upper decile
of the common-scale contrast over valid Brazilian land grid cells.

When available, an additional condition is evaluated using:

    DHW_TmaxOnly_intensity_trend_decade > 0
    and DHW_TmaxOnly_intensity_pvalue <= 0.05

This significance condition applies only to the DHW Tmax-only trend. No
standalone p-value is assigned to the DHW−HHW difference of slopes.

Usage
-----
python supplementary_hotspot_threshold_sensitivity.py \
    --trend-nc /path/to/figure_01_heatwave_trends_1990_2024.nc \
    --states-shapefile /path/to/brazil_states.shp \
    --biomes-shapefile /path/to/brazil_biomes.shp \
    --output-dir ./outputs/hotspot_threshold_sensitivity
"""

import os
import re
import glob
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, ListedColormap
from shapely.geometry import Point
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

SCRIPT_VERSION = "1.0.0"
EARTH_RADIUS_KM = 6371.0
REGION_ORDER = ["Brazil", "North", "Northeast", "Central-West", "Southeast", "South"]
UF_TO_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}
BRAZIL_EXTENT = [-75.5, -32.0, -35.5, 6.5]

# Publication-figure font configuration.
# All panel titles, axis labels, tick labels, legends, colorbars and annotations
# are set to 12 pt.
FONT_SIZE = 12
TITLE_SIZE = 12
LABEL_SIZE = 12
TICK_SIZE = 12
LEGEND_SIZE = 12
ANNOT_SIZE = 12

# State-label rendering. "selected" adds small UF labels only to states that
# tend to be visually subtle in the raster maps. This is a visual QC aid and
# does not alter any calculation.
DEFAULT_UF_LABEL_MODE = "none"   # one of: "none", "selected", "all"
DEFAULT_SELECTED_UFS = []


plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": TITLE_SIZE,
})


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_name(s) -> str:
    s = str(s).strip().lower()
    repl = {
        "á": "a", "à": "a", "â": "a", "ã": "a", "ä": "a",
        "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o", "õ": "o",
        "ú": "u", "ç": "c", "º": "", "ª": "",
        "β": "beta", "–": "_", "—": "_", "−": "_", "-": "_",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def open_dataset_robust(path: str) -> xr.Dataset:
    """
    Open the Figure 01 trend NetCDF using a sequence of xarray backends.

    The product contains static spatial trend fields, so CF time decoding is
    disabled.
    """
    attempts = [
        ("h5netcdf", {"phony_dims": "sort"}, {"decode_times": False, "mask_and_scale": True}),
        ("h5netcdf", {"phony_dims": "sort"}, {"decode_times": False, "mask_and_scale": False}),
        ("scipy", {}, {"decode_times": False, "mask_and_scale": True}),
        (None, {}, {"decode_times": False, "mask_and_scale": True}),
    ]

    errors = []
    for engine, backend_kwargs, open_kwargs in attempts:
        try:
            if engine is None:
                ds = xr.open_dataset(path, **open_kwargs)
                used = "default"
            else:
                ds = xr.open_dataset(
                    path,
                    engine=engine,
                    backend_kwargs=backend_kwargs,
                    **open_kwargs,
                )
                used = engine
            print(
                f"[INFO] Opened trend NetCDF with xarray engine='{used}' "
                f"(decode_times=False): {Path(path).name}"
            )
            return ds
        except Exception as exc:
            errors.append(
                f"{engine or 'default'}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Could not open Figure 01 trend NetCDF with available "
        "backends. This is an I/O/backend problem, not a scientific-method "
        "problem.\n"
        f"File: {path}\nAttempts:\n" + "\n".join(errors)
    )


def standardize_grid(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
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
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(f"Could not find lat/lon coordinates. Coords={list(ds.coords)} dims={list(ds.dims)}")
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds


def has_transition_or_components(ds: xr.Dataset) -> Tuple[bool, str]:
    """Check whether the dataset contains common-scale Tmax-only fields."""
    vars_ = set(ds.data_vars)
    direct = "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    dhw = "DHW_TmaxOnly_intensity_trend_decade"
    hhw = "HHW_TmaxOnly_intensity_trend_decade"
    if direct in vars_:
        return True, f"direct common-scale contrast: {direct}"
    if dhw in vars_ and hhw in vars_:
        return True, f"common-scale components: {dhw} - {hhw}"
    return False, "missing common-scale Tmax-only contrast/components"


def select_component_var(ds: xr.Dataset, explicit: Optional[str], role: str, required: bool = False) -> Optional[str]:
    expected = {
        "hhw_trend": "HHW_TmaxOnly_intensity_trend_decade",
        "hhw_pvalue": "HHW_TmaxOnly_intensity_pvalue",
    }
    if explicit:
        if explicit not in ds.data_vars:
            raise KeyError(f"Explicit {role} variable not found: {explicit}. Available: {list(ds.data_vars)}")
        return explicit
    name = expected.get(role)
    if name and name in ds.data_vars:
        print(f"[INFO] Selected {role} variable: {name}")
        return name
    if required:
        raise KeyError(f"Required {role} variable not found. Expected: {name}. Available: {list(ds.data_vars)}")
    return None


def open_trend_dataset(path: str, search_dir: str = "") -> Tuple[xr.Dataset, str]:
    """Open and validate the explicitly supplied Figure 01 trend dataset."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)

    ds = standardize_grid(open_dataset_robust(path))
    ok, reason = has_transition_or_components(ds)
    print(f"[INFO] Trend dataset check: {reason}")
    print(f"[INFO] Available variables: {list(ds.data_vars)}")

    if not ok:
        ds.close()
        raise RuntimeError(
            "The supplied Figure 01 trend dataset does not contain the "
            "common-scale DHW−HHW contrast or both Tmax-only component trends."
        )
    return ds, path

def select_var(ds: xr.Dataset, explicit: Optional[str], role: str, required: bool = True) -> Optional[str]:
    expected = {
        "transition": "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "dhw_trend": "DHW_TmaxOnly_intensity_trend_decade",
        "dhw_pvalue": "DHW_TmaxOnly_intensity_pvalue",
    }
    if explicit:
        if explicit not in ds.data_vars:
            raise KeyError(f"Explicit {role} variable not found: {explicit}. Available: {list(ds.data_vars)}")
        return explicit
    name = expected.get(role)
    if name and name in ds.data_vars:
        print(f"[INFO] Selected {role} variable: {name}")
        return name
    if required:
        raise KeyError(f"Could not find {role} variable. Expected: {name}. Available variables: {list(ds.data_vars)}")
    print(f"[WARN] Optional {role} variable not found. Related sensitivity will be skipped.")
    return None


def read_states(path: str) -> gpd.GeoDataFrame:
    """Read Brazilian state boundaries and robustly assign official UF codes.

    The function first searches for exact two-letter UF codes and then falls
    back to full state-name mapping. UF codes are never inferred from the first
    two letters of state names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Repair invalid geometries conservatively for plotting/spatial joins.
    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    valid_ufs = set(UF_TO_REGION.keys())

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

    preferred_uf_cols = [
        "uf", "UF", "sigla", "SIGLA", "sigla_uf", "SIGLA_UF",
        "cd_uf", "CD_UF", "uf_sigla", "UF_SIGLA"
    ]

    # 1) Prefer columns that actually contain exact two-letter UF codes.
    chosen_col = None
    best_count = -1
    best_vals = None

    candidate_columns = []
    for c in preferred_uf_cols:
        if c in gdf.columns and c not in candidate_columns:
            candidate_columns.append(c)
    for c in gdf.columns:
        if c != "geometry" and c not in candidate_columns:
            candidate_columns.append(c)

    for c in candidate_columns:
        vals = gdf[c].astype(str).str.strip().str.upper()
        # Exact UF code only. Do not extract first two letters from longer names.
        exact = vals.where(vals.str.fullmatch(r"[A-Z]{2}", na=False))
        count = int(exact.isin(valid_ufs).sum())
        if count > best_count:
            best_count = count
            best_vals = exact
            chosen_col = c

    if best_count >= 20:
        gdf["uf"] = best_vals
        print(f"[INFO] UF codes identified from exact-code column: {chosen_col}")
    else:
        # 2) Fallback: map full state names to UF codes using normalized names.
        chosen_name_col = None
        best_count = -1
        best_mapped = None

        for c in candidate_columns:
            vals_norm = gdf[c].astype(str).map(normalize_name)
            mapped = vals_norm.map(state_name_to_uf_norm)
            count = int(mapped.isin(valid_ufs).sum())
            if count > best_count:
                best_count = count
                best_mapped = mapped
                chosen_name_col = c

        if best_count >= 20:
            gdf["uf"] = best_mapped
            print(f"[INFO] UF codes mapped from state-name column: {chosen_name_col}")
        else:
            print("[ERROR] Could not robustly identify Brazilian UF/state codes.")
            print(f"[ERROR] Columns available: {list(gdf.columns)}")
            for c in gdf.columns:
                if c != "geometry":
                    vals = gdf[c].astype(str).head(10).tolist()
                    print(f"  sample {c}: {vals}")
            raise ValueError(
                "Could not identify UF/state column. Provide a shapefile with either "
                "two-letter UF codes or full Brazilian state names."
            )

    gdf = gdf[gdf["uf"].isin(valid_ufs)].copy()
    gdf["region"] = gdf["uf"].map(UF_TO_REGION)

    out = gdf[["uf", "region", "geometry"]].dissolve(by=["uf", "region"], as_index=False)

    found = sorted(out["uf"].astype(str).unique().tolist())
    missing = sorted(set(UF_TO_REGION) - set(found))
    print(f"[INFO] State polygons detected in shapefile: {len(found)}/27")
    print(f"[INFO] UFs detected: {', '.join(found)}")
    if missing:
        print(f"[WARN] Missing UF polygons in supplied shapefile: {', '.join(missing)}")
        print("[WARN] Use a complete state-boundary shapefile if these limits must appear.")
    return out

def make_brazil_mask(lats: np.ndarray, lons: np.ndarray, states: gpd.GeoDataFrame) -> np.ndarray:
    # Uses point-in-polygon on grid-cell centers. For ~10-20k Brazil cells this is fast enough.
    lon2d, lat2d = np.meshgrid(lons, lats)
    pts = gpd.GeoDataFrame(
        {"iy": np.repeat(np.arange(len(lats)), len(lons)), "ix": np.tile(np.arange(len(lons)), len(lats))},
        geometry=[Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())],
        crs="EPSG:4326",
    )
    brazil_poly = states.dissolve().reset_index(drop=True)
    joined = gpd.sjoin(pts, brazil_poly[["geometry"]], predicate="within", how="inner")
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    mask[joined["iy"].values, joined["ix"].values] = True
    return mask


def cell_area_km2(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    # Approximate spherical quadrilateral area for regular lat/lon centers.
    if len(lats) > 1:
        dlat = float(np.nanmedian(np.abs(np.diff(lats))))
    else:
        dlat = 0.25
    if len(lons) > 1:
        dlon = float(np.nanmedian(np.abs(np.diff(lons))))
    else:
        dlon = 0.25
    lat_edges1 = np.radians(lats - dlat / 2.0)
    lat_edges2 = np.radians(lats + dlat / 2.0)
    dlon_rad = math.radians(dlon)
    row_area = (EARTH_RADIUS_KM ** 2) * dlon_rad * np.abs(np.sin(lat_edges2) - np.sin(lat_edges1))
    return np.repeat(row_area[:, None], len(lons), axis=1)


def grid_points_dataframe(lats, lons, mask, area, transition) -> gpd.GeoDataFrame:
    iy, ix = np.where(mask & np.isfinite(transition))
    df = pd.DataFrame({
        "iy": iy,
        "ix": ix,
        "lat": lats[iy],
        "lon": lons[ix],
        "area_km2": area[iy, ix],
        "transition": transition[iy, ix],
    })
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


def attach_regions(points: gpd.GeoDataFrame, states: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(points, states[["uf", "region", "geometry"]], predicate="within", how="left")
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined["region"] = joined["region"].fillna("Unknown")
    joined["uf"] = joined["uf"].fillna("Unknown")
    return joined


def read_biomes(path: Optional[str]) -> Optional[gpd.GeoDataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] Biome shapefile not found: {path}. Skipping biome distribution.")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    # Identify likely biome-name column.
    biome_col = None
    for c in gdf.columns:
        if c == "geometry":
            continue
        nc = normalize_name(c)
        if any(k in nc for k in ["bioma", "biome", "nome"]):
            biome_col = c
            break
    if biome_col is None:
        # Fallback: first non-geometry string column.
        for c in gdf.columns:
            if c != "geometry" and gdf[c].dtype == object:
                biome_col = c
                break
    if biome_col is None:
        print(f"[WARN] Could not identify biome name column in {path}. Skipping biome distribution.")
        return None
    out = gdf[[biome_col, "geometry"]].copy().rename(columns={biome_col: "biome"})
    out["biome"] = out["biome"].astype(str)
    return out.dissolve(by="biome", as_index=False)


def attach_biomes(points: gpd.GeoDataFrame, biomes: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if biomes is None:
        points["biome"] = "Not evaluated"
        return points
    joined = gpd.sjoin(points, biomes[["biome", "geometry"]], predicate="within", how="left")
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined["biome"] = joined["biome"].fillna("Unknown")
    return joined


def mask_from_point_table(points: gpd.GeoDataFrame, shape: Tuple[int, int], selected: pd.Series) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    sub = points.loc[selected]
    out[sub["iy"].values.astype(int), sub["ix"].values.astype(int)] = True
    return out


def build_criteria(points: gpd.GeoDataFrame, sig_available: bool) -> Dict[str, pd.Series]:
    trans = points["transition"]
    p75 = float(np.nanpercentile(trans, 75))
    p90 = float(np.nanpercentile(trans, 90))
    criteria = {
        "gt3": trans > 3.0,
        "gt4_primary": trans > 4.0,
        "gt5": trans > 5.0,
        "top_quartile": trans >= p75,
        "top_decile": trans >= p90,
    }
    if sig_available and "dhw_sig_positive" in points.columns:
        sig = points["dhw_sig_positive"].astype(bool)
        for key, val in list(criteria.items()):
            criteria[key + "_sigDHW"] = val & sig
    points.attrs["p75_threshold"] = p75
    points.attrs["p90_threshold"] = p90
    return criteria


def summarize_criteria(points: gpd.GeoDataFrame, criteria: Dict[str, pd.Series], primary_key="gt4_primary") -> pd.DataFrame:
    total_area = float(points["area_km2"].sum())
    total_cells = int(len(points))
    primary = criteria[primary_key]
    primary_area = float(points.loc[primary, "area_km2"].sum())
    rows = []
    for key, sel in criteria.items():
        sel = sel.fillna(False).astype(bool)
        area = float(points.loc[sel, "area_km2"].sum())
        cells = int(sel.sum())
        inter = sel & primary
        union = sel | primary
        inter_area = float(points.loc[inter, "area_km2"].sum())
        union_area = float(points.loc[union, "area_km2"].sum())
        rows.append({
            "criterion": key,
            "n_grid_cells": cells,
            "area_km2": area,
            "area_percent_of_brazil_mask": 100.0 * area / total_area if total_area else np.nan,
            "mean_transition": float(points.loc[sel, "transition"].mean()) if cells else np.nan,
            "median_transition": float(points.loc[sel, "transition"].median()) if cells else np.nan,
            "primary_overlap_area_km2": inter_area,
            "primary_overlap_percent_of_primary": 100.0 * inter_area / primary_area if primary_area else np.nan,
            "primary_overlap_percent_of_current": 100.0 * inter_area / area if area else np.nan,
            "jaccard_area_overlap_with_primary": inter_area / union_area if union_area else np.nan,
            "total_brazil_mask_cells": total_cells,
            "total_brazil_mask_area_km2": total_area,
        })
    return pd.DataFrame(rows)


def distribution_table(points: gpd.GeoDataFrame, criteria: Dict[str, pd.Series], group_col: str) -> pd.DataFrame:
    rows = []
    for key, sel in criteria.items():
        sel = sel.fillna(False).astype(bool)
        sub = points.loc[sel].copy()
        total_area = float(sub["area_km2"].sum())
        if sub.empty:
            continue
        g = sub.groupby(group_col, as_index=False).agg(
            n_grid_cells=("transition", "size"),
            area_km2=("area_km2", "sum"),
            median_transition=("transition", "median"),
            mean_transition=("transition", "mean"),
        )
        g["area_percent_within_criterion"] = 100.0 * g["area_km2"] / total_area if total_area else np.nan
        g["criterion"] = key
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["criterion", group_col, "n_grid_cells", "area_km2", "area_percent_within_criterion", "median_transition", "mean_transition"]]


def ranking_stability(dist: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if dist.empty:
        return pd.DataFrame()
    rows = []
    for crit, g in dist.groupby("criterion"):
        gg = g.sort_values("area_percent_within_criterion", ascending=False).reset_index(drop=True)
        for i, r in gg.iterrows():
            rows.append({
                "criterion": crit,
                group_col: r[group_col],
                "rank_by_area_share": i + 1,
                "area_percent_within_criterion": r["area_percent_within_criterion"],
                "area_km2": r["area_km2"],
            })
    return pd.DataFrame(rows)


def prepare_states_for_plot(states: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return cleaned state geometries for robust plotting of all Brazilian boundaries.

    This helper repairs invalid geometries, preserves/dissolves features by UF
    where available, and keeps internal state boundaries visible in all panels.
    """
    gdf = states.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    if "uf" in gdf.columns:
        gdf = gdf.dissolve(by="uf", as_index=False)

    return gdf


def draw_brazil_state_backdrop(ax, states: gpd.GeoDataFrame, *, facecolor="#ececec", edgecolor="none", zorder=0):
    """Draw a light polygon backdrop for all Brazilian states.

    This rendering-only layer keeps all state territories visible beneath the
    raster field.
    """
    states_plot = prepare_states_for_plot(states)
    states_plot.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.0, zorder=zorder)
    return states_plot


def draw_brazil_state_boundaries(ax, states: gpd.GeoDataFrame, *, state_lw=0.95, outer_lw=1.25, halo_lw=2.00):
    """Draw state boundaries with a strong white halo plus dark line for visibility.

    A white halo and dark outline improve boundary visibility without affecting
    calculations.
    """
    states_plot = prepare_states_for_plot(states)

    # White halo underneath internal boundaries.
    states_plot.boundary.plot(ax=ax, color="white", linewidth=halo_lw, zorder=30)
    # Dark state boundaries on top.
    states_plot.boundary.plot(ax=ax, color="black", linewidth=state_lw, zorder=31)

    # Slightly stronger national outline.
    try:
        brazil_outline = states_plot.dissolve().boundary
        brazil_outline.plot(ax=ax, color="white", linewidth=outer_lw + 1.00, zorder=32)
        brazil_outline.plot(ax=ax, color="black", linewidth=outer_lw, zorder=33)
    except Exception:
        pass

    return states_plot


def draw_uf_labels(ax, states: gpd.GeoDataFrame, mode: str = DEFAULT_UF_LABEL_MODE, selected_ufs=None):
    """Draw UF acronyms as a visual check that state polygons are present.

    mode="selected" labels only DEFAULT_SELECTED_UFS (MG, RN, RS by default).
    mode="all" labels all states but can be visually crowded.
    mode="none" disables labels.

    Labels are drawn with a small white box so they remain visible over both
    red hotspot cells and pale no-data/non-hotspot areas.
    """
    if selected_ufs is None:
        selected_ufs = DEFAULT_SELECTED_UFS
    mode = str(mode).lower().strip()
    if mode == "none":
        return

    states_plot = prepare_states_for_plot(states)
    if "uf" not in states_plot.columns:
        print("[WARN] UF labels requested, but 'uf' column is not available after state preparation.")
        return

    if mode == "selected":
        states_plot = states_plot[states_plot["uf"].isin(selected_ufs)].copy()
    elif mode != "all":
        return

    for _, row in states_plot.iterrows():
        try:
            pt = row.geometry.representative_point()
            x, y = float(pt.x), float(pt.y)
            ax.text(
                x, y, str(row["uf"]),
                fontsize=FONT_SIZE,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=80,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.45, boxstyle="round,pad=0.12", alpha=0.85),
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            )
        except Exception:
            continue

def plot_map(ax, states, lons, lats, field, title, cmap="RdBu_r", vmin=-6, vmax=6, cbar_label="Tmax-standardized severity decade$^{-1}$"):
    draw_brazil_state_backdrop(ax, states, facecolor="#e8e8e8", zorder=0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(lons, lats, field, shading="auto", cmap=cmap, norm=norm, zorder=1)
    draw_brazil_state_boundaries(ax, states)
    draw_uf_labels(ax, states)
    ax.set_xlim(BRAZIL_EXTENT[0], BRAZIL_EXTENT[1])
    ax.set_ylim(BRAZIL_EXTENT[2], BRAZIL_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude", fontsize=LABEL_SIZE)
    ax.grid(True, linewidth=0.25, alpha=0.22)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.012)
    cb.set_label(cbar_label, fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)


def make_figure(out_dir: str, states, points, criteria, summary, region_dist, lats, lons, transition, primary_mask, p75, p90):
    """Create supplementary hotspot-sensitivity figure.

    Panel (b) shows valid Brazilian land in light grey and primary hotspot cells
    in dark red.
    """
    fig = plt.figure(figsize=(17.5, 12.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.955, bottom=0.075, wspace=0.30, hspace=0.28)

    # Build a mask of all valid Brazil land cells used in the analysis.
    # This is preferable for panel (b), because outside-Brazil and missing-data
    # cells remain white, while valid non-hotspot land appears in light grey.
    valid_brazil_mask = np.zeros_like(primary_mask, dtype=bool)
    valid_brazil_mask[points["iy"].values.astype(int), points["ix"].values.astype(int)] = True

    # Panel a: continuous transition with primary hotspot points.
    ax = fig.add_subplot(gs[0, 0])
    plot_map(ax, states, lons, lats, np.where(np.isfinite(transition), transition, np.nan),
             "(a) DHW−HHW Tmax-only contrast\ncontinuous field", vmin=-6, vmax=6)
    hot = points.loc[criteria["gt4_primary"]]
    ax.scatter(hot["lon"], hot["lat"], s=1.0, c="black", alpha=0.45, linewidth=0, zorder=5, label=">4 diagnostic hotspot")
    ax.legend(loc="lower left", fontsize=LEGEND_SIZE, frameon=True)

    # Panel b: primary hotspot map with an explicit full-state polygon backdrop.
    ax = fig.add_subplot(gs[0, 1])

    # Draw all Brazilian state polygons first. This backdrop is independent of the raster mask.
    draw_brazil_state_backdrop(ax, states, facecolor="#e6e6e6", zorder=0)

    # Overlay ONLY the primary hotspot cells in red. Non-hotspot and missing cells
    # are left transparent so the complete state-polygon backdrop remains visible.
    hotspot_only = np.full_like(transition, np.nan, dtype=float)
    hotspot_only[primary_mask] = 1.0
    red_cmap = ListedColormap(["#8b001c"])
    ax.pcolormesh(lons, lats, hotspot_only, shading="auto", cmap=red_cmap, vmin=1, vmax=1, zorder=2)

    # Draw boundaries after the raster overlay.
    draw_brazil_state_boundaries(ax, states)
    draw_uf_labels(ax, states)

    ax.set_title("(b) Primary diagnostic hotspot\nDHW−HHW Tmax-only > 4", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlim(BRAZIL_EXTENT[0], BRAZIL_EXTENT[1]); ax.set_ylim(BRAZIL_EXTENT[2], BRAZIL_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", fontsize=LABEL_SIZE); ax.set_ylabel("Latitude", fontsize=LABEL_SIZE)
    ax.grid(True, linewidth=0.25, alpha=0.22); ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(
        handles=[
            Patch(facecolor="#e6e6e6", edgecolor="black", label="Brazilian states / non-hotspot"),
            Patch(facecolor="#8b001c", edgecolor="black", label="primary hotspot"),
        ],
        loc="lower left", fontsize=LEGEND_SIZE, frameon=True
    )

    # Panel c: area and primary-hotspot recovery summary.
    ax = fig.add_subplot(gs[0, 2])
    show_order = ["gt3", "gt4_primary", "gt5", "top_quartile", "top_decile"]
    lab = [">3", ">4 primary", ">5", f"top quartile\n(≥{p75:.2f})", f"top decile\n(≥{p90:.2f})"]
    dat = summary.set_index("criterion").loc[[k for k in show_order if k in summary["criterion"].values]]
    y = np.arange(len(dat))
    ax.barh(y, dat["area_percent_of_brazil_mask"].values, alpha=0.78, label="hotspot area")
    ax.plot(dat["primary_overlap_percent_of_primary"].values, y, "ko", label="primary hotspot recovered")
    ax.set_yticks(y)
    ax.set_yticklabels([lab[show_order.index(k)] for k in dat.index], fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Area of Brazil mask / primary hotspot recovered (%)", fontsize=LABEL_SIZE)
    ax.set_title("(c) Hotspot area and recovery\nof primary definition", fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="lower right")

    # Panel d: regional distribution heatmap by area share.
    ax = fig.add_subplot(gs[1, :2])
    if not region_dist.empty:
        rd = region_dist[region_dist["region"].isin([r for r in REGION_ORDER if r != "Brazil"])]
        mat = rd.pivot_table(index="region", columns="criterion", values="area_percent_within_criterion", aggfunc="sum")
        col_order = [k for k in show_order if k in mat.columns]
        row_order = [r for r in REGION_ORDER if r != "Brazil" and r in mat.index]
        mat = mat.reindex(row_order)[col_order]
        im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1)
        ax.set_xticks(np.arange(len(col_order)))
        ax.set_xticklabels([lab[show_order.index(k)] for k in col_order], rotation=20, ha="right", fontsize=TICK_SIZE)
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=LABEL_SIZE)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=ANNOT_SIZE,
                            color="white" if val > 40 else "black")
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cb.set_label("% hotspot area within criterion", fontsize=TICK_SIZE)
        cb.ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("(d) Regional distribution of hotspot area", fontsize=TITLE_SIZE, fontweight="bold")

    # Panel e: sensitivity summary with unambiguous overlap language.
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    sidx = summary.set_index("criterion")
    lines = []
    lines.append("Diagnostic thresholds:")
    lines.append("• absolute cutoffs: >3, >4, >5 Tmax-standardized severity decade$^{-1}$")
    lines.append("• percentile cutoffs: top quartile and top decile")
    if any(k.endswith("_sigDHW") for k in criteria):
        lines.append("• repeated with significant positive DHW trend condition")
    lines.append("")
    primary_row = sidx.loc["gt4_primary"]
    lines.append(f"Primary hotspot (>4): {primary_row['area_percent_of_brazil_mask']:.1f}% of Brazil mask")
    lines.append(f"Primary grid cells: {int(primary_row['n_grid_cells']):,}")
    lines.append("")
    # Report two nonambiguous quantities: total area and recovery of the primary mask.
    # For stricter subsets, 'contained within primary' is not the same as 'primary recovered'.
    for key, label in [("gt3", ">3"), ("gt5", ">5"), ("top_quartile", f"top quartile (≥{p75:.2f})"), ("top_decile", f"top decile (≥{p90:.2f})")]:
        if key in sidx.index:
            r = sidx.loc[key]
            lines.append(
                f"{label}: {r['area_percent_of_brazil_mask']:.1f}% area; "
                f"recovers {r['primary_overlap_percent_of_primary']:.1f}% of primary"
            )
    if "gt4_primary_sigDHW" in sidx.index:
        r = sidx.loc["gt4_primary_sigDHW"]
        lines.append(
            f">4 + sig. DHW: {r['area_percent_of_brazil_mask']:.1f}% area; "
            f"recovers {r['primary_overlap_percent_of_primary']:.1f}% of primary"
        )
    ax.text(0.0, 1.0, "(e) Sensitivity summary", fontsize=TITLE_SIZE, fontweight="bold", va="top")
    ax.text(0.0, 0.88, "\n".join(lines), fontsize=FONT_SIZE, va="top")

    out_base = os.path.join(out_dir, "Supplementary_Figure_Hotspot_Threshold_Sensitivity_DHW_HHW")
    fig.savefig(out_base + ".jpeg", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_base}.jpeg")
    print(f"[OK] Saved figure: {out_base}.pdf")

def main():
    global DEFAULT_UF_LABEL_MODE, DEFAULT_SELECTED_UFS

    ap = argparse.ArgumentParser(
        description=(
            "Hotspot-threshold sensitivity for the common-scale "
            "DHW−HHW Tmax-only contrast."
        )
    )
    ap.add_argument(
        "--trend-nc", "--trend-nc",
        dest="trend_nc",
        required=True,
        help="Figure 01 NetCDF containing the common-scale DHW−HHW contrast or its components.",
    )
    ap.add_argument(
        "--transition-var", "--transition_var",
        dest="transition_var",
        default=None,
        help="Optional explicit variable name for the DHW−HHW contrast.",
    )
    ap.add_argument(
        "--dhw-trend-var", "--dhw_trend_var",
        dest="dhw_trend_var",
        default=None,
        help="Optional explicit DHW Tmax-only trend variable.",
    )
    ap.add_argument(
        "--hhw-trend-var", "--hhw_trend_var",
        dest="hhw_trend_var",
        default=None,
        help="Optional explicit HHW Tmax-only trend variable.",
    )
    ap.add_argument(
        "--dhw-pvalue-var", "--dhw_pvalue_var",
        dest="dhw_pvalue_var",
        default=None,
        help="Optional DHW Tmax-only p-value variable.",
    )
    ap.add_argument(
        "--hhw-pvalue-var", "--hhw_pvalue_var",
        dest="hhw_pvalue_var",
        default=None,
        help="Optional HHW Tmax-only p-value variable.",
    )
    ap.add_argument(
        "--p-threshold", "--p_threshold",
        dest="p_threshold",
        type=float,
        default=0.05,
        help="Significance threshold for the optional positive-DHW condition.",
    )
    ap.add_argument(
        "--states-shapefile", "--states-shapefile",
        dest="shp_uf",
        required=True,
        help="Brazilian state shapefile.",
    )
    ap.add_argument(
        "--biomes-shapefile", "--biomes-shapefile",
        dest="biome_shp",
        default=None,
        help="Optional Brazilian biome shapefile.",
    )
    ap.add_argument(
        "--output-dir", "--output-dir",
        dest="out_dir",
        required=True,
        help="Output directory.",
    )
    ap.add_argument(
        "--uf-label-mode", "--uf_label_mode",
        dest="uf_label_mode",
        default=DEFAULT_UF_LABEL_MODE,
        choices=["none", "selected", "all"],
    )
    ap.add_argument(
        "--selected-ufs", "--selected_ufs",
        dest="selected_ufs",
        default=",".join(DEFAULT_SELECTED_UFS),
        help="Comma-separated UF labels used when --uf-label-mode selected.",
    )
    args = ap.parse_args()

    DEFAULT_UF_LABEL_MODE = args.uf_label_mode
    DEFAULT_SELECTED_UFS = [u.strip().upper() for u in args.selected_ufs.split(",") if u.strip()]

    ensure_dir(args.out_dir)
    print("[START] Hotspot-threshold sensitivity")
    print("[INFO] Interpretation: diagnostic hotspot cutoff sensitivity; not a physical threshold test.")
    print("[INFO] Trend fields are static; CF time decoding is disabled.")

    ds, trend_path = open_trend_dataset(args.trend_nc)
    print(f"[INFO] Trend dataset: {trend_path}")
    print(f"[INFO] Available variables: {list(ds.data_vars)}")

    # Use only the common-scale Tmax-only contrast.
    transition_name = select_var(ds, args.transition_var, "transition", required=False)
    if transition_name is not None:
        transition = ds[transition_name].squeeze().values.astype(float)
        transition_source = transition_name
    else:
        dhw_component_name = select_var(ds, args.dhw_trend_var, "dhw_trend", required=True)
        hhw_component_name = select_component_var(ds, args.hhw_trend_var, "hhw_trend", required=True)
        transition = (ds[dhw_component_name].squeeze() - ds[hhw_component_name].squeeze()).values.astype(float)
        transition_name = "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
        transition_source = f"{dhw_component_name} - {hhw_component_name}"
        print(f"[INFO] Computed common-scale transition field as: {transition_source}")

    dhw_trend_name = select_var(ds, args.dhw_trend_var, "dhw_trend", required=False)
    dhw_p_name = select_var(ds, args.dhw_pvalue_var, "dhw_pvalue", required=False)

    legacy_tokens = [
        "DHW_intensity_trend_decade",
        "HHW_intensity_trend_decade",
        "dry_minus_humid_intensity_trend",
    ]
    legacy_used = [tok for tok in legacy_tokens if tok in str(transition_source)]
    if legacy_used:
        raise ValueError(
            "Non-commensurate intensity field detected in transition source: "
            f"{transition_source}. Refusing to continue."
        )

    lats = ds["lat"].values.astype(float)
    lons = ds["lon"].values.astype(float)
    if transition.shape != (len(lats), len(lons)):
        raise ValueError(f"Transition field shape {transition.shape} does not match lat/lon {(len(lats), len(lons))}.")

    states = read_states(args.shp_uf)
    # Optional UF-label diagnostics.
    for _uf in DEFAULT_SELECTED_UFS:
        _sub = states[states["uf"] == _uf]
        if _sub.empty:
            print(f"[WARN] Selected UF label requested but polygon is missing from shapefile: {_uf}")
        else:
            _pt = _sub.geometry.iloc[0].representative_point()
            print(f"[INFO] Selected UF present for plotting: {_uf} at lon={_pt.x:.2f}, lat={_pt.y:.2f}")
    brazil_mask = make_brazil_mask(lats, lons, states)
    area = cell_area_km2(lats, lons)
    points = grid_points_dataframe(lats, lons, brazil_mask, area, transition)
    print(f"[INFO] Brazil mask valid grid cells with transition data: {len(points):,}")

    points = attach_regions(points, states)
    biomes = read_biomes(args.biome_shp)
    points = attach_biomes(points, biomes)

    sig_available = False
    if dhw_trend_name and dhw_p_name:
        dhw_trend = ds[dhw_trend_name].squeeze().values.astype(float)
        dhw_p = ds[dhw_p_name].squeeze().values.astype(float)
        if dhw_trend.shape == transition.shape and dhw_p.shape == transition.shape:
            vals_tr = dhw_trend[points["iy"].values.astype(int), points["ix"].values.astype(int)]
            vals_p = dhw_p[points["iy"].values.astype(int), points["ix"].values.astype(int)]
            points["dhw_trend"] = vals_tr
            points["dhw_pvalue"] = vals_p
            points["dhw_sig_positive"] = np.isfinite(vals_tr) & np.isfinite(vals_p) & (vals_tr > 0) & (vals_p <= args.p_threshold)
            sig_available = True
            print(f"[INFO] Significant-positive-DHW condition available: {points['dhw_sig_positive'].sum():,} grid cells")
        else:
            print("[WARN] DHW trend/pvalue shapes do not match transition field. Skipping significance-constrained sensitivity.")

    criteria = build_criteria(points, sig_available=sig_available)
    p75 = points.attrs["p75_threshold"]
    p90 = points.attrs["p90_threshold"]
    print(f"[INFO] Percentile cutoffs among Brazil grid cells: P75={p75:.3f}, P90={p90:.3f}")

    summary = summarize_criteria(points, criteria)
    region_dist = distribution_table(points, criteria, "region")
    region_rank = ranking_stability(region_dist, "region")
    biome_dist = distribution_table(points, criteria, "biome") if "biome" in points.columns else pd.DataFrame()
    biome_rank = ranking_stability(biome_dist, "biome") if not biome_dist.empty else pd.DataFrame()

    # Save outputs
    pts_cols = ["iy", "ix", "lat", "lon", "area_km2", "transition", "uf", "region", "biome"]
    for c in ["dhw_trend", "dhw_pvalue", "dhw_sig_positive"]:
        if c in points.columns:
            pts_cols.append(c)
    points[pts_cols].to_csv(os.path.join(args.out_dir, "Hotspot_Sensitivity_gridcell_base_table.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "Hotspot_Sensitivity_summary_by_criterion.csv"), index=False)
    region_dist.to_csv(os.path.join(args.out_dir, "Hotspot_Sensitivity_region_distribution.csv"), index=False)
    region_rank.to_csv(os.path.join(args.out_dir, "Hotspot_Sensitivity_region_ranking_stability.csv"), index=False)
    if not biome_dist.empty:
        biome_dist.to_csv(os.path.join(args.out_dir, "Hotspot_Sensitivity_biome_distribution.csv"), index=False)
        biome_rank.to_csv(os.path.join(args.out_dir, "Hotspot_Sensitivity_biome_ranking_stability.csv"), index=False)

    with pd.ExcelWriter(os.path.join(args.out_dir, "Supplementary_Tables_Hotspot_Threshold_Sensitivity_DHW_HHW.xlsx")) as writer:
        summary.to_excel(writer, sheet_name="criteria_summary", index=False)
        region_dist.to_excel(writer, sheet_name="region_distribution", index=False)
        region_rank.to_excel(writer, sheet_name="region_ranking", index=False)
        if not biome_dist.empty:
            biome_dist.to_excel(writer, sheet_name="biome_distribution", index=False)
            biome_rank.to_excel(writer, sheet_name="biome_ranking", index=False)
        points[pts_cols].to_excel(writer, sheet_name="gridcell_base", index=False)

    metadata = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "trend_dataset": trend_path,
        "transition_variable": transition_name,
        "transition_source": transition_source,
        "dhw_trend_variable": dhw_trend_name,
        "dhw_pvalue_variable": dhw_p_name,
        "p_threshold": args.p_threshold,
        "p75_transition_threshold": p75,
        "p90_transition_threshold": p90,
        "interpretation": "Diagnostic hotspot-cutoff sensitivity applied to the common-scale Tmax-only DHW-HHW contrast; thresholds are not physical thresholds and no p-value is assigned to the slope difference.",
    }
    with open(os.path.join(args.out_dir, "software_metadata_hotspot_threshold_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    primary_mask = mask_from_point_table(points, transition.shape, criteria["gt4_primary"])
    make_figure(args.out_dir, states, points, criteria, summary, region_dist, lats, lons, transition, primary_mask, p75, p90)

    print("\n[SUMMARY] Hotspot sensitivity relative to primary definition DHW−HHW > 4:")
    for _, r in summary.iterrows():
        print(
            f"  {r['criterion']:<18} cells={int(r['n_grid_cells']):6d} "
            f"area={r['area_percent_of_brazil_mask']:5.1f}% "
            f"overlap_primary={r['primary_overlap_percent_of_primary']:5.1f}% "
            f"jaccard={r['jaccard_area_overlap_with_primary']:.2f}"
        )
    print(f"\n[DONE] Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()


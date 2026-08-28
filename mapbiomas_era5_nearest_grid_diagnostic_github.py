#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MapBiomas–ERA5 nearest-grid diagnostic integration.

Purpose
-------
Build municipality–biome land-cover metrics from MapBiomas and attach ERA5
heatwave-trend fields from the nearest grid cell to each municipality
representative point.

This nearest-grid workflow is intended as a diagnostic/preprocessing product.
The primary municipality aggregation used in Figure 02 is performed by the
Figure 02 workflow using cos(latitude)-weighted zonal aggregation, with
nearest-grid extraction only as a fallback where no ERA5 grid-cell centre
falls within a municipality.

Land-cover quality control
--------------------------
MapBiomas municipality–biome intersections smaller than ``MIN_AREA_HA`` are
excluded because very small residual intersections can produce unstable
fractional-change estimates. The default threshold is 1000 ha.

Cross-regime metric
-------------------
When available, the DHW–HHW comparison is read directly from the common-scale
Figure 01 field:

    DHW_minus_HHW_TmaxOnly_intensity_trend_decade

Primary regime-specific DHW and HHW intensity trends are not subtracted.

Usage
-----
python mapbiomas_era5_nearest_grid_diagnostic.py \
    --mapbiomas-coverage /path/to/mapbiomas_coverage.xlsx \
    --urban-module /path/to/mapbiomas_urban_module.xlsx \
    --era5-trends /path/to/figure_01_heatwave_trends_1990_2024.nc \
    --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
    --output-dir ./outputs/mapbiomas_era5_diagnostic
"""

import os
import re
import warnings
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree


# ============================================================
# Runtime paths
# ============================================================

COVERAGE_XLSX = None
URBAN_XLSX = None
ERA5_TRENDS_NC = None
MUNICIPALITY_SHP = None
OUT_DIR = None


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a MapBiomas–ERA5 nearest-grid diagnostic table."
    )
    parser.add_argument(
        "--mapbiomas-coverage",
        required=True,
        help="MapBiomas municipality/state/biome coverage workbook.",
    )
    parser.add_argument(
        "--urban-module",
        default=None,
        help="Optional MapBiomas urban-module workbook.",
    )
    parser.add_argument(
        "--era5-trends",
        required=True,
        help="Figure 01 trend NetCDF.",
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
    return parser.parse_args()


def configure_paths(args):
    global COVERAGE_XLSX, URBAN_XLSX, ERA5_TRENDS_NC, MUNICIPALITY_SHP, OUT_DIR

    COVERAGE_XLSX = str(Path(args.mapbiomas_coverage).expanduser().resolve())
    URBAN_XLSX = (
        str(Path(args.urban_module).expanduser().resolve())
        if args.urban_module else None
    )
    ERA5_TRENDS_NC = str(Path(args.era5_trends).expanduser().resolve())
    MUNICIPALITY_SHP = str(
        Path(args.municipality_shapefile).expanduser().resolve()
    )
    OUT_DIR = str(Path(args.output_dir).expanduser().resolve())

    required = [
        (COVERAGE_XLSX, "MapBiomas coverage workbook"),
        (ERA5_TRENDS_NC, "Figure 01 trend file"),
        (MUNICIPALITY_SHP, "municipality shapefile"),
    ]
    for path, label in required:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    if URBAN_XLSX is not None and not os.path.isfile(URBAN_XLSX):
        raise FileNotFoundError(f"MapBiomas urban-module workbook not found: {URBAN_XLSX}")

    os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

YEAR0 = 1985
YEAR1 = 2024

# Minimum municipality-biome area to retain.
# Rows below this threshold are usually residual sliver intersections.
MIN_AREA_HA = 1000.0

# Land-cover groups from MapBiomas class hierarchy.
# MapBiomas class groups used to construct land-cover metrics.
NATIVE_LEVEL1 = {
    "1. forest",
    "2. non forest natural formation",
}

AGRO_LEVEL1 = {
    "3. farming",
}

URBAN_LEVEL2 = {
    "4.2. urban area",
}

WATER_LEVEL1 = {
    "5. water and marine environment",
}


# ============================================================
# Helpers
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


def clean_municipality_name_from_urban(x):
    """
    Urban module names often appear as 'Alta Floresta D'Oeste (RO)'.
    This returns 'Alta Floresta D'Oeste'.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s*\([A-Z]{2}\)\s*$", "", s)
    return s


def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where((den > 0) & np.isfinite(den), num / den, np.nan)


def find_municipality_shapefile():
    """Return the configured municipality shapefile."""
    if MUNICIPALITY_SHP and os.path.isfile(MUNICIPALITY_SHP):
        return MUNICIPALITY_SHP
    return None


def detect_column(columns, candidates):
    cols_norm = {normalize_text(c): c for c in columns}

    for cand in candidates:
        nc = normalize_text(cand)
        if nc in cols_norm:
            return cols_norm[nc]

    for c in columns:
        nc = normalize_text(c)
        for cand in candidates:
            if normalize_text(cand) in nc:
                return c

    return None


# ============================================================
# MapBiomas land-cover table
# ============================================================

def read_coverage():
    print("[INFO] Reading MapBiomas coverage statistics.")
    df = pd.read_excel(COVERAGE_XLSX, sheet_name="COVERAGE_10.1")

    needed = [
        "biome", "state", "state_acronym", "municipality",
        "class_level_0", "class_level_1", "class_level_2",
        YEAR0, YEAR1
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in coverage file: {missing}")

    for y in range(YEAR0, YEAR1 + 1):
        if y in df.columns:
            df[y] = pd.to_numeric(df[y], errors="coerce")

    df["municipality_norm"] = df["municipality"].map(normalize_text)
    df["state_acronym_norm"] = df["state_acronym"].map(normalize_text)
    df["biome_norm"] = df["biome"].map(normalize_text)
    df["class1_norm"] = df["class_level_1"].map(normalize_class)
    df["class2_norm"] = df["class_level_2"].map(normalize_class)

    return df


def build_landcover_metrics(df):
    print("[INFO] Building municipality-biome land-cover metrics.")

    id_cols = [
        "country", "biome", "state", "state_acronym",
        "municipality", "municipality_norm", "state_acronym_norm", "biome_norm"
    ]

    total = (
        df.groupby(id_cols, dropna=False)[[YEAR0, YEAR1]]
          .sum()
          .rename(columns={YEAR0: "total_area_1985_ha", YEAR1: "total_area_2024_ha"})
          .reset_index()
    )

    def aggregate_group(mask, prefix):
        g = (
            df.loc[mask]
              .groupby(id_cols, dropna=False)[[YEAR0, YEAR1]]
              .sum()
              .rename(columns={YEAR0: f"{prefix}_1985_ha", YEAR1: f"{prefix}_2024_ha"})
              .reset_index()
        )
        return g

    native_mask = df["class1_norm"].isin(NATIVE_LEVEL1)
    agro_mask = df["class1_norm"].isin(AGRO_LEVEL1)
    urban_mask = df["class2_norm"].isin(URBAN_LEVEL2)
    water_mask = df["class1_norm"].isin(WATER_LEVEL1)

    out = total.copy()

    for prefix, mask in [
        ("native", native_mask),
        ("agro", agro_mask),
        ("urban", urban_mask),
        ("water", water_mask),
    ]:
        g = aggregate_group(mask, prefix)
        out = out.merge(g, on=id_cols, how="left")

    area_cols = [c for c in out.columns if c.endswith("_ha")]
    out[area_cols] = out[area_cols].fillna(0.0)

    # ---------------- QC: remove tiny municipality-biome residuals ----------------
    before = len(out)
    removed = out[out["total_area_2024_ha"] < MIN_AREA_HA].copy()
    out = out[out["total_area_2024_ha"] >= MIN_AREA_HA].copy()
    after = len(out)

    qc_out = os.path.join(OUT_DIR, "QC_removed_small_area_municipality_biome_rows.csv")
    removed.to_csv(qc_out, index=False)

    print(f"[QC] Removed {before - after} municipality-biome rows with total_area_2024_ha < {MIN_AREA_HA:.0f} ha.")
    print(f"[QC] Saved removed rows to: {qc_out}")

    # Use each year's total area as denominator for that year's fraction.
    den_1985 = out["total_area_1985_ha"].values
    den_2024 = out["total_area_2024_ha"].values

    for group in ["native", "agro", "urban", "water"]:
        out[f"{group}_frac_1985"] = safe_div(out[f"{group}_1985_ha"].values, den_1985)
        out[f"{group}_frac_2024"] = safe_div(out[f"{group}_2024_ha"].values, den_2024)

        out[f"{group}_pct_1985"] = 100.0 * out[f"{group}_frac_1985"]
        out[f"{group}_pct_2024"] = 100.0 * out[f"{group}_frac_2024"]

        out[f"{group}_change_ha"] = out[f"{group}_2024_ha"] - out[f"{group}_1985_ha"]
        out[f"{group}_change_pct_points"] = out[f"{group}_pct_2024"] - out[f"{group}_pct_1985"]

    # Physically interpretable predictors.
    out["veg_loss_pct_points"] = out["native_pct_1985"] - out["native_pct_2024"]
    out["agri_gain_pct_points"] = out["agro_pct_2024"] - out["agro_pct_1985"]
    out["urban_gain_pct_points"] = out["urban_pct_2024"] - out["urban_pct_1985"]

    out["veg_loss_fraction"] = out["veg_loss_pct_points"] / 100.0
    out["agri_gain_fraction"] = out["agri_gain_pct_points"] / 100.0
    out["urban_gain_fraction"] = out["urban_gain_pct_points"] / 100.0

    # Remove tiny numerical noise.
    for c in ["veg_loss_pct_points", "agri_gain_pct_points", "urban_gain_pct_points"]:
        out[c] = out[c].where(np.abs(out[c]) > 1e-8, 0.0)

    # Clamp fractions only for interpretability. Do not alter hectares.
    frac_cols = [c for c in out.columns if c.endswith("_frac_1985") or c.endswith("_frac_2024")]
    for c in frac_cols:
        out[c] = out[c].clip(lower=0.0, upper=1.0)

    pct_cols = [c for c in out.columns if c.endswith("_pct_1985") or c.endswith("_pct_2024")]
    for c in pct_cols:
        out[c] = out[c].clip(lower=0.0, upper=100.0)

    return out


def read_urban_module():
    if URBAN_XLSX is None or not os.path.exists(URBAN_XLSX):
        print("[WARN] Urban module file not found. Skipping detailed urban variables.")
        return None

    print("[INFO] Reading MapBiomas urban module.")

    try:
        urb = pd.read_excel(URBAN_XLSX, sheet_name="UrbanVegetation")
    except Exception as e:
        print(f"[WARN] Could not read UrbanVegetation sheet: {e}")
        return None

    needed = ["munCD", "munNM", "stateNM", "biomaNM", "classNM", YEAR0, YEAR1]
    missing = [c for c in needed if c not in urb.columns]
    if missing:
        print(f"[WARN] Urban module missing columns {missing}. Skipping.")
        return None

    urb["municipality"] = urb["munNM"].map(clean_municipality_name_from_urban)
    urb["municipality_norm"] = urb["municipality"].map(normalize_text)
    urb["state_acronym_norm"] = urb["stateNM"].map(normalize_text)
    urb["biome_norm"] = urb["biomaNM"].map(normalize_text)
    urb["class_norm"] = urb["classNM"].map(normalize_text)

    for y in range(YEAR0, YEAR1 + 1):
        if y in urb.columns:
            urb[y] = pd.to_numeric(urb[y], errors="coerce")

    id_cols = ["munCD", "municipality_norm", "state_acronym_norm", "biome_norm", "class_norm"]

    out = (
        urb.groupby(id_cols, dropna=False)[[YEAR0, YEAR1]]
           .sum()
           .rename(columns={YEAR0: "urban_module_1985_ha", YEAR1: "urban_module_2024_ha"})
           .reset_index()
    )

    out_piv = out.pivot_table(
        index=["munCD", "municipality_norm", "state_acronym_norm", "biome_norm"],
        columns="class_norm",
        values=["urban_module_1985_ha", "urban_module_2024_ha"],
        aggfunc="sum"
    )

    out_piv.columns = [
        normalize_text("_".join([str(a), str(b)])).replace(" ", "_").replace("/", "_")
        for a, b in out_piv.columns
    ]
    out_piv = out_piv.reset_index()

    return out_piv


# ============================================================
# Municipality shapefile and representative points
# ============================================================

def read_municipality_centroids():
    shp = find_municipality_shapefile()

    if shp is None:
        print("[WARN] Municipality shapefile not found.")
        print("[WARN] Land-cover table will be created, but nearest-grid ERA5 extraction requires representative points.")
        return None

    print(f"[INFO] Reading municipality shapefile: {shp}")
    gdf = gpd.read_file(shp).to_crs(epsg=4326)

    name_col = detect_column(gdf.columns, [
        "NM_MUN", "NM_MUNICIP", "NM_MUNICIPIO", "NOME", "MUNICIPIO", "municipality", "name"
    ])

    state_col = detect_column(gdf.columns, [
        "SIGLA_UF", "UF", "state_acronym", "CD_UF", "SIGLA"
    ])

    code_col = detect_column(gdf.columns, [
        "CD_MUN", "CD_MUNICIPIO", "CD_GEOCMU", "GEOCODIGO", "munCD", "code_muni", "COD_MUN"
    ])

    if name_col is None:
        raise ValueError(
            "Could not detect municipality name column in shapefile. "
            f"Available columns: {list(gdf.columns)}"
        )

    if state_col is None:
        print("[WARN] Could not detect state column. Merge will use municipality name only, less safe.")

    # representative_point() returns a point located within each polygon.
    pts = gdf.geometry.representative_point()

    cent = pd.DataFrame({
        "shp_municipality": gdf[name_col].astype(str),
        "municipality_norm": gdf[name_col].map(normalize_text),
        "centroid_lon": pts.x,
        "centroid_lat": pts.y,
    })

    if state_col is not None:
        cent["state_acronym_norm"] = gdf[state_col].map(normalize_text)
    else:
        cent["state_acronym_norm"] = ""

    if code_col is not None:
        cent["mun_code"] = gdf[code_col].astype(str)

    cent = cent.drop_duplicates(subset=["municipality_norm", "state_acronym_norm"])
    return cent


# ============================================================
# ERA5 trend extraction by nearest grid
# ============================================================

def extract_nearest_era5_trends(land, centroids):
    if not os.path.exists(ERA5_TRENDS_NC):
        print(f"[WARN] ERA5 trend file not found: {ERA5_TRENDS_NC}")
        print("[WARN] Returning land-cover table without ERA5 integration.")
        return land

    print("[INFO] Extracting nearest-grid ERA5 trends for municipality diagnostics.")
    ds = xr.open_dataset(ERA5_TRENDS_NC)

    lon = ds["lon"].values
    lat = ds["lat"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    tree = cKDTree(points)

    if "state_acronym_norm" in centroids.columns and centroids["state_acronym_norm"].astype(str).str.len().gt(0).any():
        merged = land.merge(
            centroids,
            on=["municipality_norm", "state_acronym_norm"],
            how="left"
        )
    else:
        merged = land.merge(
            centroids.drop(columns=["state_acronym_norm"], errors="ignore"),
            on="municipality_norm",
            how="left"
        )

    n_missing = merged["centroid_lon"].isna().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} municipality-biome rows did not match the shapefile centroids.")

    ok = merged["centroid_lon"].notna() & merged["centroid_lat"].notna()

    nearest_lat = np.full(len(merged), np.nan)
    nearest_lon = np.full(len(merged), np.nan)
    nearest_dist_deg = np.full(len(merged), np.nan)

    query = np.column_stack([merged.loc[ok, "centroid_lat"].values, merged.loc[ok, "centroid_lon"].values])
    dist, idx = tree.query(query)

    nearest_lat[ok.values] = points[idx, 0]
    nearest_lon[ok.values] = points[idx, 1]
    nearest_dist_deg[ok.values] = dist

    merged["era5_nearest_lat"] = nearest_lat
    merged["era5_nearest_lon"] = nearest_lon
    merged["era5_nearest_distance_deg"] = nearest_dist_deg

    trend_vars = [v for v in ds.data_vars if v.endswith("_trend_decade") or v.endswith("_pvalue")]

    for v in trend_vars:
        arr = ds[v].values
        vals = np.full(len(merged), np.nan)

        if ok.any():
            lat_idx = np.array([np.argmin(np.abs(lat - la)) for la in nearest_lat[ok.values]])
            lon_idx = np.array([np.argmin(np.abs(lon - lo)) for lo in nearest_lon[ok.values]])
            vals[ok.values] = arr[lat_idx, lon_idx]

        merged[v] = vals

    for kind in ["HHW", "DHW", "HW", "CHW"]:
        for metric in ["frequency", "duration", "intensity"]:
            col = f"{kind}_{metric}_trend_decade"
            if col not in merged.columns:
                merged[col] = np.nan

    common_contrast = "DHW_minus_HHW_TmaxOnly_intensity_trend_decade"
    if common_contrast not in merged.columns:
        ds.close()
        raise RuntimeError(
            "Figure 01 trend file is missing the common-scale DHW–HHW field: "
            f"{common_contrast}. Primary DHW and HHW intensity trends are not "
            "subtracted because their intensity definitions are not commensurate."
        )

    merged["dry_minus_humid_intensity_trend"] = pd.to_numeric(
        merged[common_contrast],
        errors="coerce",
    )

    ds.close()
    return merged


# ============================================================
# Diagnostics
# ============================================================

def print_basic_diagnostics(df, label):
    print(f"\n[DIAGNOSTICS] {label}")
    print(f"Rows: {len(df)}")

    cols = [
        "total_area_2024_ha",
        "veg_loss_pct_points",
        "agri_gain_pct_points",
        "urban_gain_pct_points",
    ]

    available = [c for c in cols if c in df.columns]
    if available:
        print(df[available].describe().to_string())

    if "biome" in df.columns:
        print("\nRows by biome:")
        print(df["biome"].value_counts().to_string())


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    configure_paths(args)

    coverage = read_coverage()
    land = build_landcover_metrics(coverage)

    urb = read_urban_module()
    if urb is not None:
        print("[INFO] Merging urban-module variables.")
        land = land.merge(
            urb,
            on=["municipality_norm", "state_acronym_norm", "biome_norm"],
            how="left"
        )

    print_basic_diagnostics(land, "MapBiomas land-cover table after small-area QC")

    land_out = os.path.join(
        OUT_DIR,
        "MapBiomas_municipality_landcover_change_1985_2024.csv"
    )
    land.to_csv(land_out, index=False)
    print(f"[OK] Saved land-cover metrics: {land_out}")

    centroids = read_municipality_centroids()

    if centroids is None:
        integrated = land.copy()
    else:
        integrated = extract_nearest_era5_trends(land, centroids)

    integrated_out = os.path.join(
        OUT_DIR,
        "MapBiomas_ERA5_nearest_grid_diagnostic.csv"
    )
    integrated.to_csv(integrated_out, index=False)
    print(f"[OK] Saved integrated MapBiomas + ERA5 table: {integrated_out}")

    keep_cols = [
        "country", "biome", "state", "state_acronym", "municipality",
        "municipality_norm", "state_acronym_norm", "biome_norm",
        "total_area_2024_ha",
        "native_pct_1985", "native_pct_2024", "veg_loss_pct_points",
        "agro_pct_1985", "agro_pct_2024", "agri_gain_pct_points",
        "urban_pct_1985", "urban_pct_2024", "urban_gain_pct_points",
        "centroid_lon", "centroid_lat", "era5_nearest_lon", "era5_nearest_lat",
        "HHW_intensity_trend_decade", "DHW_intensity_trend_decade",
        "HW_intensity_trend_decade", "CHW_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
        "DHW_TmaxOnly_intensity_trend_decade",
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "dry_minus_humid_intensity_trend",
    ]

    keep_cols_existing = [c for c in keep_cols if c in integrated.columns]
    compact = integrated[keep_cols_existing].copy()

    compact_out = os.path.join(
        OUT_DIR,
        "MapBiomas_ERA5_nearest_grid_diagnostic_compact.csv"
    )
    compact.to_csv(compact_out, index=False)
    print(f"[OK] Saved compact diagnostic table: {compact_out}")

    print("\n[DONE]")
    print("Main outputs:")
    print("  1)", land_out)
    print("  2)", integrated_out)
    print("  3)", compact_out)
    print("\nQC output:")
    print("  ", os.path.join(OUT_DIR, "QC_removed_small_area_municipality_biome_rows.csv"))
    print("\nInterpretation variables:")
    print("  veg_loss_pct_points       = native vegetation fraction lost from 1985 to 2024")
    print("  agri_gain_pct_points      = agro/farming fraction gained from 1985 to 2024")
    print("  urban_gain_pct_points     = urban fraction gained from 1985 to 2024")
    print("  dry_minus_humid trend     = common-scale DHW-HHW Tmax-only trend contrast")
    print(f"\nSmall-area QC:")
    print(f"  rows with total_area_2024_ha < {MIN_AREA_HA:.0f} ha were removed.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

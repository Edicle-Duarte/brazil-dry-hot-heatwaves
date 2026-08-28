#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary material for Figure 03
====================================

Purpose
-------
Generate Supplementary Tables S9–S12 from the Figure 03 atmospheric-drying
products without recomputing ERA5 trend fields.

Required upstream products
--------------------------
From ``figure_03_atmospheric_drying.py``:

  figure_03_municipality_mechanism_table_1990_2024.csv
  figure_03_ERA5_mechanistic_trends_with_dryhot_1990_2024.nc

Scientific alignment
--------------------
1. DHW–HHW comparison
   Uses the common-scale metric already propagated into Figure 03:

       dry_minus_humid_intensity_trend

   with upstream definition:

       DHW_TmaxOnly_intensity_trend_decade
       - HHW_TmaxOnly_intensity_trend_decade

   Primary regime-specific DHW and HHW intensity fields are not subtracted.

2. Atmospheric drying
   VPD trends are expressed in kPa decade^-1.
   RH trends are expressed in percent decade^-1.

3. Municipality aggregation
   Municipality-level ERA5 trends are inherited from Figure 03, where
   continuous fields are aggregated with cos(latitude) grid-cell weights and
   p-values are not averaged.

4. Association diagnostics
   LOWESS is descriptive. Weighted Spearman correlations use weights
   proportional to the square root of municipal area and two-sided permutation
   p-values based on 999 permutations.

5. Controlled regressions
   Weighted least squares (WLS) uses weights proportional to the square root of
   municipal area and HC3 heteroscedasticity-robust standard errors. These
   models quantify spatial associations and are not interpreted as causal or
   mediation analyses.

Outputs
-------
Supplementary_Table_S9.csv/.xlsx
Supplementary_Table_S10.csv/.xlsx
Supplementary_Table_S11.csv/.xlsx
Supplementary_Table_S12.csv/.xlsx
figure_03_supplementary_lowess_summary.csv
figure_03_supplementary_QA.csv

Usage
-----
python figure_03_supplementary_material.py \
    --figure3-output-dir /path/to/figure_03_outputs \
    --output-dir ./outputs/figure_03_supplementary
"""

import os
import argparse
import warnings
from pathlib import Path


import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


# ============================================================
# Configuration consistent with Figure 03
# ============================================================

YEAR0, YEAR1 = 1990, 2024

N_BOOT = 300
N_PERMUTATIONS = 999
RANDOM_SEED = 42
MIN_N_FOR_RELATIONSHIP = 60

REGION_ORDER = [
    "Amazon",
    "Cerrado",
    "MATOPIBA",
    "Semi-arid Northeast",
    "Urban Southeast",
    "Pantanal",
    "Atlantic Forest",
    "Pampa",
]

REGIONAL_RELATIONSHIP_REGIONS = [
    "Amazon",
    "Cerrado",
    "MATOPIBA",
    "Atlantic Forest",
    "Urban Southeast",
]

REQUIRED_MUNICIPAL_COLUMNS = [
    "municipality_norm",
    "state_acronym_norm",
    "total_area_2024_ha",
    "focus_region",
    "dominant_biome",
    "VPD_trend_decade",
    "RH_trend_decade",
    "dry_minus_humid_intensity_trend",
    "transformed_non_native_pct_2024",
    "agri_gain_pct_points",
]

OPTIONAL_SPATIAL_CONTROL_COLUMNS = [
    "rep_lat",
    "rep_lon",
]

OPTIONAL_CORRECTED_COLUMNS = [
    "DHW_TmaxOnly_intensity_trend_decade",
    "HHW_TmaxOnly_intensity_trend_decade",
    "DHW_TmaxOnly_significant_positive_fraction",
]


# ============================================================
# CLI and safe I/O
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Tables S9–S12 for Figure 03."
    )
    parser.add_argument(
        "--figure3-output-dir",
        required=True,
        help="Directory produced by figure_03_atmospheric_drying.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Supplementary output directory. Default: "
            "<figure3-output-dir>/supplementary"
        ),
    )
    return parser.parse_args()


def _available_netcdf_engines():
    engines = xr.backends.list_engines()
    return [name for name in ("h5netcdf", "netcdf4", "scipy") if name in engines]


def safe_open_dataset(path, *, decode_times=False, chunks=None):
    errors = []
    for engine in _available_netcdf_engines():
        try:
            return xr.open_dataset(
                path,
                engine=engine,
                decode_times=decode_times,
                chunks=chunks,
            )
        except Exception as exc:
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not open NetCDF file: {path}\n" + "\n".join(errors)
    )


def save_table_dual(df, output_dir, stem):
    csv_path = output_dir / f"{stem}.csv"
    xlsx_path = output_dir / f"{stem}.xlsx"

    df.to_csv(csv_path, index=False)

    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save {xlsx_path.name}: {exc}")

    print(f"[OK] Saved: {csv_path}")
    return csv_path, xlsx_path


# ============================================================
# QA / compatibility
# ============================================================

def validate_figure3_products(mun, trends):
    missing = [c for c in REQUIRED_MUNICIPAL_COLUMNS if c not in mun.columns]
    if missing:
        raise ValueError(
            "Figure 03 municipality table is missing fields required "
            "for the core supplementary analyses: " + ", ".join(missing)
        )

    missing_spatial = [
        c for c in OPTIONAL_SPATIAL_CONTROL_COLUMNS
        if c not in mun.columns
    ]
    if missing_spatial:
        print(
            "[INFO] Optional spatial-control columns are absent: "
            + ", ".join(missing_spatial)
            + ". Spatial WLS variants will be skipped."
        )

    # The common-scale contrast must exist and be finite.
    dryhot = pd.to_numeric(
        mun["dry_minus_humid_intensity_trend"],
        errors="coerce",
    )
    if dryhot.notna().sum() == 0:
        raise ValueError(
            "dry_minus_humid_intensity_trend contains no finite values."
        )

    # Validate the gridded product and physical units.
    required_grid = [
        "VPD_trend_decade",
        "RH_trend_decade",
        "dry_minus_humid_intensity_trend",
        "DHW_TmaxOnly_intensity_trend_decade",
        "DHW_TmaxOnly_intensity_pvalue",
    ]
    missing_grid = [v for v in required_grid if v not in trends.data_vars]
    if missing_grid:
        raise ValueError(
            "Figure 03 trend product is missing: "
            + ", ".join(missing_grid)
        )

    vpd_units = str(trends["VPD_trend_decade"].attrs.get("units", "")).lower()
    if "kpa" not in vpd_units:
        raise ValueError(
            "VPD_trend_decade is not explicitly labelled in kPa decade-1. "
            f"Found units='{trends['VPD_trend_decade'].attrs.get('units', '')}'."
        )

    source = trends.attrs.get("cross_regime_source", "")
    if source and source != "DHW_minus_HHW_TmaxOnly_intensity_trend_decade":
        raise ValueError(
            "Figure 03 trend product does not declare the "
            "common-scale DHW-HHW source."
        )

    # Never silently reconstruct the contrast from primary intensity metrics.
    return True


def build_qa_table(mun, trends):
    rows = []

    def add(name, value, expected, status):
        rows.append({
            "Check": name,
            "Observed": value,
            "Expected": expected,
            "Status": status,
        })

    vpd_units = trends["VPD_trend_decade"].attrs.get("units", "")
    add(
        "VPD trend units",
        vpd_units,
        "kPa decade-1",
        "PASS" if "kpa" in str(vpd_units).lower() else "FAIL",
    )

    source = trends.attrs.get("cross_regime_source", "")
    add(
        "DHW–HHW contrast source",
        source or "not declared",
        "DHW_minus_HHW_TmaxOnly_intensity_trend_decade",
        "PASS" if source == "DHW_minus_HHW_TmaxOnly_intensity_trend_decade" else "CHECK",
    )

    add(
        "Municipal DHW–HHW contrast finite N",
        int(pd.to_numeric(
            mun["dry_minus_humid_intensity_trend"],
            errors="coerce",
        ).notna().sum()),
        ">0",
        "PASS",
    )

    add(
        "Weighted Spearman inference",
        f"{N_PERMUTATIONS} permutations",
        "999 permutations",
        "PASS" if N_PERMUTATIONS == 999 else "FAIL",
    )

    add(
        "LOWESS bootstrap",
        f"{N_BOOT} replicates",
        "300 replicates",
        "PASS" if N_BOOT == 300 else "FAIL",
    )

    for col in OPTIONAL_CORRECTED_COLUMNS:
        add(
            f"Optional common-scale field: {col}",
            "present" if col in mun.columns else "absent",
            "optional",
            "PASS",
        )

    for col in OPTIONAL_SPATIAL_CONTROL_COLUMNS:
        add(
            f"Optional spatial-control field: {col}",
            "present" if col in mun.columns else "absent",
            "optional; spatial WLS variants are skipped if absent",
            "PASS",
        )

    return pd.DataFrame(rows)


# ============================================================
# Weighting / association helpers
# ============================================================

def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan

    values = values[ok]
    weights = weights[ok]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cdf, q)])


def weighted_mean_values(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan

    return float(np.average(values[ok], weights=weights[ok]))


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
    """
    Weighted Spearman rho with a two-sided permutation p-value.

    This intentionally replaces the classical t approximation used in the old
    supplementary workflow, which is not exact for a weighted rank statistic.
    """
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
        r_perm = _weighted_rank_correlation(
            x,
            rng.permutation(y),
            w,
        )
        if not np.isfinite(r_perm):
            continue

        valid_perm += 1
        if abs(r_perm) >= abs(observed):
            extreme += 1

    if valid_perm == 0:
        p = np.nan
    else:
        # +1 Monte-Carlo correction prevents a zero p-value.
        p = (extreme + 1.0) / (valid_perm + 1.0)

    return float(observed), float(p)


def prepare_relationship_data(
    df,
    x_col,
    y_col,
    regions=None,
):
    required = [
        x_col,
        y_col,
        "focus_region",
        "total_area_2024_ha",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing relationship fields: {missing}"
        )

    d = df[required].copy()

    if regions is not None:
        d = d[d["focus_region"].isin(regions)].copy()
    else:
        d = d[d["focus_region"].isin(REGION_ORDER)].copy()

    for col in [
        x_col,
        y_col,
        "total_area_2024_ha",
    ]:
        d[col] = pd.to_numeric(
            d[col],
            errors="coerce",
        )

    d = d.dropna(
        subset=[
            x_col,
            y_col,
            "total_area_2024_ha",
        ]
    )
    d = d[d["total_area_2024_ha"] > 0].copy()

    # Match Figure 03: 2nd-98th percentile restriction for both
    # variables to avoid leverage from extreme tails.
    if len(d) > 20:
        for col in [x_col, y_col]:
            q02, q98 = np.nanpercentile(
                d[col],
                [2, 98],
            )
            d = d[
                (d[col] >= q02)
                & (d[col] <= q98)
            ].copy()

    # Association weights are proportional to the square root of municipal area.
    weights = np.sqrt(
        d["total_area_2024_ha"].to_numpy(dtype=float)
    )
    d["weight"] = weights / np.nanmean(weights)

    return d


def lowess_with_bootstrap(
    df,
    x_col,
    y_col,
    frac,
):
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    q02, q98 = np.nanpercentile(x, [2, 98])
    x_grid = np.linspace(q02, q98, 180)

    order = np.argsort(x)
    fitted = lowess(
        y[order],
        x[order],
        frac=frac,
        it=1,
        return_sorted=True,
    )

    xu, idx = np.unique(
        fitted[:, 0],
        return_index=True,
    )
    yhat = np.interp(
        x_grid,
        xu,
        fitted[:, 1][idx],
    )

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_curves = []

    for _ in range(N_BOOT):
        idxb = rng.integers(
            0,
            len(x),
            size=len(x),
        )
        xb = x[idxb]
        yb = y[idxb]

        try:
            order_b = np.argsort(xb)
            fb = lowess(
                yb[order_b],
                xb[order_b],
                frac=frac,
                it=1,
                return_sorted=True,
            )

            xbu, ib = np.unique(
                fb[:, 0],
                return_index=True,
            )

            if len(xbu) >= 3:
                bootstrap_curves.append(
                    np.interp(
                        x_grid,
                        xbu,
                        fb[:, 1][ib],
                    )
                )
        except Exception:
            continue

    if len(bootstrap_curves) >= 20:
        boot = np.vstack(bootstrap_curves)
        lo = np.nanpercentile(
            boot,
            2.5,
            axis=0,
        )
        hi = np.nanpercentile(
            boot,
            97.5,
            axis=0,
        )
    else:
        lo = np.full_like(yhat, np.nan)
        hi = np.full_like(yhat, np.nan)

    rho, p = weighted_spearman(
        x,
        y,
        df["weight"].to_numpy(dtype=float),
    )

    return {
        "x": x_grid,
        "yhat": yhat,
        "lo": lo,
        "hi": hi,
        "rho": rho,
        "p": p,
        "n": len(df),
        "x_obs": x,
        "bootstrap_successes": len(bootstrap_curves),
    }


# ============================================================
# Supplementary Table S9
# ============================================================

def build_s9(mun, output_dir):
    """
    Municipal VPD/RH trend distributions.

    Both unweighted municipality summaries and municipal-area-weighted central estimates
    are reported. This makes the descriptive target explicit rather than mixing
    municipality-count and area-weighted interpretations.
    """
    rows = []

    def add_rows(subset, region_name):
        weights = pd.to_numeric(
            subset["total_area_2024_ha"],
            errors="coerce",
        ).to_numpy(dtype=float)

        for var, label, unit in [
            (
                "VPD_trend_decade",
                "VPD trend",
                "kPa decade^-1",
            ),
            (
                "RH_trend_decade",
                "Relative humidity trend",
                "% decade^-1",
            ),
        ]:
            vals = pd.to_numeric(
                subset[var],
                errors="coerce",
            ).to_numpy(dtype=float)

            ok = (
                np.isfinite(vals)
                & np.isfinite(weights)
                & (weights > 0)
            )
            if ok.sum() == 0:
                continue

            vv = vals[ok]
            ww = weights[ok]

            rows.append({
                "Region": region_name,
                "Variable": label,
                "Unit": unit,
                "N_municipalities": int(ok.sum()),
                "Municipality_median": float(np.nanmedian(vv)),
                "Municipality_Q1": float(np.nanpercentile(vv, 25)),
                "Municipality_Q3": float(np.nanpercentile(vv, 75)),
                "Municipality_IQR": float(
                    np.nanpercentile(vv, 75)
                    - np.nanpercentile(vv, 25)
                ),
                "Municipality_mean": float(np.nanmean(vv)),
                "Municipality_SD": float(
                    np.nanstd(vv, ddof=1)
                ) if len(vv) > 1 else np.nan,
                "Minimum": float(np.nanmin(vv)),
                "Maximum": float(np.nanmax(vv)),
                "Area_weighted_median": weighted_quantile(vv, ww, 0.50),
                "Area_weighted_Q1": weighted_quantile(vv, ww, 0.25),
                "Area_weighted_Q3": weighted_quantile(vv, ww, 0.75),
                "Area_weighted_mean": weighted_mean_values(vv, ww),
                "Note": (
                    "VPD inherited from Figure 03 in kPa decade^-1; "
                    "RH inherited in % decade^-1."
                ),
            })

    add_rows(mun, "Brazil")

    for region in REGION_ORDER:
        subset = mun[
            mun["focus_region"] == region
        ].copy()
        if not subset.empty:
            add_rows(subset, region)

    out = pd.DataFrame(rows)
    save_table_dual(
        out,
        output_dir,
        "Supplementary_Table_S9",
    )
    return out


# ============================================================
# Supplementary Tables S10-S11
# ============================================================

def build_relationships(mun):
    results = {}
    rows = []

    # S10: national VPD trend -> DHW–HHW contrast
    d = prepare_relationship_data(
        mun,
        "VPD_trend_decade",
        "dry_minus_humid_intensity_trend",
        regions=REGION_ORDER,
    )

    if len(d) >= MIN_N_FOR_RELATIONSHIP:
        res = lowess_with_bootstrap(
            d,
            "VPD_trend_decade",
            "dry_minus_humid_intensity_trend",
            frac=0.55,
        )
        results["vpd_to_dryhot"] = res

        rows.append({
            "relationship": "vpd_to_dryhot",
            "region": "Brazil",
            "n": res["n"],
            "weighted_spearman_rho": res["rho"],
            "permutation_p_value": res["p"],
            "n_permutations": N_PERMUTATIONS,
            "lowess_frac": 0.55,
            "bootstrap_replicates": N_BOOT,
            "bootstrap_successes": res["bootstrap_successes"],
        })
    else:
        results["vpd_to_dryhot"] = None

    # S11: regional transformed-land -> VPD
    results["transformed_to_vpd_regions"] = {}

    for region in REGIONAL_RELATIONSHIP_REGIONS:
        dreg = prepare_relationship_data(
            mun,
            "transformed_non_native_pct_2024",
            "VPD_trend_decade",
            regions=[region],
        )

        if len(dreg) < MIN_N_FOR_RELATIONSHIP:
            results["transformed_to_vpd_regions"][region] = None
            rows.append({
                "relationship": "transformed_to_vpd",
                "region": region,
                "n": len(dreg),
                "weighted_spearman_rho": np.nan,
                "permutation_p_value": np.nan,
                "n_permutations": N_PERMUTATIONS,
                "lowess_frac": 0.65,
                "bootstrap_replicates": N_BOOT,
                "bootstrap_successes": np.nan,
            })
            continue

        res = lowess_with_bootstrap(
            dreg,
            "transformed_non_native_pct_2024",
            "VPD_trend_decade",
            frac=0.65,
        )
        results["transformed_to_vpd_regions"][region] = res

        rows.append({
            "relationship": "transformed_to_vpd",
            "region": region,
            "n": res["n"],
            "weighted_spearman_rho": res["rho"],
            "permutation_p_value": res["p"],
            "n_permutations": N_PERMUTATIONS,
            "lowess_frac": 0.65,
            "bootstrap_replicates": N_BOOT,
            "bootstrap_successes": res["bootstrap_successes"],
        })

    return results, pd.DataFrame(rows)


def build_s10(results, output_dir):
    res = results.get("vpd_to_dryhot")
    rows = []

    if res is not None:
        x = np.asarray(
            res["x_obs"],
            dtype=float,
        )
        x = x[np.isfinite(x)]

        rows.append({
            "Relationship": "VPD trend vs DHW–HHW contrast",
            "Region": "Brazil",
            "N": int(res["n"]),
            "Weighted_Spearman_rho": float(res["rho"]),
            "Permutation_P_value": float(res["p"]),
            "Spearman_permutations": N_PERMUTATIONS,
            "Observed_x_Q02": float(np.nanpercentile(x, 2)),
            "Observed_x_Q50": float(np.nanpercentile(x, 50)),
            "Observed_x_Q98": float(np.nanpercentile(x, 98)),
            "LOWESS_frac": 0.55,
            "Bootstrap_replicates": N_BOOT,
            "Bootstrap_successes": int(res["bootstrap_successes"]),
            "Response_definition": (
                "DHW_TmaxOnly_intensity_trend_decade - "
                "HHW_TmaxOnly_intensity_trend_decade"
            ),
            "Response_unit": "Tmax-standardized severity decade^-1",
            "Interpretation": (
                "Descriptive nonlinear association; not causal attribution "
                "or a formal physical-threshold estimate."
            ),
        })

    out = pd.DataFrame(rows)
    save_table_dual(
        out,
        output_dir,
        "Supplementary_Table_S10",
    )
    return out


def build_s11(results, output_dir):
    rows = []
    reg = results.get(
        "transformed_to_vpd_regions",
        {},
    )

    for region in REGIONAL_RELATIONSHIP_REGIONS:
        res = reg.get(region)

        if res is None:
            rows.append({
                "Relationship": (
                    "Transformed/non-native land fraction vs VPD trend"
                ),
                "Region": region,
                "N": np.nan,
                "Weighted_Spearman_rho": np.nan,
                "Permutation_P_value": np.nan,
                "Spearman_permutations": N_PERMUTATIONS,
                "Observed_x_Q02": np.nan,
                "Observed_x_Q50": np.nan,
                "Observed_x_Q98": np.nan,
                "LOWESS_frac": 0.65,
                "Bootstrap_replicates": N_BOOT,
                "Bootstrap_successes": np.nan,
                "Status": "Insufficient data",
            })
            continue

        x = np.asarray(
            res["x_obs"],
            dtype=float,
        )
        x = x[np.isfinite(x)]

        rows.append({
            "Relationship": (
                "Transformed/non-native land fraction vs VPD trend"
            ),
            "Region": region,
            "N": int(res["n"]),
            "Weighted_Spearman_rho": float(res["rho"]),
            "Permutation_P_value": float(res["p"]),
            "Spearman_permutations": N_PERMUTATIONS,
            "Observed_x_Q02": float(np.nanpercentile(x, 2)),
            "Observed_x_Q50": float(np.nanpercentile(x, 50)),
            "Observed_x_Q98": float(np.nanpercentile(x, 98)),
            "LOWESS_frac": 0.65,
            "Bootstrap_replicates": N_BOOT,
            "Bootstrap_successes": int(res["bootstrap_successes"]),
            "Status": "OK",
        })

    out = pd.DataFrame(rows)
    save_table_dual(
        out,
        output_dir,
        "Supplementary_Table_S11",
    )
    return out


# ============================================================
# Supplementary Table S12: controlled diagnostic WLS
# ============================================================

def zscore_column(d, col, weight_col="weight"):
    x = pd.to_numeric(
        d[col],
        errors="coerce",
    ).to_numpy(dtype=float)
    w = pd.to_numeric(
        d[weight_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if ok.sum() < 3:
        return np.full(len(d), np.nan)

    mu = np.average(x[ok], weights=w[ok])
    sd = np.sqrt(
        np.average(
            (x[ok] - mu) ** 2,
            weights=w[ok],
        )
    )

    if not np.isfinite(sd) or sd <= 0:
        return np.full(len(d), np.nan)

    return (x - mu) / sd


def fit_controlled_wls(
    df,
    response,
    predictor,
    controls,
    model_name,
):
    required = [
        response,
        predictor,
        "weight",
    ] + controls

    d = df.copy()

    numeric_candidates = [
        response,
        predictor,
        "weight",
        "rep_lat",
        "rep_lon",
    ]
    for col in numeric_candidates:
        if col in d.columns:
            d[col] = pd.to_numeric(
                d[col],
                errors="coerce",
            )

    missing = [c for c in required if c not in d.columns]
    if missing:
        return [{
            "model": model_name,
            "response": response,
            "predictor": predictor,
            "n": 0,
            "term": "MODEL_NOT_FIT_MISSING_FIELDS",
            "missing_fields": ", ".join(missing),
            "coef_per_1sd_predictor": np.nan,
            "se_HC3": np.nan,
            "p_HC3": np.nan,
            "r2": np.nan,
            "aic": np.nan,
        }]

    d = d.dropna(
        subset=required
    ).copy()
    d = d[d["weight"] > 0].copy()

    if len(d) < 80:
        return [{
            "model": model_name,
            "response": response,
            "predictor": predictor,
            "n": len(d),
            "term": "MODEL_NOT_FIT_N_LT_80",
            "coef_per_1sd_predictor": np.nan,
            "se_HC3": np.nan,
            "p_HC3": np.nan,
            "r2": np.nan,
            "aic": np.nan,
        }]

    zcol = predictor + "_z"
    d[zcol] = zscore_column(
        d,
        predictor,
        weight_col="weight",
    )
    d = d.dropna(
        subset=[
            zcol,
            response,
            "weight",
        ]
    ).copy()

    formula_terms = [zcol]

    for control in controls:
        if control == "dominant_biome":
            formula_terms.append("C(dominant_biome)")
        elif control == "focus_region":
            formula_terms.append("C(focus_region)")
        elif control in {"rep_lat", "rep_lon"}:
            formula_terms.append(control)

    formula = (
        f"{response} ~ "
        + " + ".join(formula_terms)
    )

    try:
        fit = sm.WLS.from_formula(
            formula,
            data=d,
            weights=d["weight"],
        ).fit(cov_type="HC3")

        ci = fit.conf_int()

        return [{
            "model": model_name,
            "formula": formula,
            "response": response,
            "predictor": predictor,
            "n": int(fit.nobs),
            "term": zcol,
            "coef_per_1sd_predictor": float(
                fit.params.get(zcol, np.nan)
            ),
            "se_HC3": float(
                fit.bse.get(zcol, np.nan)
            ),
            "t_HC3": float(
                fit.tvalues.get(zcol, np.nan)
            ),
            "p_HC3": float(
                fit.pvalues.get(zcol, np.nan)
            ),
            "ci95_low": float(
                ci.loc[zcol, 0]
            ) if zcol in fit.params.index else np.nan,
            "ci95_high": float(
                ci.loc[zcol, 1]
            ) if zcol in fit.params.index else np.nan,
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "aic": float(fit.aic),
            "weighting": "weights proportional to sqrt(municipality area)",
            "covariance": "HC3 heteroscedasticity-robust standard errors",
            "interpretation_note": (
                "Conditional spatial association; not causal attribution or mediation."
            ),
        }]

    except Exception as exc:
        return [{
            "model": model_name,
            "formula": formula,
            "response": response,
            "predictor": predictor,
            "n": len(d),
            "term": f"FIT_FAILED: {exc}",
            "coef_per_1sd_predictor": np.nan,
            "se_HC3": np.nan,
            "p_HC3": np.nan,
            "r2": np.nan,
            "aic": np.nan,
        }]


def build_s12(mun, output_dir):
    """
    Build controlled diagnostic WLS tables aligned with Figure 03.

    rep_lat/rep_lon are optional. Spatially adjusted variants are fitted only
    when both are available; core biome/focus-region models remain valid
    without them.
    """
    df = mun.copy()

    # WLS weights are proportional to the square root of municipal area.
    df["weight"] = np.sqrt(
        pd.to_numeric(
            df["total_area_2024_ha"],
            errors="coerce",
        )
    )
    df["weight"] = (
        df["weight"]
        / np.nanmean(df["weight"])
    )

    df = df[
        df["focus_region"].isin(REGION_ORDER)
    ].copy()

    # Core models do not require representative coordinates.
    model_specs = [
        ("agri_to_vpd_biomeFE", "VPD_trend_decade", "agri_gain_pct_points", ["dominant_biome"]),
        ("agri_to_vpd_focusFE", "VPD_trend_decade", "agri_gain_pct_points", ["focus_region"]),
        ("vpd_to_dryhot_biomeFE", "dry_minus_humid_intensity_trend", "VPD_trend_decade", ["dominant_biome"]),
        ("vpd_to_dryhot_focusFE", "dry_minus_humid_intensity_trend", "VPD_trend_decade", ["focus_region"]),
        ("agri_to_dryhot_biomeFE", "dry_minus_humid_intensity_trend", "agri_gain_pct_points", ["dominant_biome"]),
        ("agri_to_dryhot_focusFE", "dry_minus_humid_intensity_trend", "agri_gain_pct_points", ["focus_region"]),
        ("transformed_to_vpd_biomeFE", "VPD_trend_decade", "transformed_non_native_pct_2024", ["dominant_biome"]),
        ("transformed_to_dryhot_biomeFE", "dry_minus_humid_intensity_trend", "transformed_non_native_pct_2024", ["dominant_biome"]),
    ]

    has_spatial_controls = all(
        col in df.columns for col in OPTIONAL_SPATIAL_CONTROL_COLUMNS
    )

    if has_spatial_controls:
        print("[INFO] rep_lat/rep_lon available: including spatial WLS variants.")
        model_specs.extend([
            ("agri_to_vpd_biomeFE_spatial", "VPD_trend_decade", "agri_gain_pct_points", ["dominant_biome", "rep_lat", "rep_lon"]),
            ("vpd_to_dryhot_biomeFE_spatial", "dry_minus_humid_intensity_trend", "VPD_trend_decade", ["dominant_biome", "rep_lat", "rep_lon"]),
            ("agri_to_dryhot_biomeFE_spatial", "dry_minus_humid_intensity_trend", "agri_gain_pct_points", ["dominant_biome", "rep_lat", "rep_lon"]),
            ("transformed_to_vpd_biomeFE_spatial", "VPD_trend_decade", "transformed_non_native_pct_2024", ["dominant_biome", "rep_lat", "rep_lon"]),
            ("transformed_to_dryhot_biomeFE_spatial", "dry_minus_humid_intensity_trend", "transformed_non_native_pct_2024", ["dominant_biome", "rep_lat", "rep_lon"]),
        ])
    else:
        missing_spatial = [
            col for col in OPTIONAL_SPATIAL_CONTROL_COLUMNS
            if col not in df.columns
        ]
        print(
            "[INFO] Spatial-control columns absent "
            f"({', '.join(missing_spatial)}). "
            "Skipping only spatially adjusted WLS variants."
        )


    rows = []
    for model_name, response, predictor, controls in model_specs:
        rows.extend(
            fit_controlled_wls(
                df,
                response,
                predictor,
                controls,
                model_name,
            )
        )

    out = pd.DataFrame(rows)
    save_table_dual(
        out,
        output_dir,
        "Supplementary_Table_S12",
    )
    return out


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    figure3_dir = Path(
        args.figure3_output_dir
    ).expanduser().resolve()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else figure3_dir / "supplementary"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    municipality_csv = (
        figure3_dir
        / "figure_03_municipality_mechanism_table_1990_2024.csv"
    )
    trend_nc = (
        figure3_dir
        / "figure_03_ERA5_mechanistic_trends_with_dryhot_1990_2024.nc"
    )

    for path, label in [
        (municipality_csv, "Figure 03 municipality table"),
        (trend_nc, "Figure 03 gridded trend product"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    print("=" * 86)
    print("FIGURE 03 — SUPPLEMENTARY MATERIAL")
    print("=" * 86)
    print(f"Figure 03 products : {figure3_dir}")
    print(f"Municipality table : {municipality_csv}")
    print(f"Trend product      : {trend_nc}")
    print(f"Output             : {output_dir}")
    print("DHW-HHW metric    : common Tmax-only contrast")
    print(f"Spearman inference : {N_PERMUTATIONS} permutations")
    print("=" * 86)

    mun = pd.read_csv(
        municipality_csv
    )

    with safe_open_dataset(
        trend_nc,
        decode_times=False,
        chunks=None,
    ) as trends:
        validate_figure3_products(
            mun,
            trends,
        )
        qa = build_qa_table(
            mun,
            trends,
        )

    qa_path = (
        output_dir
        / "figure_03_supplementary_QA.csv"
    )
    qa.to_csv(
        qa_path,
        index=False,
    )
    print(f"[OK] Saved: {qa_path}")

    s9 = build_s9(
        mun,
        output_dir,
    )

    relationships, relationship_summary = build_relationships(
        mun
    )

    summary_path = (
        output_dir
        / "figure_03_supplementary_lowess_summary.csv"
    )
    relationship_summary.to_csv(
        summary_path,
        index=False,
    )
    print(f"[OK] Saved: {summary_path}")

    s10 = build_s10(
        relationships,
        output_dir,
    )
    s11 = build_s11(
        relationships,
        output_dir,
    )
    s12 = build_s12(
        mun,
        output_dir,
    )

    print("\n[DONE]")
    print("Generated:")
    print(f"  Supplementary Table S9  : {len(s9)} rows")
    print(f"  Supplementary Table S10 : {len(s10)} rows")
    print(f"  Supplementary Table S11 : {len(s11)} rows")
    print(f"  Supplementary Table S12 : {len(s12)} rows")
    print(f"  Output directory        : {output_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

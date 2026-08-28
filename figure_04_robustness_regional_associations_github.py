#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 04 — Robustness, temporal stability, regional structure, and controlled
association diagnostics for preferential dry-hot heatwave amplification in
Brazil, 1990–2024.

This workflow combines outputs from Figures 01–03 without reconstructing
upstream metrics.

Panel (a): threshold and event-definition sensitivity
    Uses Figure 01 Supplementary Table S3 and the common Tmax-only contrast:
        DHW_TmaxOnly_intensity_trend_decade
        - HHW_TmaxOnly_intensity_trend_decade
    Units: Tmax-standardized severity decade^-1.

Panel (b): temporal robustness of primary DHW intensity
    Uses Figure 01 Supplementary Table S4.
    This panel evaluates the primary DHW intensity metric and is not a
    DHW–HHW difference-of-slopes analysis.

Panel (c): regional DHW–HHW structure
    Uses the Figure 02 municipality table. Regional medians, interquartile
    ranges, and positive fractions are weighted by municipal area.

Panel (d): controlled association diagnostics
    Uses Figure 03 weighted least-squares (WLS) models. These models use
    weights proportional to the square root of municipal area and HC3
    heteroscedasticity-robust standard errors. They quantify conditional
    spatial associations and are not interpreted as causal or mediation
    analyses.

Required inputs
---------------
--fig1-sensitivity
    Figure 01 Supplementary_Table_S3.csv.
--fig1-extreme-year
    Figure 01 Supplementary_Table_S4.csv.
--fig2-municipality
    Figure 02 municipality-level CSV.
--fig3-controlled
    Figure 03 controlled-regression CSV or Supplementary_Table_S12.csv.
--out-dir
    Output directory for Figure 04.

Outputs
-------
Figure_4.jpeg
Figure_4.pdf
Figure_4_summary_tables.xlsx
Figure_4_compiled_data.csv
Supplementary_Table_S13.csv/.xlsx
Supplementary_Table_S14.csv/.xlsx
Supplementary_Table_S15.csv/.xlsx
Supplementary_Table_S16.csv/.xlsx
software_metadata.json

Usage
-----
python figure_04_robustness_regional_associations.py \
    --fig1-sensitivity /path/to/Supplementary_Table_S3.csv \
    --fig1-extreme-year /path/to/Supplementary_Table_S4.csv \
    --fig2-municipality /path/to/figure_02_municipality_dhw_hhw_contrast_1990_2024.csv \
    --fig3-controlled /path/to/Supplementary_Table_S12.csv \
    --out-dir ./outputs/figure_04
"""

import os
import re
import glob
import json
import warnings
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================


SOFTWARE_VERSION = "2.1.0"
FIG_DPI = 350
FIG_FONT_SIZE = 14
RANDOM_SEED = 42

REGION_ORDER = [
    "Amazon",
    "Cerrado",
    "MATOPIBA",
    "Semi-arid Northeast",
    "Pantanal",
    "Atlantic Forest",
    "Pampa",
    "Urban Southeast",
]

REGION_ALIASES = {
    "Amazônia": "Amazon",
    "Amazonia": "Amazon",
    "Amazon": "Amazon",
    "AMAZÔNIA": "Amazon",
    "AMAZONIA": "Amazon",
    "CERRADO": "Cerrado",
    "Caatinga": "Semi-arid Northeast",
    "CAATINGA": "Semi-arid Northeast",
    "Semiárido": "Semi-arid Northeast",
    "Semiarido": "Semi-arid Northeast",
    "SEMIARIDO": "Semi-arid Northeast",
    "Atlantic Forest": "Atlantic Forest",
    "Mata Atlântica": "Atlantic Forest",
    "Mata Atlantica": "Atlantic Forest",
    "MATA ATLÂNTICA": "Atlantic Forest",
    "MATA ATLANTICA": "Atlantic Forest",
    "Pampas": "Pampa",
    "Pampa": "Pampa",
    "PAMPA": "Pampa",
    "Pantanal": "Pantanal",
    "PANTANAL": "Pantanal",
    "Sudeste urbano": "Urban Southeast",
    "SUDESTE URBANO": "Urban Southeast",
    "Urban Southeast": "Urban Southeast",
}

plt.rcParams.update({
    "font.size": FIG_FONT_SIZE,
    "axes.titlesize": FIG_FONT_SIZE,
    "axes.labelsize": FIG_FONT_SIZE,
    "xtick.labelsize": FIG_FONT_SIZE,
    "ytick.labelsize": FIG_FONT_SIZE,
    "legend.fontsize": FIG_FONT_SIZE,
    "figure.titlesize": FIG_FONT_SIZE,
})

PANEL_A_DEFAULT_ORDER = [
    "primary_P75_P95_minlen3",
    "DHW_VPD_P70_minlen3",
    "DHW_VPD_P80_minlen3",
    "HHW_TWB_P90_minlen3",
    "primary_thresholds_minlen2",
    "primary_thresholds_minlen4",
]


# ============================================================
# Utility helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_table_dual(df: pd.DataFrame, out_dir: str, stem: str):
    """Save a table as CSV and XLSX using the Supplementary_Table_S* naming convention."""
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    xlsx_path = os.path.join(out_dir, f"{stem}.xlsx")
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save {stem}.xlsx: {exc}")
    print(f"[OK] Saved {stem}: {csv_path}")
    return csv_path, xlsx_path


def normalize_text(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def first_existing(candidates: List[str], required: bool = False, label: str = "file") -> Optional[str]:
    expanded = []
    for c in candidates:
        expanded.extend(glob.glob(c, recursive=True))
    expanded = sorted(set(expanded))
    for path in expanded:
        if os.path.exists(path):
            print(f"[INFO] Using {label}: {path}")
            return path
    if required:
        msg = "No valid " + label + " found among:\n" + "\n".join(candidates)
        raise FileNotFoundError(msg)
    print(f"[WARN] No valid {label} found.")
    return None


def read_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def resolve_fig3_controlled_path(requested_path: str) -> str:
    """
    Resolve equivalent Figure 03 controlled-regression outputs.

    If a nested supplementary/Supplementary_Table_S12.csv path is supplied but
    the table was saved directly in Figure_03, the root
    location is used. Legacy Figure 03 directories are never searched.
    """
    requested = os.path.abspath(requested_path)
    if os.path.isfile(requested):
        return requested

    parent = os.path.dirname(requested)
    basename = os.path.basename(requested)

    if os.path.basename(parent).lower() == "supplementary":
        root = os.path.dirname(parent)
    else:
        root = parent

    candidates = [
        os.path.join(root, basename),
        os.path.join(root, "Supplementary_Table_S12.csv"),
        os.path.join(root, "figure_03_controlled_mechanism_regressions_1990_2024.csv"),
        os.path.join(root, "supplementary", "Supplementary_Table_S12.csv"),
    ]

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    for candidate in candidates:
        if os.path.isfile(candidate):
            print(
                "[INFO] Requested Figure 03 controlled table was not found; "
                f"using equivalent Figure 03 product: {candidate}"
            )
            return candidate

    raise FileNotFoundError(
        "Figure 03 controlled-regression table not found. "
        "Requested: " + requested + "\nChecked:\n" + "\n".join(candidates)
    )


def find_col(df: pd.DataFrame, exact: List[str] = None, contains: List[str] = None,
             exclude: List[str] = None, required: bool = False, label: str = "column") -> Optional[str]:
    exact = exact or []
    contains = contains or []
    exclude = exclude or []
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    for e in exact:
        if e.lower() in lower:
            return lower[e.lower()]

    contains_l = [s.lower() for s in contains]
    exclude_l = [s.lower() for s in exclude]
    for c in cols:
        cl = c.lower()
        if all(s in cl for s in contains_l) and not any(s in cl for s in exclude_l):
            return c

    if required:
        raise ValueError(f"Could not find {label}. Available columns: {cols}")
    return None


def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(values, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    ok = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    if ok.sum() == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    values = values[ok]
    sample_weight = sample_weight[ok]
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    cdf /= np.sum(sample_weight)
    return np.interp(quantiles, cdf, values)


def clean_region_name(x):
    s = normalize_text(x)
    return REGION_ALIASES.get(s, s)


def safe_p_to_text(p):
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 1e-99:
        return "p<1e−99"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def color_by_sign(v, pos="#B2182B", neg="#2166AC", neutral="0.45"):
    if not np.isfinite(v):
        return neutral
    return pos if v >= 0 else neg


def clean_scenario_label(label: str) -> str:
    """Readable labels for panel-a sensitivity tests."""
    s = str(label)
    sl = s.lower()

    if "primary" in sl and "minlen3" in sl:
        return "Primary\n(P75/P95, ≥3 d)"
    if "primary_figure1" in sl:
        return "Primary\n(P75/P95, ≥3 d)"
    if "minlen2" in sl:
        return "Events ≥2 d"
    if "minlen4" in sl:
        return "Events ≥4 d"
    if "vpd" in sl and "70" in sl:
        return "DHW VPD ≥P70"
    if "vpd" in sl and "75" in sl:
        return "DHW VPD ≥P75"
    if "vpd" in sl and "80" in sl:
        return "DHW VPD ≥P80"
    if "twb" in sl and "90" in sl:
        return "HHW Twb ≥P90"
    if "twb" in sl and "95" in sl:
        return "HHW Twb ≥P95"

    s = re.sub(r"ALL_?", "", s, flags=re.I)
    s = s.replace("_", " ")
    s = s.replace("minlen", "≥")
    s = s.replace("DHW VPDmean", "DHW VPD")
    s = s.replace("HHW Twbmax", "HHW Twb")
    return s


def clean_period_label(label: str) -> str:
    """Readable labels for period/truncation tests."""
    s = str(label)
    m = re.search(r"(1990)[_–-](20\d{2})", s)
    if m:
        return f"{m.group(1)}–{m.group(2)}"
    m = re.search(r"(20\d{2})", s)
    if "1990" in s and m:
        return f"1990–{m.group(1)}"
    s = s.replace("truncated_", "").replace("primary_", "").replace("_", " ")
    return s


# ============================================================
# Panel A — threshold sensitivity
# ============================================================

def build_panel_a_sensitivity(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Build panel (a) from Figure 01 Supplementary Table S3.

    The table must represent the common-scale Tmax-only DHW-HHW contrast.
    Primary regime-specific intensity medians are never subtracted here.
    """
    required = [
        "scenario",
        "region",
        "kind",
        "metric",
        "units",
        "trend_median",
        "trend_q25",
        "trend_q75",
    ]

    if df is None or df.empty:
        raise ValueError("Figure 01 Supplementary Table S3 is empty.")

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    missing = [col for col in required if col not in d.columns]
    if missing:
        raise ValueError(
            "Figure 01 S3 is incompatible with the expected common-scale sensitivity table. "
            "Missing columns: " + ", ".join(missing)
        )

    region = d["region"].astype(str).str.strip().str.lower()
    kind = d["kind"].astype(str).str.strip()
    metric = d["metric"].astype(str).str.strip()

    mask = (
        region.isin(["brazil", "brasil", "national"])
        & kind.eq("DHW_minus_HHW")
        & metric.eq("TmaxOnly_intensity")
    )
    d = d.loc[mask].copy()

    if d.empty:
        raise ValueError(
            "Figure 01 S3 contains no Brazil rows for "
            "kind='DHW_minus_HHW', metric='TmaxOnly_intensity'."
        )

    # Validate units rather than silently relabelling another metric.
    units_norm = (
        d["units"]
        .astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    if not units_norm.str.contains("standardizedseveritydecade-1", regex=False).all():
        raise ValueError(
            "Figure 01 S3 DHW-HHW sensitivity rows do not have the expected "
            "standardized-severity-per-decade units."
        )

    out = pd.DataFrame({
        "scenario": d["scenario"].astype(str),
        "median": pd.to_numeric(d["trend_median"], errors="coerce"),
        "q25": pd.to_numeric(d["trend_q25"], errors="coerce"),
        "q75": pd.to_numeric(d["trend_q75"], errors="coerce"),
        "n": pd.to_numeric(d.get("n_grid_cells", np.nan), errors="coerce"),
        "source_note": (
            "Figure 01 S3; common Tmax-only DHW-HHW contrast"
        ),
        "units": "Tmax-standardized severity decade-1",
    })

    out = out.dropna(subset=["median"]).drop_duplicates(
        subset=["scenario"],
        keep="first",
    )

    order = {
        scenario.lower(): idx
        for idx, scenario in enumerate(PANEL_A_DEFAULT_ORDER)
    }
    out["_order"] = (
        out["scenario"]
        .str.lower()
        .map(order)
        .fillna(999)
    )
    out = (
        out.sort_values(["_order", "scenario"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return out


# ============================================================
# Panel B — recent-year robustness
# ============================================================

def build_panel_b_extreme_year(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Build temporal robustness from Figure 01 S4.

    S4 tests the PRIMARY DHW intensity trend:
        0.5*z+(Tmax) + 0.5*z+(VPD)
    under truncated periods and leave-one-year-out diagnostics.
    """
    empty = {
        "periods": pd.DataFrame(
            columns=["label", "median", "q25", "q75"]
        ),
        "leave_one_year_out": pd.DataFrame(
            columns=["excluded_year", "median"]
        ),
        "loyo_summary": pd.DataFrame(
            columns=["label", "median", "min", "max"]
        ),
    }

    if df is None or df.empty:
        raise ValueError("Figure 01 Supplementary Table S4 is empty.")

    required = [
        "scenario",
        "region",
        "kind",
        "metric",
        "units",
        "trend_median",
        "trend_q25",
        "trend_q75",
    ]
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    missing = [col for col in required if col not in d.columns]
    if missing:
        raise ValueError(
            "Figure 01 S4 is incompatible with the expected temporal-robustness table. "
            "Missing columns: " + ", ".join(missing)
        )

    region = d["region"].astype(str).str.strip().str.lower()
    mask = (
        region.isin(["brazil", "brasil", "national"])
        & d["kind"].astype(str).str.strip().eq("DHW")
        & d["metric"].astype(str).str.strip().eq("primary_intensity")
    )
    d = d.loc[mask].copy()

    if d.empty:
        raise ValueError(
            "Figure 01 S4 contains no Brazil rows for "
            "kind='DHW', metric='primary_intensity'."
        )

    d["trend_median"] = pd.to_numeric(d["trend_median"], errors="coerce")
    d["trend_q25"] = pd.to_numeric(d["trend_q25"], errors="coerce")
    d["trend_q75"] = pd.to_numeric(d["trend_q75"], errors="coerce")

    period_map = {
        "primary_1990_2024": "1990–2024",
        "truncated_1990_2022": "1990–2022",
        "truncated_1990_2023": "1990–2023",
    }

    periods = []
    for scenario, label in period_map.items():
        sub = d[d["scenario"].astype(str).eq(scenario)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if not np.isfinite(row["trend_median"]):
            continue
        periods.append({
            "label": label,
            "median": float(row["trend_median"]),
            "q25": float(row["trend_q25"]) if np.isfinite(row["trend_q25"]) else np.nan,
            "q75": float(row["trend_q75"]) if np.isfinite(row["trend_q75"]) else np.nan,
        })

    def scenario_value(name):
        sub = d[d["scenario"].astype(str).eq(name)]
        if sub.empty:
            return np.nan
        return float(pd.to_numeric(sub.iloc[0]["trend_median"], errors="coerce"))

    loyo_med = scenario_value("LOYO_median")
    loyo_min = scenario_value("LOYO_min")
    loyo_max = scenario_value("LOYO_max")

    if all(np.isfinite(v) for v in [loyo_med, loyo_min, loyo_max]):
        loyo_summary = pd.DataFrame([{
            "label": "Leave-one-\nyear-out",
            "median": loyo_med,
            "min": loyo_min,
            "max": loyo_max,
            "source_note": "Figure 01 S4 leave-one-year-out summary",
        }])
    else:
        loyo_summary = empty["loyo_summary"]

    return {
        "periods": pd.DataFrame(periods),
        "leave_one_year_out": empty["leave_one_year_out"],
        "loyo_summary": loyo_summary,
    }


# ============================================================
# Panel C — regional emergence from municipality table
# ============================================================

def build_panel_c_region_emergence(
    fig2_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Regional summaries from the Figure 02 municipality table.

    Median, IQR, and positive fraction are weighted by municipality area.
    """
    if fig2_df is None or fig2_df.empty:
        raise ValueError("Figure 02 municipality table is empty.")

    d = fig2_df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    required = [
        "dry_minus_humid_intensity_trend",
        "focus_region",
        "total_area_2024_ha",
    ]
    missing = [col for col in required if col not in d.columns]
    if missing:
        raise ValueError(
            "Figure 02 municipality table is missing: "
            + ", ".join(missing)
        )

    # Presence of the common components is a compatibility check, not used to
    # reconstruct the response.
    common_components = [
        "DHW_TmaxOnly_intensity_trend_decade",
        "HHW_TmaxOnly_intensity_trend_decade",
    ]
    missing_components = [
        col for col in common_components if col not in d.columns
    ]
    if missing_components:
        raise ValueError(
            "Figure 02 municipality table does not contain the required common-scale "
            "DHW/HHW components: " + ", ".join(missing_components)
        )

    d["region"] = d["focus_region"].map(clean_region_name)
    d["value"] = pd.to_numeric(
        d["dry_minus_humid_intensity_trend"],
        errors="coerce",
    )
    d["weight"] = pd.to_numeric(
        d["total_area_2024_ha"],
        errors="coerce",
    )

    rows = []

    for region, g in d.groupby("region"):
        vals = g["value"].to_numpy(dtype=float)
        weights = g["weight"].to_numpy(dtype=float)

        ok = (
            np.isfinite(vals)
            & np.isfinite(weights)
            & (weights > 0)
        )
        if ok.sum() < 5:
            continue

        vals = vals[ok]
        weights = weights[ok]

        q25, med, q75 = weighted_quantile(
            vals,
            [0.25, 0.50, 0.75],
            weights,
        )

        positive = vals > 0
        positive_area_fraction = float(
            np.sum(weights[positive]) / np.sum(weights)
        )

        rows.append({
            "region": region,
            "median": float(med),
            "q25": float(q25),
            "q75": float(q75),
            "positive_area_fraction": positive_area_fraction,
            "n": int(ok.sum()),
            "units": "Tmax-standardized severity decade-1",
        })

    out = pd.DataFrame(rows)

    order = {region: idx for idx, region in enumerate(REGION_ORDER)}
    out["_order"] = out["region"].map(order).fillna(999)
    out = (
        out.sort_values(["_order", "region"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return out


# ============================================================
# Panel D — controlled association diagnostics
# ============================================================

def build_panel_d_associations(
    controlled_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize Figure 03 controlled WLS association diagnostics.

    Spatially controlled biome-FE models are preferred when present. If
    representative coordinates were not saved and spatial variants are absent,
    the corresponding corrected non-spatial biome-FE model is used.
    """
    if controlled_df is None or controlled_df.empty:
        raise ValueError("Figure 03 controlled-regression table is empty.")

    d = controlled_df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    required = [
        "model",
        "coef_per_1sd_predictor",
        "p_HC3",
    ]
    missing = [col for col in required if col not in d.columns]
    if missing:
        raise ValueError(
            "Figure 03 controlled-regression table is missing: "
            + ", ".join(missing)
        )

    specs = [
        (
            [
                "transformed_to_vpd_biomeFE_spatial",
                "transformed_to_vpd_biomeFE",
            ],
            "Transformed land 2024 → VPD trend",
            "kPa decade$^{-1}$ per 1 s.d.",
        ),
        (
            [
                "vpd_to_dryhot_biomeFE_spatial",
                "vpd_to_dryhot_biomeFE",
            ],
            "VPD trend → DHW−HHW",
            "Tmax-standardized severity decade$^{-1}$ per 1 s.d.",
        ),
        (
            [
                "transformed_to_dryhot_biomeFE_spatial",
                "transformed_to_dryhot_biomeFE",
            ],
            "Transformed land 2024 → DHW−HHW",
            "Tmax-standardized severity decade$^{-1}$ per 1 s.d.",
        ),
    ]

    rows = []

    for model_candidates, label, units in specs:
        selected = None
        for model_name in model_candidates:
            sub = d[
                d["model"].astype(str).eq(model_name)
            ]
            if not sub.empty:
                selected = sub.iloc[0]
                break

        if selected is None:
            raise ValueError(
                f"Required Figure 03 model not found for: {label}. "
                f"Tried: {model_candidates}"
            )

        coef = pd.to_numeric(
            selected["coef_per_1sd_predictor"],
            errors="coerce",
        )
        if not np.isfinite(coef):
            raise ValueError(
                f"Selected Figure 03 model has no finite coefficient: "
                f"{selected['model']}"
            )

        rows.append({
            "association": label,
            "model": str(selected["model"]),
            "coef": float(coef),
            "ci_low": float(pd.to_numeric(
                selected.get("ci95_low", np.nan),
                errors="coerce",
            )),
            "ci_high": float(pd.to_numeric(
                selected.get("ci95_high", np.nan),
                errors="coerce",
            )),
            "p": float(pd.to_numeric(
                selected.get("p_HC3", np.nan),
                errors="coerce",
            )),
            "n": int(pd.to_numeric(
                selected.get("n", np.nan),
                errors="coerce",
            )) if np.isfinite(pd.to_numeric(
                selected.get("n", np.nan),
                errors="coerce",
            )) else np.nan,
            "response_units": units,
            "interpretation": (
                "Conditional spatial association; not causal attribution or mediation."
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def annotate_missing(ax, title):
    ax.set_title(title, fontsize=FIG_FONT_SIZE, fontweight="bold", loc="left")
    ax.text(
        0.5, 0.5,
        "Required table not found\nor columns could not be inferred",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=FIG_FONT_SIZE,
        color="0.35",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9)
    )
    ax.set_axis_off()


def plot_panel_a(ax, sens: pd.DataFrame):
    ax.set_title("(a) Threshold sensitivity of DHW−HHW", fontsize=FIG_FONT_SIZE, fontweight="bold", loc="left")

    if sens is None or sens.empty:
        annotate_missing(ax, "(a) Threshold sensitivity of DHW−HHW")
        return

    d = sens.copy().reset_index(drop=True)
    labels = [clean_scenario_label(x) for x in d["scenario"].astype(str).tolist()]
    y = np.arange(len(d))[::-1]
    med = d["median"].to_numpy(dtype=float)
    q25 = d["q25"].to_numpy(dtype=float)
    q75 = d["q75"].to_numpy(dtype=float)

    for i, yi in enumerate(y):
        c = color_by_sign(med[i])
        if np.isfinite(q25[i]) and np.isfinite(q75[i]):
            ax.plot([q25[i], q75[i]], [yi, yi], color=c, linewidth=3.0, alpha=0.35, solid_capstyle="round")
        ax.scatter(med[i], yi, s=46, color=c, edgecolor="white", linewidth=0.7, zorder=5)

    ax.axvline(0, color="0.3", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FIG_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.18, linewidth=0.45)
    ax.set_xlabel("National DHW−HHW response\n(Tmax-standardized severity decade$^{-1}$)", fontsize=FIG_FONT_SIZE)
    ax.text(
        0.98, 0.04,
        "Point: national median; line: IQR",
        transform=ax.transAxes,
        fontsize=FIG_FONT_SIZE,
        ha="right",
        va="bottom",
        color="0.28",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
    )


def plot_panel_b(ax, robust: Dict[str, pd.DataFrame]):
    ax.set_title("(b) Temporal robustness of DHW intensification", fontsize=FIG_FONT_SIZE, fontweight="bold", loc="left")

    periods = robust.get("periods", pd.DataFrame())
    loo = robust.get("leave_one_year_out", pd.DataFrame())
    loyo_summary = robust.get("loyo_summary", pd.DataFrame())

    if (periods is None or periods.empty) and (loo is None or loo.empty) and (loyo_summary is None or loyo_summary.empty):
        annotate_missing(ax, "(b) Robustness to truncated analysis periods")
        return

    x_positions = []
    x_labels = []
    values_for_limits = []

    # Period/truncated tests as points.
    if periods is not None and not periods.empty:
        p = periods.copy().reset_index(drop=True)

        def end_year(label):
            matches = re.findall(r"(20\d{2})", str(label))
            return int(matches[-1]) if matches else 9999

        p["_order"] = p["label"].map(end_year)
        p = p.sort_values(["_order", "label"]).drop(columns="_order")

        for _, r in p.iterrows():
            xpos = len(x_positions)
            med = float(r["median"])
            q25 = float(r["q25"]) if np.isfinite(r.get("q25", np.nan)) else np.nan
            q75 = float(r["q75"]) if np.isfinite(r.get("q75", np.nan)) else np.nan
            c = color_by_sign(med)

            if np.isfinite(q25) and np.isfinite(q75):
                ax.plot([xpos, xpos], [q25, q75], color=c, linewidth=3.0, alpha=0.35)
            ax.scatter(xpos, med, s=56, color=c, edgecolor="white", linewidth=0.7, zorder=5)

            x_positions.append(xpos)
            x_labels.append(clean_period_label(r["label"]))
            values_for_limits.append(med)
            if np.isfinite(q25):
                values_for_limits.append(q25)
            if np.isfinite(q75):
                values_for_limits.append(q75)

    # Compact leave-one-year-out summary as a vertical range (min-max) with median point.
    if loyo_summary is not None and not loyo_summary.empty:
        r = loyo_summary.iloc[0]
        xpos = len(x_positions)
        med = float(r["median"])
        lo = float(r["min"])
        hi = float(r["max"])
        c = color_by_sign(med)
        ax.plot([xpos, xpos], [lo, hi], color=c, linewidth=4.2, alpha=0.28, solid_capstyle="round")
        ax.scatter(xpos, med, s=68, color=c, edgecolor="white", linewidth=0.8, zorder=6)
        ax.plot([xpos - 0.10, xpos + 0.10], [lo, lo], color=c, linewidth=1.2, alpha=0.55)
        ax.plot([xpos - 0.10, xpos + 0.10], [hi, hi], color=c, linewidth=1.2, alpha=0.55)
        x_positions.append(xpos)
        x_labels.append("Leave-one-\nyear-out")
        values_for_limits.extend([lo, med, hi])

    # If individual leave-one-year-out rows exist but no compact summary, show a boxplot.
    elif loo is not None and not loo.empty:
        vals = loo["median"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            xpos = len(x_positions)
            ax.boxplot(
                vals,
                positions=[xpos],
                widths=0.45,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="0.88", edgecolor="0.35"),
                medianprops=dict(color="#B2182B", linewidth=1.5),
                whiskerprops=dict(color="0.35"),
                capprops=dict(color="0.35"),
            )
            rng = np.random.default_rng(RANDOM_SEED)
            jitter = rng.normal(0, 0.035, size=vals.size)
            ax.scatter(np.zeros(vals.size) + xpos + jitter, vals, s=16, color="0.25", alpha=0.45, zorder=3)
            x_positions.append(xpos)
            x_labels.append("Leave-one-\nyear-out")
            values_for_limits.extend(vals.tolist())

    ax.axhline(0, color="0.3", linestyle="--", linewidth=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.45)
    ax.set_ylabel("Primary DHW intensity trend\n(standardized severity decade$^{-1}$)", fontsize=FIG_FONT_SIZE)

    if values_for_limits:
        vals = np.asarray(values_for_limits, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            pad = 0.15 * max(1e-6, vals.max() - vals.min())
            ax.set_ylim(vals.min() - pad, vals.max() + pad)

    # The leave-one-year-out point is shown as min–max range with a median point.
    # This is explained in the caption/figure footnote to avoid clutter in the panel.


def plot_panel_c(ax, reg: pd.DataFrame):
    ax.set_title("(c) Regional DHW−HHW structure", fontsize=FIG_FONT_SIZE, fontweight="bold", loc="left")

    if reg is None or reg.empty:
        annotate_missing(ax, "(c) Regional DHW−HHW structure")
        return

    d = reg.copy()
    y = np.arange(len(d))[::-1]
    med = d["median"].to_numpy(dtype=float)
    q25 = d["q25"].to_numpy(dtype=float)
    q75 = d["q75"].to_numpy(dtype=float)

    pantanal_caution = False
    for i, yi in enumerate(y):
        c = color_by_sign(med[i])
        ax.plot([q25[i], q75[i]], [yi, yi], color=c, linewidth=5, alpha=0.22, solid_capstyle="round")
        ax.scatter(med[i], yi, s=52, color=c, edgecolor="white", linewidth=0.7, zorder=5)

        region_i = str(d.iloc[i]["region"])
        n_i = d.iloc[i]["n"] if "n" in d.columns else np.nan
        caution = region_i.lower().startswith("pantanal") and np.isfinite(n_i) and n_i < 30
        if caution:
            pantanal_caution = True
            ax.scatter(med[i], yi, s=88, facecolors="none", edgecolors="0.25", linewidth=1.0, zorder=6)

        if "positive_area_fraction" in d.columns and np.isfinite(d.iloc[i]["positive_area_fraction"]):
            suffix = "*" if caution else ""
            ax.text(
                max(q75[i], med[i]) + 0.12,
                yi,
                f"{100*d.iloc[i]['positive_area_fraction']:.0f}% area >0{suffix}",
                va="center",
                ha="left",
                fontsize=FIG_FONT_SIZE,
                color="0.35",
            )

    ax.axvline(0, color="0.3", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["region"], fontsize=FIG_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.18, linewidth=0.45)
    ax.set_xlabel("DHW−HHW\n(Tmax-standardized severity decade$^{-1}$)", fontsize=FIG_FONT_SIZE)

    finite_vals = np.concatenate([med[np.isfinite(med)], q75[np.isfinite(q75)]])
    if finite_vals.size:
        ax.set_xlim(left=min(-0.4, np.nanmin(finite_vals) - 0.4),
                    right=np.nanmax(finite_vals) + 1.0)

    # Pantanal caution is indicated by an open circle and explained in the figure footer.


def plot_panel_d(ax, path: pd.DataFrame):
    ax.set_title("(d) Controlled association diagnostics", fontsize=FIG_FONT_SIZE, fontweight="bold", loc="left")

    if path is None or path.empty:
        annotate_missing(ax, "(d) Controlled association diagnostics")
        return

    ax.axis("off")

    # Display three controlled associations with their coefficients.
    x_land, x_vpd, x_dhw = 0.10, 0.50, 0.88
    y_main = 0.60

    node_kw = dict(boxstyle="round,pad=0.45,rounding_size=0.08", facecolor="white", edgecolor="0.25", linewidth=1.0)
    ax.text(x_land, y_main, "Transformed\nland 2024", transform=ax.transAxes,
            ha="center", va="center", fontsize=FIG_FONT_SIZE, fontweight="bold", bbox=node_kw)
    ax.text(x_vpd, y_main, "VPD trend", transform=ax.transAxes,
            ha="center", va="center", fontsize=FIG_FONT_SIZE, fontweight="bold", bbox=node_kw)
    ax.text(x_dhw, y_main, "Preferential dry-hot\namplification\n(DHW−HHW ↑)", transform=ax.transAxes,
            ha="center", va="center", fontsize=FIG_FONT_SIZE, fontweight="bold", bbox=node_kw)


    ax.text(
        0.30, 0.47, "association", transform=ax.transAxes,
        ha="center", va="center", fontsize=FIG_FONT_SIZE, color="0.45"
    )
    ax.text(
        0.70, 0.47, "association", transform=ax.transAxes,
        ha="center", va="center", fontsize=FIG_FONT_SIZE, color="0.45"
    )

    def get_row(startswith):
        sub = path[path["association"].str.startswith(startswith, na=False)]
        return sub.iloc[0] if not sub.empty else None

    r1 = get_row("Transformed land 2024 → VPD")
    r2 = get_row("VPD trend → DHW")
    r3 = get_row("Transformed land 2024 → DHW")

    def format_coef(r):
        if r is None:
            return "not available"
        beta = r["coef"]
        p = r["p"]
        return f"β={beta:+.2f}, {safe_p_to_text(p)}"

    ax.text(0.30, y_main + 0.07, format_coef(r1), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=FIG_FONT_SIZE, color=color_by_sign(r1["coef"]) if r1 is not None else "0.35")
    ax.text(0.70, y_main + 0.07, format_coef(r2), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=FIG_FONT_SIZE, color=color_by_sign(r2["coef"]) if r2 is not None else "0.35")
    ax.text(0.50, y_main - 0.23, format_coef(r3), transform=ax.transAxes,
            ha="center", va="top", fontsize=FIG_FONT_SIZE, color=color_by_sign(r3["coef"]) if r3 is not None else "0.35")

    ax.text(
        0.02, 0.12,
        "Controlled WLS models use weights proportional to sqrt(municipal area) and\n"
        "HC3 heteroscedasticity-robust standard errors; biome fixed effects and\n"
        "latitude/longitude controls are included where available.\n"
        "β values are per 1 s.d. increase in the predictor; not causal or mediation effects.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FIG_FONT_SIZE,
        color="0.28",
    )


def plot_figure(panel_a, panel_b, panel_c, panel_d, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.5), constrained_layout=False)
    axes = axes.ravel()

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.885,
        bottom=0.095,
        wspace=0.30,
        hspace=0.34,
    )

    plot_panel_a(axes[0], panel_a)
    plot_panel_b(axes[1], panel_b)
    plot_panel_c(axes[2], panel_c)
    plot_panel_d(axes[3], panel_d)

    fig.suptitle(
        "Robustness, temporal stability and regional structure of preferential dry-hot heatwave amplification",
        fontsize=FIG_FONT_SIZE,
        fontweight="bold",
        y=0.962,
    )

    # Figure footnotes are intentionally omitted from the artwork; explanatory text belongs in the caption.

    out_base = os.path.join(out_dir, "Figure_4")
    fig.savefig(out_base + ".jpeg", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved: {out_base}.jpeg")
    print(f"[OK] Saved: {out_base}.pdf")


# ============================================================
# Output tables
# ============================================================

def save_compiled_outputs(out_dir, panel_a, panel_b, panel_c, panel_d, input_files):
    xlsx = os.path.join(out_dir, "Figure_4_summary_tables.xlsx")
    with pd.ExcelWriter(xlsx) as writer:
        panel_a.to_excel(writer, index=False, sheet_name="threshold_sensitivity")
        panel_b.get("periods", pd.DataFrame()).to_excel(writer, index=False, sheet_name="recent_year_periods")
        panel_b.get("leave_one_year_out", pd.DataFrame()).to_excel(writer, index=False, sheet_name="leave_one_year_out")
        panel_b.get("loyo_summary", pd.DataFrame()).to_excel(writer, index=False, sheet_name="loyo_summary")
        panel_c.to_excel(writer, index=False, sheet_name="regional_emergence")
        panel_d.to_excel(writer, index=False, sheet_name="controlled_associations")
        pd.DataFrame([input_files]).T.reset_index().rename(columns={"index": "input_key", 0: "path"}).to_excel(
            writer, index=False, sheet_name="input_files"
        )
    print(f"[OK] Saved: {xlsx}")

    compiled = []
    for name, df in [
        ("threshold_sensitivity", panel_a),
        ("recent_year_periods", panel_b.get("periods", pd.DataFrame())),
        ("leave_one_year_out", panel_b.get("leave_one_year_out", pd.DataFrame())),
        ("loyo_summary", panel_b.get("loyo_summary", pd.DataFrame())),
        ("regional_emergence", panel_c),
        ("controlled_associations", panel_d),
    ]:
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp.insert(0, "table", name)
            compiled.append(tmp.astype(str))
    if compiled:
        out_csv = os.path.join(out_dir, "Figure_4_compiled_data.csv")
        pd.concat(compiled, ignore_index=True, sort=False).to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")

    # Supplementary tables cited with the standardized manuscript numbering.
    save_table_dual(panel_a, out_dir, "Supplementary_Table_S13")
    temporal_parts = []
    for name, df in [
        ("recent_year_periods", panel_b.get("periods", pd.DataFrame())),
        ("leave_one_year_out", panel_b.get("leave_one_year_out", pd.DataFrame())),
        ("loyo_summary", panel_b.get("loyo_summary", pd.DataFrame())),
    ]:
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp.insert(0, "diagnostic", name)
            temporal_parts.append(tmp.astype(str))
    panel_b_table = pd.concat(temporal_parts, ignore_index=True, sort=False) if temporal_parts else pd.DataFrame()
    save_table_dual(panel_b_table, out_dir, "Supplementary_Table_S14")
    save_table_dual(panel_c, out_dir, "Supplementary_Table_S15")
    save_table_dual(panel_d, out_dir, "Supplementary_Table_S16")

    meta = {
        "software_version": SOFTWARE_VERSION,
        "input_files": input_files,
        "interpretation": "robustness and synthesis figure using the common-scale DHW-HHW contrast and conditional spatial associations",
    }
    with open(os.path.join(out_dir, "software_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[OK] Saved: software_metadata.json")


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build Figure 04 from Figure 01/02/03 products."
        )
    )
    parser.add_argument(
        "--fig1-sensitivity",
        required=True,
        help="Figure 01 Supplementary_Table_S3.csv.",
    )
    parser.add_argument(
        "--fig1-extreme-year",
        required=True,
        help="Figure 01 Supplementary_Table_S4.csv.",
    )
    parser.add_argument(
        "--fig2-municipality",
        required=True,
        help="Figure 02 municipality CSV.",
    )
    parser.add_argument(
        "--fig3-controlled",
        required=True,
        help=(
            "Figure 03 controlled-regression CSV, "
            "or Supplementary_Table_S12.csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for Figure 04.",
    )

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    input_files = {
        "fig1_sensitivity": os.path.abspath(str(args.fig1_sensitivity)),
        "fig1_extreme_year": os.path.abspath(str(args.fig1_extreme_year)),
        "fig2_municipality": os.path.abspath(str(args.fig2_municipality)),
        "fig3_controlled": resolve_fig3_controlled_path(str(args.fig3_controlled)),
    }

    for key, path in input_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required input not found ({key}): {path}"
            )
        print(f"[INFO] Using {key}: {path}")

    df_sens = read_table(input_files["fig1_sensitivity"])
    df_extreme = read_table(input_files["fig1_extreme_year"])
    df_fig2_mun = read_table(input_files["fig2_municipality"])
    df_controlled = read_table(input_files["fig3_controlled"])

    panel_a = build_panel_a_sensitivity(df_sens)
    panel_b = build_panel_b_extreme_year(df_extreme)
    panel_c = build_panel_c_region_emergence(df_fig2_mun)
    panel_d = build_panel_d_associations(df_controlled)

    print("[DIAGNOSTICS]")
    print(f"  Panel a sensitivity rows : {len(panel_a)}")
    print(f"  Panel b period rows                : {len(panel_b.get('periods', pd.DataFrame()))}")
    print(f"  Panel b LOYO summary rows          : {len(panel_b.get('loyo_summary', pd.DataFrame()))}")
    print(f"  Panel c regional rows              : {len(panel_c)}")
    print(f"  Panel d controlled models          : {len(panel_d)}")

    save_compiled_outputs(
        args.out_dir,
        panel_a,
        panel_b,
        panel_c,
        panel_d,
        input_files,
    )

    plot_figure(
        panel_a,
        panel_b,
        panel_c,
        panel_d,
        args.out_dir,
    )

    print("[DONE] Figure 04 workflow completed.")
    print(f"[INFO] Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

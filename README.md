# brazil-dry-hot-heatwaves
Reproducible analysis code for the study “Brazilian heatwaves are shifting toward dry-hot thermodynamic regimes in transformed landscapes”

## Overview

This repository contains the Python workflows used to derive heatwave-regime metrics from ERA5, quantify preferential dry-hot heatwave amplification, examine its spatial association with cumulative land transformation and atmospheric drying, evaluate robustness and sensitivity, and compare ERA5 diagnostics with INMET observations.

The analysis covers **Brazil, 1990–2024**, with a **1991–2020 climatological baseline** for the main ERA5 heatwave analysis.

The central cross-regime comparison is:

```text
DHW_minus_HHW_TmaxOnly_intensity_trend_decade
```

This field compares DHW and HHW trends on the same **Tmax-standardized severity** scale. Primary DHW and HHW regime-specific intensity metrics are not subtracted because their definitions and units are not directly commensurate.

## Main data inputs

| Input | Purpose | Used by |
|---|---|---|
| ERA5 hourly single-level data | Main meteorological input; T2m, D2m, U10 and V10 are used to derive daily thermodynamic variables and heatwave metrics | Figure 01 |
| Figure 01 daily ERA5 cache | Reused for sensitivity, atmospheric-drying, VPD decomposition and validation workflows | Figures 01–03 and supplementary diagnostics |
| MapBiomas Collection 10.1 municipality/state/biome coverage workbook | Native and transformed/non-native land-cover metrics | Figure 02 and land-transformation diagnostics |
| Brazilian municipality shapefile | Municipality-level spatial aggregation | Figures 02–03 and supplementary land analyses |
| Brazilian state/boundary shapefile | Brazil mask, state boundaries and validation maps | Figure 01, supplementary diagnostics and validation |
| Brazilian biome shapefile | Biome summaries and map overlays | Figure 01, Figure 02 and supplementary analyses |
| South America country shapefile | Optional geographic context in maps | Figure 01 and selected supplementary figures |
| INMET hourly station observations | Daily station-based ERA5 evaluation | ERA5–INMET validation scripts |
| INMET 1991–2020 climatological normals | Monthly climatology evaluation and station metadata | Climatological-normal validation |

Data files are not distributed with this repository. Paths are supplied at runtime through command-line arguments.

## Recommended repository layout

```text
brazil-dry-hot-heatwaves/
├── figure_01_heatwave_trends.py
├── figure_01_supplementary_robustness.py
├── figure_02_land_transformation.py
├── figure_03_atmospheric_drying.py
├── figure_03_supplementary_material.py
├── figure_04_robustness_regional_associations.py
├── supplementary_hotspot_threshold_sensitivity.py
├── supplementary_nested_common_core_heatwave_sensitivity.py
├── supplementary_vpd_circularity_diagnostics.py
├── supplementary_vpd_decomposition_diagnostics.py
├── annual_mapbiomas_land_atmosphere_diagnostics.py
├── controlled_background_warming_aridity.py
├── mapbiomas_era5_nearest_grid_diagnostic.py
├── era5_inmet_climatological_normals_validation.py
├── era5_inmet_trends_heatwave_metrics_validation.py
├── era5_inmet_station_daily_taylor_validation.py
└── README.md
```

The commands below assume these repository filenames.

## Python environment

The workflows use standard scientific Python packages, including:

```text
numpy
pandas
xarray
h5netcdf
scipy
geopandas
shapely
matplotlib
statsmodels
openpyxl
pymannkendall
```

Depending on the NetCDF format available locally, `netCDF4` may also be used.

A typical environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate

pip install numpy pandas xarray h5netcdf scipy geopandas shapely \
            matplotlib statsmodels openpyxl pymannkendall netCDF4
```

## Execution order

The main manuscript workflow should be run in the following order:

```text
ERA5 hourly data
      │
      ▼
Figure 01
      │
      ├──────────────► Figure 01 supplementary robustness
      │
      ├──────────────► hotspot-threshold sensitivity
      │
      ├──────────────► nested/common-core sensitivity
      │
      ├──────────────► VPD circularity diagnostics
      │
      └──────────────► VPD decomposition diagnostics
      │
      ▼
Figure 02
      │
      ▼
Figure 03
      │
      ├──────────────► Figure 03 supplementary material
      │
      ├──────────────► annual MapBiomas diagnostics
      │
      └──────────────► background-warming/aridity controls
      │
      ▼
Figure 04
```

The ERA5–INMET evaluation workflows are methodologically independent of Figures 02–04, but they reuse Figure 01 ERA5 products where appropriate and are therefore most conveniently run after Figure 01.

---

# 1. Figure 01 — Heatwave-regime trends

**Script**

```text
figure_01_heatwave_trends.py
```

**Purpose**

Processes hourly ERA5 data and derives the main HW, HHW, DHW and CHW metrics for 1990–2024. It calculates daily variables, climatological thresholds, annual heatwave metrics, Theil–Sen trends, autocorrelation-aware Mann–Kendall significance and the common Tmax-only DHW−HHW comparison.

**Main inputs**

- ERA5 hourly single-level files
- Brazil boundary shapefile
- optional biome shapefile
- optional South America boundary shapefile

Required ERA5 fields are 2-m air temperature, 2-m dew-point temperature, 10-m zonal wind and 10-m meridional wind.

**Main outputs**

```text
outputs/figure_01/
├── cache/
│   ├── era5_daily/ERA5_daily_Brazil_YYYY.nc
│   └── figure_01_heatwave_thresholds_1991_2020.nc
└── data/
    ├── figure_01_annual_heatwave_metrics_1990_2024.nc
    └── figure_01_heatwave_trends_1990_2024.nc
```

The workflow also generates the Figure 01 map products.

**Run**

```bash
python figure_01_heatwave_trends.py \
  --era5-dir /path/to/era5_hourly \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --south-america-shapefile /path/to/south_america_countries.shp \
  --output-dir ./outputs/figure_01
```

The biome and South America shapefiles are optional.

To rebuild existing intermediate products:

```bash
python figure_01_heatwave_trends.py \
  --era5-dir /path/to/era5_hourly \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --output-dir ./outputs/figure_01 \
  --overwrite-daily \
  --recompute-thresholds \
  --recompute-annual-trends
```

---

# 2. Figure 01 supplementary robustness

**Script**

```text
figure_01_supplementary_robustness.py
```

**Purpose**

Generates Supplementary Tables S1–S4. The workflow summarizes the primary Figure 01 results, evaluates threshold and persistence sensitivity, and tests temporal robustness.

**Main inputs**

- complete Figure 01 output directory
- Brazil shapefile
- optional biome shapefile

**Main outputs**

```text
Supplementary_Table_S1.csv/.xlsx
Supplementary_Table_S2.csv/.xlsx
Supplementary_Table_S3.csv/.xlsx
Supplementary_Table_S4.csv/.xlsx
Diagnostic_Table_threshold_duration_sensitivity_all_metrics.csv
```

S2 and S3 use the common Tmax-only DHW−HHW contrast. S4 evaluates temporal robustness of the **primary DHW intensity**, not the DHW−HHW contrast.

**Run**

```bash
python figure_01_supplementary_robustness.py \
  --figure1-output-dir ./outputs/figure_01 \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --output-dir ./outputs/figure_01_supplementary
```

---

# 3. Hotspot-threshold sensitivity

**Script**

```text
supplementary_hotspot_threshold_sensitivity.py
```

**Purpose**

Tests whether the spatial hotspot pattern is sensitive to the diagnostic DHW−HHW cutoff. The primary `>4` threshold is compared with `>3`, `>5`, the upper quartile and the upper decile.

The optional significance condition refers only to the positive DHW Tmax-only trend and its p-value. No p-value is assigned to the DHW−HHW difference of slopes.

**Main inputs**

- Figure 01 trend NetCDF
- Brazil/state shapefile
- optional biome shapefile

**Main outputs**

```text
Hotspot_Sensitivity_summary_by_criterion.csv
Hotspot_Sensitivity_region_distribution.csv
Hotspot_Sensitivity_region_ranking_stability.csv
Hotspot_Sensitivity_biome_distribution.csv
Hotspot_Sensitivity_biome_ranking_stability.csv
Hotspot_Sensitivity_gridcell_base_table.csv
Supplementary_Tables_Hotspot_Threshold_Sensitivity_DHW_HHW.xlsx
Supplementary_Figure_Hotspot_Threshold_Sensitivity_DHW_HHW.pdf/.jpeg
```

**Run**

```bash
python supplementary_hotspot_threshold_sensitivity.py \
  --trend-nc ./outputs/figure_01/data/figure_01_heatwave_trends_1990_2024.nc \
  --states-shapefile /path/to/brazil_states.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --output-dir ./outputs/hotspot_threshold_sensitivity
```

---

# 4. Nested/common-core sensitivity

**Script**

```text
supplementary_nested_common_core_heatwave_sensitivity.py
```

**Purpose**

Tests the dry-hot versus humid contrast within a common persistent Tmax-defined heatwave core. Core-event days are partitioned into mutually exclusive DRY, HUMID, MIXED and NEUTRAL states.

The main nested comparison is:

```text
NESTED_DRY_intensity_trend_decade
-
NESTED_HUMID_intensity_trend_decade
```

Both components use the same Tmax-standardized severity scale.

**Main inputs**

- Figure 01 daily ERA5 cache
- Figure 01 threshold climatology
- Figure 01 trend product
- Brazil shapefile
- optional biome and South America shapefiles

**Main outputs**

```text
Nested_CommonCore_annual_metrics_1990_2024.nc
Nested_CommonCore_trends_1990_2024.nc
Nested_CommonCore_regional_summary.csv
Nested_CommonCore_gridcell_comparison.csv
Supplementary_Nested_CommonCore_Heatwave_Sensitivity.pdf/.jpeg
```

**Run**

```bash
python supplementary_nested_common_core_heatwave_sensitivity.py \
  --figure1-output-dir ./outputs/figure_01 \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --south-america-shapefile /path/to/south_america_countries.shp \
  --output-dir ./outputs/nested_common_core
```

Use `--recompute` to ignore existing row caches.

---

# 5. VPD circularity diagnostics

**Script**

```text
supplementary_vpd_circularity_diagnostics.py
```

**Purpose**

Tests whether preferential dry-hot amplification remains when VPD is excluded from the **cross-regime intensity metric**. It compares the primary DHW intensity trend with the DHW Tmax-only trend and also examines duration, frequency, actual atmospheric vapour pressure and dew-point trends.

VPD still defines DHW occurrence; it is excluded only from the common Tmax-only intensity used for direct DHW−HHW comparison.

**Main inputs**

- Figure 01 annual heatwave metrics
- Figure 01 trend product
- Figure 01 daily ERA5 cache
- Brazil shapefile

**Main outputs**

```text
VPD_Circularity_Tonly_annual_metrics_1990_2024.nc
VPD_Circularity_Tonly_trends_1990_2024.nc
VPD_Circularity_independent_moisture_annual_1990_2024.nc
VPD_Circularity_independent_moisture_trends_1990_2024.nc
VPD_Circularity_summary_statistics_1990_2024.csv/.xlsx
Supplementary_VPD_Circularity_Diagnostics_1990_2024.pdf/.jpeg
```

**Run**

```bash
python supplementary_vpd_circularity_diagnostics.py \
  --figure1-output-dir ./outputs/figure_01 \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --output-dir ./outputs/vpd_circularity
```

Use `--recompute-moisture` to rebuild the auxiliary atmospheric-moisture products.

---

# 6. VPD decomposition diagnostics

**Script**

```text
supplementary_vpd_decomposition_diagnostics.py
```

**Purpose**

Separates VPD into:

```text
VPD = es(T) - ea
```

where `es(T)` represents temperature-controlled saturation vapour pressure and `ea` represents actual atmospheric vapour pressure.

Actual atmospheric vapour pressure is derived from dew point when available, or from RH and Tmean as a fallback. It is not reconstructed from VPD.

The workflow retains the directly estimated VPD trend separately from the component-slope approximation `βes − βea`, because Theil–Sen slopes are not exactly additive.

**Main inputs**

- Figure 01 daily ERA5 cache
- Figure 01 trend NetCDF
- Brazil shapefile
- optional biome shapefile

**Main outputs**

```text
VPD_Decomposition_annual_metrics_1990_2024.nc
VPD_Decomposition_trends_1990_2024.nc
VPD_Decomposition_component_correlations_1990_2024.csv/.xlsx
VPD_Decomposition_regional_summary_1990_2024.csv/.xlsx
Supplementary_VPD_Decomposition_Diagnostics_1990_2024.pdf/.jpeg
```

**Run**

```bash
python supplementary_vpd_decomposition_diagnostics.py \
  --figure1-output-dir ./outputs/figure_01 \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --output-dir ./outputs/vpd_decomposition
```

Use `--recompute` to rebuild annual and trend products.

---

# 7. Figure 02 — Land transformation

**Script**

```text
figure_02_land_transformation.py
```

**Purpose**

Aggregates the Figure 01 DHW−HHW common-scale contrast to Brazilian municipalities and relates preferential dry-hot amplification to cumulative land transformation from MapBiomas.

The main transformed-land metric is:

```text
transformed_non_native_pct_2024 = 100 - native vegetation cover
```

Continuous ERA5 fields are aggregated to municipalities using cos(latitude)-weighted grid-cell values. Nearest-grid extraction is used only when no ERA5 grid-cell centre falls inside a municipality.

**Main inputs**

- `figure_01_heatwave_trends_1990_2024.nc`
- MapBiomas Collection 10.1 municipality/state/biome coverage workbook
- 2024 Brazilian municipality shapefile

**Main outputs**

```text
figure_02_land_transformation.pdf/.jpeg
figure_02_municipality_dhw_hhw_contrast_1990_2024.csv
figure_02_region_summary_dhw_hhw_contrast_1990_2024.csv
figure_02_regional_model_summary_dhw_hhw_contrast_1990_2024.csv
Supplementary_Table_S5.csv/.xlsx
Supplementary_Table_S6.csv/.xlsx
Supplementary_Table_S7.csv/.xlsx
Supplementary_Table_S8.csv/.xlsx
Supplementary_Tables_S5_S8.xlsx
```

**Run**

```bash
python figure_02_land_transformation.py \
  --figure1-trends ./outputs/figure_01/data/figure_01_heatwave_trends_1990_2024.nc \
  --mapbiomas-xlsx /path/to/mapbiomas_municipality_coverage.xlsx \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --output-dir ./outputs/figure_02
```

---

# 8. Figure 03 — Atmospheric drying

**Script**

```text
figure_03_atmospheric_drying.py
```

**Purpose**

Calculates ERA5 VPD and RH trends and evaluates their spatial association with the municipal DHW−HHW contrast and cumulative transformed-land fraction.

Continuous ERA5 municipality fields use cos(latitude)-weighted aggregation. P-values are not averaged. LOWESS curves are descriptive. Controlled WLS models use weights proportional to the square root of municipal area and HC3 heteroscedasticity-robust standard errors.

**Main inputs**

- complete Figure 01 output directory
- Figure 02 municipality table
- MapBiomas Collection 10.1 workbook
- municipality shapefile

**Main outputs**

```text
figure_03_ERA5_annual_mechanism_metrics_1990_2024.nc
figure_03_ERA5_mechanistic_trends_1990_2024.nc
figure_03_ERA5_mechanistic_trends_with_dryhot_1990_2024.nc
figure_03_municipality_mechanism_table_1990_2024.csv/.nc
figure_03_controlled_mechanism_regressions_1990_2024.csv/.xlsx
figure_03_mechanism_regression_summary_1990_2024.csv
software_metadata.json
```

The workflow also generates the Figure 03 PDF/JPEG products.

**Run**

```bash
python figure_03_atmospheric_drying.py \
  --figure1-output-dir ./outputs/figure_01 \
  --figure2-table ./outputs/figure_02/figure_02_municipality_dhw_hhw_contrast_1990_2024.csv \
  --mapbiomas-xlsx /path/to/mapbiomas_municipality_coverage.xlsx \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --output-dir ./outputs/figure_03
```

---

# 9. Figure 03 supplementary material

**Script**

```text
figure_03_supplementary_material.py
```

**Purpose**

Generates Supplementary Tables S9–S12 from the already computed Figure 03 products. ERA5 trends are not recomputed.

The tables include circularity-related diagnostics, atmospheric-drying summaries, regional relationships and controlled association models.

**Main inputs**

From Figure 03:

```text
figure_03_municipality_mechanism_table_1990_2024.csv
figure_03_ERA5_mechanistic_trends_with_dryhot_1990_2024.nc
```

**Main outputs**

```text
Supplementary_Table_S9.csv/.xlsx
Supplementary_Table_S10.csv/.xlsx
Supplementary_Table_S11.csv/.xlsx
Supplementary_Table_S12.csv/.xlsx
figure_03_supplementary_QA.csv
figure_03_supplementary_lowess_summary.csv
```

**Run**

```bash
python figure_03_supplementary_material.py \
  --figure3-output-dir ./outputs/figure_03 \
  --output-dir ./outputs/figure_03_supplementary
```

---

# 10. Annual MapBiomas land–atmosphere diagnostics

**Script**

```text
annual_mapbiomas_land_atmosphere_diagnostics.py
```

**Purpose**

Builds annual municipality-level land-transformation metrics and compares cumulative transformed-land state and annual land-cover change metrics with ERA5-derived atmospheric and heatwave outcomes.

The controlled models are spatial association diagnostics, not land-cover sensitivity experiments or causal attribution.

**Main inputs**

- MapBiomas Collection 10.1 workbook
- municipality shapefile
- Figure 03 or Figure 02 municipality-level atmospheric/heatwave table containing the common-scale DHW−HHW metric

**Run**

```bash
python annual_mapbiomas_land_atmosphere_diagnostics.py \
  --mapbiomas-coverage /path/to/mapbiomas_coverage.xlsx \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --atmospheric-table ./outputs/figure_03/figure_03_municipality_mechanism_table_1990_2024.csv \
  --output-dir ./outputs/annual_mapbiomas_diagnostics
```

This is a supplementary diagnostic and is not required to generate Figures 01–04.

---

# 11. Background-warming and climatological-aridity controls

**Script**

```text
controlled_background_warming_aridity.py
```

**Purpose**

Tests whether the association between transformed land and DHW−HHW remains after controlling for background warming, climatological aridity, biome structure and broad latitude/longitude gradients.

The models use WLS with weights proportional to the square root of municipal area and HC3 heteroscedasticity-robust standard errors.

**Main inputs**

- Figure 03 municipality table
- Figure 03 annual ERA5 atmospheric metrics
- Figure 01 daily ERA5 cache
- municipality shapefile

**Main outputs**

```text
Supplementary_ERA5_annual_Tmean_1990_2024.nc
Supplementary_BackgroundWarming_Aridity_gridded_covariates_1990_2024.nc
Supplementary_Table_municipality_with_background_warming_aridity_covariates.csv
Supplementary_Table_Controlled_BackgroundWarming_Aridity_models.csv
Supplementary_Tables_Controlled_BackgroundWarming_Aridity.xlsx
software_metadata.json
```

**Run**

```bash
python controlled_background_warming_aridity.py \
  --figure3-municipality-table ./outputs/figure_03/figure_03_municipality_mechanism_table_1990_2024.csv \
  --figure3-annual-metrics ./outputs/figure_03/figure_03_ERA5_annual_mechanism_metrics_1990_2024.nc \
  --figure1-daily-dir ./outputs/figure_01/cache/era5_daily \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --output-dir ./outputs/background_warming_aridity
```

Use `--overwrite` to rebuild cached covariates and municipality tables.

---

# 12. Figure 04 — Robustness and regional synthesis

**Script**

```text
figure_04_robustness_regional_associations.py
```

**Purpose**

Combines the robustness analyses into the final synthesis figure.

- Panel (a): threshold and event-definition sensitivity from S3.
- Panel (b): temporal robustness of the **primary DHW intensity** from S4.
- Panel (c): regional structure of the municipal DHW−HHW contrast.
- Panel (d): controlled spatial-association diagnostics from Figure 03/S12.

**Main inputs**

```text
Supplementary_Table_S3.csv
Supplementary_Table_S4.csv
figure_02_municipality_dhw_hhw_contrast_1990_2024.csv
Supplementary_Table_S12.csv
```

**Main outputs**

```text
Figure_4.pdf
Figure_4.jpeg
Figure_4_compiled_data.csv
Figure_4_summary_tables.xlsx
Supplementary_Table_S13.csv/.xlsx
Supplementary_Table_S14.csv/.xlsx
Supplementary_Table_S15.csv/.xlsx
Supplementary_Table_S16.csv/.xlsx
software_metadata.json
```

**Run**

```bash
python figure_04_robustness_regional_associations.py \
  --fig1-sensitivity ./outputs/figure_01_supplementary/Supplementary_Table_S3.csv \
  --fig1-extreme-year ./outputs/figure_01_supplementary/Supplementary_Table_S4.csv \
  --fig2-municipality ./outputs/figure_02/figure_02_municipality_dhw_hhw_contrast_1990_2024.csv \
  --fig3-controlled ./outputs/figure_03_supplementary/Supplementary_Table_S12.csv \
  --out-dir ./outputs/figure_04
```

---

# 13. MapBiomas–ERA5 nearest-grid diagnostic

**Script**

```text
mapbiomas_era5_nearest_grid_diagnostic.py
```

**Purpose**

Creates a municipality-level MapBiomas–ERA5 table using the nearest ERA5 grid cell to each municipality representative point.

This is a **diagnostic/preprocessing workflow**, not the primary Figure 02 municipality aggregation. Figure 02 uses cos(latitude)-weighted zonal aggregation and nearest-grid extraction only as a fallback.

**Main inputs**

- MapBiomas municipality coverage workbook
- optional MapBiomas urban-module workbook
- Figure 01 trend NetCDF
- municipality shapefile

**Main outputs**

```text
MapBiomas_municipality_landcover_change_1985_2024.csv
MapBiomas_ERA5_nearest_grid_diagnostic.csv
MapBiomas_ERA5_nearest_grid_diagnostic_compact.csv
QC_removed_small_area_municipality_biome_rows.csv
```

**Run**

```bash
python mapbiomas_era5_nearest_grid_diagnostic.py \
  --mapbiomas-coverage /path/to/mapbiomas_coverage.xlsx \
  --urban-module /path/to/mapbiomas_urban_module.xlsx \
  --era5-trends ./outputs/figure_01/data/figure_01_heatwave_trends_1990_2024.nc \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --output-dir ./outputs/mapbiomas_era5_diagnostic
```

`--urban-module` is optional.

---

# ERA5–INMET evaluation

These workflows evaluate ERA5 independently from the MapBiomas analyses. They can be run after Figure 01 because they can reuse its daily ERA5 products.

## 14. ERA5–INMET climatological-normal evaluation

**Script**

```text
era5_inmet_climatological_normals_validation.py
```

**Purpose**

Compares ERA5 monthly climatologies with INMET 1991–2020 climatological normals at station locations.

This evaluates the baseline climatology rather than individual heatwave events.

**Main inputs**

INMET normal files:

```text
Normal-Climatologica-ESTAÇÕES.xlsx
Normal-Climatologica-TMAX.xlsx
Normal-Climatologica-TMIN.xlsx
Normal-Climatologica-TORV.xlsx
Normal-Climatologica-UR.xlsx
Normal-Climatologica-VENTIN.xlsx
Normal-Climatologica-TMEDUMID.xlsx
```

and either:

- Figure 01 daily ERA5 files, or
- hourly ERA5 files.

A Brazilian state shapefile is also required.

**Main outputs**

```text
Supplementary_Table_ERA5_INMET_climatological_normals_station_month_pairs_1991_2020.csv
Supplementary_Table_ERA5_INMET_climatological_normals_station_metrics_1991_2020.csv
Supplementary_Table_ERA5_INMET_climatological_normals_taylor_metrics_1991_2020.csv
Supplementary_Table_ERA5_INMET_climatological_normals_nearest_grid_index.csv
Supplementary_Table_INMET_climatological_normal_station_metadata.csv
Supplementary_Tables_ERA5_INMET_climatological_normals_validation_1991_2020.xlsx
software_metadata_climatological_normals_validation.json
```

**Recommended daily-mode run**

```bash
python era5_inmet_climatological_normals_validation.py \
  --normal-dir /path/to/inmet_normals_1991_2020 \
  --era5-daily-dir ./outputs/figure_01/cache/era5_daily \
  --source daily \
  --states-shapefile /path/to/brazil_states.shp \
  --output-dir ./outputs/era5_inmet_climatology \
  --reuse-era5-cache
```

---

## 15. ERA5–INMET trend and heatwave-metric evaluation

**Script**

```text
era5_inmet_trends_heatwave_metrics_validation.py
```

**Purpose**

Evaluates ERA5 against quality-controlled INMET observations for annual thermodynamic trends and HW, HHW and DHW frequency, duration and accumulated intensity.

Thresholds are calculated separately for ERA5 and INMET over the validation baseline, using the same threshold and persistence logic.

**Main inputs**

- INMET hourly observations
- Figure 01 daily ERA5 files or a precomputed station-level ERA5 cache
- optional INMET station metadata
- optional Brazilian state shapefile

**Recommended run**

```bash
python era5_inmet_trends_heatwave_metrics_validation.py \
  --inmet-hourly-dir /path/to/inmet_hourly \
  --era5-daily-dir ./outputs/figure_01/cache/era5_daily \
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
```

The main QC defaults include at least 18 hourly observations per day and at least 70% valid-day coverage for retained station-years.

---

## 16. ERA5–INMET daily station/Taylor evaluation

**Script**

```text
era5_inmet_station_daily_taylor_validation.py
```

**Purpose**

Performs station-level daily ERA5–INMET comparison for temperature, relative humidity and 10-m wind speed using Pearson correlation, bias, RMSE, MAE, standard-deviation ratio and Taylor diagrams.

**Main inputs**

- INMET hourly observations
- ERA5 station cache, Figure 01 daily files, or hourly ERA5 files
- optional official station metadata
- optional trusted station-selection table
- Brazilian state shapefile

**Recommended cache-mode run**

```bash
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
```

Main outputs include paired daily observations, station metrics, aggregated Taylor metrics, nearest-grid metadata, outlier-QC summaries, figures and software metadata.

---

# Minimal manuscript reproduction sequence

For reproduction of the main Figures 01–04 and their directly associated supplementary tables, the shortest sequence is:

```bash
# 1. Figure 01
python figure_01_heatwave_trends.py \
  --era5-dir /path/to/era5_hourly \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --output-dir ./outputs/figure_01

# 2. Figure 01 supplementary Tables S1–S4
python figure_01_supplementary_robustness.py \
  --figure1-output-dir ./outputs/figure_01 \
  --brazil-shapefile /path/to/brazil_boundary.shp \
  --biomes-shapefile /path/to/brazil_biomes.shp \
  --output-dir ./outputs/figure_01_supplementary

# 3. Figure 02
python figure_02_land_transformation.py \
  --figure1-trends ./outputs/figure_01/data/figure_01_heatwave_trends_1990_2024.nc \
  --mapbiomas-xlsx /path/to/mapbiomas_municipality_coverage.xlsx \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --output-dir ./outputs/figure_02

# 4. Figure 03
python figure_03_atmospheric_drying.py \
  --figure1-output-dir ./outputs/figure_01 \
  --figure2-table ./outputs/figure_02/figure_02_municipality_dhw_hhw_contrast_1990_2024.csv \
  --mapbiomas-xlsx /path/to/mapbiomas_municipality_coverage.xlsx \
  --municipality-shapefile /path/to/brazil_municipalities_2024.shp \
  --output-dir ./outputs/figure_03

# 5. Figure 03 supplementary Tables S9–S12
python figure_03_supplementary_material.py \
  --figure3-output-dir ./outputs/figure_03 \
  --output-dir ./outputs/figure_03_supplementary

# 6. Figure 04
python figure_04_robustness_regional_associations.py \
  --fig1-sensitivity ./outputs/figure_01_supplementary/Supplementary_Table_S3.csv \
  --fig1-extreme-year ./outputs/figure_01_supplementary/Supplementary_Table_S4.csv \
  --fig2-municipality ./outputs/figure_02/figure_02_municipality_dhw_hhw_contrast_1990_2024.csv \
  --fig3-controlled ./outputs/figure_03_supplementary/Supplementary_Table_S12.csv \
  --out-dir ./outputs/figure_04
```

## Reproducibility notes

- All file paths are supplied through command-line arguments; repository scripts should not contain machine-specific paths.
- The main analysis uses a 365-day calendar with 29 February removed.
- Missing whole days break event persistence.
- Figure 01 thresholds are local, grid-cell-specific and day-of-year-specific using a centred 31-day climatological window.
- Trend magnitudes are Theil–Sen median slopes per decade.
- Trend significance uses the original two-sided Mann–Kendall test unless significant detrended lag-1 rank autocorrelation is detected, in which case Hamed–Rao lag-1 variance correction is used.
- Municipality aggregation of continuous ERA5 fields uses cos(latitude) grid-cell weighting.
- P-values are not spatially averaged.
- The `DHW_minus_HHW_TmaxOnly_intensity_trend_decade` field is the required common-scale metric for direct DHW–HHW intensity comparison.
- Diagnostic hotspot cutoffs and controlled regressions are not interpreted as physical thresholds, causal attribution or mediation tests.

## Citation

If this repository is used in research, please cite the associated article:

> **Brazilian heatwaves are shifting toward dry-hot thermodynamic regimes in transformed landscapes**

Full bibliographic information will be added after publication.

## Code availability

All data processing, statistical analyses and figure generation were performed in Python 3. The scripts in this repository reproduce the analyses and figures described in the associated study. Data are obtained from their respective official providers and are not redistributed here.

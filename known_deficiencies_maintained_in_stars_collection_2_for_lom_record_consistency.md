# Known Deficiencies Maintained in STARS Collection 2 for LOM Record Consistency

**Product:** ECOSTRESS Collection 2 Level-2 Tiled Ancillary NDVI and Albedo (`ECO_L2T_STARS`, Build `0700`)  
**Scope:** Operational PGE (`L2T_STARS.py` / `process_ECOSTRESS_data_fusion.jl`)  
**Purpose:** Document known algorithmic, state-persistence, and configuration behaviors intentionally retained in Collection 2 to ensure Life-of-Mission (LOM) data record consistency across reprocessing and forward processing streams.

---

## Executive Summary

During Collection 2 validation and dry-run I/O auditing of the `L2T_STARS` Product Generation Executive (PGE), several architectural and configuration behaviors were identified in how prior model states, coarse-to-fine calibration, and temporal spin-up windows are handled. Modifying these behaviors mid-mission would introduce step-change discontinuities in the `NDVI`, `NDVI-UQ`, `albedo`, and `albedo-UQ` time series. Consequently, these behaviors are documented below as **known deficiencies maintained in Collection 2** to preserve Life-of-Mission (LOM) record consistency.

---

## 1. Reset of Coarse-Scale Spatial Bias and Bias Uncertainty Priors (`load_prior`)

### Mechanism
In `L2T_STARS.py`, the `DataFusionState` passed into the Julia STARS solver (`process_ECOSTRESS_data_fusion.jl`) requires four prior state rasters from the preceding timestep ($t_{\text{prior}}$):
1. Fine-scale state mean (`NDVI` / `albedo` at $70\text{ m}$)
2. Fine-scale state uncertainty (`NDVI.UQ` / `albedo.UQ` at $70\text{ m}$)
3. Coarse-scale sensor bias mean (`NDVI.bias` / `albedo.bias` at $490\text{ m}$ / $980\text{ m}$)
4. Coarse-scale sensor bias uncertainty (`NDVI.bias.UQ` / `albedo.bias.UQ` at $490\text{ m}$ / $980\text{ m}$)

However, the published `ECO_L2T_STARS` archive granule (`.zip`) only packages the four fine-scale layers (`NDVI`, `NDVI-UQ`, `albedo`, `albedo-UQ`). In addition, under default production flags (`remove_posterior=True`), the intermediate `.bias` and `.bias.UQ` GeoTIFFs written to `L2T_STARS_MODEL` are deleted upon granule completion, or are unavailable when a granule is processed on a different compute node or during distributed reprocessing.

When `load_prior()` unpacks `L2T_STARS_PRIOR` and finds no pre-existing `.bias` or `.bias.UQ` GeoTIFFs in `L2T_STARS_MODEL/<tile>/<prior_date>/`, it synthesizes constant rasters:
* `initial_bias_mean` $= 0.0$
* `initial_bias_UQ` $= 0.05$

### Impact on the Data Record
* While the fine-scale mean and uncertainty states propagate continuously from `L2T_STARS_PRIOR`, the coarse-scale VIIRS-to-HLS bias state (`mean_bias`, `SD_bias`) resets to a zero-mean, uniform-uncertainty prior at every granule execution.
* Over the short assimilation window between consecutive ECOSTRESS overpasses (typically 1–5 days), the random-walk bias state (`offset_ar = [1, 0.0]`, `offset_var = [1e-5, 1e-5]`) partially re-adapts rather than inheriting the accumulated spatial bias field from $t_{\text{prior}}$.

### LOM Retention Rationale
Packaging new layers into the `ECO_L2T_STARS` product specification or altering prior bias initialization would break schema compatibility with downstream Collection 2 PGEs (`L3T_JET`, `L4T_ESI`, `L4T_WUE`) and introduce a systematic shift in bias-corrected `NDVI` and `albedo` relative to previously processed Collection 2 granules.

---

## 2. Uncalibrated Fine-Scale HLS Inputs (`DEFAULT_CALIBRATE_FINE = False`)

### Mechanism
`L2T_STARS.py` defines a helper function `calibrate_fine_to_coarse(fine_image, coarse_image)` that applies an ordinary least-squares linear regression (`scipy.stats.linregress`) to match aggregated $70\text{ m}$ HLS observations to the corresponding coarse VIIRS grid ($490\text{ m}$ for NDVI, $980\text{ m}$ for albedo) whenever $\ge 30$ valid co-located pixels exist.

Although the helper functions `generate_STARS_inputs()` and `process_STARS_product()` default their local parameter signatures to `calibrate_fine: bool = True`, the top-level PGE entrypoint `L2T_STARS()` overrides this via the module constant:
```python
DEFAULT_CALIBRATE_FINE = False
```
In addition, `L2T_STARS.xml` does not expose `calibrate_fine` as a configurable run-config parameter.

### Impact on the Data Record
* Fine-resolution HLS `NDVI` and `albedo` inputs written to `DOWNSAMPLED_products` are assimilated directly in native HLS radiometric/BRDF space without linear cross-calibration to VIIRS (`VCD43NRT`).
* Any cross-sensor offset between HLS (Landsat 8/9 OLI and Sentinel-2 MSI) and VIIRS (`VNP09GA` / `VCD43NRT`) is absorbed entirely by the STARS coarse-scale bias state (`mean_bias`) rather than pre-harmonized via linear regression.

### LOM Retention Rationale
Enabling `DEFAULT_CALIBRATE_FINE = True` would alter the radiometric baseline of HLS observations entering the Kalman filter and change the magnitude of `NDVI` and `albedo` across the entire mission record, particularly in scenes with high spatial heterogeneity or sparse valid pixel overlap.

---

## 3. Asymmetric Temporal Windows for Spatial Covariance vs. Kalman Fusion

### Mechanism
When a valid prior granule (`using_prior == True`) falls within the default 7-day spin-up window (`DEFAULT_SPINUP_DAYS = 7`), `L2T_STARS.py` defines two distinct start dates:
* **VIIRS window (`VIIRS_start_date` to `end_date`):** Always spans `spinup_days + 1` days (8 days total, e.g., $t - 7$ through $t$).
* **HLS / Fusion window (`HLS_start_date` to `end_date`):** Truncated to `prior_date_UTC + 1 day` through `end_date` (e.g., 4 days when `prior_date_UTC` is $t - 4$).

Inside `process_ECOSTRESS_data_fusion.jl`:
1. `covariance_images` ingests the full 8-day VIIRS stack (`VIIRS_start_date:Day(1):VIIRS_end_date`) to estimate the Matérn spatial variance parameter (`sp_var = fast_var_est(covariance_images, n_eff_agg = n_eff)`).
2. `coarse_images` and `fine_images` are constructed only over `dates = HLS_start_date:Day(1):HLS_end_date`, reusing the in-memory VIIRS slices from that truncated sub-window.

### Impact on the Data Record
* Spatial covariance parameters (`cov_pars_raster`) are estimated using a fixed 8-day rolling window of coarse observations (which overlaps dates prior to `prior_date_UTC`), whereas the state-space data fusion filter only assimilates VIIRS and HLS observations strictly following `prior_date_UTC`.
* When no prior is available (`using_prior == False`), both `fast_var_est` and `coarse_fine_data_fusion` operate over the identical 8-day window.

### LOM Retention Rationale
Aligning the covariance estimation window with `HLS_start_date` would reduce the sample size available to `fast_var_est` during short inter-overpass intervals (down to 1–2 days), increasing noise in `sp_var` and causing temporal variance fluctuations between dense and sparse ECOSTRESS revisit periods.

---

## 4. Zero-Value Masking Treated as Missing Data (`0 == NaN`)

### Mechanism
In `L2T_STARS.py`, all four input ingestion functions explicitly mask exact zero values to `NaN` prior to writing GeoTIFFs for Julia:
* `generate_NDVI_coarse_image`: `coarse_image = rt.where(coarse_image == 0, np.nan, coarse_image)`
* `generate_NDVI_fine_image`: `fine_image = rt.where(fine_image == 0, np.nan, fine_image)`
* `generate_albedo_coarse_image`: `coarse_image = rt.where(coarse_image == 0, np.nan, coarse_image)`
* `generate_albedo_fine_image`: `fine_image = rt.where(fine_image == 0, np.nan, fine_image)`

### Impact on the Data Record
* While intended to filter unmasked fill values (`0`) along tile boundaries or water/nodata edges, any legitimate surface pixel whose computed `NDVI` or `albedo` equals exactly `0.0` is converted to `NaN` in the observation slice.
* Those pixels rely on temporal persistence from the prior and spatial interpolation from neighboring pixels rather than direct measurement updates on that date.

### LOM Retention Rationale
Changing the nodata sentinel logic risks propagating `0.0` fill pixels from upstream HLS or `VCD43NRT` tile margins into the Kalman filter, which would corrupt spatial variance estimates (`fast_var_est`) and bias fields along swath and tile edges.

---

## 5. Fixed Matérn Spatial Covariance Hyperparameters and Unsmoothed Filtering

### Mechanism
In `process_ECOSTRESS_data_fusion.jl`, the spatial covariance structure and Kalman filter execution parameters are hardcoded across all global tiles and land-cover types:
* **Matérn range:** $200\text{ m}$ (`cov_pars_raster[:, :, 2] .= 200`)
* **Nugget variance:** $10^{-10}$ (`cov_pars_raster[:, :, 3] .= 1e-10`)
* **Matérn smoothness:** $\nu = 1.5$ (`cov_pars_raster[:, :, 4] .= 1.5`)
* **Spatial buffer distance:** $100\text{ m}$ (`buffer_distance = 100.`)
* **Temporal smoothing:** Disabled (`smooth = false`), executing forward-only filtering up to `target_times = [target_date]`.

### Impact on the Data Record
* Spatial correlation length scales do not vary by biome, topography, or agricultural parcel size.
* Because `smooth = false` is enforced, observations occurring on `target_date` update the posterior state, whereas the forward-only filter does not perform backward Rauch–Tung–Striebel (RTS) smoothing across the spin-up window.

### LOM Retention Rationale
Maintaining fixed Matérn hyperparameters and forward-only filtering ensures deterministic uncertainty quantification (`NDVI-UQ`, `albedo-UQ`) and consistent spatial texture across all Collection 2 granules globally.

---

## Summary Matrix

| ID | Component | Location | Behavior Maintained in Collection 2 | Primary Consistency Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | Prior Bias State | `L2T_STARS.py` (`load_prior`) | `.bias` and `.bias.UQ` initialize to `0.0` and `0.05` when absent from `L2T_STARS_MODEL` | Preserves 4-layer granule schema and avoids mid-mission baseline shifts |
| **DEF-02** | HLS Calibration | `L2T_STARS.py` (`DEFAULT_CALIBRATE_FINE`) | Fine-scale HLS inputs are not linearly regressed to VIIRS (`False`) | Prevents radiometric step-changes across the LOM archive |
| **DEF-03** | Temporal Windows | `L2T_STARS.py` / `process_ECOSTRESS_data_fusion.jl` | Covariance uses 8-day VIIRS window; fusion uses post-prior window | Stabilizes `fast_var_est` during short revisit intervals |
| **DEF-04** | Zero-Value Masking | `L2T_STARS.py` (`generate_*_image`) | Exact `0.0` values are coerced to `NaN` prior to fusion | Prevents `0`-padded tile/swath borders from corrupting covariance |
| **DEF-05** | Spatial Hyperparameters | `process_ECOSTRESS_data_fusion.jl` | Fixed $200\text{ m}$ range, $\nu=1.5$, $100\text{ m}$ buffer, `smooth = false` | Guarantees uniform spatial texture and UQ scaling globally |
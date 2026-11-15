# Development Notes: Translating the Julia STARS Data Fusion Pipeline to Rust/PyO3

## 1. Executive Summary & Motivation

The consolidated, pruned Julia script (`process_ECOSTRESS_data_fusion.jl`) establishes a self-contained mathematical specification for the Spatial and Temporal Adaptive Reflection Fusion Model (STARS) used in the ECOSTRESS Collection 2 pipeline. While pruning dead code and inlining dependencies removed external package fragility (`STARS.jl`, `GeoDataFrames`, `sentinel2.geojson`), each Julia invocation still incurs **~110–130 seconds of startup and JIT compilation overhead** per product (`~220–260 seconds` per granule across `NDVI` and `albedo`), while actual compute takes only ~23 seconds (NDVI) and ~52 seconds (albedo) on 8 threads.

Translating the core fusion engine into a **PyO3 / Rust extension module** (`stars_fusion`) built with `maturin` eliminates JIT compilation latency entirely, removes subprocess spawning and intermediate GeoTIFF serialization overhead from `L2T_STARS.py`, and provides deterministic multithreading via `rayon`.

---

## 2. Target Crate & Package Architecture

### 2.1 Directory Layout
```text
stars_fusion/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs            # PyO3 module bindings & NumPy array conversions
│   ├── covariance.rs     # Matérn-3/2 kernel, compute_n_eff, fast_var_est
│   ├── grid.rs           # Coordinate axes, bounding boxes, window slicing, & stitching
│   ├── obs_operator.rs   # Uniform spatial aggregation matrix (H)
│   └── kalman.rs         # Forward Kalman filter step & moving-window execution
└── python/
    └── stars_fusion/
        ├── __init__.py   # Re-exports Rust functions + optional rasterio CLI wrapper
        └── cli.py        # Drop-in replacement for process_ECOSTRESS_data_fusion.jl
```

### 2.2 Recommended Rust Crates (`Cargo.toml`)
* **`pyo3` (`0.22+`, feature `extension-module`)**: Exposes Rust functions directly as Python callables.
* **`numpy` (`0.22+`)**: Zero-copy `PyReadonlyArray2`/`PyReadonlyArray3` views into NumPy arrays and `PyArray2` outputs.
* **`ndarray` (`0.16+`)**: N-dimensional array slicing and views for `(row, col, time)` data cubes.
* **`faer` (`0.19+`) or `nalgebra` (`0.33+`)**: Pure-Rust dense linear algebra with fast Cholesky (`LLT`) factorization and triangular solves without external OpenBLAS/LAPACK linking issues across macOS ARM64 and Linux containers.
* **`rayon` (`1.10+`)**: Work-stealing thread pool for parallelizing the independent coarse-pixel moving windows.

### 2.3 Build Configuration (`pyproject.toml`)
* Build backend: `maturin` (`build-backend = "maturin"`), allowing standard `pip install .` or `maturin develop --release` inside the `ECOSTRESS` mamba environment.
* Set `RUSTFLAGS="-C target-cpu=native"` for local builds or standard baseline targets (`x86-64-v3` / `aarch64-apple-darwin`) for portable wheels.

---

## 3. Mathematical & Algorithmic Specification

To maintain bit-for-bit / machine-epsilon parity (`MAE ~ 1e-12`, `r = 1.000000`) with the verified Julia reference outputs, the Rust implementation must preserve the exact floating-point (`f64`) equations and indexing conventions below.

### 3.1 Coordinate Conventions & Memory Layout
* **Julia vs. NumPy/Rust Axis Order:**
  * In `Rasters.jl`, GeoTIFFs are loaded in `(X, Y)` order (`dims(ref, X)` has length `x_size` = columns/easting, `dims(ref, Y)` has length `y_size` = rows/northing).
  * In Python/NumPy (`rasterio`), arrays are `(row, col)` = `(Y, X)`.
  * **Convention Recommendation:** Standardize the Rust API on standard NumPy `(Y, X)` or `(T, Y, X)` layout, and explicitly pass the 1D coordinate vectors `x_coords: &[f64]` (easting, ascending) and `y_coords: &[f64]` (northing, descending) along with `cell_width: f64` (`> 0`) and `cell_height: f64` (`< 0`, or positive magnitude `dy = |cell_height|`).
* **Pixel Interval Semantics (`Intervals(Start())`):**
  * Each grid coordinate $x_i$ represents the **top-left (west/north) corner** of the cell, NOT the cell center.
  * Cell bounds for index $(r, c)$:
    * $x \in [x_c, x_c + \Delta x]$
    * $y \in [y_r - \Delta y, y_r]$ (where $\Delta y > 0$ is the positive vertical cell size).
  * This corner-referenced convention is critical when computing `target_bbox`, `window_bbox`, and `unif_weighted_obs_operator`.

### 3.2 Spatial Innovation Variance Estimation (`compute_n_eff` & `fast_var_est`)
1. **Effective Sample Size (`compute_n_eff`):**
   * Parameters: `agg_scale = round(coarse_cell_size / fine_cell_size)` (`7` for NDVI 490m/70m; `14` for albedo 980m/70m), `range = 2.0`, `smoothness = 1.5`.
   * Construct an `agg_scale × agg_scale` grid of integer coordinates $(i, j) \in \{1, \dots, \text{agg\_scale}\}^2$ (`N = agg_scale^2` points).
   * Compute the $N \times N$ Matérn-3/2 correlation matrix $C$ with nugget `reg = 1e-16` and length scale $\sigma = 2.0$:
     $$C_{ab} = \left(1 + \frac{\sqrt{3} r_{ab}}{\sigma}\right) \exp\left(-\frac{\sqrt{3} r_{ab}}{\sigma}\right) + \text{reg} \cdot \delta_{ab}$$
     where $r_{ab} = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}$.
   * Return:
     $$n_{\text{eff}} = \frac{\text{agg\_scale}^4}{\sum_{a=1}^N \sum_{b=1}^N C_{ab}}$$

2. **Pixel-Wise Temporal Difference Variance (`fast_var_est`):**
   * Input: `coarse_images` of shape `(n_cov_times, y_coarse, x_coarse)` (default 8 daily VIIRS steps).
   * For each coarse pixel $(r, c)$:
     * Count valid (non-`NaN`) observations across time: $n_{\text{obs}} = \sum_{t=1}^{T_{\text{cov}}} \mathbb{I}(\neg\text{isnan}(z_t))$.
     * If $n_{\text{obs}} < 8$ (`min_num_obs = 8`), set $\sigma^2_{\text{sp}}(r, c) = 10^{-4}$ (`default_var = 1e-4`).
     * Otherwise, compute first differences along time: $d_t = z_{t+1} - z_t$ for $t = 1, \dots, T_{\text{cov}} - 1$.
     * Filter out `NaN` values from $\{d_t\}$, compute sample variance (with Bessel's correction, denominator $m - 1$), and multiply by $n_{\text{eff}}$:
       $$\sigma^2_{\text{sp}}(r, c) = \text{Var}_{\text{sample}}(\{d_t : \neg\text{isnan}(d_t)\}) \cdot n_{\text{eff}}$$
   * Covariance parameter vector for coarse cell $(r, c)$:
     $$\theta(r, c) = [\sigma^2_{\text{sp}}(r, c),\; \sigma = 200.0,\; \text{nugget} = 10^{-10},\; \nu = 1.5]$$

### 3.3 Moving-Window Geometry & Bounds
For each coarse cell $(r_c, c_c)$ in the tile:
1. **Target Bounding Box (`target_bbox`):**
   * Start with the coarse cell extent $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$.
   * Expand by `fine_cell_width` (`70.0` m) on all four sides:
     $$\text{target\_bbox} = [x_{\min} - \Delta x_{\text{fine}},\; x_{\max} + \Delta x_{\text{fine}}] \times [y_{\min} - \Delta y_{\text{fine}},\; y_{\max} + \Delta y_{\text{fine}}]$$
2. **Window Bounding Box (`window_bbox`):**
   * Expand `target_bbox` by `buffer_distance` (`100.0` m) on all four sides:
     $$\text{window\_bbox} = [\text{target\_bbox}.x_{\min} - 100.0,\; \text{target\_bbox}.x_{\max} + 100.0] \times [\text{target\_bbox}.y_{\min} - 100.0,\; \text{target\_bbox}.y_{\max} + 100.0]$$
3. **Subsetting Pixels (`Rasters.Between`):**
   * A pixel belongs to a bounding box if its **cell interval** overlaps/falls within the bounding box interval (matching `DimensionalData` `Between` on `Intervals(Start())` lookups: cell $[x_i, x_i + \Delta x]$ is selected when $x_i \ge \text{bbox}.x_{\min}$ and $x_i + \Delta x \le \text{bbox}.x_{\max}$, or verify boundary inclusivity against `test_ECOSTRESS_data_fusion.py`).
   * *Note on typical window dimensions (interior tile pixel):*
     * **NDVI (`490m` coarse / `70m` fine, `170m` total pad):**
       * `target_bbox` width = $490 + 140 = 630\text{ m}$ ($9 \times 9$ fine pixels = $81$).
       * `window_bbox` width = $490 + 340 = 830\text{ m}$ ($3 \times 3$ coarse pixels = $9$; $\sim 11 \times 11$ to $13 \times 13$ fine pixels $\approx 121\text{–}169$).
       * State vector dimension: $n_{\text{state}} = 9 + 144 = 153$.
     * **Albedo (`980m` coarse / `70m` fine, `170m` total pad):**
       * `window_bbox` width = $980 + 340 = 1320\text{ m}$ ($3 \times 3$ coarse pixels = $9$; $\sim 19 \times 19$ fine pixels $= 361$).
       * State vector dimension: $n_{\text{state}} = 9 + 361 = 370$.
   * **Important Optimization Opportunity:** For all interior coarse pixels where the window does not clip the tile boundary, the relative geometry (`H`, `matern32_cor`, `F`, and the `ar_vars` block of `Q`) is **identical** up to scalar multiplication by $\sigma^2_{\text{sp}}(r_c, c_c)$! Caching the template `H` and unit-variance Matérn correlation matrix $C_0$ per window shape `(n_coarse_rows, n_coarse_cols, n_fine_rows, n_fine_cols)` eliminates >99% of pairwise distance and `exp()` evaluations.

### 3.4 Observation Operator (`unif_weighted_obs_operator`)
Given sensor grid top-left coordinates $(s_x, s_y)$ with cell size $(w_s, h_s)$ ($w_s, h_s > 0$) and target fine grid top-left coordinates $(t_x, t_y)$ with cell size $(w_t, h_t)$ ($w_t, h_t > 0$):
* Center-to-center distance threshold:
  $$d_{1, ij} = \left| \left(s_{x,i} + \frac{w_s}{2}\right) - \left(t_{x,j} + \frac{w_t}{2}\right) \right| \le \frac{w_s}{2}$$
  $$d_{2, ij} = \left| \left(s_{y,i} - \frac{h_s}{2}\right) - \left(t_{y,j} - \frac{h_t}{2}\right) \right| \le \frac{h_s}{2}$$
  *(Note: In Julia, `cell_height` is negative, so `sensor[:, 2] .+ sensor_res[2] / 2` subtracts half the positive cell height).*
* Binary indicator $B_{ij} = d_{1,ij} \land d_{2,ij}$, row-normalized:
  $$H_{ij} = \frac{B_{ij}}{\sum_{k} B_{ik}}$$
* Stack coarse and fine observation matrices vertically:
  $$H_{\text{obs}} = \begin{bmatrix} H_{\text{coarse}} \\ H_{\text{fine}} \end{bmatrix} \in \mathbb{R}^{(n_c + n_f) \times n_f}$$
* Prepend the coarse additive bias identity block ($n_{\text{bias}} = n_c$, since `offset_ar = [1.0, 0.0]`):
  $$H = \begin{bmatrix} I_{n_c} & H_{\text{coarse}} \\ 0_{n_f \times n_c} & H_{\text{fine}} \end{bmatrix} \in \mathbb{R}^{(n_c + n_f) \times (n_c + n_f)}$$

### 3.5 State-Space Model & Forward Kalman Filter (`STARS_fusion_step`)
* **State vector ($x \in \mathbb{R}^{n_c + n_f}$):**
  $$x_0 = \begin{bmatrix} \mu_{\text{bias, prior}} \\ \mu_{\text{fine, prior}} \end{bmatrix}, \quad P_0 = \text{Diag}\left(\begin{bmatrix} \sigma^2_{\text{bias, prior}} \\ \sigma^2_{\text{fine, prior}} \end{bmatrix}\right)$$
  *(Default priors when `prior` is `None`: $\mu_{\text{fine}} = 0.3$, $\sigma_{\text{fine}} = 0.01$, $\mu_{\text{bias}} = 0.0$, $\sigma_{\text{bias}} = 10^{-5}$).*
* **Transition & Process Noise Matrices:**
  * With `offset_ar = [1.0, 0.0]` and `offset_var = [1e-5, 1e-5]`:
    $$F = I_{n_c + n_f}$$
    $$Q = \begin{bmatrix} 10^{-5} I_{n_c} & 0 \\ 0 & \sigma^2_{\text{sp}} C_{\text{Matérn-3/2}} \end{bmatrix}$$
* **Observation Noise Variances:**
  * `coarse_err_var = 1e-6`, `fine_err_var = 1e-6`.
* **Time Loop ($t = 1, \dots, T_{\text{target}}$):**
  1. **Predict:**
     $$x_{\text{pred}} = F x_{\text{filt}} = x_{\text{filt}}$$
     $$P_{\text{pred}} = F P_{\text{filt}} F^\top + Q = P_{\text{filt}} + Q$$
  2. **Mask Missing Observations:**
     * Let $\mathcal{M}_t = \{i : \neg\text{isnan}(y_{i,t})\}$.
     * If $|\mathcal{M}_t| == 0$, set $x_{\text{filt}} = x_{\text{pred}}, P_{\text{filt}} = P_{\text{pred}}$ and continue.
  3. **Update (Cholesky Solve):**
     * Subset rows $H_t = H[\mathcal{M}_t, :]$, $y_t = y[\mathcal{M}_t, t]$, $R_t = \text{Diag}(\text{err\_vars}[\mathcal{M}_t, t])$.
     * Innovation: $r_t = y_t - H_t x_{\text{pred}}$.
     * Cross-covariance: $A = P_{\text{pred}} H_t^\top \in \mathbb{R}^{(n_c + n_f) \times |\mathcal{M}_t|}$.
     * Innovation covariance: $S = H_t A + R_t \in \mathbb{R}^{|\mathcal{M}_t| \times |\mathcal{M}_t|}$ (symmetric positive definite).
     * Compute Cholesky factorization $S = U^\top U$ (or $L L^\top$).
     * Solve $K = A S^{-1}$ via triangular solve (`K = (S \ A')'`) rather than explicit matrix inversion for superior numerical stability and speed.
     * Posterior mean & covariance:
       $$x_{\text{filt}} = x_{\text{pred}} + K r_t$$
       $$P_{\text{filt}} = P_{\text{pred}} - K A^\top$$
* **Outputs at $t = T_{\text{target}}$:**
  * `bias_mean` = $x_{\text{filt}}[0 : n_c]$
  * `fine_mean` = $x_{\text{filt}}[n_c : n_c + n_f]$
  * `bias_sd` = $\sqrt{\text{diag}(P_{\text{filt}})[0 : n_c]}$
  * `fine_sd` = $\sqrt{\text{diag}(P_{\text{filt}})[n_c : n_c + n_f]}$

### 3.6 Tile Stitching Order (`coarse_fine_data_fusion`)
* Parallelize the Kalman filter loop across all $N_{\text{coarse}} = \text{coarse\_rows} \times \text{coarse\_cols}$ windows using `rayon::par_iter()`, storing per-window results in a pre-allocated `Vec<WindowResult>`.
* Stitch sequentially in **column-major order** (`for row in 0..coarse_rows { for col in 0..coarse_cols { ... } }` matching Julia's `m1` ordering):
  * Write values falling inside `target_bbox` into the output grids **only where the output pixel is currently `NaN`** (`if out[r, c].is_nan() { out[r, c] = win_val }`).
  * Because `target_bbox` expands each coarse cell by `fine_cell_width` (1 fine pixel overlap on each edge), adjacent windows overlap by 2 fine pixels; writing only to `NaN` cells reproduces Julia's `inds = isnan.(subset); subset[inds] = resampled_subset[inds]` exact tie-breaking behavior.

---

## 4. Proposed Python/Rust API Surface

```python
import numpy as np
import stars_fusion

# 1. Compute effective sample size
n_eff: float = stars_fusion.compute_n_eff(
    agg_scale=7,
    range_m=2.0,
    smoothness=1.5,
)

# 2. Estimate spatial innovation variance (2D float64 array: coarse_rows x coarse_cols)
sp_var: np.ndarray = stars_fusion.fast_var_est(
    coarse_images,       # shape: (n_cov_dates, coarse_rows, coarse_cols), float64
    n_eff_agg=n_eff,
    min_num_obs=8,
    default_var=1e-4,
)

# 3. Run moving-window Kalman filter data fusion
mean, sd, mean_bias, sd_bias = stars_fusion.coarse_fine_data_fusion(
    coarse_images=coarse_images,       # (n_dates, coarse_rows, coarse_cols), float64
    fine_images=fine_images,           # (n_dates, fine_rows, fine_cols), float64
    x_coarse=x_coarse,                 # (coarse_cols,), float64 (top-left easting)
    y_coarse=y_coarse,                 # (coarse_rows,), float64 (top-left northing)
    x_fine=x_fine,                     # (fine_cols,), float64 (top-left easting)
    y_fine=y_fine,                     # (fine_rows,), float64 (top-left northing)
    coarse_cell_size=490.0,
    fine_cell_size=70.0,
    sp_var=sp_var,                     # (coarse_rows, coarse_cols), float64
    prior_mean=prior_mean,             # Optional[(fine_rows, fine_cols)], float64
    prior_sd=prior_sd,                 # Optional[(fine_rows, fine_cols)], float64
    prior_bias_mean=prior_bias_mean,   # Optional[(coarse_rows, coarse_cols)], float64
    prior_bias_sd=prior_bias_sd,     # Optional[(coarse_rows, coarse_cols)], float64
    target_time_index=-1,              # Default: last timestep
    buffer_distance=100.0,
    coarse_err_var=1e-6,
    fine_err_var=1e-6,
    offset_ar=1.0,
    offset_var=1e-5,
)
```

---

## 5. Verification & Rollout Checklist

1. **Unit Tests (`cargo test`):**
   * Verify `compute_n_eff(7, 2.0, 1.5)` and `compute_n_eff(14, 2.0, 1.5)` match Julia to `1e-15`.
   * Verify `unif_weighted_obs_operator` row sums equal `1.0` and boundary inclusions match Julia's `Between` logic on both interior and edge windows.
   * Verify single-window `kalman_filter_step` against Julia with all-NaN observations, partial-NaN observations, and full observations.
2. **End-to-End Regression (`test_ECOSTRESS_data_fusion.py`):**
   * Add `--backend rust` to `test_ECOSTRESS_data_fusion.py` invoking `python -m stars_fusion.cli` with the exact same CLI positional arguments as `process_ECOSTRESS_data_fusion.jl`.
   * Confirm `Overall result: PASS` across all 8 layers (`NDVI` and `albedo` posterior mean, UQ, bias, and bias UQ) on tile `11SPS` against `~/data/STARS_julia_isolated_test_bundle/reference_outputs`.
3. **Performance Target:**
   * Expected wall-clock runtime on 8 threads (Apple M-series / Xeon): **~2–5 seconds** for `NDVI` and **~6–12 seconds** for `albedo` (down from `143.77s` and `92.35s` in Julia—a **~15–20× end-to-end speedup**).
4. **Pipeline Integration (`L2T_STARS.py`):**
   * Replace `subprocess.run(["julia", ...])` in `L2T_STARS.py` with direct in-memory calls to `stars_fusion.coarse_fine_data_fusion(...)`, eliminating intermediate GeoTIFF writes and removing Julia from the operational Docker container.
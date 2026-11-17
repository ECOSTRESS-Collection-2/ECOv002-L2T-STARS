# Development Plan: Aligning `ECOv002-L2T-STARS` with `ECOSTRESS-Collection-2`

**Target Repository:** `ECOv002-L2T-STARS`  
**Reference Implementation:** `ECOSTRESS-Collection-2/L2T_STARS`  
**Objective:** Bring the standalone `ECOv002-L2T-STARS` package into full algorithmic, I/O, and validation parity with the refactored `L2T_STARS` submodule from `ECOSTRESS-Collection-2`, while preserving its standalone Python package structure (`pyproject.toml`, internal `exit_codes.py`, `runconfig.py`, `daterange/`, and `timer/`).

---

## 1. Executive Summary & Architectural Delta

The `L2T_STARS` submodule in `ECOSTRESS-Collection-2` has undergone significant optimization and stabilization compared to the original `ECOv002-L2T-STARS` standalone repository:

1. **Self-Contained Julia Data Fusion Engine:** Replaced the multi-file embedded `STARS_jl` package (which depended on `GeoDataFrames`, `sentinel2.geojson`, `HTTP`, `Plots`, and runtime `Pkg.add("OpenSSL")`) with a standalone, zero-internal-package script (`process_ECOSTRESS_data_fusion.jl`) that infers spatial dimensions directly from input GeoTIFFs.
2. **Unified Hierarchical Input Staging (`DOWNSAMPLED_products`):** Replaced four separate flat staging folders (`NDVI_coarse`, `NDVI_fine`, `albedo_coarse`, `albedo_fine`) with a deterministic date/tile directory tree (`DOWNSAMPLED_products/<YYYY>/<YYYY-MM-DD>/<tile>/STARS_<var>_<tile>_<res>m.tif`) that caches intermediate rasters and avoids redundant regeneration.
3. **Prior State & Bias Initialization (`load_prior`):** Fixed the resolution mismatch for coarse bias priors (`490 m` for NDVI and `980 m` for albedo instead of `70 m`) and added automatic synthesis of zero-mean (`0.0`), uniform-uncertainty (`0.05`) bias rasters when prior `.bias` / `.bias.UQ` files are absent.
4. **Isolated Regression Testing & LOM Documentation:** Added a self-contained test bundle (`test_bundle/`), an isolated numerical verification harness (`test_ECOSTRESS_data_fusion.py`), Life-of-Mission (LOM) consistency documentation, and a technical specification for a future Rust/PyO3 backend (`stars_fusion`).

---

## 2. Phased Implementation Plan

### Phase 1: Package Packaging & Configuration Fixes (`pyproject.toml`, `makefile`)
- [ ] **Fix Package Discovery in `pyproject.toml`:**
  - [ ] Update `[tool.setuptools.packages.find]` from `include = ["ECOv002_L2T_JET"]` to `include = ["ECOv002_L2T_STARS*"]` so all subpackages (`daterange`, `timer`) are properly bundled in wheels and editable installs.
  - [ ] Update `[tool.setuptools.package-data]` to include `["*.txt", "*.xml", "*.jl", "Project.toml"]`.
- [ ] **Align Dependencies:**
  - [ ] Verify `ECOv002-granules`, `VCD43NRT`, `harmonized-landsat-sentinel`, `sentinel-tiles`, and `colored-logging` versions match the APIs used in `ECOSTRESS-Collection-2` (e.g., `VIIRSDownloaderNDVI` / `VIIRSDownloaderAlbedo` vs. `VCD43NRT`).
- [ ] **Update `makefile` Targets:**
  - [x] Add `repomix` target to generate repository bundles cleanly.
  - [ ] Refine `repomix` target to ignore `repomix*.xml` and `test_bundle/**`.
  - [ ] Update `test-julia` (currently referencing non-existent `julia_smoke_test.jl` and `process_ECOSTRESS_data_fusion_distributed_bias.jl`) to invoke `python verification/test_ECOSTRESS_data_fusion.py`.
  - [ ] Keep `JULIA_PROJECT_NAME = julia_env` (or `ECOv002_L2T_STARS`) aligned with the pruned Julia environment.

---

### Phase 2: Replace Embedded `STARS_jl` with Standalone `process_ECOSTRESS_data_fusion.jl`

- [x] **Replace `ECOv002_L2T_STARS/process_ECOSTRESS_data_fusion.jl`:**
  - [x] Port the self-contained `process_ECOSTRESS_data_fusion.jl` from `ECOSTRESS-Collection-2`.
  - [x] Remove runtime `Pkg.add("OpenSSL")` network calls during PGE execution.
  - [x] Remove dependency on `STARS_jl` (`STARS.jl`, `HLS.jl`, `VNP43.jl`, `VIIRS.jl`, `MODLAND.jl`, `sentinel_tiles.jl`, `sentinel2.geojson`).
  - [x] Extract `(X, Y)` grid dimensions and CRS directly from reference GeoTIFFs via `extract_tile_dims(ref_tif)` instead of looking up `sentinel2.geojson`.
  - [x] Read inputs from the unified `DOWNSAMPLED_products` tree using `downsampled_tif_path(dir, d, tile, product, cell_size)`.
  - [x] Write `fusion_results.mean_bias` and `fusion_results.SD_bias` using `(x_coarse, y_coarse, Band(1:1))` dimensions with explicit `force=true, driver="GTiff"` to all `Rasters.write` calls.
- [x] **Prune `ECOv002_L2T_STARS/Project.toml`:**
  - [x] Update `Project.toml` to the pruned dependency list (`ArchGDAL`, `BlockDiagonals`, `Dates`, `DimensionalData`, `Distances`, `GaussianRandomFields`, `Glob`, `LinearAlgebra`, `Logging`, `Plots`, `ProgressMeter`, `Rasters`, `SparseArrays`, `Statistics`)—*optional: remove unused `DimensionalData`, `Glob`, and `Plots` entries*.
- [x] **Remove `ECOv002_L2T_STARS/STARS_jl/`:**
  - [x] Remove the legacy `ECOv002_L2T_STARS/STARS_jl/` subtree and `sentinel2.geojson`.

---

### Phase 3: Align `ECOv002_L2T_STARS/L2T_STARS.py`

Update `ECOv002_L2T_STARS/L2T_STARS.py` to match `ECOSTRESS-Collection-2/L2T_STARS.py` while **retaining package-internal imports** (`.exit_codes`, `.runconfig`, `.daterange`, `.timer`, `.PGEVersion`) and the tile-only `granule_ID` convention:

- [x] **3.1 Prior Loading (`load_prior`):**
  - [x] Update function signature to accept `NDVI_resolution: int` and `albedo_resolution: int`, plus `initial_bias_mean: float = 0.0` and `initial_bias_UQ: float = 0.05`.
  - [x] Generate `.bias` and `.bias.UQ` filenames at **coarse resolution** (`NDVI_resolution` = `490`, `albedo_resolution` = `980`) rather than `target_resolution` (`70`).
  - [x] Synthesize and write constant rasters on `SentinelTileGrid(target_resolution=...)` (`0.0` for bias mean, `0.05` for bias UQ) when prior `.bias` or `.bias.UQ` GeoTIFFs do not exist on disk.
- [x] **3.2 Input Staging (`generate_downsampled_filename` & `generate_STARS_inputs`):**
  - [x] Add `generate_downsampled_filename(directory, variable, date_UTC, tile, cell_size)` producing `<sources_directory>/DOWNSAMPLED_products/<YYYY>/<YYYY-MM-DD>/<tile>/STARS_<variable>_<tile>_<cell_size>m.tif`.
  - [x] Update `generate_STARS_inputs` to write directly into `downsampled_directory` (`DOWNSAMPLED_products`), skip regenerating existing GeoTIFFs (`exists(...)`), fix `calibrate_fine: bool = True`, and log exceptions on coarse albedo failures.
- [x] **3.3 Julia Subprocess Execution (`process_julia_data_fusion`):**
  - [x] Remove `install_STARS_jl()` and `instantiate_STARS_jl()`.
  - [x] Pass `downsampled_directory` and `product_name` (`"NDVI"` or `"albedo"`) instead of separate `coarse_directory` and `fine_directory` paths.
  - [x] Support `julia +1.11` when `juliaup` is available, optional `--sysimage` detection via `STARS_JULIA_SYSIMAGE` or `julia_env/stars_sysimage.{so,dylib,dll}`, and `--project` resolution (with fallback to `ECOv002_L2T_STARS/`).
  - [x] Add `check=True` to `subprocess.run(command, shell=True, check=True)`.
- [x] **3.4 Posterior Cleanup & VIIRS Prefetching (`process_STARS_product` & `L2T_STARS`):**
  - [x] Always re-write and retain posterior `.bias` and `.bias.UQ` GeoTIFFs in `L2T_STARS_MODEL` (only gate the fine-resolution posterior GeoTIFF re-writes behind `if not remove_posterior:`).
  - [x] Ensure `VNP09GA` prefetching runs in standard execution mode (`else:` branch in `L2T_STARS()`) as well as `sources_only=True`.
  - [x] Preserve standalone package imports (`.exit_codes`, `.runconfig`, `.daterange`, `.timer`, `.PGEVersion`) and tile-only `granule_ID` format (`ECOv002_L2T_STARS_<tile>_<YYYYMMDD>_<build>_<counter>`).

---

### Phase 4: Port Validation Suite & Documentation

- [ ] **4.1 Isolated Fusion Test Runner (`verification/test_ECOSTRESS_data_fusion.py`):**
  - [x] Copy `test_ECOSTRESS_data_fusion.py` and `test_bundle/` (`11SPS`, target `2026-10-27`, prior `2026-10-23`) into the repository.
  - [ ] Update `DEFAULT_BUNDLE_DIR` and `candidate_julia_paths` in `verification/test_ECOSTRESS_data_fusion.py` to resolve `../test_bundle` and `../ECOv002_L2T_STARS/process_ECOSTRESS_data_fusion.jl` from the `verification/` subdirectory.
  - [ ] Ensure `test_bundle/` is excluded from Git commits (via `.gitignore` or external fixture hosting) so large GeoTIFFs do not bloat repository history.
- [ ] **4.2 End-to-End Integration Script (`verification/test_L2T_STARS.py` & `example_L2T_STARS_with_download.py`):**
  - [x] Copy `test_L2T_STARS.py` into `verification/`.
  - [ ] Update `verification/test_L2T_STARS.py` imports from `L2T_STARS` to `ECOv002_L2T_STARS` and fix `open_granule("test_bundle/ECOv002_L2T_STARS_11SPS_20261027_0700_01")`.
- [ ] **4.3 Include LOM & Rust Migration Documentation:**
  - [ ] Add `known_deficiencies_maintained_in_stars_collection_2_for_lom_record_consistency.md` to document DEF-01 through DEF-05.
  - [ ] Add `stars_rust_pyo3_translation_development_notes.md` as the blueprint for replacing the Julia subprocess with a native Rust/PyO3 extension module (`stars_fusion`).

---

## 3. Verification Checklist

- [ ] **Editable Install:** `pip install -e '.[dev]'` installs `ECOv002_L2T_STARS` with all subpackages and `.jl`/`.xml`/`.txt` assets included.
- [ ] **Julia Environment:** `make install-julia` instantiates and precompiles `process_ECOSTRESS_data_fusion.jl` without network calls at runtime.
- [ ] **Prior State Generation:** `load_prior()` creates `490 m` (NDVI) and `980 m` (albedo) `.bias` and `.bias.UQ` rasters when missing from `L2T_STARS_MODEL`.
- [ ] **Isolated Bundle Parity:** `python verification/test_ECOSTRESS_data_fusion.py` achieves `Overall result: PASS` across all 8 layers (`NDVI`, `NDVI-UQ`, `NDVI.bias`, `NDVI.bias.UQ`, `albedo`, `albedo-UQ`, `albedo.bias`, `albedo.bias.UQ`).
- [ ] **Full PGE Run:** `python verification/test_L2T_STARS.py` exits with code `0` (`SUCCESS_EXIT_CODE`) and matches reference granule layers within tolerance.
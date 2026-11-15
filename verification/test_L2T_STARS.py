import glob
import logging
from os.path import join, expanduser
from os import makedirs
import os
import tempfile
import zipfile

import numpy as np
import rasterio

from ECOv002_CMR import download_ECOSTRESS_granule
from ECOv002_granules import open_granule
from L2T_STARS import generate_L2T_STARS_runconfig, L2T_STARS
from L2T_STARS import L2TSTARSConfig

# Disable logger output
# logging.getLogger().handlers = []

logger = logging.getLogger(__name__)

L2T_STARS_expected_output = open_granule("L2T_STARS/test_bundle/ECOv002_L2T_STARS_11SPS_20261027_0700_01")

# Set working directory
working_directory = expanduser(join("~", "data", "ECOSTRESS_collection_2_testing"))
ECOv002_download_directory = join(working_directory, "ECOv002_downloads")
diff_output_directory = join(working_directory, "L2T_STARS_diff")
makedirs(working_directory, exist_ok=True)
makedirs(ECOv002_download_directory, exist_ok=True)
makedirs(diff_output_directory, exist_ok=True)

logger.info("retrieving collection 2 L2T LSTE granule")
# Retrieve LST LSTE granule from CMR API for target date
L2T_LSTE_granule = download_ECOSTRESS_granule(
    product="L2T_LSTE",
    tile="11SPS",
    aquisition_date="2026-10-27",
    parent_directory=ECOv002_download_directory
)

# Retrieve L2T STARS granule from CMR API as prior
L2T_STARS_granule = download_ECOSTRESS_granule(
    product="L2T_STARS",
    tile="11SPS",
    aquisition_date="2026-10-23",
    parent_directory=ECOv002_download_directory
)

# # Retrieve the published L2T STARS granule for the target date to compare against
# L2T_STARS_expected_output = download_ECOSTRESS_granule(
#     product="L2T_STARS",
#     tile="11SPS",
#     aquisition_date="2026-10-27",
#     parent_directory=ECOv002_download_directory
# )

# Generate XML run-config file for L2T STARS PGE run
runconfig_filename = generate_L2T_STARS_runconfig(
    L2T_LSTE_filename=L2T_LSTE_granule.product_filename,
    prior_L2T_STARS_filename=L2T_STARS_granule.product_filename,
    working_directory=working_directory
)

print("Runconfig generated:", runconfig_filename)

# Run L2T STARS PGE
exit_code = L2T_STARS(
    runconfig_filename=runconfig_filename,
    # threads=1,
    # num_workers=8,
    remove_input_staging=False,
    remove_prior=False,
    remove_posterior=False,
    # overwrite=True
)

print("L2T STARS PGE exit code:", exit_code)

# Compare the locally generated product against the downloaded product
COMPARISON_VARIABLES = [
    "NDVI",
    "NDVI-UQ",
    "albedo",
    "albedo-UQ",
]


def resolve_granule_path(path: str, temp_root: str, label: str) -> str:
    """Resolve a directory or extract a .zip into an isolated temp directory."""
    if os.path.isdir(path):
        return path

    if path.endswith(".zip") and os.path.isfile(path):
        extract_dir = os.path.join(temp_root, label)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)

        root_tifs = glob.glob(os.path.join(extract_dir, "*.tif"))
        if not root_tifs:
            subfolders = [
                os.path.join(extract_dir, d)
                for d in os.listdir(extract_dir)
                if os.path.isdir(os.path.join(extract_dir, d))
            ]
            if subfolders:
                return subfolders[0]
        return extract_dir

    raise FileNotFoundError(f"Granule path not found: {path}")


def get_template_profile(granule_dir: str, variable: str) -> dict:
    """Read spatial metadata (CRS, transform, dimensions) from the source GeoTIFF."""
    matches = glob.glob(os.path.join(granule_dir, f"*_{variable}.tif"))
    if not matches:
        raise FileNotFoundError(f"Template GeoTIFF not found for {variable} in {granule_dir}")
    with rasterio.open(matches[0]) as src:
        profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan,
        compress="deflate",
    )
    return profile


def write_diff_rasters(
    local: np.ndarray,
    downloaded: np.ndarray,
    profile: dict,
    output_dir: str,
    granule_stem: str,
    variable: str,
) -> tuple[str, str]:
    """Write signed difference (local - downloaded) and symmetric percent difference GeoTIFFs."""
    os.makedirs(output_dir, exist_ok=True)

    a = local.astype(np.float64)
    b = downloaded.astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(b)

    diff_grid = np.full(a.shape, np.nan, dtype=np.float32)
    diff_grid[valid] = (a[valid] - b[valid]).astype(np.float32)

    pct_diff_grid = np.full(a.shape, np.nan, dtype=np.float32)
    denom = np.abs(a) + np.abs(b)
    nonzero_valid = valid & (denom > 0)
    pct_diff_grid[nonzero_valid] = (
        200.0 * (a[nonzero_valid] - b[nonzero_valid]) / denom[nonzero_valid]
    ).astype(np.float32)
    pct_diff_grid[valid & (denom == 0)] = 0.0

    diff_path = os.path.join(output_dir, f"{granule_stem}_{variable}_diff.tif")
    pct_diff_path = os.path.join(output_dir, f"{granule_stem}_{variable}_pct_diff.tif")

    with rasterio.open(diff_path, "w", **profile) as dst:
        dst.write(diff_grid, 1)

    with rasterio.open(pct_diff_path, "w", **profile) as dst:
        dst.write(pct_diff_grid, 1)

    return diff_path, pct_diff_path


def compare_variable(variable: str, local: np.ndarray, downloaded: np.ndarray) -> dict:
    row = {
        "variable": variable,
        "status": "OK",
        "n": None,
        "nan_diff": None,
        "identical_pct": None,
        "mean_local": None,
        "mean_ref": None,
        "bias": None,
        "pct_bias": None,
        "mae": None,
        "rmse": None,
        "max_abs_diff": None,
        "smape_pct": None,
        "r": None,
    }

    if local.shape != downloaded.shape:
        row["status"] = f"shape mismatch {local.shape} vs {downloaded.shape}"
        return row

    local = local.astype(np.float64)
    downloaded = downloaded.astype(np.float64)
    local_nan = ~np.isfinite(local)
    downloaded_nan = ~np.isfinite(downloaded)
    valid = ~(local_nan | downloaded_nan)
    nan_disagreement = int(np.count_nonzero(local_nan != downloaded_nan))

    row["nan_diff"] = nan_disagreement

    if not valid.any():
        row["status"] = "no jointly valid pixels"
        return row

    a = local[valid]
    b = downloaded[valid]
    diff = a - b
    identical = 100.0 * np.mean(a == b)
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

    # Symmetric percent difference (sMAPE): 200 * |a - b| / (|a| + |b|), excluding 0/0 pairs
    denom = np.abs(a) + np.abs(b)
    nonzero = denom > 0
    sym_pct_diff = np.mean(200.0 * np.abs(diff[nonzero]) / denom[nonzero]) if nonzero.any() else 0.0

    # Signed percent bias relative to downloaded reference mean
    mean_b = b.mean()
    pct_bias = 100.0 * diff.mean() / abs(mean_b) if mean_b != 0 else np.nan

    row.update(
        {
            "n": int(a.size),
            "identical_pct": float(identical),
            "mean_local": float(a.mean()),
            "mean_ref": float(mean_b),
            "bias": float(diff.mean()),
            "pct_bias": float(pct_bias),
            "mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "max_abs_diff": float(np.abs(diff).max()),
            "smape_pct": float(sym_pct_diff),
            "r": float(corr),
        }
    )
    return row


def format_comparison_table(rows: list[dict]) -> str:
    headers = [
        "Layer",
        "Valid n",
        "NaN Diff",
        "Identical",
        "Mean Local",
        "Mean Ref",
        "Bias (%)",
        "MAE",
        "RMSE",
        "Max |Diff|",
        "sMAPE",
        "r",
    ]

    formatted_rows = []
    for r in rows:
        if r["status"] != "OK":
            formatted_rows.append([
                r["variable"],
                "-",
                str(r["nan_diff"]) if r["nan_diff"] is not None else "-",
                r["status"],
                "-", "-", "-", "-", "-", "-", "-", "-",
            ])
        else:
            bias_str = (
                f"{r['bias']:+.5f} ({r['pct_bias']:+.2f}%)"
                if np.isfinite(r["pct_bias"])
                else f"{r['bias']:+.5f}"
            )
            formatted_rows.append([
                r["variable"],
                f"{r['n']:,}",
                f"{r['nan_diff']:,}",
                f"{r['identical_pct']:.2f}%",
                f"{r['mean_local']:.5f}",
                f"{r['mean_ref']:.5f}",
                bias_str,
                f"{r['mae']:.5f}",
                f"{r['rmse']:.5f}",
                f"{r['max_abs_diff']:.5f}",
                f"{r['smape_pct']:.2f}%",
                f"{r['r']:.5f}" if np.isfinite(r["r"]) else "NaN",
            ])

    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in formatted_rows))
        for i in range(len(headers))
    ]

    def fmt_line(cells: list[str]) -> str:
        aligned = [
            cells[0].ljust(col_widths[0]),
            *(cells[i].rjust(col_widths[i]) for i in range(1, len(cells))),
        ]
        return "| " + " | ".join(aligned) + " |"

    separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    lines = [fmt_line(headers), separator, *(fmt_line(r) for r in formatted_rows)]
    return "\n".join(lines)


if exit_code == 0:
    runconfig = L2TSTARSConfig(runconfig_filename)

    with tempfile.TemporaryDirectory(prefix="ecov002_compare_") as temp_root:
        local_path = resolve_granule_path(runconfig.L2T_STARS_zip_filename, temp_root, "local")
        downloaded_path = resolve_granule_path(
            L2T_STARS_expected_output.product_filename, temp_root, "downloaded"
        )
        granule_stem = os.path.splitext(os.path.basename(runconfig.L2T_STARS_zip_filename))[0]

        local_granule = open_granule(local_path)
        downloaded_granule = open_granule(downloaded_path)

        print("\nLocal vs downloaded L2T STARS comparison")
        print(f"  local:      {local_path}")
        print(f"  downloaded: {downloaded_path}")
        print(f"  diff dir:   {diff_output_directory}\n")

        summary_rows = []
        for variable in COMPARISON_VARIABLES:
            try:
                local_data = np.array(local_granule.variable(variable))
                downloaded_data = np.array(downloaded_granule.variable(variable))
                summary_rows.append(compare_variable(variable, local_data, downloaded_data))

                if local_data.shape == downloaded_data.shape:
                    profile = get_template_profile(local_path, variable)
                    diff_path, pct_diff_path = write_diff_rasters(
                        local_data,
                        downloaded_data,
                        profile,
                        diff_output_directory,
                        granule_stem,
                        variable,
                    )
                    print(
                        f"Wrote diff rasters for {variable}: "
                        f"{os.path.basename(diff_path)}, {os.path.basename(pct_diff_path)}"
                    )
            except Exception as e:
                summary_rows.append({
                    "variable": variable,
                    "status": f"error: {e}",
                    "nan_diff": None,
                })

        print("\nLayer-by-Layer Granule Comparison Summary:\n")
        print(format_comparison_table(summary_rows))
else:
    print("L2T STARS PGE did not succeed; skipping comparison")
import glob
import os
import tempfile
import zipfile
import numpy as np
import rasterio
from ECOv002_granules import open_granule

# ==========================================
# CONFIGURATION: Update paths to your files
# ==========================================
LOCAL_PRODUCT_PATH = os.path.expanduser(
    "~/data/ECOv002_testing/L2T_STARS_output/ECOv002_L2T_STARS_11SPS_20261027_0700_01"
)
DOWNLOADED_PRODUCT_PATH = os.path.expanduser(
    "~/data/ECOv002_testing/ECOv002_L2T_STARS_11SPS_20261027_0713_01"
)
DIFF_OUTPUT_DIR = os.path.expanduser(
    "~/data/ECOv002_testing/L2T_STARS_diff"
)

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


def compare_variable(local: np.ndarray, downloaded: np.ndarray) -> str:
    if local.shape != downloaded.shape:
        return f"shape mismatch: local {local.shape}, downloaded {downloaded.shape}"

    local = local.astype(np.float64)
    downloaded = downloaded.astype(np.float64)
    local_nan = ~np.isfinite(local)
    downloaded_nan = ~np.isfinite(downloaded)
    valid = ~(local_nan | downloaded_nan)
    nan_disagreement = int(np.count_nonzero(local_nan != downloaded_nan))

    if not valid.any():
        return f"no jointly valid pixels (NaN-mask disagreements: {nan_disagreement})"

    a = local[valid]
    b = downloaded[valid]
    diff = a - b
    identical = 100 * np.mean(a == b)
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

    # Symmetric percent difference (sMAPE): 200 * |a - b| / (|a| + |b|), excluding 0/0 pairs
    denom = np.abs(a) + np.abs(b)
    nonzero = denom > 0
    sym_pct_diff = np.mean(200.0 * np.abs(diff[nonzero]) / denom[nonzero]) if nonzero.any() else 0.0

    # Signed percent bias relative to downloaded reference mean
    mean_b = b.mean()
    pct_bias = 100.0 * diff.mean() / abs(mean_b) if mean_b != 0 else np.nan

    return (
        f"n={a.size}  NaN-mask disagreements={nan_disagreement}  identical={identical:.2f}%\n"
        f"    mean local={a.mean():.6f}  mean downloaded={mean_b:.6f}  bias={diff.mean():.6f} ({pct_bias:+.2f}%)\n"
        f"    MAE={np.abs(diff).mean():.6f}  RMSE={np.sqrt((diff ** 2).mean()):.6f}  "
        f"max|diff|={np.abs(diff).max():.6f}  mean_pct_diff={sym_pct_diff:.2f}%  r={corr:.6f}"
    )


def main():
    with tempfile.TemporaryDirectory(prefix="ecov002_compare_") as temp_root:
        local_path = resolve_granule_path(LOCAL_PRODUCT_PATH, temp_root, "local")
        downloaded_path = resolve_granule_path(DOWNLOADED_PRODUCT_PATH, temp_root, "downloaded")
        granule_stem = os.path.basename(local_path.rstrip("/"))

        print("Opening local experimental product:", local_path)
        print("Opening downloaded reference product:", downloaded_path)
        print("Writing difference rasters to:", DIFF_OUTPUT_DIR)

        local_granule = open_granule(local_path)
        downloaded_granule = open_granule(downloaded_path)

        print("\n========================================")
        print("Local vs Downloaded Product Comparison")
        print("========================================")

        for variable in COMPARISON_VARIABLES:
            print(f"\nVariable: {variable}")
            try:
                local_data = np.array(local_granule.variable(variable))
                downloaded_data = np.array(downloaded_granule.variable(variable))
                print("  " + compare_variable(local_data, downloaded_data))

                if local_data.shape == downloaded_data.shape:
                    profile = get_template_profile(local_path, variable)
                    diff_path, pct_diff_path = write_diff_rasters(
                        local_data,
                        downloaded_data,
                        profile,
                        DIFF_OUTPUT_DIR,
                        granule_stem,
                        variable,
                    )
                    print(f"    wrote: {os.path.basename(diff_path)}")
                    print(f"    wrote: {os.path.basename(pct_diff_path)}")
            except Exception as e:
                print(f"  could not compare: {e}")


if __name__ == "__main__":
    main()
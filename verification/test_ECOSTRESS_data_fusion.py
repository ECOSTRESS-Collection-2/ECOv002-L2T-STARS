"""
Run isolated tests of `process_ECOSTRESS_data_fusion.jl` from a collected test
bundle (`manifest.json`) and quantitatively compare all generated posterior
GeoTIFF layers against the known reference outputs.

Compares all four output rasters per product (`NDVI` and `albedo`):
  - posterior (`STARS_<product>_<date>_<tile>_70m.tif`)
  - posterior_UQ (`STARS_<product>.UQ_<date>_<tile>_70m.tif`)
  - posterior_bias (`STARS_<product>.bias_<date>_<tile>_<coarse>m.tif`)
  - posterior_bias_UQ (`STARS_<product>.bias.UQ_<date>_<tile>_<coarse>m.tif`)

Optionally writes signed difference (`_diff.tif`) and symmetric percent
difference (`_pct_diff.tif`) GeoTIFFs plus a machine-readable JSON summary.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import rasterio


OUTPUT_KEYS = (
    "posterior",
    "posterior_UQ",
    "posterior_bias",
    "posterior_bias_UQ",
)

DEFAULT_BUNDLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_bundle",
)


def _find_julia_command() -> list[str]:
    if shutil.which("juliaup"):
        return ["julia", "+1.11"]
    return ["julia"]


def _find_sysimage(julia_env_dir: str) -> str | None:
    env_sysimage = os.environ.get("STARS_JULIA_SYSIMAGE")
    if env_sysimage and os.path.exists(env_sysimage):
        return env_sysimage
    for ext in ("so", "dylib", "dll"):
        candidate = os.path.join(julia_env_dir, f"stars_sysimage.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _build_julia_cmd(
    julia_script: str,
    bundle_dir: str,
    output_root: str,
    run_spec: dict[str, Any],
) -> list[str]:
    julia_script = os.path.abspath(os.path.expanduser(julia_script))
    julia_env_dir = os.path.abspath(
        os.path.join(os.path.dirname(julia_script), "..", "julia_env")
    )

    cmd = _find_julia_command()

    sysimage_path = _find_sysimage(julia_env_dir)
    if sysimage_path:
        cmd.append(f"--sysimage={sysimage_path}")

    if os.path.isdir(julia_env_dir) and not os.environ.get("JULIA_LOAD_PATH"):
        cmd.append(f"--project={julia_env_dir}")

    cmd.extend(
        [
            "--threads",
            "auto",
            julia_script,
            str(run_spec["tile"]),
            str(run_spec["coarse_cell_size"]),
            str(run_spec["fine_cell_size"]),
            str(run_spec["viirs_start_date"]),
            str(run_spec["viirs_end_date"]),
            str(run_spec["hls_start_date"]),
            str(run_spec["hls_end_date"]),
            os.path.join(bundle_dir, run_spec["inputs"]["downsampled_directory"]),
            str(run_spec["product"]),
            os.path.join(output_root, run_spec["isolated_test_outputs"]["posterior"]),
            os.path.join(output_root, run_spec["isolated_test_outputs"]["posterior_UQ"]),
            os.path.join(output_root, run_spec["isolated_test_outputs"]["posterior_bias"]),
            os.path.join(output_root, run_spec["isolated_test_outputs"]["posterior_bias_UQ"]),
        ]
    )

    priors = run_spec["inputs"].get("priors", {})
    if len(priors) == 4:
        cmd.extend(
            [
                os.path.join(bundle_dir, priors["prior"]),
                os.path.join(bundle_dir, priors["prior_UQ"]),
                os.path.join(bundle_dir, priors["prior_bias"]),
                os.path.join(bundle_dir, priors["prior_bias_UQ"]),
            ]
        )

    return cmd


def _write_diff_rasters(
    test_arr: np.ndarray,
    ref_arr: np.ndarray,
    profile: dict[str, Any],
    diff_dir: str,
    stem: str,
) -> tuple[str, str]:
    os.makedirs(diff_dir, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan,
        compress="deflate",
    )

    a = test_arr.astype(np.float64)
    b = ref_arr.astype(np.float64)
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

    diff_path = os.path.join(diff_dir, f"{stem}_diff.tif")
    pct_diff_path = os.path.join(diff_dir, f"{stem}_pct_diff.tif")

    with rasterio.open(diff_path, "w", **out_profile) as dst:
        dst.write(diff_grid, 1)

    with rasterio.open(pct_diff_path, "w", **out_profile) as dst:
        dst.write(pct_diff_grid, 1)

    return diff_path, pct_diff_path


def compare_rasters(
    test_path: str,
    ref_path: str,
    max_mae: float = 5e-4,
    max_bias: float = 1e-4,
    min_r: float = 0.99,
    diff_dir: str | None = None,
) -> dict[str, Any]:
    """Compare a newly generated GeoTIFF against a known reference GeoTIFF."""
    if not os.path.exists(test_path):
        return {"passed": False, "error": f"missing test output: {test_path}"}
    if not os.path.exists(ref_path):
        return {"passed": False, "error": f"missing reference output: {ref_path}"}

    with rasterio.open(test_path) as test_src, rasterio.open(ref_path) as ref_src:
        test_data = test_src.read(1)
        ref_data = ref_src.read(1)
        crs_match = test_src.crs == ref_src.crs
        transform_match = test_src.transform.almost_equals(ref_src.transform)
        profile = test_src.profile.copy()

    if test_data.shape != ref_data.shape:
        return {
            "passed": False,
            "error": f"shape mismatch: test={test_data.shape}, ref={ref_data.shape}",
        }

    a_all = test_data.astype(np.float64)
    b_all = ref_data.astype(np.float64)

    a_nan = ~np.isfinite(a_all)
    b_nan = ~np.isfinite(b_all)
    nan_disagreements = int(np.count_nonzero(a_nan != b_nan))
    valid = ~(a_nan | b_nan)

    if not np.any(valid):
        return {
            "passed": nan_disagreements == 0 and crs_match and transform_match,
            "valid_pixels": 0,
            "nan_disagreements": nan_disagreements,
            "crs_match": crs_match,
            "transform_match": transform_match,
        }

    a = a_all[valid]
    b = b_all[valid]
    diff = a - b
    abs_diff = np.abs(diff)

    exact_pct = float(100.0 * np.mean(a == b))
    mae = float(abs_diff.mean())
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs_diff = float(abs_diff.max())
    p99_abs_diff = float(np.percentile(abs_diff, 99))
    mean_test = float(a.mean())
    mean_ref = float(b.mean())
    bias = float(diff.mean())
    pct_bias = float(100.0 * bias / abs(mean_ref)) if mean_ref != 0 else float("nan")

    denom = np.abs(a) + np.abs(b)
    nonzero = denom > 0
    sym_pct_diff = (
        float(np.mean(200.0 * abs_diff[nonzero] / denom[nonzero]))
        if np.any(nonzero)
        else 0.0
    )

    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if (a.std() > 0 and b.std() > 0)
        else (1.0 if mae <= max_mae else float("nan"))
    )

    within_tol = bool(
        mae <= max_mae
        and abs(bias) <= max_bias
        and (np.isnan(corr) or corr >= min_r)
    )
    passed = bool(
        within_tol and nan_disagreements == 0 and crs_match and transform_match
    )

    diff_files = {}
    if diff_dir:
        stem = os.path.splitext(os.path.basename(test_path))[0]
        diff_path, pct_diff_path = _write_diff_rasters(
            test_data, ref_data, profile, diff_dir, stem
        )
        diff_files = {
            "diff_tif": diff_path,
            "pct_diff_tif": pct_diff_path,
        }

    return {
        "passed": passed,
        "within_tolerance": within_tol,
        "max_mae": max_mae,
        "max_bias": max_bias,
        "min_r": min_r,
        "crs_match": crs_match,
        "transform_match": transform_match,
        "valid_pixels": int(a.size),
        "nan_disagreements": nan_disagreements,
        "exact_match_pct": exact_pct,
        "mean_test": mean_test,
        "mean_ref": mean_ref,
        "bias": bias,
        "pct_bias": pct_bias,
        "mae": mae,
        "rmse": rmse,
        "max_abs_diff": max_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "sym_pct_diff": sym_pct_diff,
        "pearson_r": corr,
        **diff_files,
    }


def _execute_bundle(
    bundle_dir: str,
    output_root: str,
    julia_script: str | None = None,
    products: list[str] | None = None,
    skip_julia: bool = False,
    max_mae: float = 5e-4,
    max_bias: float = 1e-4,
    min_r: float = 0.99,
    write_diffs: bool = False,
    report_json: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in bundle: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_julia_paths = [
        os.path.join(script_dir, "L2T_STARS", "process_ECOSTRESS_data_fusion.jl"),
        os.path.join(script_dir, "process_ECOSTRESS_data_fusion.jl"),
    ]
    if manifest.get("runs"):
        candidate_julia_paths.append(manifest["runs"][0]["julia_script"])

    if julia_script:
        resolved_julia_script = os.path.abspath(os.path.expanduser(julia_script))
    else:
        resolved_julia_script = next(
            (p for p in candidate_julia_paths if os.path.exists(p)),
            candidate_julia_paths[0],
        )

    diff_dir = os.path.join(output_root, "diff_outputs") if write_diffs else None

    env = os.environ.copy()
    env.setdefault("JULIA_NUM_THREADS", "auto")

    all_passed = True
    run_reports: list[dict[str, Any]] = []

    for run_spec in manifest.get("runs", []):
        product = run_spec["product"]
        if products and product not in products:
            continue

        for rel_out in run_spec["isolated_test_outputs"].values():
            os.makedirs(os.path.dirname(os.path.join(output_root, rel_out)), exist_ok=True)

        elapsed_s = 0.0
        if not skip_julia:
            cmd = _build_julia_cmd(resolved_julia_script, bundle_dir, output_root, run_spec)
            print(f"\n=== Running Julia data fusion: {product} (tile={run_spec['tile']}) ===")
            print("Command:", " ".join(cmd))
            t0 = time.perf_counter()
            subprocess.run(cmd, env=env, check=True)
            elapsed_s = time.perf_counter() - t0
            print(f"Completed {product} in {elapsed_s:.2f}s")

        layer_results: dict[str, Any] = {}
        print(f"\n--- Comparison Summary: {product} (tile={run_spec['tile']}) ---")
        for key in OUTPUT_KEYS:
            test_path = os.path.join(output_root, run_spec["isolated_test_outputs"][key])
            ref_rel = run_spec.get("reference_outputs", {}).get(key)
            if not ref_rel:
                layer_results[key] = {"passed": False, "error": "no reference output in manifest"}
                all_passed = False
                print(f"  [{key}] FAIL: no reference output recorded in manifest.json")
                continue

            ref_path = os.path.join(bundle_dir, ref_rel)
            metrics = compare_rasters(
                test_path=test_path,
                ref_path=ref_path,
                max_mae=max_mae,
                max_bias=max_bias,
                min_r=min_r,
                diff_dir=diff_dir,
            )
            layer_results[key] = metrics

            if "error" in metrics:
                all_passed = False
                print(f"  [{key}] FAIL: {metrics['error']}")
            else:
                status = "PASS" if metrics["passed"] else "FAIL"
                if not metrics["passed"]:
                    all_passed = False
                print(
                    f"  [{key}] {status} | "
                    f"n={metrics['valid_pixels']} nan_diff={metrics['nan_disagreements']} "
                    f"exact={metrics['exact_match_pct']:.2f}% | "
                    f"bias={metrics['bias']:+.6e} ({metrics['pct_bias']:+.2f}%) | "
                    f"MAE={metrics['mae']:.6e} RMSE={metrics['rmse']:.6e} "
                    f"max|diff|={metrics['max_abs_diff']:.6e} r={metrics['pearson_r']:.6f}"
                )

        run_reports.append(
            {
                "product": product,
                "tile": run_spec["tile"],
                "elapsed_seconds": round(elapsed_s, 3),
                "layers": layer_results,
            }
        )

    summary = {
        "bundle_dir": bundle_dir,
        "output_dir": output_root,
        "julia_script": resolved_julia_script,
        "all_passed": all_passed,
        "runs": run_reports,
    }

    if report_json:
        report_path = os.path.abspath(os.path.expanduser(report_json))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved comparison report to: {report_path}")

    return all_passed, summary


def run_and_compare_bundle(
    bundle_dir: str = DEFAULT_BUNDLE_DIR,
    julia_script: str | None = None,
    products: list[str] | None = None,
    skip_julia: bool = False,
    max_mae: float = 5e-4,
    max_bias: float = 1e-4,
    min_r: float = 0.99,
    write_diffs: bool = False,
    report_json: str | None = None,
    output_dir: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Execute isolated Julia fusion tests from `bundle_dir` and compare against known outputs."""
    bundle_dir = os.path.abspath(os.path.expanduser(bundle_dir))

    if output_dir or skip_julia:
        resolved_out = os.path.abspath(os.path.expanduser(output_dir or bundle_dir))
        return _execute_bundle(
            bundle_dir=bundle_dir,
            output_root=resolved_out,
            julia_script=julia_script,
            products=products,
            skip_julia=skip_julia,
            max_mae=max_mae,
            max_bias=max_bias,
            min_r=min_r,
            write_diffs=write_diffs,
            report_json=report_json,
        )

    with tempfile.TemporaryDirectory(prefix="stars_fusion_test_") as tmp_out:
        return _execute_bundle(
            bundle_dir=bundle_dir,
            output_root=tmp_out,
            julia_script=julia_script,
            products=products,
            skip_julia=False,
            max_mae=max_mae,
            max_bias=max_bias,
            min_r=min_r,
            write_diffs=write_diffs,
            report_json=report_json,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated process_ECOSTRESS_data_fusion.jl tests and compare outputs against known reference outputs."
    )
    parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help="Path to the collected test bundle directory containing manifest.json (defaults to L2T_STARS/test_bundle).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to persist generated test_outputs/ and diff_outputs/ (defaults to an ephemeral temp dir).",
    )
    parser.add_argument(
        "--julia-script",
        default=None,
        help="Path to process_ECOSTRESS_data_fusion.jl (defaults to repo copy).",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=["NDVI", "albedo"],
        default=None,
        help="Optional subset of products to run/compare (default: all in manifest).",
    )
    parser.add_argument(
        "--skip-julia",
        action="store_true",
        help="Skip executing Julia and only compare existing files in --output-dir (or --bundle-dir) against reference_outputs/.",
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=5e-4,
        help="Maximum allowed Mean Absolute Error (default: 5e-4).",
    )
    parser.add_argument(
        "--max-bias",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute mean bias (default: 1e-4).",
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=0.99,
        help="Minimum allowed Pearson correlation coefficient (default: 0.99).",
    )
    parser.add_argument(
        "--write-diffs",
        action="store_true",
        help="Write signed difference and symmetric percent difference GeoTIFFs into <output-dir>/diff_outputs/.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a JSON summary report.",
    )
    args = parser.parse_args()

    all_passed, _ = run_and_compare_bundle(
        bundle_dir=args.bundle_dir,
        julia_script=args.julia_script,
        products=args.products,
        skip_julia=args.skip_julia,
        max_mae=args.max_mae,
        max_bias=args.max_bias,
        min_r=args.min_r,
        write_diffs=args.write_diffs,
        report_json=args.report_json,
        output_dir=args.output_dir,
    )

    print(f"\nOverall result: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
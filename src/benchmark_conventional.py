#!/usr/bin/env python3
"""Synchronized post-delay timing of all reviewer-comparison beamformers."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_beamforming import (
    FdmasFilterConfiguration,
    MvConfiguration,
    capon_minimum_variance,
    design_fdmas_fir,
    fdmas_analytic_image,
    method_complexities,
    receive_cf_das,
    signed_sqrt_pair_sum,
)
from method_names import ANGLE_NSI, CF_DAS, DAS, DMAS, METHODS, MV, RECEIVE_NSI
from nsi_core import nsi_envelope


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark six beamformers from a common delayed-channel boundary"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "conventional_timing",
    )
    parser.add_argument("--device", default=os.environ.get("NSI_CUDA_DEVICE", "0"))
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--elements", type=int, default=64)
    parser.add_argument("--angles", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--mv-chunk-pixels", type=int, default=32)
    parser.add_argument("--mv-temporal-half-window", type=int, default=0)
    parser.add_argument("--nsi-c", type=float, default=0.05)
    parser.add_argument("--center-frequency-mhz", type=float, default=5.208333)
    parser.add_argument("--sound-speed-m-s", type=float, default=1540.0)
    parser.add_argument("--axial-spacing-mm", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    integer_positive = (
        "nx",
        "nz",
        "elements",
        "angles",
        "warmups",
        "repetitions",
        "mv_chunk_pixels",
    )
    if any(getattr(args, name) < 1 for name in integer_positive):
        raise SystemExit("Grid sizes, dimensions, and timing counts must be positive.")
    if args.elements < 4:
        raise SystemExit("At least four receive elements are required.")
    if args.angles < 3 or args.angles % 2 == 0:
        raise SystemExit("--angles must be odd and at least three.")
    if args.mv_temporal_half_window < 0:
        raise SystemExit("MV temporal half-window cannot be negative.")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (
            args.nsi_c,
            args.center_frequency_mhz,
            args.sound_speed_m_s,
            args.axial_spacing_mm,
        )
    ):
        raise SystemExit("Physical and NSI parameters must be finite and positive.")


def version_or_none(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def execute_method(
    method: str,
    delayed_iq: Any,
    delayed_rf: Any,
    active_mask: Any,
    receive_sign: Any,
    angular_weights: Any,
    fdmas_coefficients: Any,
    mv_configuration: MvConfiguration,
    nsi_c: float,
    xp: Any,
) -> Any:
    """Run one method from the common already-delayed channel tensor."""

    if method == DAS:
        return xp.abs(xp.sum(delayed_iq, axis=(0, 3)))
    if method == CF_DAS:
        compounded = xp.zeros(delayed_iq.shape[1:3], dtype=delayed_iq.dtype)
        for angle_index in range(delayed_iq.shape[0]):
            compounded += receive_cf_das(
                delayed_iq[angle_index], active_mask, xp=xp
            )
        return xp.abs(compounded)
    if method == MV:
        compounded = xp.zeros(delayed_iq.shape[1:3], dtype=delayed_iq.dtype)
        for angle_index in range(delayed_iq.shape[0]):
            compounded += capon_minimum_variance(
                delayed_iq[angle_index],
                active_mask,
                grid_shape=delayed_iq.shape[1:3],
                configuration=mv_configuration,
                xp=xp,
            )
        return xp.abs(compounded)
    if method == DMAS:
        pair_sum = xp.zeros(delayed_rf.shape[1:3], dtype=xp.float32)
        for angle_index in range(delayed_rf.shape[0]):
            pair_sum += signed_sqrt_pair_sum(
                delayed_rf[angle_index], active_mask, xp=xp
            )
        return xp.abs(
            fdmas_analytic_image(
                pair_sum, fdmas_coefficients, depth_axis=1, xp=xp
            )
        )
    if method == RECEIVE_NSI:
        uniform = xp.sum(delayed_iq, axis=(0, 3))
        null = xp.sum(delayed_iq * receive_sign[None, ...], axis=(0, 3))
        return nsi_envelope(uniform, null, nsi_c, xp=xp)
    if method == ANGLE_NSI:
        per_angle = xp.sum(delayed_iq, axis=3)
        uniform = xp.sum(per_angle, axis=0)
        null = xp.sum(per_angle * angular_weights[:, None, None], axis=0)
        return nsi_envelope(uniform, null, nsi_c, xp=xp)
    raise ValueError(f"Unknown conventional timing method: {method}")


def summarize(method: str, elapsed_ms: list[float], args: argparse.Namespace) -> dict[str, Any]:
    values = np.asarray(elapsed_ms, dtype=float)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    complexity = method_complexities()[method]
    return {
        "method": method,
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "sample_sd_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "iqr_ms": float(q3 - q1),
        "n": int(values.size),
        "warmups": args.warmups,
        "angles": args.angles,
        "receive_elements": args.elements,
        "grid_nx": args.nx,
        "grid_nz": args.nz,
        "pixels": args.nx * args.nz,
        "scope": "post-delay beamformer kernel",
        "input_residency": "all delayed channel samples already on GPU",
        "output_scope": "one final envelope remains on GPU",
        "asymptotic_order_per_pixel_angle": complexity["order"],
        "complexity_note": complexity["note"],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.8, 3.8), dpi=180)
    positions = np.arange(len(rows))
    medians = [row["median_ms"] for row in rows]
    axis.bar(positions, medians, color=["#6a3d9a", "#2ca02c", "#ff7f0e", "#8c564b", "#d62728", "#1f77b4"])
    axis.set_xticks(positions, [row["method"] for row in rows], rotation=24, ha="right")
    axis.set_ylabel("Median kernel time [ms]")
    axis.set_yscale("log")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.nx = min(args.nx, 2)
        args.nz = min(args.nz, 24)
        args.elements = min(args.elements, 16)
        args.angles = min(args.angles, 5)
        if args.angles % 2 == 0:
            args.angles -= 1
        args.warmups = min(args.warmups, 1)
        args.repetitions = min(args.repetitions, 3)
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import cupy as cp
    except ImportError as error:
        raise SystemExit("The conventional timing benchmark requires CuPy.") from error
    cp.cuda.Device(0).use()

    rng = np.random.default_rng(args.seed)
    shape = (args.angles, args.nx, args.nz, args.elements)
    delayed_iq_host = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)
    # Add a coherent component so validation exercises finite, nonzero outputs.
    delayed_iq_host += np.complex64(0.25 + 0.1j)
    delayed_rf_host = delayed_iq_host.real.astype(np.float32, copy=True)
    active_mask_host = np.ones((args.nx, args.nz, args.elements), dtype=np.float32)
    receive_sign_host = np.ones_like(active_mask_host)
    receive_sign_host[..., : args.elements // 2] = -1.0
    angular_weights_host = np.sign(
        np.linspace(-1.0, 1.0, args.angles, dtype=np.float32)
    )
    delayed_iq = cp.asarray(delayed_iq_host)
    delayed_rf = cp.asarray(delayed_rf_host)
    active_mask = cp.asarray(active_mask_host)
    receive_sign = cp.asarray(receive_sign_host)
    angular_weights = cp.asarray(angular_weights_host)
    depth_sampling_hz = args.sound_speed_m_s / (2.0 * args.axial_spacing_mm * 1e-3)
    coefficients, filter_metadata = design_fdmas_fir(
        args.center_frequency_mhz * 1e6,
        depth_sampling_hz,
        FdmasFilterConfiguration(),
    )
    fdmas_coefficients = cp.asarray(coefficients)
    mv_configuration = MvConfiguration(
        temporal_half_window_samples=args.mv_temporal_half_window,
        chunk_pixels=args.mv_chunk_pixels,
    )

    def invoke(method: str) -> Any:
        return execute_method(
            method,
            delayed_iq,
            delayed_rf,
            active_mask,
            receive_sign,
            angular_weights,
            fdmas_coefficients,
            mv_configuration,
            args.nsi_c,
            cp,
        )

    rows = []
    raw_timings: dict[str, list[float]] = {}
    for method in METHODS:
        validation = invoke(method)
        cp.cuda.Stream.null.synchronize()
        validation_host = cp.asnumpy(validation)
        if validation_host.shape != (args.nx, args.nz) or not np.all(np.isfinite(validation_host)):
            raise RuntimeError(f"{method} failed pre-timing output validation.")
        for _ in range(args.warmups):
            invoke(method)
        cp.cuda.Stream.null.synchronize()
        elapsed_ms = []
        for _ in range(args.repetitions):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            invoke(method)
            end.record()
            end.synchronize()
            elapsed_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
        raw_timings[method] = elapsed_ms
        row = summarize(method, elapsed_ms, args)
        rows.append(row)
        print(f"{method:16s} median {row['median_ms']:.3f} ms")

    device_properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device_properties.get("name", "")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    publication_ready = bool(
        not args.quick
        and args.warmups >= 10
        and args.repetitions >= 50
        and {row["method"] for row in rows} == set(METHODS)
    )
    csv_path = output_dir / "conventional_timing_summary.csv"
    json_path = output_dir / "conventional_timing_summary.json"
    runs_path = output_dir / "conventional_timing_runs.csv"
    figure_path = output_dir / "conventional_timing_summary.png"
    write_csv(csv_path, rows)
    run_rows = [
        {"method": method, "repetition": index + 1, "elapsed_ms": elapsed}
        for method in METHODS
        for index, elapsed in enumerate(raw_timings[method])
    ]
    write_csv(runs_path, run_rows)
    plot_summary(figure_path, rows)
    summary = {
        "script": Path(__file__).name,
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "quick_engineering_run": bool(args.quick),
        "publication_ready": publication_ready,
        "requested_warmups": args.warmups,
        "requested_repetitions": args.repetitions,
        "benchmark_boundary": {
            "scope": "post-delay beamformer kernel",
            "common_input": "complex IQ and real RF delayed receive-channel tensors resident on GPU",
            "excluded": "raw-data H2D transfer, receive-delay calculation, interpolation, and final D2H transfer",
            "included": "method-specific reduction; DMAS FIR and Hilbert transform; MV covariance and solve",
        },
        "dimensions": {
            "nx": args.nx,
            "nz": args.nz,
            "pixels": args.nx * args.nz,
            "receive_elements": args.elements,
            "transmit_angles": args.angles,
        },
        "mv_configuration": mv_configuration.to_dict(),
        "fdmas_filter": filter_metadata,
        "rows": rows,
        "gpu": {
            "device": str(device_name),
            "cupy_version": version_or_none("cupy-cuda12x") or version_or_none("cupy"),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version_or_none("scipy"),
        },
        "artifacts": {
            "summary_csv": str(csv_path.resolve()),
            "runs_csv": str(runs_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
    }
    write_json(json_path, summary)
    print(f"Conventional timing summary: {json_path}")
    print(f"Publication-ready: {publication_ready}")


if __name__ == "__main__":
    main()

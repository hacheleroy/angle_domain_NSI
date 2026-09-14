#!/usr/bin/env python3
"""Experimental PSF and spatial-variability analysis for PICMUS.

The official ``resolution_distorsion`` RF acquisition contains five near-axis
point targets at different depths and two off-axis targets near 37.5 mm.  This
script reconstructs local maps and fine lateral/axial profiles with the same
delays and dynamic receive aperture for DAS, angular CF-DAS, Receive-NSI, and
Angle-NSI.  It also evaluates the NSI offset ``c`` without repeating
beamforming.  ``--metadata-only`` validates dataset access without a GPU.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsi_core import (
    angular_sign_weights,
    coherence_factor_from_moments,
    nsi_envelope,
)
from picmus_io import load_picmus_dataset, load_picmus_phantom, load_picmus_scan
from psf_metrics import amplitude_to_db, measure_connected_width, peak_to_background_db


METHODS = ("DAS", "Angular CF-DAS", "Receive-NSI", "Angle-NSI")
STYLES = {
    "DAS": ("#6a3d9a", "o", "-"),
    "Angular CF-DAS": ("#2ca02c", "P", ":"),
    "Receive-NSI": ("#d62728", "s", "--"),
    "Angle-NSI": ("#1f77b4", "^", "-."),
}


@dataclass(frozen=True)
class TargetGrid:
    points_m: np.ndarray
    map_x_mm: np.ndarray
    map_z_mm: np.ndarray
    lateral_x_mm: np.ndarray
    axial_z_mm: np.ndarray
    map_slice: slice
    lateral_slice: slice
    axial_slice: slice


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("Provide comma-separated positive c values.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("c values must not be duplicated.")
    return values


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get(
        "PICMUS_RESOLUTION_DIR",
        root / "data" / "PICMUS" / "resolution_distorsion",
    ))
    parser = argparse.ArgumentParser(description="PICMUS experimental PSF analysis")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=data_root / "resolution_distorsion_expe_dataset_rf.hdf5",
    )
    parser.add_argument(
        "--phantom",
        type=Path,
        default=data_root / "resolution_distorsion_expe_phantom.hdf5",
    )
    parser.add_argument(
        "--scan",
        type=Path,
        default=data_root / "resolution_distorsion_expe_scan.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "picmus_experimental_psf",
    )
    parser.add_argument("--device", default=os.environ.get("NSI_CUDA_DEVICE", "0"))
    parser.add_argument("--carrier-frequency-mhz", type=float, default=5.208333)
    parser.add_argument("--f-number", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument(
        "--c-values", type=parse_float_list, default=(0.02, 0.05, 0.1, 0.2)
    )
    parser.add_argument("--angle-count", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--map-half-width-mm", type=float, default=0.8)
    parser.add_argument("--map-spacing-mm", type=float, default=0.02)
    parser.add_argument("--profile-half-width-mm", type=float, default=1.0)
    parser.add_argument("--lateral-spacing-mm", type=float, default=0.002)
    parser.add_argument("--axial-spacing-mm", type=float, default=0.005)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate all HDF5 metadata and exit before importing CuPy.",
    )
    return parser.parse_args()


def centered_axis(center_mm: float, half_width_mm: float, spacing_mm: float) -> np.ndarray:
    count = int(round(2.0 * half_width_mm / spacing_mm)) + 1
    return center_mm + np.linspace(-half_width_mm, half_width_mm, count)


def build_target_grid(
    target_x_mm: float,
    target_z_mm: float,
    *,
    map_half_width_mm: float,
    map_spacing_mm: float,
    profile_half_width_mm: float,
    lateral_spacing_mm: float,
    axial_spacing_mm: float,
) -> TargetGrid:
    map_x = centered_axis(target_x_mm, map_half_width_mm, map_spacing_mm)
    map_z = centered_axis(target_z_mm, map_half_width_mm, map_spacing_mm)
    map_x_grid, map_z_grid = np.meshgrid(map_x, map_z, indexing="ij")
    map_points = np.column_stack(
        [map_x_grid.ravel(), np.zeros(map_x_grid.size), map_z_grid.ravel()]
    )
    lateral_x = centered_axis(
        target_x_mm, profile_half_width_mm, lateral_spacing_mm
    )
    lateral_points = np.column_stack(
        [lateral_x, np.zeros(lateral_x.size), np.full(lateral_x.size, target_z_mm)]
    )
    axial_z = centered_axis(target_z_mm, profile_half_width_mm, axial_spacing_mm)
    axial_points = np.column_stack(
        [np.full(axial_z.size, target_x_mm), np.zeros(axial_z.size), axial_z]
    )
    points_mm = np.concatenate([map_points, lateral_points, axial_points], axis=0)
    map_end = map_points.shape[0]
    lateral_end = map_end + lateral_points.shape[0]
    return TargetGrid(
        points_m=(points_mm * 1e-3).astype(np.float32),
        map_x_mm=map_x,
        map_z_mm=map_z,
        lateral_x_mm=lateral_x,
        axial_z_mm=axial_z,
        map_slice=slice(0, map_end),
        lateral_slice=slice(map_end, lateral_end),
        axial_slice=slice(lateral_end, points_mm.shape[0]),
    )


def choose_angle_indices(total: int, requested: int | None) -> np.ndarray:
    count = total if requested is None else requested
    if count < 3 or count > total or count % 2 == 0:
        raise ValueError("--angle-count must be odd, at least 3, and no larger than available.")
    start = (total - count) // 2
    return np.arange(start, start + count, dtype=int)


def dynamic_aperture_tables(
    points_m: np.ndarray,
    probe_geometry_m: np.ndarray,
    *,
    f_number: float,
    sound_speed_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return receive travel time, aperture mask, and split-aperture signs."""

    x_pixels = points_m[:, 0].astype(np.float64)
    z_pixels = points_m[:, 2].astype(np.float64)
    x_elements = probe_geometry_m[:, 0].astype(np.float64)
    z_elements = probe_geometry_m[:, 2].astype(np.float64)
    elements = x_elements.size
    pitch = float(np.median(np.diff(np.sort(x_elements))))
    if pitch <= 0.0:
        raise ValueError("Probe pitch could not be inferred from probe_geometry.")

    center_index = np.clip(
        np.round((x_pixels - x_elements.min()) / pitch), 0, elements - 1
    )
    full_width = z_pixels / (f_number * pitch)
    aperture_limit = 2.0 * np.minimum(center_index, (elements - 1) - center_index)
    aperture_size = np.minimum(
        np.maximum(2.0 * np.round(full_width / 2.0), 2.0), aperture_limit
    )
    half = aperture_size / 2.0
    left_half = center_index < elements / 2.0
    lower = np.where(left_half, center_index - half, center_index - half + 1.0)
    upper = np.where(left_half, center_index + half - 1.0, center_index + half)
    lower = np.clip(lower, 0, elements - 1)
    upper = np.clip(upper, 0, elements - 1)
    element_index = np.arange(elements, dtype=float)[None, :]
    aperture = (element_index >= lower[:, None]) & (element_index <= upper[:, None])
    midpoint = lower + half
    receive_sign = 2.0 * (element_index >= midpoint[:, None]) - 1.0
    receive_time = np.hypot(
        x_pixels[:, None] - x_elements[None, :],
        z_pixels[:, None] - z_elements[None, :],
    ) / sound_speed_m_s
    return (
        receive_time.astype(np.float32),
        aperture,
        receive_sign.astype(np.float32),
    )


def demodulate_rf(rf: np.ndarray, fs_hz: float, fc_hz: float, cp: Any) -> Any:
    samples = rf.shape[0]
    rf_gpu = cp.asarray(rf.real, dtype=cp.float32)
    analytic_mask = cp.zeros((samples, 1, 1), dtype=cp.float32)
    analytic_mask[0] = 1.0
    if samples % 2 == 0:
        analytic_mask[samples // 2] = 1.0
        analytic_mask[1 : samples // 2] = 2.0
    else:
        analytic_mask[1 : (samples + 1) // 2] = 2.0
    time_s = (cp.arange(samples, dtype=cp.float32) / fs_hz)[:, None, None]
    carrier = cp.exp(-2j * cp.pi * fc_hz * time_s).astype(cp.complex64)
    analytic = cp.fft.ifft(cp.fft.fft(rf_gpu, axis=0) * analytic_mask, axis=0)
    return (analytic * carrier).astype(cp.complex64)


def reconstruct_target(
    iq_gpu: Any,
    angles_rad: np.ndarray,
    angular_weights: np.ndarray,
    grid: TargetGrid,
    probe_geometry_m: np.ndarray,
    *,
    fs_hz: float,
    fc_hz: float,
    sound_speed_m_s: float,
    initial_time_s: float,
    f_number: float,
    c_values: tuple[float, ...],
    primary_c: float,
    cp: Any,
) -> tuple[dict[str, np.ndarray], dict[float, dict[str, np.ndarray]]]:
    receive_time, aperture, receive_sign = dynamic_aperture_tables(
        grid.points_m,
        probe_geometry_m,
        f_number=f_number,
        sound_speed_m_s=sound_speed_m_s,
    )
    receive_time_gpu = cp.asarray(receive_time)
    aperture_gpu = cp.asarray(aperture, dtype=cp.float32)
    receive_sign_gpu = cp.asarray(receive_sign, dtype=cp.float32)
    column_gpu = cp.arange(probe_geometry_m.shape[0], dtype=cp.int32)[None, :]
    weights_gpu = cp.asarray(angular_weights, dtype=cp.float32)

    point_count = grid.points_m.shape[0]
    uniform = cp.zeros(point_count, dtype=cp.complex64)
    angle_null = cp.zeros(point_count, dtype=cp.complex64)
    receive_null = cp.zeros(point_count, dtype=cp.complex64)
    angle_power = cp.zeros(point_count, dtype=cp.float32)
    omega = 2.0 * np.pi * fc_hz

    for angle_index, angle_rad in enumerate(angles_rad):
        direction = np.asarray(
            [np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=np.float32
        )
        transmit_time = (
            grid.points_m @ direction / sound_speed_m_s + initial_time_s
        )
        tau = cp.asarray(transmit_time)[:, None] + receive_time_gpu
        sample_position = (tau - initial_time_s) * fs_hz
        floor_index = cp.floor(sample_position)
        fraction = (sample_position - floor_index).astype(cp.float32)
        valid_time = (sample_position >= 0.0) & (
            sample_position <= iq_gpu.shape[0] - 2
        )
        mask = aperture_gpu * valid_time.astype(cp.float32)
        safe_index = cp.clip(floor_index, 0, iq_gpu.shape[0] - 2).astype(cp.int32)
        iq_angle = iq_gpu[:, :, angle_index]
        lower = iq_angle[safe_index, column_gpu]
        upper = iq_angle[safe_index + 1, column_gpu]
        interpolated = lower * (1.0 - fraction) + upper * fraction
        base = interpolated * cp.exp(1j * omega * tau).astype(cp.complex64) * mask
        angle_image = cp.sum(base, axis=1)
        uniform += angle_image
        angle_null += weights_gpu[angle_index] * angle_image
        receive_null += cp.sum(base * receive_sign_gpu, axis=1)
        angle_power += cp.abs(angle_image) ** 2

    cf = coherence_factor_from_moments(
        uniform, angle_power, len(angles_rad), xp=cp
    )
    primary_gpu = {
        "DAS": cp.abs(uniform),
        "Angular CF-DAS": cf * cp.abs(uniform),
        "Receive-NSI": nsi_envelope(uniform, receive_null, primary_c, xp=cp),
        "Angle-NSI": nsi_envelope(uniform, angle_null, primary_c, xp=cp),
    }
    sensitivity_gpu = {
        value: {
            "Receive-NSI": nsi_envelope(uniform, receive_null, value, xp=cp),
            "Angle-NSI": nsi_envelope(uniform, angle_null, value, xp=cp),
        }
        for value in c_values
    }
    cp.cuda.Stream.null.synchronize()
    primary = {name: cp.asnumpy(value) for name, value in primary_gpu.items()}
    sensitivity = {
        c_value: {name: cp.asnumpy(value) for name, value in methods.items()}
        for c_value, methods in sensitivity_gpu.items()
    }
    return primary, sensitivity


def safe_width(axis: np.ndarray, profile: np.ndarray, expected: float) -> dict[str, Any]:
    try:
        return asdict(measure_connected_width(
            axis,
            profile,
            expected_peak_mm=expected,
            search_radius_mm=0.4,
        ))
    except ValueError as error:
        return {"width_mm": None, "error": str(error)}


def metrics_for_method(
    values: np.ndarray,
    grid: TargetGrid,
    target_x_mm: float,
    target_z_mm: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    map_image = values[grid.map_slice].reshape(
        grid.map_x_mm.size, grid.map_z_mm.size
    )
    lateral = values[grid.lateral_slice]
    axial = values[grid.axial_slice]
    peak_flat = int(np.argmax(map_image))
    peak_index = np.unravel_index(peak_flat, map_image.shape)
    peak_x = float(grid.map_x_mm[peak_index[0]])
    peak_z = float(grid.map_z_mm[peak_index[1]])
    lateral_width = safe_width(grid.lateral_x_mm, lateral, target_x_mm)
    axial_width = safe_width(grid.axial_z_mm, axial, target_z_mm)
    metrics = {
        "peak_x_mm": peak_x,
        "peak_z_mm": peak_z,
        "lateral_bias_mm": peak_x - target_x_mm,
        "axial_bias_mm": peak_z - target_z_mm,
        "localization_error_mm": float(
            np.hypot(peak_x - target_x_mm, peak_z - target_z_mm)
        ),
        "lateral_width_mm": lateral_width.get("width_mm"),
        "axial_width_mm": axial_width.get("width_mm"),
        "peak_to_median_background_db": peak_to_background_db(
            map_image,
            grid.map_x_mm,
            grid.map_z_mm,
            target_x_mm=target_x_mm,
            target_z_mm=target_z_mm,
        ),
        "lateral_width_audit": lateral_width,
        "axial_width_audit": axial_width,
    }
    arrays = {"map": map_image, "lateral": lateral, "axial": axial}
    return metrics, arrays


def finite_summary(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "sample_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def belongs_to_summary_group(row: dict[str, Any], group: str) -> bool:
    """Return whether a metric row belongs to an overlapping PSF summary."""

    if group == "all":
        return True
    if group == "on-axis":
        return row["position_group"] == "on-axis"
    if group == "37.5-mm depth":
        return abs(float(row["target_z_mm"]) - 37.55) <= 0.25
    raise ValueError(f"Unknown experimental-PSF summary group: {group}")


def plot_primary(path: Path, rows: list[dict[str, Any]], representative: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 7.0), dpi=180)
    for axis, method in zip(axes[0], METHODS):
        image = representative["arrays"][method]["map"]
        db = amplitude_to_db(image)
        im = axis.imshow(
            db.T,
            extent=[
                representative["grid"].map_x_mm[0],
                representative["grid"].map_x_mm[-1],
                representative["grid"].map_z_mm[-1],
                representative["grid"].map_z_mm[0],
            ],
            cmap="gray",
            vmin=-50.0,
            vmax=0.0,
            aspect="equal",
        )
        axis.plot(
            representative["target_x_mm"], representative["target_z_mm"],
            marker="+", color="orange", markersize=6, markeredgewidth=1.0,
        )
        axis.set_title(method, fontsize=9)
        axis.set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("z [mm]")
    figure.colorbar(im, ax=axes[0].tolist(), label="Normalized amplitude [dB]", shrink=0.75)

    on_axis = [row for row in rows if row["position_group"] == "on-axis"]
    same_depth = [
        row for row in rows if abs(row["target_z_mm"] - 37.55) <= 0.25
    ]
    for method in METHODS:
        color, marker, linestyle = STYLES[method]
        depth_rows = sorted(
            (row for row in on_axis if row["method"] == method),
            key=lambda row: row["target_z_mm"],
        )
        lateral_rows = sorted(
            (row for row in same_depth if row["method"] == method),
            key=lambda row: row["target_x_mm"],
        )
        axes[1, 0].plot(
            [row["target_z_mm"] for row in depth_rows],
            [row["lateral_width_mm"] for row in depth_rows],
            color=color, marker=marker, linestyle=linestyle, label=method,
        )
        axes[1, 1].plot(
            [row["target_x_mm"] for row in lateral_rows],
            [row["lateral_width_mm"] for row in lateral_rows],
            color=color, marker=marker, linestyle=linestyle,
        )
        axes[1, 2].plot(
            [row["target_z_mm"] for row in depth_rows],
            [row["peak_to_median_background_db"] for row in depth_rows],
            color=color, marker=marker, linestyle=linestyle,
        )
        method_rows = [row for row in rows if row["method"] == method]
        axes[1, 3].plot(
            [row["target_id"] for row in method_rows],
            [row["localization_error_mm"] for row in method_rows],
            color=color, marker=marker, linestyle=linestyle,
        )
    axes[1, 0].set(xlabel="Depth [mm]", ylabel="Lateral -6 dB width [mm]")
    axes[1, 1].set(xlabel="Lateral target position [mm]", ylabel="Lateral -6 dB width [mm]")
    axes[1, 2].set(xlabel="Depth [mm]", ylabel="Peak/background [dB]")
    axes[1, 3].set(xlabel="Target ID", ylabel="Localization error [mm]")
    for axis in axes[1]:
        axis.grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_c_sensitivity(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), dpi=180)
    for method in ("Receive-NSI", "Angle-NSI"):
        color, marker, linestyle = STYLES[method]
        c_values = sorted({row["c"] for row in rows})
        for axis, metric, label in (
            (axes[0], "lateral_width_mm", "Mean on-axis -6 dB width [mm]"),
            (axes[1], "peak_to_median_background_db", "Mean on-axis peak/background [dB]"),
        ):
            means = []
            for c_value in c_values:
                selected = [
                    row[metric] for row in rows
                    if row["method"] == method
                    and row["position_group"] == "on-axis"
                    and row["c"] == c_value
                    and row[metric] is not None
                ]
                means.append(float(np.mean(selected)) if selected else np.nan)
            axis.plot(
                c_values, means, color=color, marker=marker,
                linestyle=linestyle, label=method,
            )
            axis.set(xlabel="NSI offset c", ylabel=label)
            axis.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def version_or_none(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cuda_device_name(properties: dict[str, Any]) -> str:
    """Return a JSON-safe CUDA device name across CuPy runtime versions."""

    value = properties.get("name", "")
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def main() -> None:
    args = parse_args()
    if args.c <= 0.0 or args.f_number <= 0.0 or args.carrier_frequency_mhz <= 0.0:
        raise SystemExit("c, f-number, and carrier frequency must be positive.")
    c_values = tuple(sorted({*args.c_values, args.c}))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_picmus_dataset(args.dataset)
    phantom = load_picmus_phantom(args.phantom)
    scan_x, scan_z = load_picmus_scan(args.scan)
    angle_indices = choose_angle_indices(len(dataset.angles_rad), args.angle_count)
    selected_angles = dataset.angles_rad[angle_indices]
    selected_data = dataset.data[:, :, angle_indices]
    weights, weight_diagnostics = angular_sign_weights(np.rad2deg(selected_angles))
    positions = phantom.positions_m
    if args.max_targets is not None:
        if args.max_targets < 1:
            raise SystemExit("--max-targets must be positive.")
        positions = positions[: args.max_targets]

    metadata = {
        "script": Path(__file__).name,
        "schema_version": 1,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": str(args.dataset.expanduser().resolve()),
        "phantom_file": str(args.phantom.expanduser().resolve()),
        "scan_file": str(args.scan.expanduser().resolve()),
        "data_shape_samples_elements_angles": list(dataset.data.shape),
        "selected_angle_indices": angle_indices.tolist(),
        "selected_angles_deg": np.rad2deg(selected_angles).tolist(),
        "angular_weight_diagnostics": weight_diagnostics.to_dict(),
        "target_positions_mm": (positions * 1e3).tolist(),
        "scan_limits_mm": {
            "x": [float(scan_x.min() * 1e3), float(scan_x.max() * 1e3)],
            "z": [float(scan_z.min() * 1e3), float(scan_z.max() * 1e3)],
        },
        "acquisition": {
            "sampling_frequency_hz": dataset.sampling_frequency_hz,
            "sound_speed_m_s": dataset.sound_speed_m_s,
            "initial_time_s": dataset.initial_time_s,
            "probe_elements": int(dataset.probe_geometry_m.shape[0]),
            "probe_pitch_m": float(np.median(np.diff(np.sort(dataset.probe_geometry_m[:, 0])))),
            "file_modulation_frequency_hz": dataset.modulation_frequency_hz,
            "beamforming_carrier_frequency_hz": args.carrier_frequency_mhz * 1e6,
        },
        "reconstruction": {
            "f_number": args.f_number,
            "primary_c": args.c,
            "c_values": list(c_values),
            "map_half_width_mm": args.map_half_width_mm,
            "map_spacing_mm": args.map_spacing_mm,
            "profile_half_width_mm": args.profile_half_width_mm,
            "lateral_spacing_mm": args.lateral_spacing_mm,
            "axial_spacing_mm": args.axial_spacing_mm,
            "fine_profiles_beamformed_directly": True,
            "interpolation": "linear between adjacent complex-IQ samples",
            "cf_definition": "|sum_k B_k|^2/(K sum_k |B_k|^2)",
        },
    }
    if args.metadata_only:
        path = output_dir / "picmus_experimental_psf_metadata_check.json"
        metadata["metadata_only"] = True
        with path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Validated PICMUS resolution dataset and {len(positions)} targets: {path}")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import cupy as cp
    except ImportError as error:
        raise SystemExit(
            "Full experimental PSF reconstruction requires CuPy and an NVIDIA GPU; "
            "use --metadata-only for a CPU dataset check."
        ) from error
    cp.cuda.Device(0).use()
    start = time.perf_counter()
    iq_gpu = demodulate_rf(
        selected_data,
        dataset.sampling_frequency_hz,
        args.carrier_frequency_mhz * 1e6,
        cp,
    )

    metric_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    target_audits: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    profile_archive: dict[str, np.ndarray] = {}
    for target_zero_index, position_m in enumerate(positions):
        target_id = target_zero_index + 1
        target_x_mm = float(position_m[0] * 1e3)
        target_z_mm = float(position_m[2] * 1e3)
        position_group = (
            "on-axis" if abs(target_x_mm) < 2.0 else "37.5-mm depth"
        )
        grid = build_target_grid(
            target_x_mm,
            target_z_mm,
            map_half_width_mm=args.map_half_width_mm,
            map_spacing_mm=args.map_spacing_mm,
            profile_half_width_mm=args.profile_half_width_mm,
            lateral_spacing_mm=args.lateral_spacing_mm,
            axial_spacing_mm=args.axial_spacing_mm,
        )
        print(
            f"Reconstructing target {target_id}/{len(positions)} at "
            f"({target_x_mm:.3f}, {target_z_mm:.3f}) mm on {len(grid.points_m)} points"
        )
        primary, sensitivity = reconstruct_target(
            iq_gpu,
            selected_angles,
            weights,
            grid,
            dataset.probe_geometry_m,
            fs_hz=dataset.sampling_frequency_hz,
            fc_hz=args.carrier_frequency_mhz * 1e6,
            sound_speed_m_s=dataset.sound_speed_m_s,
            initial_time_s=dataset.initial_time_s,
            f_number=args.f_number,
            c_values=c_values,
            primary_c=args.c,
            cp=cp,
        )
        target_method_audits = {}
        primary_arrays = {}
        for method in METHODS:
            metrics, arrays = metrics_for_method(
                primary[method], grid, target_x_mm, target_z_mm
            )
            row = {
                "target_id": target_id,
                "target_x_mm": target_x_mm,
                "target_z_mm": target_z_mm,
                "position_group": position_group,
                "method": method,
                "c": args.c if "NSI" in method else None,
                **{key: value for key, value in metrics.items() if not key.endswith("_audit")},
            }
            metric_rows.append(row)
            target_method_audits[method] = metrics
            primary_arrays[method] = arrays
            key = f"target_{target_id}_{method.lower().replace(' ', '_').replace('-', '_')}"
            profile_archive[f"{key}_lateral"] = arrays["lateral"]
            profile_archive[f"{key}_axial"] = arrays["axial"]

        for c_value, methods in sensitivity.items():
            for method, values in methods.items():
                metrics, _ = metrics_for_method(
                    values, grid, target_x_mm, target_z_mm
                )
                sensitivity_rows.append(
                    {
                        "target_id": target_id,
                        "target_x_mm": target_x_mm,
                        "target_z_mm": target_z_mm,
                        "position_group": position_group,
                        "method": method,
                        "c": c_value,
                        "lateral_width_mm": metrics["lateral_width_mm"],
                        "axial_width_mm": metrics["axial_width_mm"],
                        "localization_error_mm": metrics["localization_error_mm"],
                        "peak_to_median_background_db": metrics[
                            "peak_to_median_background_db"
                        ],
                    }
                )
        target_audits.append(
            {
                "target_id": target_id,
                "target_position_mm": [target_x_mm, 0.0, target_z_mm],
                "position_group": position_group,
                "method_metrics": target_method_audits,
            }
        )
        if representative is None or (
            position_group == "on-axis"
            and abs(target_z_mm - 37.6) < abs(representative["target_z_mm"] - 37.6)
        ):
            representative = {
                "target_x_mm": target_x_mm,
                "target_z_mm": target_z_mm,
                "grid": grid,
                "arrays": primary_arrays,
            }

    assert representative is not None
    summaries = {}
    for group in ("all", "on-axis", "37.5-mm depth"):
        summaries[group] = {}
        for method in METHODS:
            selected = [
                row for row in metric_rows
                if row["method"] == method
                and belongs_to_summary_group(row, group)
            ]
            summaries[group][method] = {
                "lateral_width_mm": finite_summary(
                    [row["lateral_width_mm"] for row in selected]
                ),
                "axial_width_mm": finite_summary(
                    [row["axial_width_mm"] for row in selected]
                ),
                "peak_to_median_background_db": finite_summary(
                    [row["peak_to_median_background_db"] for row in selected]
                ),
                "localization_error_mm": finite_summary(
                    [row["localization_error_mm"] for row in selected]
                ),
            }

    metrics_path = output_dir / "picmus_experimental_psf_metrics.csv"
    sensitivity_path = output_dir / "picmus_experimental_psf_c_sensitivity.csv"
    json_path = output_dir / "picmus_experimental_psf_summary.json"
    profiles_path = output_dir / "picmus_experimental_psf_profiles.npz"
    figure_path = output_dir / "picmus_experimental_psf.png"
    c_figure_path = output_dir / "picmus_experimental_psf_c_sensitivity.png"
    write_csv(metrics_path, metric_rows)
    write_csv(sensitivity_path, sensitivity_rows)
    np.savez_compressed(
        profiles_path,
        **profile_archive,
        lateral_axes_mm=np.stack([
            build_target_grid(
                float(position[0] * 1e3), float(position[2] * 1e3),
                map_half_width_mm=args.map_half_width_mm,
                map_spacing_mm=args.map_spacing_mm,
                profile_half_width_mm=args.profile_half_width_mm,
                lateral_spacing_mm=args.lateral_spacing_mm,
                axial_spacing_mm=args.axial_spacing_mm,
            ).lateral_x_mm
            for position in positions
        ]),
        axial_axes_mm=np.stack([
            build_target_grid(
                float(position[0] * 1e3), float(position[2] * 1e3),
                map_half_width_mm=args.map_half_width_mm,
                map_spacing_mm=args.map_spacing_mm,
                profile_half_width_mm=args.profile_half_width_mm,
                lateral_spacing_mm=args.lateral_spacing_mm,
                axial_spacing_mm=args.axial_spacing_mm,
            ).axial_z_mm
            for position in positions
        ]),
    )
    plot_primary(figure_path, metric_rows, representative)
    plot_c_sensitivity(c_figure_path, sensitivity_rows)

    metadata.update(
        {
            "metadata_only": False,
            "run_completed_utc": datetime.now(timezone.utc).isoformat(),
            "execution_seconds": time.perf_counter() - start,
            "hardware": {
                "gpu": cuda_device_name(cp.cuda.runtime.getDeviceProperties(0)),
                "platform": platform.platform(),
            },
            "software": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "cupy": cp.__version__,
                "h5py": version_or_none("h5py"),
            },
            "target_audits": target_audits,
            "summaries": summaries,
            "output_files": {
                "metrics_csv": metrics_path.name,
                "c_sensitivity_csv": sensitivity_path.name,
                "profiles_npz": profiles_path.name,
                "main_figure": figure_path.name,
                "c_sensitivity_figure": c_figure_path.name,
            },
        }
    )
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for path in (
        metrics_path,
        sensitivity_path,
        json_path,
        profiles_path,
        figure_path,
        c_figure_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Expected experimental-PSF output was not written: {path}")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

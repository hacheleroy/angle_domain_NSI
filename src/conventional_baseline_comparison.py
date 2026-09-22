#!/usr/bin/env python3
"""Reviewer-facing comparison with CF-DAS, MV, and filtered DMAS.

This script deliberately leaves the established four-method analyses intact.
It reconstructs one representative experimental PICMUS point target and both
PICMUS carotid views with a common delay/interpolation implementation for:

* DAS;
* conventional receive-aperture CF-DAS;
* receive-domain Capon minimum variance (MV);
* receive-domain delay-multiply-and-sum (DMAS), with the Matrone filter;
* Receive-NSI; and
* Angle-NSI.

MV and DMAS use the fixed literature/USTB conventions declared in the JSON
output.  DMAS is evaluated on a fine axial grid that supports the band near
2*f0 and is then sampled onto the common display/metric grid.  This avoids the
silent aliasing that would result from filtering the ordinary B-mode grid.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from adaptive_beamforming import (
    FdmasFilterConfiguration,
    MvConfiguration,
    USTB_REFERENCE_COMMIT,
    capon_minimum_variance,
    design_fdmas_fir,
    fdmas_analytic_image,
    receive_cf_das,
    signed_sqrt_pair_sum,
    temporal_half_window_from_wavelengths,
)
from method_names import ANGLE_NSI, CF_DAS, DAS, DMAS, METHODS, MV, RECEIVE_NSI
from nsi_core import angular_sign_weights, nsi_envelope
from picmus_experimental_psf import (
    TargetGrid,
    build_target_grid,
    choose_angle_indices,
    demodulate_rf,
    dynamic_aperture_tables,
    metrics_for_method,
)
from picmus_io import PicmusDataset, load_picmus_dataset, load_picmus_phantom
from psf_metrics import amplitude_to_db


IQ_METHODS = tuple(method for method in METHODS if method != DMAS)
FDMAS_X_CHUNK_LINES = 32
COLORS = {
    DAS: "#6a3d9a",
    CF_DAS: "#2ca02c",
    MV: "#ff7f0e",
    DMAS: "#8c564b",
    RECEIVE_NSI: "#d62728",
    ANGLE_NSI: "#1f77b4",
}
MARKERS = {
    DAS: "o",
    CF_DAS: "P",
    MV: "D",
    DMAS: "X",
    RECEIVE_NSI: "s",
    ANGLE_NSI: "^",
}
ROI_BY_VIEW = {
    "CL": {
        "label": "Longitudinal",
        "signal_center_cm": (-0.70, 1.54),
        "signal_radius_cm": 0.30,
        "background_center_cm": (0.20, 2.70),
        "background_radius_cm": 0.30,
    },
    "CC": {
        "label": "Cross-section",
        "signal_center_cm": (-0.13, 1.77),
        "signal_radius_cm": 0.30,
        "background_center_cm": (-0.10, 2.70),
        "background_radius_cm": 0.30,
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "PICMUS"
    parser = argparse.ArgumentParser(
        description="PICMUS comparison with conventional CF-DAS, MV, and DMAS"
    )
    parser.add_argument(
        "--resolution-dataset",
        type=Path,
        default=data / "resolution_distorsion" / "resolution_distorsion_expe_dataset_rf.hdf5",
    )
    parser.add_argument(
        "--resolution-phantom",
        type=Path,
        default=data / "resolution_distorsion" / "resolution_distorsion_expe_phantom.hdf5",
    )
    parser.add_argument(
        "--carotid-long",
        type=Path,
        default=data / "in_vivo" / "carotid_long" / "carotid_long_expe_dataset_rf.hdf5",
    )
    parser.add_argument(
        "--carotid-cross",
        type=Path,
        default=data / "in_vivo" / "carotid_cross" / "carotid_cross_expe_dataset_rf.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "conventional_baselines",
    )
    parser.add_argument("--device", default=os.environ.get("NSI_CUDA_DEVICE", "0"))
    parser.add_argument("--carrier-frequency-mhz", type=float, default=5.208333)
    parser.add_argument("--f-number", type=float, default=1.0)
    parser.add_argument("--nsi-c", type=float, default=0.05)
    parser.add_argument("--angle-count", type=int, default=None)
    parser.add_argument("--mv-temporal-wavelengths", type=float, default=1.5)
    parser.add_argument("--mv-chunk-pixels", type=int, default=32)
    parser.add_argument("--psf-map-half-width-mm", type=float, default=1.5)
    parser.add_argument("--psf-map-spacing-mm", type=float, default=0.02)
    parser.add_argument("--psf-profile-half-width-mm", type=float, default=1.2)
    parser.add_argument("--psf-lateral-spacing-mm", type=float, default=0.005)
    parser.add_argument("--psf-axial-spacing-mm", type=float, default=0.02)
    parser.add_argument("--carotid-x-spacing-mm", type=float, default=0.20)
    parser.add_argument("--carotid-z-spacing-mm", type=float, default=0.10)
    parser.add_argument("--fdmas-z-spacing-mm", type=float, default=0.02)
    parser.add_argument("--gcnr-bins", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Ignore valid case caches.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Five-angle engineering check; outputs are not publication-ready.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate all inputs and fixed parameters without importing CuPy.",
    )
    return parser.parse_args()


def require_positive(args: argparse.Namespace) -> None:
    names = (
        "carrier_frequency_mhz",
        "f_number",
        "nsi_c",
        "mv_temporal_wavelengths",
        "psf_map_half_width_mm",
        "psf_map_spacing_mm",
        "psf_profile_half_width_mm",
        "psf_lateral_spacing_mm",
        "psf_axial_spacing_mm",
        "carotid_x_spacing_mm",
        "carotid_z_spacing_mm",
        "fdmas_z_spacing_mm",
    )
    if any(not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0 for name in names):
        raise SystemExit("All physical, grid, and method parameters must be positive.")
    if args.mv_chunk_pixels < 1 or args.gcnr_bins < 2:
        raise SystemExit("MV chunk size must be positive and gCNR bins must be >= 2.")


def version_or_none(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def signature(value: dict[str, Any]) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def method_key(method: str) -> str:
    return method.lower().replace("-", "_").replace(" ", "_")


LEGACY_CACHE_NAMES = {
    CF_DAS: "Receive CF-DAS",
    DMAS: "F-DMAS",
}


def load_image_cache(
    cache_path: Path,
    metadata_path: Path,
    expected_signature: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != expected_signature:
            return None
        with np.load(cache_path, allow_pickle=False) as archive:
            images = {}
            for method in METHODS:
                canonical_key = f"image_{method_key(method)}"
                legacy_key = f"image_{method_key(LEGACY_CACHE_NAMES.get(method, method))}"
                key = canonical_key if canonical_key in archive.files else legacy_key
                images[method] = archive[key].copy()
            axes = {
                name[5:]: archive[name].copy()
                for name in archive.files
                if name.startswith("axis_")
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if any(not np.all(np.isfinite(image)) for image in images.values()):
        return None
    return images, axes


def save_image_cache(
    cache_path: Path,
    metadata_path: Path,
    images: dict[str, np.ndarray],
    axes: dict[str, np.ndarray],
    expected_signature: str,
) -> None:
    payload = {f"image_{method_key(method)}": images[method] for method in METHODS}
    payload.update({f"axis_{name}": value for name, value in axes.items()})
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(cache_path)
    write_json(metadata_path, {"signature": expected_signature, "methods": METHODS})


def usable_positive_image(values: np.ndarray | None) -> bool:
    """Return whether an envelope image is finite and contains signal."""

    if values is None:
        return False
    array = np.asarray(values)
    return bool(
        array.size
        and np.all(np.isfinite(array))
        and float(np.max(array)) > 0.0
    )


def usable_complete_image(values: np.ndarray | None) -> bool:
    """Return whether every lateral line of a 2-D envelope contains signal.

    DMAS is reconstructed in independent lateral batches.  A global peak
    check is insufficient because a failed batch can leave an otherwise
    finite image with one or more all-zero lateral lines.
    """

    if not usable_positive_image(values):
        return False
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0:
        return False
    line_peaks = np.max(np.abs(array), axis=1)
    return bool(np.all(np.isfinite(line_peaks)) and np.all(line_peaks > 0.0))


def prepare_fdmas_gpu(cp: Any) -> None:
    """Start filtered DMAS after releasing the preceding IQ-method arrays."""

    cp.cuda.Stream.null.synchronize()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def regular_axis(lower: float, upper: float, spacing: float) -> np.ndarray:
    count = int(np.ceil((upper - lower) / spacing)) + 1
    return np.linspace(lower, upper, count, dtype=np.float32)


def regular_points(x_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    x_grid, z_grid = np.meshgrid(x_m, z_m, indexing="ij")
    return np.column_stack(
        [x_grid.ravel(), np.zeros(x_grid.size), z_grid.ravel()]
    ).astype(np.float32)


def focused_samples(
    channel_gpu: Any,
    angle_index: int,
    angle_rad: float,
    points_m: np.ndarray,
    receive_time_gpu: Any,
    aperture_gpu: Any,
    *,
    sampling_frequency_hz: float,
    sound_speed_m_s: float,
    initial_time_s: float,
    carrier_frequency_hz: float | None,
    cp: Any,
) -> tuple[Any, Any]:
    direction = np.asarray(
        [np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=np.float32
    )
    transmit_time = points_m @ direction / sound_speed_m_s + initial_time_s
    tau = cp.asarray(transmit_time, dtype=cp.float32)[:, None] + receive_time_gpu
    sample_position = (tau - initial_time_s) * sampling_frequency_hz
    floor_index = cp.floor(sample_position)
    fraction = (sample_position - floor_index).astype(cp.float32)
    valid = (sample_position >= 0.0) & (sample_position <= channel_gpu.shape[0] - 2)
    mask = aperture_gpu * valid.astype(cp.float32)
    safe_index = cp.clip(floor_index, 0, channel_gpu.shape[0] - 2).astype(cp.int32)
    element_index = cp.arange(channel_gpu.shape[1], dtype=cp.int32)[None, :]
    channel_angle = channel_gpu[:, :, angle_index]
    lower = channel_angle[safe_index, element_index]
    upper = channel_angle[safe_index + 1, element_index]
    values = lower * (1.0 - fraction) + upper * fraction
    if carrier_frequency_hz is not None:
        omega = 2.0 * np.pi * carrier_frequency_hz
        values = values * cp.exp(1j * omega * tau).astype(cp.complex64)
    return values * mask, mask


def reconstruct_iq_methods(
    dataset: PicmusDataset,
    angle_indices: np.ndarray,
    points_m: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    carrier_frequency_hz: float,
    f_number: float,
    nsi_c: float,
    mv_configuration: MvConfiguration,
    cp: Any,
    label: str,
    compute_mv: bool = True,
    focus_chunk_pixels: int | None = None,
) -> dict[str, np.ndarray]:
    if focus_chunk_pixels is not None and focus_chunk_pixels <= 0:
        raise ValueError("focus_chunk_pixels must be positive when provided.")
    if (
        focus_chunk_pixels is not None
        and mv_configuration.temporal_half_window_samples
    ):
        raise ValueError(
            "Chunked focusing is not compatible with MV temporal averaging."
        )
    selected_data = dataset.data[:, :, angle_indices]
    selected_angles = dataset.angles_rad[angle_indices]
    angular_weights, _ = angular_sign_weights(np.rad2deg(selected_angles))
    iq_gpu = demodulate_rf(
        selected_data, dataset.sampling_frequency_hz, carrier_frequency_hz, cp
    )
    receive_time, aperture, receive_sign = dynamic_aperture_tables(
        points_m,
        dataset.probe_geometry_m,
        f_number=f_number,
        sound_speed_m_s=dataset.sound_speed_m_s,
    )
    angle_weights_gpu = cp.asarray(angular_weights, dtype=cp.float32)
    count = points_m.shape[0]
    chunk_pixels = count if focus_chunk_pixels is None else focus_chunk_pixels
    method_order = [DAS, CF_DAS, RECEIVE_NSI, ANGLE_NSI]
    if compute_mv:
        method_order.append(MV)
    output = {
        method: np.empty(count, dtype=np.float32) for method in method_order
    }
    chunk_count = int(np.ceil(count / chunk_pixels))
    adaptive_retries = 0

    # Transfer the delay/aperture tables batch by batch.  Large H2D transfers
    # of these full-field tables can silently yield finite all-zero gathers on
    # WSL/CuPy, whereas the same focused pixels are valid in small batches.
    def reconstruct_batch(begin: int, end: int) -> dict[str, np.ndarray]:
        """Reconstruct and validate one focused-pixel batch."""

        receive_time_gpu = cp.asarray(
            receive_time[begin:end], dtype=cp.float32
        )
        aperture_gpu = cp.asarray(aperture[begin:end], dtype=cp.float32)
        receive_sign_gpu = cp.asarray(
            receive_sign[begin:end], dtype=cp.float32
        )
        chunk_size = end - begin
        uniform = cp.zeros(chunk_size, dtype=cp.complex64)
        receive_cf = cp.zeros(chunk_size, dtype=cp.complex64)
        mv = cp.zeros(chunk_size, dtype=cp.complex64) if compute_mv else None
        receive_null = cp.zeros(chunk_size, dtype=cp.complex64)
        angle_null = cp.zeros(chunk_size, dtype=cp.complex64)

        for local_index, angle_rad in enumerate(selected_angles):
            base, mask = focused_samples(
                iq_gpu,
                local_index,
                float(angle_rad),
                points_m[begin:end],
                receive_time_gpu,
                aperture_gpu,
                sampling_frequency_hz=dataset.sampling_frequency_hz,
                sound_speed_m_s=dataset.sound_speed_m_s,
                initial_time_s=dataset.initial_time_s,
                carrier_frequency_hz=carrier_frequency_hz,
                cp=cp,
            )
            angle_das = cp.sum(base, axis=1)
            uniform += angle_das
            receive_cf += receive_cf_das(base, mask, xp=cp)
            if compute_mv:
                assert mv is not None
                mv += capon_minimum_variance(
                    base,
                    mask,
                    grid_shape=(chunk_size, 1),
                    configuration=mv_configuration,
                    xp=cp,
                )
            receive_null += cp.sum(base * receive_sign_gpu, axis=1)
            angle_null += angle_weights_gpu[local_index] * angle_das

        output_gpu = {
            DAS: cp.abs(uniform),
            CF_DAS: cp.abs(receive_cf),
            RECEIVE_NSI: nsi_envelope(uniform, receive_null, nsi_c, xp=cp),
            ANGLE_NSI: nsi_envelope(uniform, angle_null, nsi_c, xp=cp),
        }
        if compute_mv:
            assert mv is not None
            output_gpu[MV] = cp.abs(mv)
        cp.cuda.Stream.null.synchronize()
        return {
            method: np.asarray(cp.asnumpy(values), dtype=np.float32)
            for method, values in output_gpu.items()
        }

    def valid_batch(values: dict[str, np.ndarray]) -> bool:
        return bool(
            all(np.all(np.isfinite(array)) for array in values.values())
            and float(np.max(values[DAS])) > 0.0
        )

    for chunk_index, begin in enumerate(range(0, count, chunk_pixels), start=1):
        base_end = min(begin + chunk_pixels, count)
        pending = [(begin, base_end)]
        while pending:
            batch_begin, batch_end = pending.pop(0)
            batch_output = reconstruct_batch(batch_begin, batch_end)
            if not valid_batch(batch_output):
                if batch_end - batch_begin <= 1:
                    diagnostics = {
                        method: {
                            "finite": bool(np.all(np.isfinite(values))),
                            "maximum": float(np.nanmax(values)),
                        }
                        for method, values in batch_output.items()
                    }
                    raise RuntimeError(
                        f"{label} IQ focusing failed at pixel {batch_begin}: "
                        f"{diagnostics}"
                    )
                midpoint = batch_begin + (batch_end - batch_begin) // 2
                adaptive_retries += 1
                print(
                    f"  {label}: IQ batch pixels {batch_begin + 1}-{batch_end} "
                    "failed validation; retrying as "
                    f"{batch_begin + 1}-{midpoint} and "
                    f"{midpoint + 1}-{batch_end}."
                )
                pending[0:0] = [
                    (batch_begin, midpoint), (midpoint, batch_end)
                ]
                continue
            for method, values in batch_output.items():
                output[method][batch_begin:batch_end] = values
        if chunk_index % 10 == 0 or chunk_index == chunk_count:
            print(
                f"  {label}: IQ focus batch {chunk_index}/{chunk_count} "
                f"({base_end}/{count} pixels, {len(selected_angles)} angles)"
            )

    del iq_gpu
    cp.get_default_memory_pool().free_all_blocks()
    if adaptive_retries:
        print(
            f"  {label}: completed with {adaptive_retries} adaptive "
            "IQ batch retries."
        )
    return output


def reconstruct_fdmas(
    dataset: PicmusDataset,
    angle_indices: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    *,
    carrier_frequency_hz: float,
    f_number: float,
    filter_configuration: FdmasFilterConfiguration,
    cp: Any,
    label: str,
    x_chunk_lines: int = FDMAS_X_CHUNK_LINES,
) -> tuple[np.ndarray, dict[str, Any]]:
    if x_chunk_lines < 1:
        raise ValueError("DMAS lateral chunk size must be positive.")
    raw_gpu = cp.asarray(dataset.data[:, :, angle_indices].real, dtype=cp.float32)
    selected_angles = dataset.angles_rad[angle_indices]
    spacing_m = float(np.median(np.diff(z_m)))
    depth_sampling_hz = dataset.sound_speed_m_s / (2.0 * spacing_m)
    coefficients, filter_metadata = design_fdmas_fir(
        carrier_frequency_hz, depth_sampling_hz, filter_configuration
    )
    coefficients_gpu = cp.asarray(coefficients)
    envelope = np.full((x_m.size, z_m.size), np.nan, dtype=np.float32)
    pair_peak = 0.0
    analytic_peak = 0.0
    requested_chunk_count = int(np.ceil(x_m.size / x_chunk_lines))
    successful_batch_sizes: list[int] = []
    fallback_count = 0

    # Lateral scan lines are independent.  Chunking avoids very large CuPy
    # advanced-index gathers without changing any delay, pair product, filter,
    # or coherent angular sum.  Some CUDA/CuPy combinations have silently
    # returned zero-valued large gathers.  Every lateral line is therefore
    # validated.  A failed batch is recomputed as two smaller exact batches,
    # recursively down to one line, rather than being accepted or approximated.
    def reconstruct_batch(begin: int, end: int) -> None:
        nonlocal pair_peak, analytic_peak, fallback_count
        chunk_x = x_m[begin:end]
        points_m = regular_points(chunk_x, z_m)
        receive_time, aperture, _ = dynamic_aperture_tables(
            points_m,
            dataset.probe_geometry_m,
            f_number=f_number,
            sound_speed_m_s=dataset.sound_speed_m_s,
        )
        receive_time_gpu = cp.asarray(receive_time, dtype=cp.float32)
        aperture_gpu = cp.asarray(aperture, dtype=cp.float32)
        pair_sum = cp.zeros(points_m.shape[0], dtype=cp.float32)

        for local_index, angle_rad in enumerate(selected_angles):
            focused_rf, mask = focused_samples(
                raw_gpu,
                local_index,
                float(angle_rad),
                points_m,
                receive_time_gpu,
                aperture_gpu,
                sampling_frequency_hz=dataset.sampling_frequency_hz,
                sound_speed_m_s=dataset.sound_speed_m_s,
                initial_time_s=dataset.initial_time_s,
                carrier_frequency_hz=None,
                cp=cp,
            )
            pair_sum += signed_sqrt_pair_sum(focused_rf, mask, xp=cp)
        del focused_rf, mask

        pair_image = pair_sum.reshape(chunk_x.size, z_m.size)
        pair_line_peaks_gpu = cp.max(cp.abs(pair_image), axis=1)
        pair_line_valid_gpu = cp.all(cp.isfinite(pair_image), axis=1) & (
            pair_line_peaks_gpu > 0.0
        )
        pair_line_peaks = np.asarray(cp.asnumpy(pair_line_peaks_gpu), dtype=float)
        pair_line_valid = np.asarray(cp.asnumpy(pair_line_valid_gpu), dtype=bool)
        chunk_pair_peak = float(np.max(pair_line_peaks))
        if not np.all(pair_line_valid):
            failed = (np.flatnonzero(~pair_line_valid) + begin + 1).tolist()
            del receive_time_gpu, aperture_gpu, pair_sum, pair_image
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            if end - begin > 1:
                midpoint = begin + (end - begin) // 2
                fallback_count += 1
                print(
                    f"  {label}: DMAS batch lines {begin + 1}-{end} "
                    f"failed pair validation at image lines {failed}; "
                    f"retrying as {begin + 1}-{midpoint} and "
                    f"{midpoint + 1}-{end}."
                )
                reconstruct_batch(begin, midpoint)
                reconstruct_batch(midpoint, end)
                return
            raise RuntimeError(
                f"{label} DMAS pair accumulation remained empty or "
                f"non-finite for image line {begin + 1} after single-line "
                "fallback."
            )

        analytic = fdmas_analytic_image(
            pair_image,
            coefficients_gpu,
            depth_axis=1,
            xp=cp,
        )
        analytic_line_peaks_gpu = cp.max(cp.abs(analytic), axis=1)
        analytic_line_valid_gpu = cp.all(cp.isfinite(analytic), axis=1) & (
            analytic_line_peaks_gpu > 0.0
        )
        analytic_line_peaks = np.asarray(
            cp.asnumpy(analytic_line_peaks_gpu), dtype=float
        )
        analytic_line_valid = np.asarray(
            cp.asnumpy(analytic_line_valid_gpu), dtype=bool
        )
        chunk_analytic_peak = float(np.max(analytic_line_peaks))
        if not np.all(analytic_line_valid):
            failed = (np.flatnonzero(~analytic_line_valid) + begin + 1).tolist()
            del receive_time_gpu, aperture_gpu, pair_sum, pair_image, analytic
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            if end - begin > 1:
                midpoint = begin + (end - begin) // 2
                fallback_count += 1
                print(
                    f"  {label}: DMAS batch lines {begin + 1}-{end} "
                    f"failed filter validation at image lines {failed}; "
                    f"retrying as {begin + 1}-{midpoint} and "
                    f"{midpoint + 1}-{end}."
                )
                reconstruct_batch(begin, midpoint)
                reconstruct_batch(midpoint, end)
                return
            raise RuntimeError(
                f"{label} DMAS filtering/Hilbert output remained empty or "
                f"non-finite for image line {begin + 1} after single-line "
                "fallback."
            )

        pair_peak = max(pair_peak, chunk_pair_peak)
        analytic_peak = max(analytic_peak, chunk_analytic_peak)
        envelope[begin:end] = cp.asnumpy(cp.abs(analytic))
        successful_batch_sizes.append(end - begin)
        cp.cuda.Stream.null.synchronize()
        del receive_time_gpu, aperture_gpu, pair_sum, pair_image, analytic
        cp.get_default_memory_pool().free_all_blocks()
        print(
            f"  {label}: validated DMAS batch lines {begin + 1}-{end} "
            f"({end - begin} lines, {len(selected_angles)} angles)"
        )

    for begin in range(0, x_m.size, x_chunk_lines):
        reconstruct_batch(begin, min(begin + x_chunk_lines, x_m.size))

    if (
        pair_peak <= 0.0
        or analytic_peak <= 0.0
        or not usable_complete_image(envelope)
    ):
        raise RuntimeError(
            f"{label} DMAS output is incomplete after validated chunked "
            "reconstruction "
            f"(pair peak={pair_peak:.6e}, analytic peak={analytic_peak:.6e})."
        )
    print(
        f"  {label}: DMAS stage peaks "
        f"pair={pair_peak:.6e}, analytic={analytic_peak:.6e}; "
        f"validated batches={len(successful_batch_sizes)}, "
        f"adaptive retries={fallback_count}."
    )
    filter_metadata["requested_lateral_chunk_lines"] = int(x_chunk_lines)
    filter_metadata["requested_lateral_chunk_count"] = requested_chunk_count
    filter_metadata["validated_lateral_batch_count"] = len(successful_batch_sizes)
    filter_metadata["validated_lateral_batch_sizes"] = successful_batch_sizes
    filter_metadata["adaptive_batch_retries"] = fallback_count
    filter_metadata["lateral_chunking"] = (
        "exact independent lateral-line batches with per-line validation and "
        "recursive bisection fallback; no algorithmic approximation"
    )
    del raw_gpu, coefficients_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return envelope, filter_metadata


def fdmas_padded_axis(
    desired_min_m: float,
    desired_max_m: float,
    spacing_m: float,
    *,
    sound_speed_m_s: float,
    carrier_frequency_hz: float,
    filter_configuration: FdmasFilterConfiguration,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_sampling_hz = sound_speed_m_s / (2.0 * spacing_m)
    _, metadata = design_fdmas_fir(
        carrier_frequency_hz, depth_sampling_hz, filter_configuration
    )
    padding_m = (metadata["fir_group_delay_samples"] + 2) * spacing_m
    lower = max(spacing_m, desired_min_m - padding_m)
    upper = desired_max_m + padding_m
    return regular_axis(lower, upper, spacing_m), metadata


def resample_regular_image(
    image: np.ndarray,
    source_x: np.ndarray,
    source_z: np.ndarray,
    target_x: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    if image.shape != (source_x.size, source_z.size):
        raise ValueError("Regular image and source axes disagree.")
    along_z = np.vstack(
        [np.interp(target_z, source_z, image[row]) for row in range(source_x.size)]
    )
    output = np.vstack(
        [np.interp(target_x, source_x, along_z[:, column]) for column in range(target_z.size)]
    ).T
    return output.astype(np.float32, copy=False)


def normalized_db(image: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    peak = float(np.max(image))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("Cannot normalize an empty beamformed image.")
    floor = peak * 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(image, floor) / peak)


def circular_roi_mask(
    x_m: np.ndarray,
    z_m: np.ndarray,
    center_cm: tuple[float, float],
    radius_cm: float,
) -> np.ndarray:
    x_grid, z_grid = np.meshgrid(x_m * 100.0, z_m * 100.0, indexing="ij")
    return (
        (x_grid - center_cm[0]) ** 2 + (z_grid - center_cm[1]) ** 2
        <= radius_cm**2
    )


def compute_gcnr(signal: np.ndarray, background: np.ndarray, bins: int) -> float:
    signal = np.asarray(signal, dtype=float).ravel()
    background = np.asarray(background, dtype=float).ravel()
    lower = min(float(signal.min()), float(background.min()))
    upper = max(float(signal.max()), float(background.max()))
    if np.isclose(lower, upper):
        return 0.0
    edges = np.linspace(lower, upper, bins + 1)
    signal_hist, _ = np.histogram(signal, bins=edges)
    background_hist, _ = np.histogram(background, bins=edges)
    signal_probability = signal_hist / signal_hist.sum()
    background_probability = background_hist / background_hist.sum()
    return float(1.0 - np.minimum(signal_probability, background_probability).sum())


def carotid_metric_rows(
    view: str,
    images: dict[str, np.ndarray],
    x_m: np.ndarray,
    z_m: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    roi = ROI_BY_VIEW[view]
    signal_mask = circular_roi_mask(
        x_m, z_m, roi["signal_center_cm"], roi["signal_radius_cm"]
    )
    background_mask = circular_roi_mask(
        x_m, z_m, roi["background_center_cm"], roi["background_radius_cm"]
    )
    if not signal_mask.any() or not background_mask.any():
        raise ValueError(f"The {view} baseline grid does not cover both ROIs.")
    rows = []
    for method in METHODS:
        image = np.maximum(
            np.asarray(images[method], dtype=np.float64), np.finfo(float).tiny
        )
        image_db = normalized_db(image)
        signal = image[signal_mask]
        background = image[background_mask]
        signal_db = image_db[signal_mask]
        background_db = image_db[background_mask]
        denominator = np.sqrt(np.var(signal) + np.var(background))
        signal_mean = float(np.mean(signal, dtype=np.float64))
        background_mean = float(np.mean(background, dtype=np.float64))
        contrast_ratio_db = float(
            20.0 * (np.log10(background_mean) - np.log10(signal_mean))
        )
        rows.append(
            {
                "case": "carotid",
                "view": view,
                "method": method,
                "gcnr": compute_gcnr(signal_db, background_db, bins),
                "contrast_ratio_db": contrast_ratio_db,
                "cnr": float(
                    abs(background_mean - signal_mean)
                    / max(denominator, np.finfo(float).tiny)
                ),
                "signal_pixel_count": int(signal.size),
                "background_pixel_count": int(background.size),
                "signal_mean_linear": signal_mean,
                "background_mean_linear": background_mean,
            }
        )
    return rows


def plot_psf(
    path: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    grid: TargetGrid,
    target_x_mm: float,
    target_z_mm: float,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.0, 7.1),
        dpi=180,
        constrained_layout=True,
    )
    for axis, method in zip(axes.flat, METHODS):
        image_db = amplitude_to_db(arrays[method]["map"], floor_db=-60.0)
        im = axis.imshow(
            image_db.T,
            extent=[grid.map_x_mm[0], grid.map_x_mm[-1], grid.map_z_mm[-1], grid.map_z_mm[0]],
            cmap="gray",
            vmin=-60.0,
            vmax=0.0,
            aspect="equal",
        )
        axis.plot(target_x_mm, target_z_mm, "+", color="orange", markersize=6)
        axis.set_title(method, fontsize=9)
        axis.set_xlabel("Lateral position [mm]")
        axis.set_ylabel("Depth [mm]")
    figure.colorbar(im, ax=axes.ravel().tolist(), label="Normalized amplitude [dB]", shrink=0.82)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_psf_profiles(
    path: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    grid: TargetGrid,
    target_x_mm: float,
    target_z_mm: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), dpi=180)
    for method in METHODS:
        lateral = amplitude_to_db(arrays[method]["lateral"], floor_db=-80.0)
        axial = amplitude_to_db(arrays[method]["axial"], floor_db=-80.0)
        axes[0].plot(
            grid.lateral_x_mm - target_x_mm,
            lateral,
            label=method,
            color=COLORS[method],
        )
        axes[1].plot(
            grid.axial_z_mm - target_z_mm,
            axial,
            label=method,
            color=COLORS[method],
        )
    axes[0].set(xlabel="Lateral offset [mm]", ylabel="Normalized amplitude [dB]")
    axes[1].set(xlabel="Axial offset [mm]", ylabel="Normalized amplitude [dB]")
    for axis in axes:
        axis.set_ylim(-60.0, 1.0)
        axis.grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_carotid(
    path: Path,
    view: str,
    images: dict[str, np.ndarray],
    x_m: np.ndarray,
    z_m: np.ndarray,
) -> None:
    roi = ROI_BY_VIEW[view]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 8.1),
        dpi=180,
        constrained_layout=True,
    )
    for axis, method in zip(axes.flat, METHODS):
        im = axis.imshow(
            normalized_db(images[method]).T,
            extent=[x_m[0] * 1e3, x_m[-1] * 1e3, z_m[-1] * 1e3, z_m[0] * 1e3],
            cmap="gray",
            vmin=-60.0,
            vmax=0.0,
            aspect="equal",
        )
        for center_name, radius_name, color in (
            ("signal_center_cm", "signal_radius_cm", "#ff7f0e"),
            ("background_center_cm", "background_radius_cm", "#00bcd4"),
        ):
            center = roi[center_name]
            axis.add_patch(
                Circle(
                    (center[0] * 10.0, center[1] * 10.0),
                    roi[radius_name] * 10.0,
                    fill=False,
                    color=color,
                    linewidth=0.8,
                )
            )
        axis.set_title(method, fontsize=9)
        axis.set_xlabel("Lateral position [mm]")
        axis.set_ylabel("Depth [mm]")
    figure.colorbar(im, ax=axes.ravel().tolist(), label="Normalized amplitude [dB]", shrink=0.82)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_carotid_combined(
    path: Path,
    cases: dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
) -> None:
    """Plot both carotid views as a compact two-row manuscript figure."""

    figure, axes = plt.subplots(
        2,
        len(METHODS),
        figsize=(11.0, 5.2),
        dpi=180,
        constrained_layout=True,
    )
    image_artist = None
    for row_index, view in enumerate(("CL", "CC")):
        images, x_m, z_m = cases[view]
        roi = ROI_BY_VIEW[view]
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            image_artist = axis.imshow(
                normalized_db(images[method]).T,
                extent=[
                    x_m[0] * 1e3,
                    x_m[-1] * 1e3,
                    z_m[-1] * 1e3,
                    z_m[0] * 1e3,
                ],
                cmap="gray",
                vmin=-60.0,
                vmax=0.0,
                aspect="equal",
            )
            for center_name, radius_name, color in (
                ("signal_center_cm", "signal_radius_cm", "#ff7f0e"),
                ("background_center_cm", "background_radius_cm", "#00bcd4"),
            ):
                center = roi[center_name]
                axis.add_patch(
                    Circle(
                        (center[0] * 10.0, center[1] * 10.0),
                        roi[radius_name] * 10.0,
                        fill=False,
                        color=color,
                        linewidth=0.7,
                    )
                )
            if row_index == 0:
                axis.set_title(method, fontsize=12)
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Lateral [mm]", fontsize=10)
            if column_index == 0:
                axis.set_ylabel(
                    f"{roi['label']}\nDepth [mm]",
                    fontsize=10,
                )
            else:
                axis.tick_params(labelleft=False)
            axis.tick_params(labelsize=9)
    assert image_artist is not None
    colorbar = figure.colorbar(
        image_artist,
        ax=axes.ravel().tolist(),
        shrink=0.82,
    )
    colorbar.set_label("Normalized amplitude [dB]", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def psf_case(
    dataset: PicmusDataset,
    phantom_positions_m: np.ndarray,
    angle_indices: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    cp: Any,
    filter_configuration: FdmasFilterConfiguration,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    positions_mm = phantom_positions_m * 1e3
    target_index = int(
        np.argmin(np.abs(positions_mm[:, 0]) + np.abs(positions_mm[:, 2] - 37.5))
    )
    target_x_mm = float(positions_mm[target_index, 0])
    target_z_mm = float(positions_mm[target_index, 2])
    grid = build_target_grid(
        target_x_mm,
        target_z_mm,
        map_half_width_mm=args.psf_map_half_width_mm,
        map_spacing_mm=args.psf_map_spacing_mm,
        profile_half_width_mm=args.psf_profile_half_width_mm,
        lateral_spacing_mm=args.psf_lateral_spacing_mm,
        axial_spacing_mm=args.psf_axial_spacing_mm,
    )
    cache_inputs = {
        "dataset": file_fingerprint(args.resolution_dataset),
        "target_index": target_index,
        "angle_indices": angle_indices,
        "grid_points_m_sha256": hashlib.sha256(grid.points_m.tobytes()).hexdigest(),
        "fc": args.carrier_frequency_mhz,
        "f_number": args.f_number,
        "nsi_c": args.nsi_c,
        "fdmas_z_spacing_mm": args.fdmas_z_spacing_mm,
        "mv": asdict(MvConfiguration(chunk_pixels=args.mv_chunk_pixels)),
    }
    cache_signature = signature(cache_inputs)
    cache_path = output_dir / "conventional_baseline_psf_cache.npz"
    cache_metadata = output_dir / "conventional_baseline_psf_cache.json"
    cached = None if args.force else load_image_cache(cache_path, cache_metadata, cache_signature)
    if cached is not None:
        print("Reusing conventional-baseline PSF cache.")
        flat_images, _ = cached
        cache_changed = False
    else:
        print("Reconstructing representative experimental point target...")
        mv_configuration = MvConfiguration(
            temporal_half_window_samples=0,
            chunk_pixels=args.mv_chunk_pixels,
        )
        flat_images = reconstruct_iq_methods(
            dataset,
            angle_indices,
            grid.points_m,
            grid_shape=(grid.points_m.shape[0], 1),
            carrier_frequency_hz=args.carrier_frequency_mhz * 1e6,
            f_number=args.f_number,
            nsi_c=args.nsi_c,
            mv_configuration=mv_configuration,
            cp=cp,
            label="PSF",
        )
        cache_changed = True

    if not usable_positive_image(flat_images.get(DMAS)):
        if cached is not None:
            print(
                "Cached PSF DMAS image is empty; preserving the five valid "
                "methods and rebuilding DMAS only."
            )
        prepare_fdmas_gpu(cp)
        fdmas_spacing_m = args.fdmas_z_spacing_mm * 1e-3
        fdmas_z, _ = fdmas_padded_axis(
            min(float(grid.map_z_mm.min()), float(grid.axial_z_mm.min())) * 1e-3,
            max(float(grid.map_z_mm.max()), float(grid.axial_z_mm.max())) * 1e-3,
            fdmas_spacing_m,
            sound_speed_m_s=dataset.sound_speed_m_s,
            carrier_frequency_hz=args.carrier_frequency_mhz * 1e6,
            filter_configuration=filter_configuration,
        )
        fdmas_x = (grid.lateral_x_mm * 1e-3).astype(np.float32)
        fdmas_image, _ = reconstruct_fdmas(
            dataset,
            angle_indices,
            fdmas_x,
            fdmas_z,
            carrier_frequency_hz=args.carrier_frequency_mhz * 1e6,
            f_number=args.f_number,
            filter_configuration=filter_configuration,
            cp=cp,
            label="PSF",
        )
        fdmas_source_peak = float(np.max(fdmas_image))
        if not usable_positive_image(fdmas_image):
            raise RuntimeError(
                "DMAS produced an empty source envelope before PSF "
                "resampling. The valid IQ cache was retained."
            )
        fdmas_map = resample_regular_image(
            fdmas_image,
            fdmas_x,
            fdmas_z,
            grid.map_x_mm * 1e-3,
            grid.map_z_mm * 1e-3,
        )
        fdmas_lateral = resample_regular_image(
            fdmas_image,
            fdmas_x,
            fdmas_z,
            grid.lateral_x_mm * 1e-3,
            np.asarray([target_z_mm * 1e-3]),
        )[:, 0]
        fdmas_axial = resample_regular_image(
            fdmas_image,
            fdmas_x,
            fdmas_z,
            np.asarray([target_x_mm * 1e-3]),
            grid.axial_z_mm * 1e-3,
        )[0]
        flat_images[DMAS] = np.concatenate(
            [fdmas_map.ravel(), fdmas_lateral, fdmas_axial]
        )
        fdmas_resampled_peak = float(np.max(flat_images[DMAS]))
        if not usable_positive_image(flat_images[DMAS]):
            raise RuntimeError(
                "DMAS became empty during PSF resampling. The valid IQ "
                "cache was retained."
            )
        print(
            "  PSF: validated DMAS envelope "
            f"(source peak={fdmas_source_peak:.6e}, "
            f"resampled peak={fdmas_resampled_peak:.6e})."
        )
        cache_changed = True

    if cache_changed:
        save_image_cache(
            cache_path,
            cache_metadata,
            flat_images,
            {},
            cache_signature,
        )

    metric_rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        metrics, method_arrays = metrics_for_method(
            flat_images[method], grid, target_x_mm, target_z_mm
        )
        metric_rows.append(
            {
                "case": "experimental_psf",
                "view": "central_37p5_mm_target",
                "method": method,
                **metrics,
            }
        )
        arrays[method] = method_arrays

    map_path = output_dir / "conventional_baseline_experimental_psf.png"
    profile_path = output_dir / "conventional_baseline_experimental_psf_profiles.png"
    plot_psf(map_path, arrays, grid, target_x_mm, target_z_mm)
    plot_psf_profiles(profile_path, arrays, grid, target_x_mm, target_z_mm)
    np.savez_compressed(
        output_dir / "conventional_baseline_experimental_psf_profiles.npz",
        lateral_x_mm=grid.lateral_x_mm,
        axial_z_mm=grid.axial_z_mm,
        **{
            f"{method_key(method)}_{profile}": arrays[method][profile]
            for method in METHODS
            for profile in ("lateral", "axial")
        },
    )
    return (
        {
            "target_index_zero_based": target_index,
            "target_position_mm": [target_x_mm, 0.0, target_z_mm],
            "mv_temporal_half_window_samples": 0,
            "metrics": metric_rows,
        },
        metric_rows,
        {"map_figure": str(map_path.resolve()), "profile_figure": str(profile_path.resolve())},
    )


def carotid_bounds_m(view: str, margin_mm: float = 1.0) -> tuple[float, float, float, float]:
    roi = ROI_BY_VIEW[view]
    circles = (
        (roi["signal_center_cm"], roi["signal_radius_cm"]),
        (roi["background_center_cm"], roi["background_radius_cm"]),
    )
    x_min_mm = min((center[0] - radius) * 10.0 for center, radius in circles) - margin_mm
    x_max_mm = max((center[0] + radius) * 10.0 for center, radius in circles) + margin_mm
    z_min_mm = min((center[1] - radius) * 10.0 for center, radius in circles) - margin_mm
    z_max_mm = max((center[1] + radius) * 10.0 for center, radius in circles) + margin_mm
    return tuple(value * 1e-3 for value in (x_min_mm, x_max_mm, z_min_mm, z_max_mm))


def carotid_case(
    view: str,
    dataset: PicmusDataset,
    angle_indices: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    cp: Any,
    filter_configuration: FdmasFilterConfiguration,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    tuple[dict[str, np.ndarray], np.ndarray, np.ndarray],
]:
    x_min, x_max, z_min, z_max = carotid_bounds_m(view)
    x_m = regular_axis(x_min, x_max, args.carotid_x_spacing_mm * 1e-3)
    z_m = regular_axis(z_min, z_max, args.carotid_z_spacing_mm * 1e-3)
    points_m = regular_points(x_m, z_m)
    mv_half_window = temporal_half_window_from_wavelengths(
        args.mv_temporal_wavelengths,
        float(np.median(np.diff(z_m))),
        args.carrier_frequency_mhz * 1e6,
        dataset.sound_speed_m_s,
    )
    mv_configuration = MvConfiguration(
        temporal_half_window_samples=mv_half_window,
        chunk_pixels=args.mv_chunk_pixels,
    )
    cache_inputs = {
        "dataset": file_fingerprint(
            args.carotid_long if view == "CL" else args.carotid_cross
        ),
        "view": view,
        "angle_indices": angle_indices,
        "x_m": x_m,
        "z_m": z_m,
        "fc": args.carrier_frequency_mhz,
        "f_number": args.f_number,
        "nsi_c": args.nsi_c,
        "fdmas_z_spacing_mm": args.fdmas_z_spacing_mm,
        "mv": asdict(mv_configuration),
    }
    cache_signature = signature(cache_inputs)
    cache_path = output_dir / f"conventional_baseline_carotid_{view}_cache.npz"
    cache_metadata = output_dir / f"conventional_baseline_carotid_{view}_cache.json"
    cached = None if args.force else load_image_cache(cache_path, cache_metadata, cache_signature)
    if cached is not None:
        print(f"Reusing conventional-baseline carotid {view} cache.")
        images, cached_axes = cached
        x_m = cached_axes["x_m"]
        z_m = cached_axes["z_m"]
        cache_changed = False
    else:
        print(f"Reconstructing carotid {view} conventional baseline comparison...")
        flat = reconstruct_iq_methods(
            dataset,
            angle_indices,
            points_m,
            grid_shape=(x_m.size, z_m.size),
            carrier_frequency_hz=args.carrier_frequency_mhz * 1e6,
            f_number=args.f_number,
            nsi_c=args.nsi_c,
            mv_configuration=mv_configuration,
            cp=cp,
            label=f"carotid {view}",
        )
        images = {
            method: flat[method].reshape(x_m.size, z_m.size)
            for method in IQ_METHODS
        }
        cache_changed = True

    if not usable_complete_image(images.get(DMAS)):
        if cached is not None:
            print(
                f"Cached carotid {view} DMAS image is empty or incomplete; "
                "preserving the five valid methods and rebuilding DMAS only."
            )
        prepare_fdmas_gpu(cp)
        fdmas_z, _ = fdmas_padded_axis(
            float(z_m.min()),
            float(z_m.max()),
            args.fdmas_z_spacing_mm * 1e-3,
            sound_speed_m_s=dataset.sound_speed_m_s,
            carrier_frequency_hz=args.carrier_frequency_mhz * 1e6,
            filter_configuration=filter_configuration,
        )
        fdmas_fine, _ = reconstruct_fdmas(
            dataset,
            angle_indices,
            x_m,
            fdmas_z,
            carrier_frequency_hz=args.carrier_frequency_mhz * 1e6,
            f_number=args.f_number,
            filter_configuration=filter_configuration,
            cp=cp,
            label=f"carotid {view}",
        )
        fdmas_source_peak = float(np.max(fdmas_fine))
        if not usable_complete_image(fdmas_fine):
            raise RuntimeError(
                f"DMAS produced an empty or incomplete source envelope for carotid "
                f"{view}. The valid IQ cache was retained."
            )
        images[DMAS] = resample_regular_image(
            fdmas_fine, x_m, fdmas_z, x_m, z_m
        )
        fdmas_resampled_peak = float(np.max(images[DMAS]))
        if not usable_complete_image(images[DMAS]):
            raise RuntimeError(
                f"DMAS became empty or incomplete while resampling carotid {view}. "
                "The valid IQ cache was retained."
            )
        print(
            f"  carotid {view}: validated DMAS envelope "
            f"(source peak={fdmas_source_peak:.6e}, "
            f"resampled peak={fdmas_resampled_peak:.6e})."
        )
        cache_changed = True

    if cache_changed:
        save_image_cache(
            cache_path,
            cache_metadata,
            images,
            {"x_m": x_m, "z_m": z_m},
            cache_signature,
        )

    rows = carotid_metric_rows(view, images, x_m, z_m, args.gcnr_bins)
    figure_path = output_dir / f"conventional_baseline_carotid_{view}.png"
    plot_carotid(figure_path, view, images, x_m, z_m)
    return (
        {
            "view": view,
            "view_label": ROI_BY_VIEW[view]["label"],
            "grid": {
                "nx": int(x_m.size),
                "nz": int(z_m.size),
                "x_limits_mm": [float(x_m.min() * 1e3), float(x_m.max() * 1e3)],
                "z_limits_mm": [float(z_m.min() * 1e3), float(z_m.max() * 1e3)],
                "x_spacing_mm": float(np.median(np.diff(x_m)) * 1e3),
                "z_spacing_mm": float(np.median(np.diff(z_m)) * 1e3),
            },
            "roi": ROI_BY_VIEW[view],
            "mv_temporal_half_window_samples": mv_half_window,
            "mv_temporal_full_window_samples": 2 * mv_half_window + 1,
            "metrics": rows,
            "figure": str(figure_path.resolve()),
        },
        rows,
        str(figure_path.resolve()),
        (images, x_m, z_m),
    )


def acquisition_record(dataset: PicmusDataset, path: Path) -> dict[str, Any]:
    return {
        "file": str(path.expanduser().resolve()),
        "shape_samples_elements_angles": list(dataset.data.shape),
        "sampling_frequency_hz": dataset.sampling_frequency_hz,
        "sound_speed_m_s": dataset.sound_speed_m_s,
        "initial_time_s": dataset.initial_time_s,
        "modulation_frequency_hz": dataset.modulation_frequency_hz,
        "angle_limits_deg": [
            float(np.rad2deg(dataset.angles_rad.min())),
            float(np.rad2deg(dataset.angles_rad.max())),
        ],
    }


def main() -> None:
    args = parse_args()
    require_positive(args)
    if args.quick and args.angle_count is None:
        args.angle_count = 5
    if args.quick:
        args.psf_map_spacing_mm = max(args.psf_map_spacing_mm, 0.05)
        args.psf_lateral_spacing_mm = max(args.psf_lateral_spacing_mm, 0.02)
        args.carotid_x_spacing_mm = max(args.carotid_x_spacing_mm, 0.40)
        args.carotid_z_spacing_mm = max(args.carotid_z_spacing_mm, 0.20)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    resolution = load_picmus_dataset(args.resolution_dataset)
    phantom = load_picmus_phantom(args.resolution_phantom)
    carotid_long = load_picmus_dataset(args.carotid_long)
    carotid_cross = load_picmus_dataset(args.carotid_cross)
    datasets = {"resolution": resolution, "CL": carotid_long, "CC": carotid_cross}
    angle_indices = {
        name: choose_angle_indices(len(dataset.angles_rad), args.angle_count)
        for name, dataset in datasets.items()
    }
    filter_configuration = FdmasFilterConfiguration()
    filter_preview = {}
    for name, dataset in datasets.items():
        _, filter_preview[name] = design_fdmas_fir(
            args.carrier_frequency_mhz * 1e6,
            dataset.sound_speed_m_s / (2.0 * args.fdmas_z_spacing_mm * 1e-3),
            filter_configuration,
        )

    base_metadata: dict[str, Any] = {
        "script": Path(__file__).name,
        "schema_version": 3,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "quick_engineering_run": bool(args.quick),
        "metadata_only": bool(args.metadata_only),
        "publication_ready": False,
        "acquisitions": {
            "resolution": acquisition_record(resolution, args.resolution_dataset),
            "CL": acquisition_record(carotid_long, args.carotid_long),
            "CC": acquisition_record(carotid_cross, args.carotid_cross),
        },
        "selected_angle_indices": {
            name: indices.tolist() for name, indices in angle_indices.items()
        },
        "fixed_method_parameters": {
            "receive_cf_das": {
                "definition": "|sum_m x_m|^2/(M_active*sum_m|x_m|^2), multiplied by the complex receive DAS sum for each transmit angle before coherent compounding",
                "parameter_free": True,
                "active_element_definition": "binary dynamic receive-aperture and valid-time mask; no data-dependent activity threshold",
                "reference": "Li and Li, IEEE TUFFC 2003 (GCF with DC-only low-frequency region, i.e. conventional CF)",
            },
            "mv_point_target": MvConfiguration(
                temporal_half_window_samples=0,
                chunk_pixels=args.mv_chunk_pixels,
            ).to_dict(),
            "mv_carotid": {
                **MvConfiguration(chunk_pixels=args.mv_chunk_pixels).to_dict(),
                "temporal_half_window_wavelengths": args.mv_temporal_wavelengths,
                "forward_backward_averaging": False,
                "active_element_definition": "binary dynamic receive-aperture and valid-time mask",
                "processing_domain": "receive channels separately per transmit angle, followed by coherent angular compounding",
                "amplitude_scaling": "USTB convention: average subarray outputs, then multiply by M_active to match DAS sum scaling",
            },
            "fdmas": {
                **filter_configuration.to_dict(),
                "filter_preview_by_dataset": filter_preview,
                "processing_domain": "delayed real RF receive channels separately per transmit angle; pair sums are coherently accumulated before the linear FIR/Hilbert stages",
                "input_prefilter": "the distributed PICMUS acquisition RF is used directly; no result-dependent input filter is fitted",
                "fine_axial_spacing_mm": args.fdmas_z_spacing_mm,
                "requested_lateral_chunk_lines": FDMAS_X_CHUNK_LINES,
                "lateral_chunking": "exact independent batches with per-line validation and recursive bisection fallback; no algorithmic approximation",
                "pair_evaluation": "exact O(M) algebraic identity, unit-tested against the literal O(M^2) i<j sum",
            },
            "nsi_dc_offset": args.nsi_c,
            "f_number": args.f_number,
            "carrier_frequency_hz": args.carrier_frequency_mhz * 1e6,
            "interpolation": "linear interpolation of the same receive delays and dynamic aperture for every method",
        },
        "implementation_provenance": {
            "production": "Python/CuPy implementation in src/adaptive_beamforming.py",
            "ustb_repository": "https://github.com/unioslo/USTB",
            "ustb_reference_commit": USTB_REFERENCE_COMMIT,
            "ustb_files_checked": [
                "+postprocess/coherence_factor.m",
                "+postprocess/capon_minimum_variance.m",
                "+postprocess/delay_multiply_and_sum.m",
            ],
        },
    }
    if args.metadata_only:
        path = output_dir / "conventional_baseline_metadata_check.json"
        write_json(path, base_metadata)
        print(f"Validated all conventional-baseline inputs and parameters: {path}")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import cupy as cp
    except ImportError as error:
        raise SystemExit("The full conventional-baseline comparison requires CuPy.") from error
    cp.cuda.Device(0).use()
    device_properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device_properties.get("name", "")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")

    psf_summary, psf_rows, psf_artifacts = psf_case(
        resolution,
        phantom.positions_m,
        angle_indices["resolution"],
        args,
        output_dir,
        cp,
        filter_configuration,
    )
    view_summaries = []
    all_rows = list(psf_rows)
    figures = dict(psf_artifacts)
    carotid_cases = {}
    for view, dataset in (("CL", carotid_long), ("CC", carotid_cross)):
        view_summary, rows, figure, case = carotid_case(
            view,
            dataset,
            angle_indices[view],
            args,
            output_dir,
            cp,
            filter_configuration,
        )
        view_summaries.append(view_summary)
        all_rows.extend(rows)
        figures[f"carotid_{view}"] = figure
        carotid_cases[view] = case

    combined_carotid_path = output_dir / "conventional_baseline_carotid_combined.png"
    plot_carotid_combined(combined_carotid_path, carotid_cases)
    figures["carotid_combined"] = str(combined_carotid_path.resolve())

    metrics_path = output_dir / "conventional_baseline_metrics.csv"
    write_csv(metrics_path, all_rows)
    full_angle_run = all(
        len(angle_indices[name]) == len(dataset.angles_rad)
        for name, dataset in datasets.items()
    )
    summary = {
        **base_metadata,
        "metadata_only": False,
        "publication_ready": bool(not args.quick and full_angle_run),
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
            "matplotlib": version_or_none("matplotlib"),
        },
        "experimental_psf": psf_summary,
        "carotid_views": view_summaries,
        "artifacts": {**figures, "metrics_csv": str(metrics_path.resolve())},
        "total_wall_seconds": time.perf_counter() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "conventional_baseline_summary.json"
    write_json(summary_path, summary)
    print(f"Conventional baseline summary: {summary_path}")
    print(f"Publication-ready: {summary['publication_ready']}")


if __name__ == "__main__":
    main()

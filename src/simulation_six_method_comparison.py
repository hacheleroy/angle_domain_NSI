#!/usr/bin/env python3
"""Controlled six-method point-target comparison for manuscript Figure 1.

The established point-target script retains the dense NSI convergence and
``c``-sensitivity studies used in the Supplement.  This companion analysis
uses the same PyMUST acquisition but evaluates the six manuscript methods on a
common target-centred map.  IQ methods also receive a directly beamformed
micrometre-scale lateral profile; DMAS uses the already well-sampled map
profile because its response is much wider.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    receive_cf_das,
    signed_sqrt_pair_sum,
)
from conventional_baseline_comparison import (
    fdmas_padded_axis,
    focused_samples,
    regular_axis,
    regular_points,
    resample_regular_image,
)
from method_names import (
    ANGLE_NSI,
    CF_DAS,
    DAS,
    DMAS,
    IQ_METHODS,
    METHODS,
    MV,
    RECEIVE_NSI,
)
from nsi_core import angular_sign_weights, nsi_envelope
from psf_metrics import amplitude_to_db, measure_connected_width, peak_to_background_db


COLORS = {
    DAS: "#6a3d9a",
    CF_DAS: "#2ca02c",
    MV: "#ff7f0e",
    DMAS: "#8c564b",
    RECEIVE_NSI: "#d62728",
    ANGLE_NSI: "#1f77b4",
}
LINESTYLES = {
    DAS: "-",
    CF_DAS: "--",
    MV: "-.",
    DMAS: ":",
    RECEIVE_NSI: (0, (5, 1)),
    ANGLE_NSI: (0, (3, 1, 1, 1)),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Six-method simulated point-target comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "simulation_six_method",
    )
    parser.add_argument("--device", default=os.environ.get("NSI_CUDA_DEVICE", "0"))
    parser.add_argument("--nsi-c", type=float, default=0.05)
    parser.add_argument("--map-x-half-width-mm", type=float, default=1.5)
    parser.add_argument("--map-z-half-width-mm", type=float, default=1.5)
    parser.add_argument("--map-x-spacing-mm", type=float, default=0.020)
    parser.add_argument("--map-z-spacing-mm", type=float, default=0.020)
    parser.add_argument("--dmas-z-spacing-mm", type=float, default=0.010)
    parser.add_argument("--profile-half-width-mm", type=float, default=1.0)
    parser.add_argument("--profile-spacing-mm", type=float, default=0.000390625)
    parser.add_argument("--mv-chunk-pixels", type=int, default=32)
    parser.add_argument("--dmas-x-chunk-lines", type=int, default=16)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write the frozen configuration without importing CuPy or simulating RF.",
    )
    return parser.parse_args()


def centred_axis(centre_m: float, half_width_m: float, spacing_m: float) -> np.ndarray:
    count = int(round(half_width_m / spacing_m))
    if count < 2 or spacing_m <= 0.0:
        raise ValueError("A centred axis requires positive spacing and at least five points.")
    return centre_m + np.arange(-count, count + 1, dtype=float) * spacing_m


def pad_and_stack(arrays: list[np.ndarray], *, dtype: Any) -> np.ndarray:
    """Pad variable-length PyMUST records with zeros and stack over angles."""

    if not arrays:
        raise ValueError("No simulated records were supplied.")
    elements = arrays[0].shape[1]
    if any(array.ndim != 2 or array.shape[1] != elements for array in arrays):
        raise ValueError("All PyMUST records must have a common element dimension.")
    samples = max(array.shape[0] for array in arrays)
    output = np.zeros((samples, elements, len(arrays)), dtype=dtype)
    for index, array in enumerate(arrays):
        output[: array.shape[0], :, index] = array.astype(dtype, copy=False)
    return output


def full_aperture_tables(
    points_m: np.ndarray,
    probe_geometry_m: np.ndarray,
    sound_speed_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    receive_time = np.linalg.norm(
        points_m[:, None, :] - probe_geometry_m[None, :, :], axis=2
    ) / sound_speed_m_s
    aperture = np.ones(receive_time.shape, dtype=np.float32)
    receive_sign = np.ones(receive_time.shape, dtype=np.float32)
    receive_sign[:, : probe_geometry_m.shape[0] // 2] = -1.0
    return receive_time.astype(np.float32), aperture, receive_sign


def reconstruct_iq_methods(
    iq_gpu: Any,
    angles_rad: np.ndarray,
    points_m: np.ndarray,
    probe_geometry_m: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    sampling_frequency_hz: float,
    carrier_frequency_hz: float,
    sound_speed_m_s: float,
    initial_time_s: float,
    nsi_c: float,
    mv_configuration: MvConfiguration,
    cp: Any,
    label: str,
) -> dict[str, np.ndarray]:
    weights, _ = angular_sign_weights(np.rad2deg(angles_rad))
    receive_time, aperture, receive_sign = full_aperture_tables(
        points_m, probe_geometry_m, sound_speed_m_s
    )
    receive_time_gpu = cp.asarray(receive_time)
    aperture_gpu = cp.asarray(aperture)
    receive_sign_gpu = cp.asarray(receive_sign)
    weights_gpu = cp.asarray(weights, dtype=cp.float32)
    point_count = points_m.shape[0]
    uniform = cp.zeros(point_count, dtype=cp.complex64)
    cf_weighted = cp.zeros(point_count, dtype=cp.complex64)
    minimum_variance = cp.zeros(point_count, dtype=cp.complex64)
    receive_null = cp.zeros(point_count, dtype=cp.complex64)
    angle_null = cp.zeros(point_count, dtype=cp.complex64)

    for angle_index, angle_rad in enumerate(angles_rad):
        delayed, mask = focused_samples(
            iq_gpu,
            angle_index,
            float(angle_rad),
            points_m,
            receive_time_gpu,
            aperture_gpu,
            sampling_frequency_hz=sampling_frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
            initial_time_s=initial_time_s,
            carrier_frequency_hz=carrier_frequency_hz,
            cp=cp,
        )
        angle_das = cp.sum(delayed, axis=1)
        uniform += angle_das
        cf_weighted += receive_cf_das(delayed, mask, xp=cp)
        minimum_variance += capon_minimum_variance(
            delayed,
            mask,
            grid_shape=grid_shape,
            configuration=mv_configuration,
            xp=cp,
        )
        receive_null += cp.sum(delayed * receive_sign_gpu, axis=1)
        angle_null += weights_gpu[angle_index] * angle_das
        if (angle_index + 1) % 5 == 0 or angle_index + 1 == len(angles_rad):
            print(f"  {label}: IQ methods {angle_index + 1}/{len(angles_rad)} angles")

    output_gpu = {
        DAS: cp.abs(uniform),
        CF_DAS: cp.abs(cf_weighted),
        MV: cp.abs(minimum_variance),
        RECEIVE_NSI: nsi_envelope(uniform, receive_null, nsi_c, xp=cp),
        ANGLE_NSI: nsi_envelope(uniform, angle_null, nsi_c, xp=cp),
    }
    cp.cuda.Stream.null.synchronize()
    output = {name: cp.asnumpy(values) for name, values in output_gpu.items()}
    del (
        receive_time_gpu,
        aperture_gpu,
        receive_sign_gpu,
        uniform,
        cf_weighted,
        minimum_variance,
        receive_null,
        angle_null,
    )
    cp.get_default_memory_pool().free_all_blocks()
    return output


def reconstruct_dmas(
    raw_gpu: Any,
    angles_rad: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    probe_geometry_m: np.ndarray,
    *,
    sampling_frequency_hz: float,
    sound_speed_m_s: float,
    initial_time_s: float,
    carrier_frequency_hz: float,
    configuration: FdmasFilterConfiguration,
    x_chunk_lines: int,
    cp: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    spacing_m = float(np.median(np.diff(z_m)))
    depth_sampling_hz = sound_speed_m_s / (2.0 * spacing_m)
    coefficients, metadata = design_fdmas_fir(
        carrier_frequency_hz, depth_sampling_hz, configuration
    )
    coefficients_gpu = cp.asarray(coefficients)
    output = np.empty((x_m.size, z_m.size), dtype=np.float32)
    batch_sizes: list[int] = []

    for begin in range(0, x_m.size, x_chunk_lines):
        end = min(begin + x_chunk_lines, x_m.size)
        points_m = regular_points(x_m[begin:end], z_m)
        receive_time, aperture, _ = full_aperture_tables(
            points_m, probe_geometry_m, sound_speed_m_s
        )
        receive_time_gpu = cp.asarray(receive_time)
        aperture_gpu = cp.asarray(aperture)
        pair_sum = cp.zeros(points_m.shape[0], dtype=cp.float32)
        for angle_index, angle_rad in enumerate(angles_rad):
            delayed, mask = focused_samples(
                raw_gpu,
                angle_index,
                float(angle_rad),
                points_m,
                receive_time_gpu,
                aperture_gpu,
                sampling_frequency_hz=sampling_frequency_hz,
                sound_speed_m_s=sound_speed_m_s,
                initial_time_s=initial_time_s,
                carrier_frequency_hz=None,
                cp=cp,
            )
            pair_sum += signed_sqrt_pair_sum(delayed, mask, xp=cp)
        pair_image = pair_sum.reshape(end - begin, z_m.size)
        analytic = fdmas_analytic_image(
            pair_image, coefficients_gpu, depth_axis=1, xp=cp
        )
        line_peaks = cp.max(cp.abs(analytic), axis=1)
        valid = cp.all(cp.isfinite(analytic), axis=1) & (line_peaks > 0.0)
        if not bool(cp.all(valid).item()):
            failed = (np.flatnonzero(~cp.asnumpy(valid)) + begin + 1).tolist()
            raise RuntimeError(f"Simulation DMAS failed for lateral lines {failed}.")
        output[begin:end] = cp.asnumpy(cp.abs(analytic))
        batch_sizes.append(end - begin)
        print(
            f"  simulation: DMAS lateral lines {begin + 1}-{end}/"
            f"{x_m.size} ({len(angles_rad)} angles)"
        )
        del receive_time_gpu, aperture_gpu, pair_sum, pair_image, analytic
        cp.get_default_memory_pool().free_all_blocks()

    metadata.update(
        {
            "lateral_batch_sizes": batch_sizes,
            "full_receive_aperture": True,
            "published_method_label": DMAS,
        }
    )
    return output, metadata


def metric_row(
    method: str,
    image: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    lateral_axis_mm: np.ndarray,
    lateral_profile: np.ndarray,
    lateral_profile_z_mm: float,
) -> dict[str, Any]:
    peak_flat = int(np.argmax(image))
    peak_x_index, peak_z_index = np.unravel_index(peak_flat, image.shape)
    lateral_width = measure_connected_width(
        lateral_axis_mm,
        lateral_profile,
        expected_peak_mm=0.0,
        search_radius_mm=0.5,
    )
    axial_width = measure_connected_width(
        z_mm,
        image[peak_x_index],
        expected_peak_mm=20.0,
        search_radius_mm=0.8,
    )
    center_index = int(np.argmin(np.abs(lateral_axis_mm)))
    local_peak = float(np.max(lateral_profile))
    center_value = float(lateral_profile[center_index])
    return {
        "method": method,
        "lateral_width_mm": lateral_width.width_mm,
        "axial_width_mm": axial_width.width_mm,
        "peak_x_mm": float(x_mm[peak_x_index]),
        "peak_z_mm": float(z_mm[peak_z_index]),
        "lateral_profile_z_mm": float(lateral_profile_z_mm),
        "peak_to_median_background_db": peak_to_background_db(
            image,
            x_mm,
            z_mm,
            target_x_mm=0.0,
            target_z_mm=20.0,
            exclusion_radius_mm=0.35,
        ),
        "lateral_profile_spacing_mm": float(np.median(np.diff(lateral_axis_mm))),
        "lateral_samples_per_minus6_db_width": float(
            lateral_width.width_mm / np.median(np.diff(lateral_axis_mm))
        ),
        "center_to_profile_peak_db": float(
            20.0
            * np.log10(
                max(center_value, np.finfo(float).tiny)
                / max(local_peak, np.finfo(float).tiny)
            )
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plot_figure(
    path: Path,
    images: dict[str, np.ndarray],
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    figure = plt.figure(figsize=(10.7, 8.1), dpi=180, constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.82))
    image_axes = [
        figure.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(3)
    ]
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
    image_artist = None
    for axis, method in zip(image_axes, METHODS):
        image_artist = axis.imshow(
            amplitude_to_db(images[method], floor_db=-60.0).T,
            extent=extent,
            cmap="gray",
            vmin=-50.0,
            vmax=0.0,
            aspect="equal",
            interpolation="nearest",
        )
        axis.plot(0.0, 20.0, "+", color="#ff7f0e", markersize=5)
        axis.set_title(method, fontsize=9)
        axis.set_xlabel("Lateral position [mm]")
        axis.set_ylabel("Depth [mm]")
    assert image_artist is not None
    figure.colorbar(
        image_artist,
        ax=image_axes,
        label="Normalized amplitude [dB]",
        shrink=0.72,
    )

    profile_axis = figure.add_subplot(grid[2, :])
    for method in METHODS:
        axis_mm, profile = profiles[method]
        profile_axis.plot(
            axis_mm,
            amplitude_to_db(profile, floor_db=-80.0),
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.3,
            label=method,
        )
    profile_axis.set_xlim(-0.65, 0.65)
    profile_axis.set_ylim(-50.0, 1.0)
    profile_axis.set_xlabel("Lateral offset [mm]")
    profile_axis.set_ylabel("Normalized amplitude [dB]")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(frameon=False, fontsize=8, ncol=3)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    positive = (
        args.nsi_c,
        args.map_x_half_width_mm,
        args.map_z_half_width_mm,
        args.map_x_spacing_mm,
        args.map_z_spacing_mm,
        args.dmas_z_spacing_mm,
        args.profile_half_width_mm,
        args.profile_spacing_mm,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise SystemExit("All physical and grid arguments must be finite and positive.")
    if args.mv_chunk_pixels < 1 or args.dmas_x_chunk_lines < 1:
        raise SystemExit("MV and DMAS chunk sizes must be positive.")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    declared = {
        "schema_version": 1,
        "methods": list(METHODS),
        "nsi_c": args.nsi_c,
        "angles_deg": np.linspace(-4.0, 4.0, 17).tolist(),
        "target_mm": [0.0, 0.0, 20.0],
        "map_spacing_mm": [args.map_x_spacing_mm, args.map_z_spacing_mm],
        "dmas_z_spacing_mm": args.dmas_z_spacing_mm,
        "profile_spacing_mm": args.profile_spacing_mm,
        "full_receive_aperture": True,
        "cf_definition": "|sum_m x_m|^2/(M sum_m |x_m|^2), applied to receive DAS for each transmit angle",
        "dmas_definition": "Matrone signed-square-root pair sum, band-pass around 2 f0, analytic envelope",
        "mv": asdict(MvConfiguration(chunk_pixels=args.mv_chunk_pixels)),
    }
    if args.metadata_only:
        declared["metadata_only"] = True
        write_json(output / "simulation_six_method_metadata.json", declared)
        print(f"Validated six-method simulation configuration: {output}")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import cupy as cp
    except ImportError as error:
        raise SystemExit("The six-method simulation requires CuPy and an NVIDIA GPU.") from error
    try:
        import pymust
        from mach.io.must import linear_probe_positions
    except ImportError as error:
        raise SystemExit(
            "The six-method simulation requires PyMUST and mach-beamform."
        ) from error
    cp.cuda.Device(0).use()

    start = time.perf_counter()
    target_x_m = 0.0
    target_z_m = 20.0e-3
    param = pymust.getparam("L11-5v")
    angles_deg = np.linspace(-4.0, 4.0, 17)
    angles_rad = np.deg2rad(angles_deg)
    delays = [pymust.txdelay(param, angle) for angle in angles_rad]
    param.fs = 4 * param.fc
    sound_speed = float(param.get("c", 1540.0))
    initial_time = float(param.get("t0", 0.0))
    rf_records = []
    for index, delay in enumerate(delays):
        rf, _ = pymust.simus(
            np.asarray([target_x_m]),
            np.asarray([target_z_m]),
            np.ones(1),
            delay,
            param,
        )
        rf_records.append(np.asarray(rf))
        print(f"  simulation: RF {index + 1}/{len(delays)}")
    iq_records = [
        pymust.rf2iq(rf, param.fs, param.fc).astype(np.complex64)
        for rf in rf_records
    ]
    raw = pad_and_stack(rf_records, dtype=np.float32)
    iq = pad_and_stack(iq_records, dtype=np.complex64)
    probe_geometry = np.asarray(
        linear_probe_positions(param.Nelements, param.pitch), dtype=np.float32
    )
    iq_gpu = cp.asarray(iq)
    raw_gpu = cp.asarray(raw)

    map_x = centred_axis(
        target_x_m,
        args.map_x_half_width_mm * 1e-3,
        args.map_x_spacing_mm * 1e-3,
    ).astype(np.float32)
    map_z = centred_axis(
        target_z_m,
        args.map_z_half_width_mm * 1e-3,
        args.map_z_spacing_mm * 1e-3,
    ).astype(np.float32)
    map_points = regular_points(map_x, map_z)
    mv_configuration = MvConfiguration(
        temporal_half_window_samples=0,
        chunk_pixels=args.mv_chunk_pixels,
    )
    iq_maps_flat = reconstruct_iq_methods(
        iq_gpu,
        angles_rad,
        map_points,
        probe_geometry,
        grid_shape=(map_x.size, map_z.size),
        sampling_frequency_hz=float(param.fs),
        carrier_frequency_hz=float(param.fc),
        sound_speed_m_s=sound_speed,
        initial_time_s=initial_time,
        nsi_c=args.nsi_c,
        mv_configuration=mv_configuration,
        cp=cp,
        label="simulation map",
    )
    images = {
        method: values.reshape(map_x.size, map_z.size)
        for method, values in iq_maps_flat.items()
    }

    filter_configuration = FdmasFilterConfiguration()
    padded_z, _ = fdmas_padded_axis(
        float(map_z[0]),
        float(map_z[-1]),
        args.dmas_z_spacing_mm * 1e-3,
        sound_speed_m_s=sound_speed,
        carrier_frequency_hz=float(param.fc),
        filter_configuration=filter_configuration,
    )
    dmas_source, dmas_metadata = reconstruct_dmas(
        raw_gpu,
        angles_rad,
        map_x,
        padded_z,
        probe_geometry,
        sampling_frequency_hz=float(param.fs),
        sound_speed_m_s=sound_speed,
        initial_time_s=initial_time,
        carrier_frequency_hz=float(param.fc),
        configuration=filter_configuration,
        x_chunk_lines=args.dmas_x_chunk_lines,
        cp=cp,
    )
    images[DMAS] = resample_regular_image(
        dmas_source, map_x, padded_z, map_x, map_z
    )

    profile_x = centred_axis(
        target_x_m,
        args.profile_half_width_mm * 1e-3,
        args.profile_spacing_mm * 1e-3,
    ).astype(np.float32)
    profile_depths_m = {
        method: float(
            map_z[
                np.unravel_index(
                    np.argmax(images[method]), images[method].shape
                )[1]
            ]
        )
        for method in METHODS
    }
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for profile_depth_m in sorted({profile_depths_m[method] for method in IQ_METHODS}):
        profile_points = regular_points(
            profile_x, np.asarray([profile_depth_m], dtype=np.float32)
        )
        fine_profiles_flat = reconstruct_iq_methods(
            iq_gpu,
            angles_rad,
            profile_points,
            probe_geometry,
            grid_shape=(profile_x.size, 1),
            sampling_frequency_hz=float(param.fs),
            carrier_frequency_hz=float(param.fc),
            sound_speed_m_s=sound_speed,
            initial_time_s=initial_time,
            nsi_c=args.nsi_c,
            mv_configuration=mv_configuration,
            cp=cp,
            label=f"simulation lateral profile at {profile_depth_m * 1e3:.3f} mm",
        )
        for method in IQ_METHODS:
            if profile_depths_m[method] == profile_depth_m:
                profiles[method] = (
                    profile_x * 1e3,
                    np.asarray(fine_profiles_flat[method]).reshape(-1),
                )
    dmas_z_index = int(np.argmin(np.abs(map_z - profile_depths_m[DMAS])))
    profiles[DMAS] = (map_x * 1e3, images[DMAS][:, dmas_z_index])

    rows = [
        metric_row(
            method,
            images[method],
            map_x * 1e3,
            map_z * 1e3,
            profiles[method][0],
            profiles[method][1],
            profile_depths_m[method] * 1e3,
        )
        for method in METHODS
    ]
    figure_path = output / "simulation_six_method_comparison.png"
    plot_figure(
        figure_path,
        images,
        map_x * 1e3,
        map_z * 1e3,
        profiles,
    )
    csv_path = output / "simulation_six_method_metrics.csv"
    write_csv(csv_path, rows)
    npz_payload: dict[str, np.ndarray] = {
        "map_x_mm": map_x * 1e3,
        "map_z_mm": map_z * 1e3,
    }
    for method in METHODS:
        key = method.lower().replace("-", "_")
        npz_payload[f"image_{key}"] = images[method]
        npz_payload[f"profile_axis_{key}_mm"] = profiles[method][0]
        npz_payload[f"profile_{key}"] = profiles[method][1]
    np.savez_compressed(output / "simulation_six_method_arrays.npz", **npz_payload)
    summary = {
        **declared,
        "metadata_only": False,
        "publication_ready": True,
        "map_shape": [int(map_x.size), int(map_z.size)],
        "profile_sample_counts": {
            method: int(profiles[method][0].size) for method in METHODS
        },
        "metrics": rows,
        "dmas_filter": dmas_metadata,
        "elapsed_seconds": float(time.perf_counter() - start),
        "artifacts": {
            "figure": figure_path.name,
            "metrics_csv": csv_path.name,
            "arrays_npz": "simulation_six_method_arrays.npz",
        },
    }
    write_json(output / "simulation_six_method_summary.json", summary)
    print(f"Six-method simulation complete: {output}")


if __name__ == "__main__":
    main()

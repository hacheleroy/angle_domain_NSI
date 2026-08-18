#!/usr/bin/env python3
"""Robustness, spatial PSF and two-target validation for angle-domain NSI.

This script addresses item 5 of ``PMB_Angular_NSI_revision_report``.  It uses
the same PyMUST simulation, delayed-data geometry, two-field receive NSI and
angle-domain NSI definitions as the revised manuscript, while adding:

1. angle-count and total-angular-span sweeps
2. a missing positive-angle test, both with raw sign weights and with a
   zero-mean, L1-normalized recentering of those weights
3. additive complex-IQ noise and inter-angle phase-jitter sweeps
4. lateral PSF statistics over several depths and lateral target positions
5. equal-amplitude two-target separation tests using an explicit -6.02 dB
   inter-peak dip criterion

Every quantitative result is saved to CSV and summarized in JSON.  Four
publication-style figures use a consistent method order, colour, line style and
marker.  The default lateral spacing is 0.390625 micrometres, matching the
converged single-target analysis.  Use ``--quick`` only to check installation;
quick-run results are marked non-publication-ready.

Required packages: NumPy, Matplotlib, PyMUST, CuPy and mach-beamform.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HALF_AMPLITUDE_DB = float(20.0 * np.log10(0.5))
METHOD_ORDER = ("DAS", "Conventional NSI", "Angular NSI")
METHOD_STYLES = {
    "DAS": ("tab:purple", "-", "o"),
    "Conventional NSI": ("tab:red", "--", "s"),
    "Angular NSI": ("tab:blue", "-.", "^")
}


def version_or_none(distribution_names: tuple[str, ...]) -> str | None:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def gpu_metadata(cp: Any) -> dict[str, Any]:
    device_id = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace").rstrip("\x00")
    return {
        "name": name,
        "device_id_within_visible_set": device_id,
        "total_global_memory_bytes": int(properties.get("totalGlobalMem", 0)),
        "compute_capability": (
            f"{int(properties.get('major', 0))}."
            f"{int(properties.get('minor', 0))}"
        ),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


def parse_number_list(text: str, *, integers: bool = False) -> list[float] | list[int]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in ("inf", "+inf", "infinity"):
            value = float("inf")
        else:
            value = float(token)
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one comma-separated value is required.")
    if integers:
        if any(not np.isfinite(value) or not float(value).is_integer() for value in values):
            raise argparse.ArgumentTypeError("Every value must be a finite integer.")
        return [int(value) for value in values]
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSI angular-sampling, perturbation and resolvability study"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "NSI_ROBUSTNESS_OUTPUT_DIR",
                Path(__file__).resolve().parents[1]
                / "results"
                / "generated"
                / "robustness",
            )
        ),
    )
    parser.add_argument(
        "--device", default=os.environ.get("NSI_CUDA_DEVICE", "0")
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--probe", default="L11-5v")
    parser.add_argument("--sampling-frequency-multiple", type=float, default=4.0)
    parser.add_argument("--f-number", type=float, default=0.0)
    parser.add_argument("--dc-offset", type=float, default=0.05)
    parser.add_argument("--baseline-angle-count", type=int, default=17)
    parser.add_argument("--baseline-total-span-deg", type=float, default=8.0)
    parser.add_argument("--angle-counts", default="5,9,13,17,25")
    parser.add_argument("--total-spans-deg", default="4,8,12,16")
    parser.add_argument("--missing-positive-angle-deg", type=float, default=2.0)
    parser.add_argument("--noise-snr-db", default="inf,40,30,20,10")
    parser.add_argument("--phase-jitter-sd-deg", default="0,2,5,10,20")
    parser.add_argument("--perturbation-realizations", type=int, default=10)
    parser.add_argument("--depths-mm", default="15,20,25,30")
    parser.add_argument("--lateral-positions-mm", default="-4,0,4")
    parser.add_argument(
        "--two-target-separations-mm",
        default="0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,0.40",
    )
    parser.add_argument("--representative-separation-mm", type=float, default=0.15)
    parser.add_argument("--target-x-mm", type=float, default=0.0)
    parser.add_argument("--target-z-mm", type=float, default=20.0)
    parser.add_argument("--lateral-finest-spacing-mm", type=float, default=0.000390625)
    parser.add_argument("--lateral-half-width-mm", type=float, default=0.75)
    parser.add_argument("--axial-spacing-mm", type=float, default=0.0125)
    parser.add_argument("--axial-search-half-width-mm", type=float, default=1.5)
    parser.add_argument("--skip-angle-sweeps", action="store_true")
    parser.add_argument("--skip-perturbation-sweeps", action="store_true")
    parser.add_argument("--skip-spatial-psf", action="store_true")
    parser.add_argument("--skip-two-target", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small installation check; never use its values in the manuscript.",
    )
    return parser.parse_args()


def configure_args(args: argparse.Namespace) -> None:
    args.angle_counts = parse_number_list(args.angle_counts, integers=True)
    args.total_spans_deg = parse_number_list(args.total_spans_deg)
    args.noise_snr_db = parse_number_list(args.noise_snr_db)
    args.phase_jitter_sd_deg = parse_number_list(args.phase_jitter_sd_deg)
    args.depths_mm = parse_number_list(args.depths_mm)
    args.lateral_positions_mm = parse_number_list(args.lateral_positions_mm)
    args.two_target_separations_mm = parse_number_list(
        args.two_target_separations_mm
    )
    if args.quick:
        args.angle_counts = [5, 9]
        args.total_spans_deg = [4.0, 8.0]
        args.noise_snr_db = [float("inf"), 20.0]
        args.phase_jitter_sd_deg = [0.0, 10.0]
        args.perturbation_realizations = 1
        args.depths_mm = [args.target_z_mm]
        args.lateral_positions_mm = [args.target_x_mm]
        args.two_target_separations_mm = [0.10, 0.20]
        args.lateral_finest_spacing_mm = max(
            args.lateral_finest_spacing_mm, 0.002
        )
        args.lateral_half_width_mm = min(args.lateral_half_width_mm, 0.35)

    if args.baseline_angle_count < 3 or args.baseline_angle_count % 2 == 0:
        raise ValueError("The baseline angle count must be odd and at least three.")
    if any(count < 3 or count % 2 == 0 for count in args.angle_counts):
        raise ValueError("Every angle-count sweep value must be odd and >= 3.")
    if any(value <= 0.0 for value in args.total_spans_deg):
        raise ValueError("Every angular span must be positive.")
    if any(value <= 0.0 for value in args.two_target_separations_mm):
        raise ValueError("Two-target separations must be strictly positive.")
    if args.perturbation_realizations < 1:
        raise ValueError("At least one perturbation realization is required.")
    if args.lateral_finest_spacing_mm <= 0.0 or args.lateral_half_width_mm <= 0.0:
        raise ValueError("The lateral spacing and half-width must be positive.")
    if args.axial_spacing_mm <= 0.0 or args.axial_search_half_width_mm <= 0.0:
        raise ValueError("The axial spacing and search half-width must be positive.")
    if args.dc_offset <= 0.0 or args.sampling_frequency_multiple <= 0.0:
        raise ValueError("The DC offset and sampling multiple must be positive.")


def make_angles(count: int, total_span_deg: float) -> np.ndarray:
    angles = np.linspace(-0.5 * total_span_deg, 0.5 * total_span_deg, count)
    if not np.allclose(angles, -angles[::-1], rtol=0.0, atol=1e-12):
        raise RuntimeError("The requested angle set is not symmetric.")
    if not np.isclose(angles[count // 2], 0.0, atol=1e-12):
        raise RuntimeError("The requested angle set does not contain broadside.")
    return angles.astype(float)


def centered_axis_mm(center_mm: float, half_width_mm: float, spacing_mm: float) -> np.ndarray:
    n_each_side = int(round(half_width_mm / spacing_mm))
    if n_each_side < 2:
        raise ValueError("The requested centered axis contains too few samples.")
    axis = center_mm + np.arange(-n_each_side, n_each_side + 1) * spacing_mm
    if not np.isclose(axis[n_each_side], center_mm, rtol=0.0, atol=1e-14):
        raise RuntimeError("The centered axis misses its requested center.")
    return axis.astype(float)


def amplitude_to_db(profile: np.ndarray, reference: float | None = None) -> np.ndarray:
    values = np.maximum(np.asarray(profile, dtype=float), 0.0)
    if reference is None:
        reference = float(np.max(values))
    if not np.isfinite(reference) or reference <= 0.0:
        return np.full(values.shape, -300.0, dtype=float)
    return 20.0 * np.log10(
        np.maximum(values / reference, np.finfo(float).tiny)
    )


def measure_connected_width(
    axis_mm: np.ndarray,
    profile: np.ndarray,
    level_db: float = HALF_AMPLITUDE_DB,
    peak_index: int | None = None,
) -> dict[str, float | int]:
    axis = np.asarray(axis_mm, dtype=float).reshape(-1)
    values = np.maximum(np.asarray(profile, dtype=float).reshape(-1), 0.0)
    if axis.size != values.size or axis.size < 3:
        raise ValueError("Width axis and profile must have equal length >= 3.")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("The width axis must be strictly increasing.")
    peak_index = int(np.argmax(values)) if peak_index is None else int(peak_index)
    peak = float(values[peak_index])
    if peak <= 0.0:
        raise ValueError("Cannot measure a zero-valued profile.")
    threshold = peak * float(10.0 ** (level_db / 20.0))
    left = peak_index
    while left > 0 and values[left] >= threshold:
        left -= 1
    right = peak_index
    while right < values.size - 1 and values[right] >= threshold:
        right += 1
    if left == 0 and values[left] >= threshold:
        raise ValueError("No left threshold crossing was found.")
    if right == values.size - 1 and values[right] >= threshold:
        raise ValueError("No right threshold crossing was found.")

    def crossing(index_1: int, index_2: int) -> float:
        x1, x2 = float(axis[index_1]), float(axis[index_2])
        y1, y2 = float(values[index_1]), float(values[index_2])
        if y2 == y1:
            return 0.5 * (x1 + x2)
        return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)

    left_crossing = crossing(left, left + 1)
    right_crossing = crossing(right - 1, right)
    return {
        "peak_index": peak_index,
        "peak_x_mm": float(axis[peak_index]),
        "peak_amplitude": peak,
        "level_db": float(level_db),
        "left_crossing_mm": left_crossing,
        "right_crossing_mm": right_crossing,
        "width_mm": float(right_crossing - left_crossing),
    }


def safe_width(
    axis_mm: np.ndarray, profile: np.ndarray, level_db: float, peak_index: int
) -> dict[str, Any]:
    try:
        result = measure_connected_width(axis_mm, profile, level_db, peak_index)
        return {"ok": True, **result}
    except (ValueError, IndexError) as error:
        return {
            "ok": False,
            "width_mm": None,
            "left_crossing_mm": None,
            "right_crossing_mm": None,
            "error": str(error),
        }


def profile_metrics(
    method: str,
    x_axis_mm: np.ndarray,
    profile: np.ndarray,
    expected_x_mm: float,
) -> dict[str, Any]:
    values = np.maximum(np.asarray(profile, dtype=float), 0.0)
    search_half_width = min(
        0.5 * (x_axis_mm[-1] - x_axis_mm[0]), 0.5
    )
    candidates = np.flatnonzero(
        (x_axis_mm >= expected_x_mm - search_half_width)
        & (x_axis_mm <= expected_x_mm + search_half_width)
    )
    if candidates.size == 0:
        raise ValueError("The expected target lies outside the profile axis.")
    peak_index = int(candidates[int(np.argmax(values[candidates]))])
    minus6 = safe_width(x_axis_mm, values, HALF_AMPLITUDE_DB, peak_index)
    minus10 = safe_width(x_axis_mm, values, -10.0, peak_index)
    minus20 = safe_width(x_axis_mm, values, -20.0, peak_index)
    exclusion_half_width = max(
        0.30, 3.0 * (minus6["width_mm"] or 0.0)
    )
    background = values[np.abs(x_axis_mm - x_axis_mm[peak_index]) >= exclusion_half_width]
    background_rms = (
        float(np.sqrt(np.mean(background**2))) if background.size else None
    )
    peak = float(values[peak_index])
    peak_to_background_db = (
        float(20.0 * np.log10(peak / max(background_rms, np.finfo(float).tiny)))
        if background_rms is not None and peak > 0.0
        else None
    )
    spacing_mm = float(np.median(np.diff(x_axis_mm)))
    width = minus6["width_mm"]
    return {
        "method": method,
        "measurement_ok": bool(minus6["ok"]),
        "peak_x_mm": float(x_axis_mm[peak_index]),
        "peak_x_bias_mm": float(x_axis_mm[peak_index] - expected_x_mm),
        "peak_amplitude": peak,
        "background_rms": background_rms,
        "peak_to_background_db": peak_to_background_db,
        "lateral_fwhm_mm": width,
        "lateral_width_minus10_db_mm": minus10["width_mm"],
        "lateral_width_minus20_db_mm": minus20["width_mm"],
        "samples_per_fwhm": float(width / spacing_mm) if width is not None else None,
        "grid_spacing_mm": spacing_mm,
        "measurement_error": minus6.get("error"),
    }


def two_target_metrics(
    x_axis_mm: np.ndarray,
    profile: np.ndarray,
    left_expected_mm: float,
    right_expected_mm: float,
) -> dict[str, Any]:
    values = np.maximum(np.asarray(profile, dtype=float), 0.0)
    midpoint = 0.5 * (left_expected_mm + right_expected_mm)
    left_indices = np.flatnonzero(x_axis_mm < midpoint)
    right_indices = np.flatnonzero(x_axis_mm >= midpoint)
    if left_indices.size == 0 or right_indices.size == 0:
        raise ValueError("The two-target axis does not straddle the midpoint.")
    left_peak_index = int(
        left_indices[int(np.argmax(values[left_indices]))]
    )
    right_peak_index = int(
        right_indices[int(np.argmax(values[right_indices]))]
    )
    if left_peak_index >= right_peak_index:
        raise RuntimeError("Two-target peak ordering failed.")
    left_peak = float(values[left_peak_index])
    right_peak = float(values[right_peak_index])
    weaker_peak = min(left_peak, right_peak)
    valley_index = left_peak_index + int(
        np.argmin(values[left_peak_index : right_peak_index + 1])
    )
    valley = float(values[valley_index])
    valley_dip_db = float(
        20.0
        * np.log10(
            max(valley, np.finfo(float).tiny)
            / max(weaker_peak, np.finfo(float).tiny)
        )
    )
    expected_separation = right_expected_mm - left_expected_mm
    spacing = float(np.median(np.diff(x_axis_mm)))
    localization_tolerance = max(4.0 * spacing, 0.40 * expected_separation)
    localized = bool(
        abs(x_axis_mm[left_peak_index] - left_expected_mm)
        <= localization_tolerance
        and abs(x_axis_mm[right_peak_index] - right_expected_mm)
        <= localization_tolerance
    )
    return {
        "left_peak_x_mm": float(x_axis_mm[left_peak_index]),
        "right_peak_x_mm": float(x_axis_mm[right_peak_index]),
        "apparent_peak_separation_mm": float(
            x_axis_mm[right_peak_index] - x_axis_mm[left_peak_index]
        ),
        "left_peak_amplitude": left_peak,
        "right_peak_amplitude": right_peak,
        "valley_x_mm": float(x_axis_mm[valley_index]),
        "valley_amplitude": valley,
        "valley_dip_relative_to_weaker_peak_db": valley_dip_db,
        "peaks_localized_near_expected_targets": localized,
        "resolved_by_minus6db_dip": bool(
            localized and valley_dip_db <= HALF_AMPLITUDE_DB
        ),
        "resolution_criterion_db": HALF_AMPLITUDE_DB,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "infinity" if np.isposinf(value) else None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if value is None else value
                    for key, value in row.items()
                }
            )


def verify_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Expected output was not written correctly: {path}")
    print(f"Saved: {path.resolve()}")


def aggregate_rows(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...], value_key: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or not np.isfinite(float(value)):
            continue
        key = tuple(row.get(item) for item in group_keys)
        groups.setdefault(key, []).append(float(value))
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        array = np.asarray(values, dtype=float)
        result = {name: item for name, item in zip(group_keys, key)}
        result.update(
            {
                "metric": value_key,
                "n": int(array.size),
                "mean": float(np.mean(array)),
                "sample_sd": float(np.std(array, ddof=1))
                if array.size > 1
                else 0.0,
                "median": float(np.median(array)),
                "q1": float(np.percentile(array, 25.0)),
                "q3": float(np.percentile(array, 75.0)),
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
            }
        )
        output.append(result)
    return output


class NSISimulator:
    def __init__(
        self,
        args: argparse.Namespace,
        cp: Any,
        pymust: Any,
        wavefront: Any,
        linear_probe_positions: Any,
        scan_grid: Any,
        beamform: Any,
    ) -> None:
        self.args = args
        self.cp = cp
        self.pymust = pymust
        self.wavefront = wavefront
        self.linear_probe_positions = linear_probe_positions
        self.scan_grid = scan_grid
        self.beamform = beamform
        self.param = pymust.getparam(args.probe)
        self.param.fs = args.sampling_frequency_multiple * self.param.fc
        self.sound_speed = float(self.param.get("c", 1540.0))
        self.rx_start_s = float(self.param.get("t0", 0.0))
        self.element_count = int(self.param.Nelements)
        if self.element_count % 2:
            raise ValueError("Receive-domain NSI requires an even element count.")
        positions = linear_probe_positions(
            self.element_count, float(self.param.pitch)
        )
        self.receive_coordinates_gpu = cp.asarray(
            positions, dtype=cp.float32
        )
        self.receive_null = np.ones(self.element_count, dtype=np.float32)
        self.receive_null[: self.element_count // 2] = -1.0
        self.iq_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def parameter_metadata(self) -> dict[str, Any]:
        return {
            "probe": self.args.probe,
            "receive_elements": self.element_count,
            "pitch_mm": float(self.param.pitch * 1e3),
            "centre_frequency_mhz": float(self.param.fc * 1e-6),
            "sampling_frequency_mhz": float(self.param.fs * 1e-6),
            "sound_speed_m_s": self.sound_speed,
            "rx_start_s": self.rx_start_s,
            "f_number": self.args.f_number,
            "dc_offset": self.args.dc_offset,
            "precision": {
                "IQ": "complex64",
                "coordinates": "float32",
                "accumulators": "complex64",
            },
        }

    @staticmethod
    def scatterer_key(scatterers: list[tuple[float, float, float]]) -> tuple[Any, ...]:
        return tuple(
            (round(x_mm, 9), round(z_mm, 9), round(amplitude, 9))
            for x_mm, z_mm, amplitude in scatterers
        )

    def simulate_iq(
        self,
        scatterers: list[tuple[float, float, float]],
        angles_deg: np.ndarray,
        *,
        cache: bool = True,
    ) -> list[np.ndarray]:
        x_m = np.asarray([item[0] for item in scatterers], dtype=float) * 1e-3
        z_m = np.asarray([item[1] for item in scatterers], dtype=float) * 1e-3
        reflectivity = np.asarray([item[2] for item in scatterers], dtype=float)
        scatterer_key = self.scatterer_key(scatterers)
        output: list[np.ndarray] = []
        for angle_deg in angles_deg:
            key = (scatterer_key, round(float(angle_deg), 9))
            if cache and key in self.iq_cache:
                iq_angle = self.iq_cache[key]
            else:
                delay = self.pymust.txdelay(
                    self.param, np.deg2rad(float(angle_deg))
                )
                rf_data, _ = self.pymust.simus(
                    x_m, z_m, reflectivity, delay, self.param
                )
                iq_angle = self.pymust.rf2iq(
                    rf_data, self.param.fs, self.param.fc
                ).astype(np.complex64)
                if cache:
                    self.iq_cache[key] = iq_angle
            output.append(iq_angle)
        return output

    @staticmethod
    def perturb_iq(
        iq_list: list[np.ndarray],
        noise_snr_db: float,
        phase_jitter_sd_deg: float,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        count = sum(array.size for array in iq_list)
        total_energy = sum(
            float(
                np.sum(
                    array.real.astype(np.float64) ** 2
                    + array.imag.astype(np.float64) ** 2
                )
            )
            for array in iq_list
        )
        signal_rms = float(np.sqrt(total_energy / max(count, 1)))
        noise_rms = (
            0.0
            if np.isinf(noise_snr_db)
            else signal_rms / float(10.0 ** (noise_snr_db / 20.0))
        )
        phase_errors = rng.normal(
            0.0, phase_jitter_sd_deg, size=len(iq_list)
        )
        output: list[np.ndarray] = []
        for iq, phase_error_deg in zip(iq_list, phase_errors):
            modified = iq * np.exp(1j * np.deg2rad(phase_error_deg))
            if noise_rms > 0.0:
                noise = (
                    rng.standard_normal(iq.shape, dtype=np.float32)
                    + 1j * rng.standard_normal(iq.shape, dtype=np.float32)
                ) * np.float32(noise_rms / np.sqrt(2.0))
                modified = modified + noise
            output.append(np.ascontiguousarray(modified.astype(np.complex64)))
        return output

    def reconstruct(
        self,
        iq_list: list[np.ndarray],
        angles_deg: np.ndarray,
        x_axis_mm: np.ndarray,
        z_axis_mm: np.ndarray,
        *,
        angular_weights: np.ndarray | None = None,
        noise_snr_db: float = float("inf"),
        phase_jitter_sd_deg: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        if len(iq_list) != len(angles_deg):
            raise ValueError("The IQ and angle counts differ.")
        if rng is None:
            rng = np.random.default_rng(self.args.seed)
        if noise_snr_db != float("inf") or phase_jitter_sd_deg != 0.0:
            effective_iq = self.perturb_iq(
                iq_list, noise_snr_db, phase_jitter_sd_deg, rng
            )
        else:
            effective_iq = iq_list

        if angular_weights is None:
            angular_weights = np.sign(angles_deg)
        angular_weights = np.asarray(angular_weights, dtype=np.float32).reshape(-1)
        if angular_weights.size != angles_deg.size:
            raise ValueError("The angular weight vector has the wrong length.")

        grid_points = np.asarray(
            self.scan_grid(
                np.asarray(x_axis_mm, dtype=float) * 1e-3,
                np.array([0.0]),
                np.asarray(z_axis_mm, dtype=float) * 1e-3,
            ),
            dtype=np.float32,
        )
        scan_gpu = self.cp.asarray(grid_points, dtype=self.cp.float32)
        shape = (len(x_axis_mm), len(z_axis_mm))
        uniform_sum = self.cp.zeros(shape, dtype=self.cp.complex64)
        receive_null_sum = self.cp.zeros(shape, dtype=self.cp.complex64)
        angular_null_sum = self.cp.zeros(shape, dtype=self.cp.complex64)

        for angle_index, (angle_deg, iq_angle) in enumerate(
            zip(angles_deg, effective_iq)
        ):
            angle_rad = np.deg2rad(float(angle_deg))
            direction = np.asarray(
                [np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=np.float32
            )
            arrivals = (
                self.wavefront.plane(
                    origin_m=np.zeros(3, dtype=np.float32),
                    points_m=grid_points,
                    direction=direction,
                )
                / self.sound_speed
            )
            arrivals_gpu = self.cp.asarray(arrivals, dtype=self.cp.float32)
            frames = np.stack(
                [iq_angle, iq_angle * self.receive_null], axis=-1
            ).astype(np.complex64)
            channel_gpu = self.cp.asarray(
                np.ascontiguousarray(frames.transpose(1, 0, 2))
            )
            result = self.beamform(
                channel_data=channel_gpu,
                rx_coords_m=self.receive_coordinates_gpu,
                scan_coords_m=scan_gpu,
                tx_wave_arrivals_s=arrivals_gpu,
                f_number=self.args.f_number,
                rx_start_s=self.rx_start_s,
                sampling_freq_hz=float(self.param.fs),
                sound_speed_m_s=self.sound_speed,
                modulation_freq_hz=float(self.param.fc),
                tukey_alpha=0.0,
            )
            uniform_image = result[:, 0].reshape(shape)
            receive_null_image = result[:, 1].reshape(shape)
            uniform_sum += uniform_image
            receive_null_sum += receive_null_image
            angular_null_sum += angular_weights[angle_index] * uniform_image

        receive_plus = receive_null_sum + self.args.dc_offset * uniform_sum
        receive_minus = -receive_null_sum + self.args.dc_offset * uniform_sum
        receive_envelope = self.cp.maximum(
            0.5 * (self.cp.abs(receive_plus) + self.cp.abs(receive_minus))
            - self.cp.abs(receive_null_sum),
            0.0,
        )
        angle_plus = angular_null_sum + self.args.dc_offset * uniform_sum
        angle_minus = -angular_null_sum + self.args.dc_offset * uniform_sum
        angle_envelope = self.cp.maximum(
            0.5 * (self.cp.abs(angle_plus) + self.cp.abs(angle_minus))
            - self.cp.abs(angular_null_sum),
            0.0,
        )
        self.cp.cuda.Stream.null.synchronize()
        output = {
            "DAS": self.cp.asnumpy(self.cp.abs(uniform_sum)),
            "Conventional NSI": self.cp.asnumpy(receive_envelope),
            "Angular NSI": self.cp.asnumpy(angle_envelope),
        }
        del (
            scan_gpu,
            uniform_sum,
            receive_null_sum,
            angular_null_sum,
            receive_plus,
            receive_minus,
            receive_envelope,
            angle_plus,
            angle_minus,
            angle_envelope,
        )
        self.cp.get_default_memory_pool().free_all_blocks()
        return output


def locate_axial_peaks(
    simulator: NSISimulator,
    iq_list: list[np.ndarray],
    angles_deg: np.ndarray,
    target_x_mm: float,
    target_z_mm: float,
) -> dict[str, float]:
    z_axis = centered_axis_mm(
        target_z_mm,
        simulator.args.axial_search_half_width_mm,
        simulator.args.axial_spacing_mm,
    )
    envelopes = simulator.reconstruct(
        iq_list,
        angles_deg,
        np.asarray([target_x_mm], dtype=float),
        z_axis,
    )
    return {
        method: float(z_axis[int(np.argmax(envelopes[method][0, :]))])
        for method in METHOD_ORDER
    }


def evaluate_lateral_profiles(
    simulator: NSISimulator,
    iq_list: list[np.ndarray],
    angles_deg: np.ndarray,
    target_x_mm: float,
    z_by_method_mm: dict[str, float],
    metadata: dict[str, Any],
    *,
    angular_weights: np.ndarray | None = None,
    noise_snr_db: float = float("inf"),
    phase_jitter_sd_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    x_axis = centered_axis_mm(
        target_x_mm,
        simulator.args.lateral_half_width_mm,
        simulator.args.lateral_finest_spacing_mm,
    )
    z_axis = np.unique(
        np.asarray([z_by_method_mm[method] for method in METHOD_ORDER])
    )
    envelopes = simulator.reconstruct(
        iq_list,
        angles_deg,
        x_axis,
        z_axis,
        angular_weights=angular_weights,
        noise_snr_db=noise_snr_db,
        phase_jitter_sd_deg=phase_jitter_sd_deg,
        rng=rng,
    )
    rows: list[dict[str, Any]] = []
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in METHOD_ORDER:
        z_index = int(np.argmin(np.abs(z_axis - z_by_method_mm[method])))
        profile = np.maximum(envelopes[method][:, z_index], 0.0)
        row = dict(metadata)
        row.update(
            {
                "angle_count": int(angles_deg.size),
                "minimum_angle_deg": float(np.min(angles_deg)),
                "maximum_angle_deg": float(np.max(angles_deg)),
                "total_angle_span_deg": float(np.ptp(angles_deg)),
                "evaluation_z_mm": float(z_axis[z_index]),
                **profile_metrics(method, x_axis, profile, target_x_mm),
            }
        )
        rows.append(row)
        profiles[method] = (x_axis.copy(), profile.copy())
    return rows, profiles


def recentered_missing_angle_weights(angles_deg: np.ndarray) -> np.ndarray:
    raw = np.sign(angles_deg).astype(float)
    centered = raw - np.mean(raw)
    raw_l1 = float(np.sum(np.abs(raw)))
    centered_l1 = float(np.sum(np.abs(centered)))
    if centered_l1 <= 0.0:
        raise ValueError("Cannot recenter a zero angular-weight vector.")
    return (centered * raw_l1 / centered_l1).astype(np.float32)


def plot_angle_sweeps(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "angle_count", "angle_count", "Number of angles"),
        (axes[0, 1], "angle_span", "total_angle_span_deg", "Total angular span [deg]"),
    )
    for axis, scenario_type, x_key, x_label in panels:
        for method in METHOD_ORDER:
            selected = [
                row
                for row in rows
                if row.get("scenario_type") == scenario_type
                and row["method"] == method
                and row.get("lateral_fwhm_mm") is not None
            ]
            selected.sort(key=lambda row: float(row[x_key]))
            if not selected:
                continue
            color, line, marker = METHOD_STYLES[method]
            axis.plot(
                [row[x_key] for row in selected],
                [row["lateral_fwhm_mm"] for row in selected],
                color=color,
                linestyle=line,
                marker=marker,
                label=method,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Lateral -6.02 dB width [mm]")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")

    missing = [row for row in rows if row.get("scenario_type") == "missing_angle"]
    labels = []
    for scenario in ("symmetric baseline", "missing: raw sign", "missing: recentered"):
        if any(row.get("scenario_label") == scenario for row in missing):
            labels.append(scenario)
    positions = np.arange(len(labels))
    width_axis, pbr_axis = axes[1]
    for method_index, method in enumerate(METHOD_ORDER):
        color, line, marker = METHOD_STYLES[method]
        method_rows = {
            row["scenario_label"]: row
            for row in missing
            if row["method"] == method
        }
        widths = [method_rows[label].get("lateral_fwhm_mm") for label in labels]
        pbr = [method_rows[label].get("peak_to_background_db") for label in labels]
        offset = (method_index - 1) * 0.18
        width_axis.plot(
            positions + offset,
            widths,
            color=color,
            linestyle="none",
            marker=marker,
            label=method,
        )
        pbr_axis.plot(
            positions + offset,
            pbr,
            color=color,
            linestyle="none",
            marker=marker,
            label=method,
        )
    for axis, ylabel in (
        (width_axis, "Lateral -6.02 dB width [mm]"),
        (pbr_axis, "Peak-to-background ratio [dB]"),
    ):
        axis.set_xticks(positions, labels, rotation=14, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    width_axis.set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_perturbations(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, kind, level_key, xlabel in (
        (axes[0], "noise", "noise_snr_db", "Complex-IQ SNR [dB]"),
        (axes[1], "phase", "phase_jitter_sd_deg", "Inter-angle phase-jitter SD [deg]"),
    ):
        selected_kind = [row for row in rows if row["perturbation_type"] == kind]
        levels = []
        for row in selected_kind:
            value = row[level_key]
            if value not in levels:
                levels.append(value)
        if kind == "noise":
            levels.sort(key=lambda value: float("inf") if value == "infinity" else float(value), reverse=True)
            x_values = np.arange(len(levels))
        else:
            levels.sort(key=float)
            x_values = np.asarray([float(value) for value in levels])
        for method in METHOD_ORDER:
            means, deviations = [], []
            for level in levels:
                values = [
                    float(row["lateral_fwhm_mm"])
                    for row in selected_kind
                    if row["method"] == method
                    and row[level_key] == level
                    and row.get("lateral_fwhm_mm") is not None
                ]
                means.append(float(np.mean(values)) if values else np.nan)
                deviations.append(
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
            color, line, marker = METHOD_STYLES[method]
            axis.errorbar(
                x_values,
                means,
                yerr=deviations,
                color=color,
                linestyle=line,
                marker=marker,
                capsize=3,
                label=method,
            )
        if kind == "noise":
            axis.set_xticks(x_values, ["∞" if value == "infinity" else f"{float(value):g}" for value in levels])
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Lateral -6.02 dB width [mm]")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_psf(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    depths = sorted({float(row["target_z_mm"]) for row in rows})
    for method in METHOD_ORDER:
        means, deviations = [], []
        for depth in depths:
            values = [
                float(row["lateral_fwhm_mm"])
                for row in rows
                if row["method"] == method
                and float(row["target_z_mm"]) == depth
                and row.get("lateral_fwhm_mm") is not None
            ]
            means.append(float(np.mean(values)) if values else np.nan)
            deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        color, line, marker = METHOD_STYLES[method]
        axis.errorbar(
            depths,
            means,
            yerr=deviations,
            color=color,
            linestyle=line,
            marker=marker,
            capsize=4,
            label=method,
        )
    axis.set_xlabel("Target depth [mm]")
    axis.set_ylabel("Lateral -6.02 dB width [mm]")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_two_target(
    path: Path,
    result_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    representative_separation_mm: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for method in METHOD_ORDER:
        selected = sorted(
            [row for row in result_rows if row["method"] == method],
            key=lambda row: float(row["target_separation_mm"]),
        )
        color, line, marker = METHOD_STYLES[method]
        axes[0].plot(
            [row["target_separation_mm"] for row in selected],
            [row["valley_dip_relative_to_weaker_peak_db"] for row in selected],
            color=color,
            linestyle=line,
            marker=marker,
            label=method,
        )
    axes[0].axhline(HALF_AMPLITUDE_DB, color="0.35", linestyle=":")
    axes[0].set_xlabel("True target separation [mm]")
    axes[0].set_ylabel("Inter-peak valley relative to weaker peak [dB]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    available = sorted({float(row["target_separation_mm"]) for row in profile_rows})
    representative = min(
        available, key=lambda value: abs(value - representative_separation_mm)
    )
    for method in METHOD_ORDER:
        selected = [
            row
            for row in profile_rows
            if row["method"] == method
            and np.isclose(float(row["target_separation_mm"]), representative)
        ]
        color, line, marker = METHOD_STYLES[method]
        axes[1].plot(
            [row["x_mm"] for row in selected],
            [row["normalized_envelope_db"] for row in selected],
            color=color,
            linestyle=line,
            label=method,
        )
    axes[1].set_xlim(-max(0.35, representative), max(0.35, representative))
    axes[1].set_ylim(-40.0, 1.0)
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("Normalized envelope [dB]")
    axes[1].set_title(f"Two targets separated by {representative:g} mm")
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import cupy as cp
        import pymust
        from mach import wavefront
        from mach.io.must import linear_probe_positions, scan_grid
        from mach.kernel import beamform
    except ImportError as error:
        raise SystemExit(
            "simulation_robustness.py requires PyMUST, CuPy and "
            f"mach-beamform. Import failed: {error}"
        ) from error

    cp.cuda.Device(0).use()
    simulator = NSISimulator(
        args,
        cp,
        pymust,
        wavefront,
        linear_probe_positions,
        scan_grid,
        beamform,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_angles = make_angles(
        args.baseline_angle_count, args.baseline_total_span_deg
    )
    baseline_scatterers = [(args.target_x_mm, args.target_z_mm, 1.0)]
    baseline_iq = simulator.simulate_iq(baseline_scatterers, baseline_angles)
    baseline_z_by_method = locate_axial_peaks(
        simulator,
        baseline_iq,
        baseline_angles,
        args.target_x_mm,
        args.target_z_mm,
    )
    print(f"Baseline axial peak locations [mm]: {baseline_z_by_method}")

    angle_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    two_target_rows: list[dict[str, Any]] = []
    two_target_profile_rows: list[dict[str, Any]] = []

    if not args.skip_angle_sweeps:
        for count in args.angle_counts:
            angles = make_angles(count, args.baseline_total_span_deg)
            iq = simulator.simulate_iq(baseline_scatterers, angles)
            rows, _ = evaluate_lateral_profiles(
                simulator,
                iq,
                angles,
                args.target_x_mm,
                baseline_z_by_method,
                {
                    "scenario_type": "angle_count",
                    "scenario_label": f"{count} angles",
                    "angular_weight_strategy": "sign(angle)",
                    "angular_weight_sum": float(np.sum(np.sign(angles))),
                },
            )
            angle_rows.extend(rows)
            print(f"Completed angle-count scenario: {count}")

        for span in args.total_spans_deg:
            angles = make_angles(args.baseline_angle_count, span)
            iq = simulator.simulate_iq(baseline_scatterers, angles)
            rows, _ = evaluate_lateral_profiles(
                simulator,
                iq,
                angles,
                args.target_x_mm,
                baseline_z_by_method,
                {
                    "scenario_type": "angle_span",
                    "scenario_label": f"{span:g} deg total span",
                    "angular_weight_strategy": "sign(angle)",
                    "angular_weight_sum": float(np.sum(np.sign(angles))),
                },
            )
            angle_rows.extend(rows)
            print(f"Completed angle-span scenario: {span:g} deg")

        baseline_rows, _ = evaluate_lateral_profiles(
            simulator,
            baseline_iq,
            baseline_angles,
            args.target_x_mm,
            baseline_z_by_method,
            {
                "scenario_type": "missing_angle",
                "scenario_label": "symmetric baseline",
                "angular_weight_strategy": "sign(angle)",
                "angular_weight_sum": float(np.sum(np.sign(baseline_angles))),
            },
        )
        angle_rows.extend(baseline_rows)
        positive_target = abs(args.missing_positive_angle_deg)
        positive_indices = np.flatnonzero(baseline_angles > 0.0)
        missing_index = int(
            positive_indices[
                np.argmin(np.abs(baseline_angles[positive_indices] - positive_target))
            ]
        )
        missing_angles = np.delete(baseline_angles, missing_index)
        missing_iq = [
            iq for index, iq in enumerate(baseline_iq) if index != missing_index
        ]
        raw_missing_weights = np.sign(missing_angles).astype(np.float32)
        recentered_weights = recentered_missing_angle_weights(missing_angles)
        for label, strategy, weights in (
            ("missing: raw sign", "sign(angle), not zero mean", raw_missing_weights),
            (
                "missing: recentered",
                "zero-mean, L1-normalized recentering",
                recentered_weights,
            ),
        ):
            rows, _ = evaluate_lateral_profiles(
                simulator,
                missing_iq,
                missing_angles,
                args.target_x_mm,
                baseline_z_by_method,
                {
                    "scenario_type": "missing_angle",
                    "scenario_label": label,
                    "removed_angle_deg": float(baseline_angles[missing_index]),
                    "angular_weight_strategy": strategy,
                    "angular_weight_sum": float(np.sum(weights)),
                    "angular_weight_l1_norm": float(np.sum(np.abs(weights))),
                },
                angular_weights=weights,
            )
            angle_rows.extend(rows)
        print(
            "Completed missing-angle scenarios; removed "
            f"{baseline_angles[missing_index]:g} deg"
        )

    if not args.skip_perturbation_sweeps:
        for level_index, snr_db in enumerate(args.noise_snr_db):
            snr_label: float | str = (
                "infinity" if np.isinf(snr_db) else float(snr_db)
            )
            for realization in range(1, args.perturbation_realizations + 1):
                rng = np.random.default_rng(
                    args.seed + 100000 + 1000 * level_index + realization
                )
                rows, _ = evaluate_lateral_profiles(
                    simulator,
                    baseline_iq,
                    baseline_angles,
                    args.target_x_mm,
                    baseline_z_by_method,
                    {
                        "perturbation_type": "noise",
                        "noise_snr_db": snr_label,
                        "phase_jitter_sd_deg": 0.0,
                        "realization": realization,
                    },
                    noise_snr_db=snr_db,
                    rng=rng,
                )
                perturbation_rows.extend(rows)
            print(f"Completed noise level: {snr_label} dB")

        for level_index, phase_sd in enumerate(args.phase_jitter_sd_deg):
            for realization in range(1, args.perturbation_realizations + 1):
                rng = np.random.default_rng(
                    args.seed + 200000 + 1000 * level_index + realization
                )
                rows, _ = evaluate_lateral_profiles(
                    simulator,
                    baseline_iq,
                    baseline_angles,
                    args.target_x_mm,
                    baseline_z_by_method,
                    {
                        "perturbation_type": "phase",
                        "noise_snr_db": "infinity",
                        "phase_jitter_sd_deg": float(phase_sd),
                        "realization": realization,
                    },
                    phase_jitter_sd_deg=float(phase_sd),
                    rng=rng,
                )
                perturbation_rows.extend(rows)
            print(f"Completed phase-jitter level: {phase_sd:g} deg")

    if not args.skip_spatial_psf:
        for depth_mm in args.depths_mm:
            for lateral_mm in args.lateral_positions_mm:
                scatterers = [(float(lateral_mm), float(depth_mm), 1.0)]
                iq = simulator.simulate_iq(
                    scatterers, baseline_angles, cache=False
                )
                z_by_method = locate_axial_peaks(
                    simulator,
                    iq,
                    baseline_angles,
                    float(lateral_mm),
                    float(depth_mm),
                )
                rows, _ = evaluate_lateral_profiles(
                    simulator,
                    iq,
                    baseline_angles,
                    float(lateral_mm),
                    z_by_method,
                    {
                        "scenario_type": "spatial_psf",
                        "target_x_mm": float(lateral_mm),
                        "target_z_mm": float(depth_mm),
                    },
                )
                spatial_rows.extend(rows)
                print(
                    "Completed spatial PSF target: "
                    f"x={lateral_mm:g} mm, z={depth_mm:g} mm"
                )

    if not args.skip_two_target:
        maximum_separation = max(args.two_target_separations_mm)
        two_target_half_width = max(
            args.lateral_half_width_mm, 0.5 * maximum_separation + 0.50
        )
        x_axis = centered_axis_mm(
            args.target_x_mm,
            two_target_half_width,
            args.lateral_finest_spacing_mm,
        )
        z_axis = np.unique(
            np.asarray([baseline_z_by_method[method] for method in METHOD_ORDER])
        )
        for separation_mm in args.two_target_separations_mm:
            left_x = args.target_x_mm - 0.5 * separation_mm
            right_x = args.target_x_mm + 0.5 * separation_mm
            scatterers = [
                (left_x, args.target_z_mm, 1.0),
                (right_x, args.target_z_mm, 1.0),
            ]
            iq = simulator.simulate_iq(
                scatterers, baseline_angles, cache=False
            )
            envelopes = simulator.reconstruct(
                iq, baseline_angles, x_axis, z_axis
            )
            for method in METHOD_ORDER:
                z_index = int(
                    np.argmin(np.abs(z_axis - baseline_z_by_method[method]))
                )
                profile = np.maximum(envelopes[method][:, z_index], 0.0)
                metrics = two_target_metrics(
                    x_axis, profile, left_x, right_x
                )
                two_target_rows.append(
                    {
                        "target_separation_mm": float(separation_mm),
                        "target_left_x_mm": float(left_x),
                        "target_right_x_mm": float(right_x),
                        "target_z_mm": float(args.target_z_mm),
                        "evaluation_z_mm": float(z_axis[z_index]),
                        "method": method,
                        **metrics,
                    }
                )
                profile_db = amplitude_to_db(profile)
                for x_mm, value_db in zip(x_axis, profile_db):
                    two_target_profile_rows.append(
                        {
                            "target_separation_mm": float(separation_mm),
                            "method": method,
                            "x_mm": float(x_mm),
                            "normalized_envelope_db": float(value_db),
                        }
                    )
            print(f"Completed two-target separation: {separation_mm:g} mm")

    output_files: dict[str, str] = {}
    if angle_rows:
        path = output_dir / "robustness_angle_sweeps.csv"
        write_csv(path, angle_rows)
        verify_file(path)
        output_files["angle_sweeps_csv"] = path.name
        figure_path = output_dir / "robustness_angle_sweeps.png"
        plot_angle_sweeps(figure_path, angle_rows)
        verify_file(figure_path)
        output_files["angle_sweeps_figure"] = figure_path.name

    if perturbation_rows:
        path = output_dir / "robustness_noise_phase_sweeps.csv"
        write_csv(path, perturbation_rows)
        verify_file(path)
        output_files["perturbation_sweeps_csv"] = path.name
        figure_path = output_dir / "robustness_noise_phase_sweeps.png"
        plot_perturbations(figure_path, perturbation_rows)
        verify_file(figure_path)
        output_files["perturbation_sweeps_figure"] = figure_path.name

    if spatial_rows:
        path = output_dir / "robustness_spatial_psf.csv"
        write_csv(path, spatial_rows)
        verify_file(path)
        output_files["spatial_psf_csv"] = path.name
        figure_path = output_dir / "robustness_spatial_psf.png"
        plot_spatial_psf(figure_path, spatial_rows)
        verify_file(figure_path)
        output_files["spatial_psf_figure"] = figure_path.name

    if two_target_rows:
        result_path = output_dir / "robustness_two_target_resolvability.csv"
        profile_path = output_dir / "robustness_two_target_profiles.csv"
        write_csv(result_path, two_target_rows)
        write_csv(profile_path, two_target_profile_rows)
        verify_file(result_path)
        verify_file(profile_path)
        output_files["two_target_results_csv"] = result_path.name
        output_files["two_target_profiles_csv"] = profile_path.name
        figure_path = output_dir / "robustness_two_target_resolvability.png"
        plot_two_target(
            figure_path,
            two_target_rows,
            two_target_profile_rows,
            args.representative_separation_mm,
        )
        verify_file(figure_path)
        output_files["two_target_figure"] = figure_path.name

    angle_summary = aggregate_rows(
        angle_rows,
        ("scenario_type", "scenario_label", "method"),
        "lateral_fwhm_mm",
    )
    perturbation_summary = aggregate_rows(
        perturbation_rows,
        (
            "perturbation_type",
            "noise_snr_db",
            "phase_jitter_sd_deg",
            "method",
        ),
        "lateral_fwhm_mm",
    )
    spatial_summary = aggregate_rows(
        spatial_rows, ("target_z_mm", "method"), "lateral_fwhm_mm"
    )
    minimum_resolved_separation: dict[str, float | None] = {}
    for method in METHOD_ORDER:
        resolved = [
            float(row["target_separation_mm"])
            for row in two_target_rows
            if row["method"] == method and row["resolved_by_minus6db_dip"]
        ]
        minimum_resolved_separation[method] = min(resolved) if resolved else None

    all_sections_completed = bool(
        angle_rows and perturbation_rows and spatial_rows and two_target_rows
    )
    publication_ready = bool(
        not args.quick
        and all_sections_completed
        and args.perturbation_realizations >= 10
        and args.lateral_finest_spacing_mm <= 0.000390625 + 1e-15
    )
    summary = {
        "analysis_version": "1",
        "publication_ready_minimum_design_check": publication_ready,
        "warning": None
        if publication_ready
        else (
            "One or more full-design requirements were skipped or reduced. "
            "Do not cite quick/partial results as the complete robustness study."
        ),
        "method_order": list(METHOD_ORDER),
        "definitions": {
            "lateral_width": (
                "connected main-lobe envelope width through the selected peak"
            ),
            "noise_snr": (
                "global complex-IQ RMS divided by additive complex-noise RMS"
            ),
            "phase_error": (
                "independent Gaussian phase offset applied to each complete "
                "per-angle IQ acquisition"
            ),
            "two_target_resolution": (
                "both peaks localized near their expected positions and the "
                "inter-peak valley is at least 6.02 dB below the weaker peak"
            ),
        },
        "simulation_parameters": simulator.parameter_metadata(),
        "configuration": {
            "seed": args.seed,
            "baseline_angles_deg": baseline_angles,
            "angle_counts": args.angle_counts,
            "total_spans_deg": args.total_spans_deg,
            "missing_positive_angle_requested_deg": args.missing_positive_angle_deg,
            "noise_snr_db": args.noise_snr_db,
            "phase_jitter_sd_deg": args.phase_jitter_sd_deg,
            "perturbation_realizations": args.perturbation_realizations,
            "depths_mm": args.depths_mm,
            "lateral_positions_mm": args.lateral_positions_mm,
            "two_target_separations_mm": args.two_target_separations_mm,
            "lateral_finest_spacing_mm": args.lateral_finest_spacing_mm,
            "lateral_half_width_mm": args.lateral_half_width_mm,
            "axial_spacing_mm": args.axial_spacing_mm,
            "axial_search_half_width_mm": args.axial_search_half_width_mm,
            "baseline_axial_peak_locations_mm": baseline_z_by_method,
        },
        "angle_sweep_summary": angle_summary,
        "perturbation_summary": perturbation_summary,
        "spatial_psf_summary": spatial_summary,
        "minimum_resolved_two_target_separation_mm": minimum_resolved_separation,
        "output_files": output_files,
        "hardware": {
            "requested_cuda_visible_devices": str(args.device),
            "gpu": gpu_metadata(cp),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "matplotlib": matplotlib.__version__,
            "pymust": getattr(pymust, "__version__", None)
            or version_or_none(("pymust",)),
            "mach_beamform": version_or_none(("mach-beamform", "mach")),
        },
    }
    summary_path = output_dir / "robustness_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)
        stream.write("\n")
    verify_file(summary_path)

    print("\nMinimum separation satisfying the -6.02 dB dip criterion:")
    for method in METHOD_ORDER:
        value = minimum_resolved_separation[method]
        print(f"  {method}: {'not reached' if value is None else f'{value:g} mm'}")
    if not publication_ready:
        print("WARNING: this run is partial/quick and is not publication-ready.")


if __name__ == "__main__":
    main()

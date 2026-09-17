"""CPU-only profile and point-spread-function measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConnectedWidth:
    peak_index: int
    peak_axis_mm: float
    peak_amplitude: float
    level_db: float
    left_crossing_mm: float
    right_crossing_mm: float
    width_mm: float


def amplitude_to_db(
    values: Sequence[float], reference: float | None = None, floor_db: float = -120.0
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scale = float(np.max(array)) if reference is None else float(reference)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The dB reference must be finite and positive.")
    tiny = scale * 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(array, tiny) / scale)


def _crossing(x1: float, y1: float, x2: float, y2: float, level: float) -> float:
    if np.isclose(y1, y2):
        return 0.5 * (x1 + x2)
    return x1 + (level - y1) * (x2 - x1) / (y2 - y1)


def measure_connected_width(
    axis_mm: Sequence[float],
    amplitude: Sequence[float],
    *,
    level_db: float = -6.0,
    expected_peak_mm: float | None = None,
    search_radius_mm: float | None = None,
) -> ConnectedWidth:
    """Measure the connected component around one peak at a relative dB level."""

    axis = np.asarray(axis_mm, dtype=float).reshape(-1)
    profile = np.asarray(amplitude, dtype=float).reshape(-1)
    if axis.size != profile.size or axis.size < 3:
        raise ValueError("axis_mm and amplitude must have equal length >= 3.")
    if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(profile)):
        raise ValueError("Profile inputs must be finite.")
    if np.any(np.diff(axis) <= 0.0) or np.any(profile < 0.0):
        raise ValueError("The axis must increase and amplitude must be nonnegative.")
    if level_db >= 0.0:
        raise ValueError("level_db must be negative.")

    candidate = np.ones(axis.size, dtype=bool)
    if expected_peak_mm is not None and search_radius_mm is not None:
        candidate = np.abs(axis - expected_peak_mm) <= search_radius_mm
        if not np.any(candidate):
            raise ValueError("The requested peak-search interval is empty.")
    candidate_indices = np.flatnonzero(candidate)
    peak_index = int(candidate_indices[np.argmax(profile[candidate])])
    peak = float(profile[peak_index])
    if peak <= 0.0:
        raise ValueError("The selected peak has zero amplitude.")
    threshold = peak * 10.0 ** (level_db / 20.0)

    left = peak_index
    while left > 0 and profile[left] > threshold:
        left -= 1
    if left == 0 and profile[left] > threshold:
        raise ValueError("No left threshold crossing occurs within the profile.")
    right = peak_index
    while right < profile.size - 1 and profile[right] > threshold:
        right += 1
    if right == profile.size - 1 and profile[right] > threshold:
        raise ValueError("No right threshold crossing occurs within the profile.")

    left_crossing = _crossing(
        axis[left], profile[left], axis[left + 1], profile[left + 1], threshold
    )
    right_crossing = _crossing(
        axis[right - 1],
        profile[right - 1],
        axis[right],
        profile[right],
        threshold,
    )
    return ConnectedWidth(
        peak_index=peak_index,
        peak_axis_mm=float(axis[peak_index]),
        peak_amplitude=peak,
        level_db=float(level_db),
        left_crossing_mm=float(left_crossing),
        right_crossing_mm=float(right_crossing),
        width_mm=float(right_crossing - left_crossing),
    )


def peak_to_background_db(
    image: np.ndarray,
    x_axis_mm: Sequence[float],
    z_axis_mm: Sequence[float],
    *,
    target_x_mm: float,
    target_z_mm: float,
    exclusion_radius_mm: float = 0.35,
) -> float:
    """Peak-to-median-background ratio within a local rectangular PSF map."""

    values = np.asarray(image, dtype=float)
    x_axis = np.asarray(x_axis_mm, dtype=float)
    z_axis = np.asarray(z_axis_mm, dtype=float)
    if values.shape != (x_axis.size, z_axis.size):
        raise ValueError("image shape must be (len(x_axis_mm), len(z_axis_mm)).")
    x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="ij")
    background = np.hypot(x_grid - target_x_mm, z_grid - target_z_mm) >= exclusion_radius_mm
    samples = values[background]
    if samples.size == 0:
        raise ValueError("The exclusion radius leaves no background samples.")
    denominator = float(np.median(samples))
    peak = float(np.max(values))
    if peak <= 0.0:
        raise ValueError("The PSF map has no positive peak.")
    denominator = max(denominator, np.finfo(float).tiny)
    return float(20.0 * np.log10(peak / denominator))

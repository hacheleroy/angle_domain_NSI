"""Matched -6 dB trace-width analysis for the MBTrace experiment.

This module is imported by ``doppler_mbtrace.py`` and can also be tested
independently without GPU hardware.

This module is intentionally independent of CuPy and the beamforming code so
that peak detection, width interpolation, one-to-one matching and summary
statistics can be tested on a CPU.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PeakWidth:
    """One detected peak and its interpolated -6 dB crossings."""

    sample_index: int
    x_mm: float
    peak_db: float
    left_crossing_mm: float
    right_crossing_mm: float
    width_mm: float


@dataclass(frozen=True)
class MatchedPeak:
    """A one-to-one DAS/receive-NSI/angle-NSI peak triplet."""

    das: PeakWidth
    receive_nsi: PeakWidth
    angle_nsi: PeakWidth


def _crossing_position(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    level: float,
) -> float:
    """Linearly interpolate the x coordinate where a segment crosses level."""

    if np.isclose(y1, y2):
        return 0.5 * (x1 + x2)
    return x1 + (level - y1) * (x2 - x1) / (y2 - y1)


def measure_peak_widths(
    x_mm: Sequence[float],
    profile_db: Sequence[float],
    *,
    prominence_db: float = 6.0,
    min_distance_mm: float = 0.2,
    min_height_db: float = -25.0,
) -> list[PeakWidth]:
    """Detect peaks and measure each local full width 6 dB below its peak.

    Peaks without two valid crossings are excluded.  The returned peak and
    width always belong to the same record, avoiding the index mismatch that
    occurs when invalid widths are discarded from a separate width array.
    """

    x_arr = np.asarray(x_mm, dtype=float).reshape(-1)
    profile = np.asarray(profile_db, dtype=float).reshape(-1)
    if x_arr.size != profile.size or x_arr.size < 3:
        raise ValueError("x_mm and profile_db must have the same length >= 3.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(profile)):
        raise ValueError("x_mm and profile_db must contain only finite values.")
    dx = np.diff(x_arr)
    if not np.all(dx > 0):
        raise ValueError("x_mm must be strictly increasing.")

    representative_dx = float(np.median(dx))
    distance_samples = max(1, int(round(min_distance_mm / representative_dx)))
    peak_indices, _ = find_peaks(
        profile,
        prominence=prominence_db,
        distance=distance_samples,
        height=min_height_db,
    )

    records: list[PeakWidth] = []
    for peak_index in peak_indices:
        level = float(profile[peak_index] - 6.0)

        left = int(peak_index)
        while left > 0 and profile[left] > level:
            left -= 1
        if left == 0 and profile[left] > level:
            continue

        right = int(peak_index)
        while right < profile.size - 1 and profile[right] > level:
            right += 1
        if right == profile.size - 1 and profile[right] > level:
            continue

        left_crossing = _crossing_position(
            x_arr[left], profile[left], x_arr[left + 1], profile[left + 1], level
        )
        right_crossing = _crossing_position(
            x_arr[right - 1],
            profile[right - 1],
            x_arr[right],
            profile[right],
            level,
        )
        records.append(
            PeakWidth(
                sample_index=int(peak_index),
                x_mm=float(x_arr[peak_index]),
                peak_db=float(profile[peak_index]),
                left_crossing_mm=float(left_crossing),
                right_crossing_mm=float(right_crossing),
                width_mm=float(right_crossing - left_crossing),
            )
        )

    return records


def _one_to_one_pairs(
    reference: Sequence[PeakWidth],
    target: Sequence[PeakWidth],
    tolerance_mm: float,
) -> dict[int, int]:
    """Maximum-cardinality, minimum-distance assignment within a tolerance."""

    if not reference or not target:
        return {}

    distances = np.abs(
        np.subtract.outer(
            [record.x_mm for record in reference],
            [record.x_mm for record in target],
        )
    )
    # An invalid edge must cost more than the sum of every possible valid edge,
    # so the assignment first maximizes valid matches and then minimizes their
    # total lateral displacement.
    invalid_cost = (max(distances.shape) + 1.0) * (tolerance_mm + 1.0)
    costs = np.where(distances <= tolerance_mm, distances, invalid_cost)
    rows, columns = linear_sum_assignment(costs)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns)
        if distances[row, column] <= tolerance_mm
    }


def match_three_methods(
    das: Sequence[PeakWidth],
    receive_nsi: Sequence[PeakWidth],
    angle_nsi: Sequence[PeakWidth],
    *,
    tolerance_mm: float = 0.15,
) -> tuple[list[MatchedPeak], dict[str, dict[int, int]]]:
    """Match both NSI peak sets one-to-one to DAS and retain complete triplets."""

    receive_pairs = _one_to_one_pairs(das, receive_nsi, tolerance_mm)
    angle_pairs = _one_to_one_pairs(das, angle_nsi, tolerance_mm)
    common_das_indices = sorted(
        set(receive_pairs).intersection(angle_pairs),
        key=lambda index: das[index].x_mm,
    )
    matches = [
        MatchedPeak(
            das=das[index],
            receive_nsi=receive_nsi[receive_pairs[index]],
            angle_nsi=angle_nsi[angle_pairs[index]],
        )
        for index in common_das_indices
    ]
    return matches, {"receive_nsi": receive_pairs, "angle_nsi": angle_pairs}


def _mean_and_sample_sd(values: Sequence[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean_mm": None, "sample_sd_mm": None}
    sample_sd = float(np.std(arr, ddof=1)) if arr.size > 1 else None
    return {"mean_mm": float(np.mean(arr)), "sample_sd_mm": sample_sd}


def analyse_three_profiles(
    x_mm: Sequence[float],
    das_profile_db: Sequence[float],
    receive_profile_db: Sequence[float],
    angle_profile_db: Sequence[float],
    *,
    prominence_db: float = 6.0,
    min_distance_mm: float = 0.2,
    min_height_db: float = -25.0,
    matching_tolerance_mm: float = 0.15,
) -> dict:
    """Run detection, -6 dB width measurement, matching and paired summaries."""

    detector_options = {
        "prominence_db": prominence_db,
        "min_distance_mm": min_distance_mm,
        "min_height_db": min_height_db,
    }
    peaks = {
        "das": measure_peak_widths(x_mm, das_profile_db, **detector_options),
        "receive_nsi": measure_peak_widths(
            x_mm, receive_profile_db, **detector_options
        ),
        "angle_nsi": measure_peak_widths(x_mm, angle_profile_db, **detector_options),
    }
    matches, pair_maps = match_three_methods(
        peaks["das"],
        peaks["receive_nsi"],
        peaks["angle_nsi"],
        tolerance_mm=matching_tolerance_mm,
    )

    das_count = len(peaks["das"])
    counts = {
        "das_detected": das_count,
        "receive_nsi_detected": len(peaks["receive_nsi"]),
        "angle_nsi_detected": len(peaks["angle_nsi"]),
        "das_matched_to_receive_nsi": len(pair_maps["receive_nsi"]),
        "das_matched_to_angle_nsi": len(pair_maps["angle_nsi"]),
        "complete_triplets": len(matches),
    }
    fractions = {
        "das_matched_to_receive_nsi": (
            len(pair_maps["receive_nsi"]) / das_count if das_count else None
        ),
        "das_matched_to_angle_nsi": (
            len(pair_maps["angle_nsi"]) / das_count if das_count else None
        ),
        "das_in_complete_triplets": len(matches) / das_count if das_count else None,
    }
    width_summary = {
        "das": _mean_and_sample_sd([match.das.width_mm for match in matches]),
        "receive_nsi": _mean_and_sample_sd(
            [match.receive_nsi.width_mm for match in matches]
        ),
        "angle_nsi": _mean_and_sample_sd(
            [match.angle_nsi.width_mm for match in matches]
        ),
    }

    return {
        "parameters": {
            **detector_options,
            "matching_tolerance_mm": matching_tolerance_mm,
            "width_level_db_below_local_peak": 6.0,
            "width_crossing_interpolation": "linear",
            "standard_deviation": "sample (ddof=1)",
        },
        "peaks": peaks,
        "matches": matches,
        "counts": counts,
        "fractions": fractions,
        "width_summary": width_summary,
    }


def format_analysis_report(analysis: dict, depth_mm: float) -> str:
    """Return a terminal-readable report for one cross-section."""

    def render_fraction(value: float | None) -> str:
        return "not available" if value is None else f"{100.0 * value:.1f}%"

    lines = [
        "",
        "=" * 78,
        f"Matched -6 dB lateral trace widths at z = {depth_mm:.2f} mm",
        "=" * 78,
        "ID   DAS x/width      Conventional NSI x/width      Angular NSI x/width",
        "-" * 78,
    ]
    for match_id, match in enumerate(analysis["matches"], start=1):
        lines.append(
            f"{match_id:2d}   "
            f"{match.das.x_mm:6.2f}/{match.das.width_mm:6.3f} mm   "
            f"{match.receive_nsi.x_mm:6.2f}/{match.receive_nsi.width_mm:6.3f} mm       "
            f"{match.angle_nsi.x_mm:6.2f}/{match.angle_nsi.width_mm:6.3f} mm"
        )

    lines.extend(["-" * 78, "Paired mean +/- sample SD:"])
    labels = {
        "das": "DAS",
        "receive_nsi": "Conventional NSI",
        "angle_nsi": "Angular NSI",
    }
    for key, label in labels.items():
        summary = analysis["width_summary"][key]
        if summary["mean_mm"] is None:
            rendered = "not available"
        elif summary["sample_sd_mm"] is None:
            rendered = f"{summary['mean_mm']:.3f} mm (n=1; SD undefined)"
        else:
            rendered = (
                f"{summary['mean_mm']:.3f} +/- "
                f"{summary['sample_sd_mm']:.3f} mm"
            )
        lines.append(f"  {label:12s}: {rendered}")

    counts = analysis["counts"]
    fractions = analysis["fractions"]
    lines.extend(
        [
            f"Detected peaks: DAS={counts['das_detected']}, "
            f"receive NSI={counts['receive_nsi_detected']}, "
            f"angle NSI={counts['angle_nsi_detected']}",
            f"Complete one-to-one triplets: {counts['complete_triplets']}",
            "Fraction of DAS peaks matched: "
            f"receive NSI={render_fraction(fractions['das_matched_to_receive_nsi'])}, "
            f"angle NSI={render_fraction(fractions['das_matched_to_angle_nsi'])}, "
            f"both={render_fraction(fractions['das_in_complete_triplets'])}",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


def save_analysis_outputs(
    analysis: dict,
    output_dir: str | Path,
    *,
    depth_mm: float,
    stem: str = "microbubble_trace_widths",
) -> tuple[Path, Path]:
    """Write complete-case measurements to CSV and audit metadata to JSON."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}.csv"
    json_path = destination / f"{stem}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "match_id",
                "das_x_mm",
                "das_width_mm",
                "receive_nsi_x_mm",
                "receive_nsi_width_mm",
                "angle_nsi_x_mm",
                "angle_nsi_width_mm",
            ]
        )
        for match_id, match in enumerate(analysis["matches"], start=1):
            writer.writerow(
                [
                    match_id,
                    match.das.x_mm,
                    match.das.width_mm,
                    match.receive_nsi.x_mm,
                    match.receive_nsi.width_mm,
                    match.angle_nsi.x_mm,
                    match.angle_nsi.width_mm,
                ]
            )

    serializable = {
        "depth_mm": float(depth_mm),
        "parameters": analysis["parameters"],
        "counts": analysis["counts"],
        "fractions": analysis["fractions"],
        "width_summary": analysis["width_summary"],
        "detected_peaks": {
            key: [asdict(record) for record in records]
            for key, records in analysis["peaks"].items()
        },
        "complete_matches": [
            {
                "match_id": match_id,
                "das": asdict(match.das),
                "receive_nsi": asdict(match.receive_nsi),
                "angle_nsi": asdict(match.angle_nsi),
            }
            for match_id, match in enumerate(analysis["matches"], start=1)
        ],
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, indent=2)
        stream.write("\n")

    return csv_path, json_path

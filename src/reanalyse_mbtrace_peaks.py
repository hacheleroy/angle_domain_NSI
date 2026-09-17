#!/usr/bin/env python3
"""Reanalyse saved MBTrace peak lists without repeating GPU beamforming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_width_analysis import (
    PeakWidth,
    diagnose_peak_displacements,
    matching_tolerance_sweep,
    save_concordance_outputs,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="MBTrace peak concordance reanalysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "results" / "reported" / "doppler" / "microbubble_trace_widths_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "doppler_concordance",
    )
    parser.add_argument("--matching-tolerance-mm", type=float, default=0.15)
    parser.add_argument("--tracking-radius-mm", type=float, default=0.35)
    return parser.parse_args()


def load_peaks(path: Path) -> dict[str, list[PeakWidth]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    source = payload.get("detected_peaks")
    if not isinstance(source, dict):
        raise ValueError(f"{path} has no detected_peaks mapping.")
    return {
        method: [PeakWidth(**record) for record in records]
        for method, records in source.items()
    }


def main() -> None:
    args = parse_args()
    peaks = load_peaks(args.input.expanduser().resolve())
    tolerances = [round(0.05 + 0.01 * index, 10) for index in range(36)]
    concordance = {}
    for method in ("receive_nsi", "angle_nsi"):
        concordance[method] = {
            "reference": "DAS detected peaks (not independent ground truth)",
            "interpretation": (
                "Positional concordance only; it must not be interpreted as "
                "sensitivity or target-detection probability."
            ),
            "tolerance_sweep": matching_tolerance_sweep(
                peaks["das"], peaks[method], tolerances
            ),
            "displacements": diagnose_peak_displacements(
                peaks["das"],
                peaks[method],
                matching_tolerance_mm=args.matching_tolerance_mm,
                tracking_radius_mm=args.tracking_radius_mm,
            ),
        }
    analysis = {
        "parameters": {"matching_tolerance_mm": args.matching_tolerance_mm},
        "concordance": concordance,
    }
    paths = save_concordance_outputs(
        analysis, args.output_dir.expanduser().resolve()
    )
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

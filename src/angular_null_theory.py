#!/usr/bin/env python3
"""Generate the small-angle angular-null derivation diagnostics.

The scalar paired-angle model is not used as evidence for experimental image
quality.  It visualizes why antisymmetric angle weights create a spatial null,
checks the first-order expansion, and records how the normalized null slope
depends on angle count, angular span, and pair symmetry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsi_core import angular_sign_weights


def angular_fields(
    displacement_m: np.ndarray,
    angles_deg: np.ndarray,
    *,
    wavenumber_rad_m: float,
    amplitudes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference and sign-weighted fields in the paired-angle model."""

    angles = np.asarray(angles_deg, dtype=float)
    weights, _ = angular_sign_weights(angles)
    if amplitudes is None:
        amplitudes = np.ones_like(angles)
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    phase = np.exp(
        -1j
        * wavenumber_rad_m
        * np.asarray(displacement_m, dtype=float)[:, None]
        * np.sin(np.deg2rad(angles))[None, :]
    )
    contributions = amplitudes[None, :] * phase
    return contributions.sum(axis=1), (contributions * weights[None, :]).sum(axis=1)


def normalized_null_slope_per_m(
    angles_deg: np.ndarray, wavenumber_rad_m: float
) -> float:
    """First-order |Z| slope divided by the in-focus reference sum |U(0)|."""

    angles = np.asarray(angles_deg, dtype=float)
    positive = angles[angles > 0.0]
    return float(
        2.0 * wavenumber_rad_m * np.sum(np.sin(np.deg2rad(positive))) / angles.size
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Angular-null small-angle model")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "results" / "generated" / "angular_null_theory",
    )
    parser.add_argument("--frequency-mhz", type=float, default=5.21)
    parser.add_argument("--sound-speed-m-s", type=float, default=1540.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frequency_mhz <= 0.0 or args.sound_speed_m_s <= 0.0:
        raise SystemExit("Frequency and sound speed must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    k0 = 2.0 * np.pi * args.frequency_mhz * 1e6 / args.sound_speed_m_s
    displacement_mm = np.linspace(-0.25, 0.25, 2001)
    displacement_m = displacement_mm * 1e-3
    baseline_angles = np.linspace(-4.0, 4.0, 17)
    uniform, null = angular_fields(
        displacement_m, baseline_angles, wavenumber_rad_m=k0
    )
    slope = normalized_null_slope_per_m(baseline_angles, k0)
    first_order = slope * np.abs(displacement_m)
    exact = np.abs(null) / abs(uniform[len(uniform) // 2])

    rows = []
    for count in (5, 9, 17, 33):
        for span in (4.0, 8.0, 12.0, 16.0):
            angles = np.linspace(-span / 2.0, span / 2.0, count)
            weights, diagnostics = angular_sign_weights(angles)
            rows.append(
                {
                    "angle_count": count,
                    "total_span_deg": span,
                    "normalized_null_slope_per_mm": (
                        normalized_null_slope_per_m(angles, k0) * 1e-3
                    ),
                    "weight_sum": diagnostics.weight_sum,
                    "weight_l1_norm": diagnostics.weight_l1_norm,
                    "broadside_weight": float(weights[count // 2]),
                }
            )

    # Demonstrate the in-focus residual created by one missing positive angle.
    missing_angles = np.delete(baseline_angles, np.flatnonzero(baseline_angles > 0)[3])
    missing_weights = np.sign(missing_angles)
    missing_in_focus = float(abs(np.sum(missing_weights)))
    metadata = {
        "model": "B_theta(dx)=A_theta exp[-i k0 dx sin(theta)]",
        "paired_first_order_result": (
            "Z_theta(dx)=-2 i k0 dx sum_{theta>0} A_theta sin(theta)+O(dx^3)"
        ),
        "uniform_reference_at_focus": "U(0)=sum_theta A_theta",
        "frequency_hz": args.frequency_mhz * 1e6,
        "sound_speed_m_s": args.sound_speed_m_s,
        "wavenumber_rad_m": k0,
        "baseline_angles_deg": baseline_angles.tolist(),
        "baseline_normalized_slope_per_mm": slope * 1e-3,
        "maximum_first_order_absolute_error_over_plot": float(
            np.max(np.abs(exact - first_order))
        ),
        "missing_angle_example": {
            "angles_deg": missing_angles.tolist(),
            "raw_sign_weight_sum": float(np.sum(missing_weights)),
            "normalized_in_focus_null_magnitude": (
                missing_in_focus / missing_angles.size
            ),
            "interpretation": (
                "Without paired contributions the sign weights need not create "
                "an in-focus null; zero-sum redesign or exclusion is required."
            ),
        },
        "slope_rows": rows,
    }

    csv_path = output_dir / "angular_null_slope.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "angular_null_theory.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=180)
    axes[0].plot(
        displacement_mm, exact, color="tab:blue", linewidth=2.0,
        label="Exact paired-angle field",
    )
    axes[0].plot(
        displacement_mm, first_order, color="0.25", linestyle="--",
        linewidth=1.3, label="First-order slope",
    )
    axes[0].set(
        xlabel="Lateral displacement $\\Delta x$ [mm]",
        ylabel="$|Z_\\theta|/|U(0)|$",
        title="Field surrounding the angular null",
    )
    axes[0].legend(frameon=False, fontsize=8)
    for count in (5, 9, 17, 33):
        selected = [row for row in rows if row["angle_count"] == count]
        axes[1].plot(
            [row["total_span_deg"] for row in selected],
            [row["normalized_null_slope_per_mm"] for row in selected],
            marker="o", linewidth=1.4, label=f"K={count}",
        )
    axes[1].set(
        xlabel="Total angular span [deg]",
        ylabel="Normalized null slope [mm$^{-1}$]",
        title="Slope versus angular sampling",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure_path = output_dir / "angular_null_small_angle_derivation.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    for path in (csv_path, json_path, figure_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Expected theory output was not written: {path}")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize the frozen PMB-revision results as manuscript assets.

The script is the only bridge between numerical outputs and LaTeX. It checks
that every publication analysis is complete, migrates the two legacy method
labels used by early revision runs, writes tables/macros, and copies the five
agreed main figures plus supplementary diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method_names import (  # noqa: E402
    ANGLE_NSI,
    CF_DAS,
    DAS,
    DMAS,
    METHODS,
    MV,
    RECEIVE_NSI,
    canonical_method_name,
)
from simulation_robustness import (  # noqa: E402
    plot_angle_sweeps,
    plot_perturbations,
    plot_spatial_psf,
)


NSI_METHODS = (DAS, RECEIVE_NSI, ANGLE_NSI)
TIMING_METHODS = {
    DAS: "DAS / coherent compounding",
    RECEIVE_NSI: "Conventional NSI (two fields)",
    ANGLE_NSI: "Angular NSI (streaming)",
}
EXPECTED_C_VALUES = {0.02, 0.05, 0.10, 0.20}
CONVENTIONAL_BASELINE_SCHEMA_VERSION = 3
PICMUS_FULL_PHANTOM_SCHEMA_VERSION = 3
COLORS = {
    DAS: "#6a3d9a",
    CF_DAS: "#2ca02c",
    MV: "#ff7f0e",
    DMAS: "#8c564b",
    RECEIVE_NSI: "#d62728",
    ANGLE_NSI: "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize PMB revision results as LaTeX assets"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results" / "generated" / "revision",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "generated" / "revision" / "manuscript_assets",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write diagnostic assets even when publication checks fail.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_typed_csv(path: Path) -> list[dict[str, Any]]:
    """Read analysis CSV rows and restore simple scalar types for plotting."""

    rows: list[dict[str, Any]] = []
    for source in read_csv(path):
        row: dict[str, Any] = {}
        for key, value in source.items():
            if value == "":
                row[key] = None
            elif value == "True":
                row[key] = True
            elif value == "False":
                row[key] = False
            elif value == "infinity":
                row[key] = value
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        if "method" in row:
            row["method"] = canonical_method_name(str(row["method"]))
        rows.append(row)
    return rows


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {label}: {value!r}")
    return number


def fmt(value: Any, digits: int = 3) -> str:
    return f"{finite(value, 'LaTeX value'):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * finite(value, 'percentage'):.{digits}f}"


def tex_macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    if path.stat().st_size == 0:
        raise OSError(f"Failed to write {path}")


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Required figure is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def first(rows: Iterable[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    joined = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
    raise KeyError(f"No result row matched {joined}")


def canonical_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        if "method" in row:
            row["method"] = canonical_method_name(str(row["method"]))
        output.append(row)
    return output


def selected_c_values(rows: Iterable[dict[str, Any]], method: str) -> set[float]:
    """Return finite c values for one method, rounded for JSON/CSV stability."""

    return {
        round(finite(row["c"], f"{method} c value"), 8)
        for row in rows
        if row.get("method") == method
    }


def latex_table(
    column_spec: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    lines = [
        rf"\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def timing_figure(
    path: Path,
    scaling_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.3, 7.0), dpi=180)
    families = (
        ("grid_pixels", "Image size [pixels]"),
        ("receive_elements", "Receive elements"),
        ("transmit_angles", "Plane waves"),
    )
    for axis, (family, xlabel) in zip(axes.flat[:3], families):
        for method in NSI_METHODS:
            timing_name = TIMING_METHODS[method]
            rows = sorted(
                (
                    row
                    for row in scaling_rows
                    if row.get("family") == family
                    and row.get("method") == timing_name
                ),
                key=lambda row: finite(
                    row["family_value"], "timing family value"
                ),
            )
            axis.plot(
                [finite(row["family_value"], "timing x") for row in rows],
                [finite(row["median_ms"], "timing median") for row in rows],
                marker="o",
                linewidth=1.4,
                color=COLORS[method],
                label=method,
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("End-to-end time [ms]")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    kernel_by_method = {row["method"]: row for row in kernel_rows}
    medians = [
        finite(kernel_by_method[method]["median_ms"], method)
        for method in METHODS
    ]
    axes[1, 1].bar(
        np.arange(len(METHODS)),
        medians,
        color=[COLORS[method] for method in METHODS],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(
        np.arange(len(METHODS)), METHODS, rotation=35, ha="right"
    )
    axes[1, 1].set_ylabel("Post-delay kernel time [ms]")
    axes[1, 1].grid(axis="y", alpha=0.25, which="both")
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            0.02,
            0.96,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results = args.results_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    simulation = read_json(
        results
        / "simulation_six_method"
        / "simulation_six_method_summary.json"
    )
    full_phantom = read_json(
        results / "picmus_full_phantom" / "picmus_full_phantom_summary.json"
    )
    doppler_rows = canonical_rows(
        read_csv(results / "doppler" / "mbtrace_three_method_comparison.csv")
    )
    displacement_rows = read_csv(
        results / "doppler" / "microbubble_peak_concordance_displacements.csv"
    )
    doppler_c = read_json(results / "doppler" / "mbtrace_c_sensitivity.json")
    bmode = read_json(results / "bmode" / "bmode_nsi_results.json")
    experimental = read_json(
        results
        / "experimental_psf"
        / "picmus_experimental_psf_summary.json"
    )
    timing = read_json(
        results / "timing_scaling" / "nsi_timing_scaling_summary.json"
    )
    conventional = read_json(
        results
        / "conventional_baselines"
        / "conventional_baseline_summary.json"
    )
    conventional_timing = read_json(
        results
        / "conventional_timing"
        / "conventional_timing_summary.json"
    )
    robustness_root = results / "robustness"
    if not (robustness_root / "robustness_angle_sweeps.csv").is_file():
        robustness_root = ROOT / "results" / "reported" / "robustness"
    robustness_angle_rows = read_typed_csv(
        robustness_root / "robustness_angle_sweeps.csv"
    )
    robustness_perturbation_rows = read_typed_csv(
        robustness_root / "robustness_noise_phase_sweeps.csv"
    )
    robustness_spatial_rows = read_typed_csv(
        robustness_root / "robustness_spatial_psf.csv"
    )

    simulation_rows = canonical_rows(simulation.get("metrics", []))
    conventional_psf_rows = canonical_rows(
        conventional.get("experimental_psf", {}).get("metrics", [])
    )
    conventional_views = []
    for source in conventional.get("carotid_views", []):
        view = dict(source)
        view["metrics"] = canonical_rows(source.get("metrics", []))
        conventional_views.append(view)
    conventional_timing_rows = canonical_rows(
        conventional_timing.get("rows", [])
    )
    profile_rows = canonical_rows(
        full_phantom.get("profile_diagnostics", [])
    )

    problems: list[str] = []
    expected = set(METHODS)
    if (
        simulation.get("metadata_only") is not False
        or simulation.get("publication_ready") is not True
    ):
        problems.append("six-method simulation is not publication-ready")
    if {row.get("method") for row in simulation_rows} != expected:
        problems.append("six-method simulation is incomplete")
    if (
        int(full_phantom.get("schema_version", 0))
        != PICMUS_FULL_PHANTOM_SCHEMA_VERSION
        or full_phantom.get("metadata_only") is not False
        or full_phantom.get("publication_ready") is not True
    ):
        problems.append("full PICMUS phantom comparison is not publication-ready")
    if set(full_phantom.get("methods", [])) != expected:
        problems.append("full PICMUS phantom comparison is incomplete")
    expected_profile_pairs = {
        (target_id, method)
        for target_id in range(1, 8)
        for method in NSI_METHODS
    }
    observed_profile_pairs = {
        (int(row.get("target_id", -1)), row.get("method"))
        for row in profile_rows
    }
    if observed_profile_pairs != expected_profile_pairs:
        problems.append("full-phantom target-wise profile audit is incomplete")
    if any(
        "central_notch_depth_db" not in row
        or "central_notch_detected" not in row
        for row in profile_rows
    ):
        problems.append("full-phantom central-notch diagnostic is stale")
    if {row.get("method") for row in doppler_rows} != set(NSI_METHODS):
        problems.append(
            "MBTrace comparison does not contain DAS and both NSI methods"
        )
    if (
        int(conventional.get("schema_version", 0))
        != CONVENTIONAL_BASELINE_SCHEMA_VERSION
        or conventional.get("metadata_only") is not False
        or conventional.get("publication_ready") is not True
    ):
        problems.append("conventional comparison is not publication-ready")
    if {row.get("method") for row in conventional_psf_rows} != expected:
        problems.append("conventional point-target comparison is incomplete")
    view_methods = {
        view.get("view"): {
            row.get("method") for row in view.get("metrics", [])
        }
        for view in conventional_views
    }
    if set(view_methods) != {"CC", "CL"} or any(
        value != expected for value in view_methods.values()
    ):
        problems.append("six-method carotid comparison is incomplete")
    if conventional_timing.get("publication_ready") is not True:
        problems.append("six-method kernel timing is not publication-ready")
    if {row.get("method") for row in conventional_timing_rows} != expected:
        problems.append("six-method kernel timing is incomplete")
    if int(conventional_timing.get("requested_warmups", 0)) < 10 or int(
        conventional_timing.get("requested_repetitions", 0)
    ) < 50:
        problems.append("six-method kernel timing is too short")
    if timing.get("failed_cases") or timing.get("quick_engineering_run"):
        problems.append("end-to-end timing suite is incomplete")
    if int(timing.get("requested_warmups", 0)) < 10 or int(
        timing.get("requested_repetitions", 0)
    ) < 50:
        problems.append("end-to-end timing suite is too short")
    for method in (RECEIVE_NSI, ANGLE_NSI):
        if (
            selected_c_values(doppler_c.get("rows", []), method)
            != EXPECTED_C_VALUES
        ):
            problems.append(f"MBTrace c sweep is incomplete for {method}")
        if (
            selected_c_values(bmode.get("c_sensitivity", []), method)
            != EXPECTED_C_VALUES
        ):
            problems.append(f"carotid c sweep is incomplete for {method}")
    if experimental.get("metadata_only") is not False:
        problems.append("experimental point-target output is metadata-only")

    if problems and not args.allow_incomplete:
        raise SystemExit(
            "Revision assets were not materialized:\n- "
            + "\n- ".join(problems)
        )

    simulation_by_method = {row["method"]: row for row in simulation_rows}
    doppler_by_method = {row["method"]: row for row in doppler_rows}
    psf_by_method = {row["method"]: row for row in conventional_psf_rows}
    carotid_by_view = {
        view["view"]: {row["method"]: row for row in view["metrics"]}
        for view in conventional_views
    }
    kernel_by_method = {
        row["method"]: row for row in conventional_timing_rows
    }
    reference_timing_rows = [
        row
        for row in timing.get("rows", [])
        if row.get("run_id") == "grid_128x256"
    ]
    end_to_end_by_method = {
        method: first(reference_timing_rows, method=timing_name)
        for method, timing_name in TIMING_METHODS.items()
    }

    macro_stems = {
        DAS: "Das",
        CF_DAS: "Cf",
        MV: "Mv",
        DMAS: "Dmas",
        RECEIVE_NSI: "ReceiveNsi",
        ANGLE_NSI: "AngleNsi",
    }
    macros = [
        "% Generated by scripts/materialize_revision_assets.py; do not edit.",
        tex_macro("RevisionAnalysisStatus", "complete"),
    ]
    for method in METHODS:
        stem = macro_stems[method]
        macros.extend(
            [
                tex_macro(
                    f"Simulation{stem}Width",
                    fmt(simulation_by_method[method]["lateral_width_mm"], 4),
                ),
                tex_macro(
                    f"Experimental{stem}Width",
                    fmt(psf_by_method[method]["lateral_width_mm"], 4),
                ),
                tex_macro(
                    f"CarotidCc{stem}Gcnr",
                    fmt(carotid_by_view["CC"][method]["gcnr"]),
                ),
                tex_macro(
                    f"CarotidCl{stem}Gcnr",
                    fmt(carotid_by_view["CL"][method]["gcnr"]),
                ),
                tex_macro(
                    f"Kernel{stem}Time",
                    fmt(kernel_by_method[method]["median_ms"], 3),
                ),
            ]
        )
    for method in NSI_METHODS:
        stem = macro_stems[method]
        macros.extend(
            [
                tex_macro(
                    f"EndToEnd{stem}Time",
                    fmt(end_to_end_by_method[method]["median_ms"], 2),
                ),
                tex_macro(
                    f"Mbtrace{stem}Width",
                    fmt(doppler_by_method[method]["mean_matched_width_mm"]),
                ),
                tex_macro(
                    f"Mbtrace{stem}Match",
                    pct(
                        doppler_by_method[method][
                            "das_concordant_fraction_at_0p15_mm"
                        ]
                    ),
                ),
            ]
        )
    for method in (RECEIVE_NSI, ANGLE_NSI):
        notch_count = sum(
            bool(row.get("central_notch_detected"))
            for row in profile_rows
            if row.get("method") == method
        )
        macros.append(
            tex_macro(
                f"Picmus{macro_stems[method]}NotchCount", str(notch_count)
            )
        )
    write_text(output / "revision_results.tex", "\n".join(macros))

    simulation_table = []
    for method in METHODS:
        row = simulation_by_method[method]
        simulation_table.append(
            [
                method,
                fmt(row["lateral_width_mm"], 4),
                fmt(row["axial_width_mm"], 4),
                fmt(row["peak_to_median_background_db"], 1),
                fmt(row["center_to_profile_peak_db"], 1),
            ]
        )
    write_text(
        output / "table_simulation_metrics.tex",
        latex_table(
            "lrrrr",
            [
                "Method",
                "Lateral width (mm)",
                "Axial width (mm)",
                "Peak/background (dB)",
                "Centre/peak (dB)",
            ],
            simulation_table,
        ),
    )

    conventional_table = []
    for method in METHODS:
        conventional_table.append(
            [
                method,
                fmt(psf_by_method[method]["lateral_width_mm"], 4),
                fmt(psf_by_method[method]["peak_to_median_background_db"], 1),
                fmt(carotid_by_view["CC"][method]["gcnr"]),
                fmt(carotid_by_view["CL"][method]["gcnr"]),
                fmt(kernel_by_method[method]["median_ms"], 3),
            ]
        )
    write_text(
        output / "table_conventional_baselines.tex",
        latex_table(
            "lrrrrr",
            [
                "Method",
                "Width (mm)",
                "Peak/background (dB)",
                "CC gCNR",
                "CL gCNR",
                "Kernel (ms)",
            ],
            conventional_table,
        ),
    )

    complexity = {
        DAS: r"$O(M)$",
        CF_DAS: r"$O(M)$",
        MV: r"$O((2K+1)(M-L+1)L^2+L^3)$",
        DMAS: r"$O(M^2)$ direct; $O(M)$ exact identity",
        RECEIVE_NSI: r"$O(M)$",
        ANGLE_NSI: r"$O(M)$",
    }
    das_kernel = finite(
        kernel_by_method[DAS]["median_ms"], "DAS kernel time"
    )
    timing_table = [
        [
            method,
            fmt(kernel_by_method[method]["median_ms"], 3),
            fmt(
                finite(kernel_by_method[method]["median_ms"], method)
                / das_kernel,
                2,
            ),
            complexity[method],
        ]
        for method in METHODS
    ]
    write_text(
        output / "table_conventional_timing.tex",
        latex_table(
            "lrrl",
            [
                "Method",
                "Median (ms)",
                "Relative to DAS",
                "Leading order per pixel/angle",
            ],
            timing_table,
        ),
    )

    mv_windows = {
        view["view"]: int(view["mv_temporal_half_window_samples"])
        for view in conventional_views
    }
    parameter_rows = [
        [
            CF_DAS,
            "Receive-aperture CF per transmit angle; no fitted parameter.",
        ],
        [
            MV,
            (
                r"$L=\lfloor M_{\mathrm{active}}/2\rfloor$; all overlapping "
                r"subarrays; loading $\mathrm{tr}(R)/(100L)$; $K=0$ for "
                f"point targets and $K={mv_windows['CC']}$ (CC), "
                f"$K={mv_windows['CL']}$ (CL)."
            ),
        ],
        [
            DMAS,
            (
                r"Signed square-root products for $i<j$; fixed Kaiser "
                r"band-pass around $2f_0$; analytic envelope."
            ),
        ],
    ]
    write_text(
        output / "table_conventional_parameters.tex",
        latex_table(
            r"lp{0.72\linewidth}",
            ["Method", "Fixed implementation choices"],
            parameter_rows,
        ),
    )

    diagnostic_table = []
    for row in profile_rows:
        diagnostic_table.append(
            [
                str(int(row["target_id"])),
                row["method"],
                fmt(row["target_x_mm"], 1),
                fmt(row["target_z_mm"], 1),
                fmt(row["center_to_local_peak_db"], 1),
                fmt(row["central_notch_depth_db"], 1),
                "yes" if row.get("central_notch_detected") else "no",
            ]
        )
    write_text(
        output / "table_picmus_center_diagnostic.tex",
        latex_table(
            "rlrrrrr",
            [
                "Target",
                "Method",
                "$x$ (mm)",
                "$z$ (mm)",
                "Centre/peak (dB)",
                "Notch depth (dB)",
                "Notch",
            ],
            diagnostic_table,
        ),
    )

    c_table = [
        [
            row["method"],
            fmt(row["c"], 2),
            str(int(row["detected_peak_count"])),
            pct(row["matched_fraction_at_0p15_mm"]) + r"\%",
            pct(row["matched_fraction_at_0p35_mm"]) + r"\%",
        ]
        for row in doppler_c.get("rows", [])
    ]
    write_text(
        output / "table_mbtrace_c_sensitivity.tex",
        latex_table(
            "lrrrr",
            [
                "Method",
                "$c$",
                "Detected peaks",
                "Match at 0.15 mm",
                "Match at 0.35 mm",
            ],
            c_table,
        ),
    )

    displacement_table = []
    for row in displacement_rows:
        label = {
            "receive_nsi": RECEIVE_NSI,
            "angle_nsi": ANGLE_NSI,
        }.get(row["method"], row["method"])
        status = row["status"].replace("_", " ")
        if row.get("possible_split", "").lower() == "true":
            status += "; possible split"
        displacement_table.append(
            [
                label,
                fmt(row["reference_x_mm"]),
                fmt(row["target_x_mm"])
                if row.get("target_x_mm")
                else "--",
                fmt(row["signed_displacement_mm"])
                if row.get("signed_displacement_mm")
                else "--",
                fmt(row["absolute_displacement_mm"])
                if row.get("absolute_displacement_mm")
                else "--",
                status,
            ]
        )
    write_text(
        output / "table_mbtrace_displacements.tex",
        latex_table(
            "lrrrrl",
            [
                "Method",
                "DAS peak",
                "Assigned peak",
                "Signed shift",
                "Absolute shift",
                "Classification",
            ],
            displacement_table,
        ),
    )

    control_rows: list[list[str]] = []
    inverse_timing_names = {
        value: key for key, value in TIMING_METHODS.items()
    }
    for run_id, label in (
        ("baseline_preloaded", "Preloaded kernel"),
        ("baseline_host_stacked", "Host-stacked input"),
    ):
        for row in timing.get("rows", []):
            if (
                row.get("run_id") != run_id
                or row.get("method") not in inverse_timing_names
            ):
                continue
            method = inverse_timing_names[row["method"]]
            control_rows.append(
                [
                    label,
                    method,
                    fmt(row["median_ms"], 2),
                    fmt(row["ratio_to_das_median"], 3),
                    f"{finite(row['timed_total_transfer_bytes'], 'control bytes') / 2**20:.2f}",
                ]
            )
    write_text(
        output / "table_timing_controls.tex",
        latex_table(
            "llrrr",
            [
                "Control",
                "Method",
                "Median (ms)",
                "Relative to DAS",
                "Transfer (MiB)",
            ],
            control_rows,
        ),
    )

    timing_path = output / "figures" / "figure5_computation_benchmark.png"
    timing_figure(
        timing_path, timing.get("rows", []), conventional_timing_rows
    )
    supplementary_output = output / "supplementary"
    supplementary_output.mkdir(parents=True, exist_ok=True)
    plot_angle_sweeps(
        supplementary_output / "figureS1_angle_sweeps.png",
        robustness_angle_rows,
    )
    plot_perturbations(
        supplementary_output / "figureS2_noise_phase_sweeps.png",
        robustness_perturbation_rows,
    )
    plot_spatial_psf(
        supplementary_output / "figureS3_spatial_psf.png",
        robustness_spatial_rows,
    )

    figure_map = {
        results
        / "simulation_six_method"
        / "simulation_six_method_comparison.png": output
        / "figures"
        / "figure1_simulation_six_method.png",
        results
        / "picmus_full_phantom"
        / "picmus_full_phantom_six_method.png": output
        / "figures"
        / "figure2_picmus_full_phantom_six_method.png",
        results
        / "doppler"
        / "power_doppler_three_method_comparison.png": output
        / "figures"
        / "figure3_microbubble_power_doppler.png",
        results
        / "conventional_baselines"
        / "conventional_baseline_carotid_combined.png": output
        / "figures"
        / "figure4_carotid_six_method.png",
        results
        / "theory"
        / "angular_null_small_angle_derivation.png": output
        / "supplementary"
        / "figureS1_angular_null_theory.png",
        results
        / "point_target"
        / "single_scatterer_c_sensitivity.png": output
        / "supplementary"
        / "figureS5_point_target_c_sensitivity.png",
        results
        / "doppler"
        / "microbubble_peak_concordance_tolerance_sweep.png": output
        / "supplementary"
        / "figureS6_mbtrace_concordance.png",
        results / "doppler" / "mbtrace_c_sensitivity.png": output
        / "supplementary"
        / "figureS7_mbtrace_c_sensitivity.png",
        results / "bmode" / "bmode_nsi_c_sensitivity.png": output
        / "supplementary"
        / "figureS8_carotid_c_sensitivity.png",
        results
        / "experimental_psf"
        / "picmus_experimental_psf_c_sensitivity.png": output
        / "supplementary"
        / "figureS9_experimental_psf_c_sensitivity.png",
        results
        / "conventional_baselines"
        / "conventional_baseline_experimental_psf.png": output
        / "supplementary"
        / "figureS10_conventional_psf_maps.png",
        results
        / "conventional_baselines"
        / "conventional_baseline_experimental_psf_profiles.png": output
        / "supplementary"
        / "figureS11_conventional_psf_profiles.png",
        results
        / "conventional_timing"
        / "conventional_timing_summary.png": output
        / "supplementary"
        / "figureS12_conventional_timing.png",
    }
    for source, destination in figure_map.items():
        copy_required(source, destination)

    manifest = {
        "publication_checks_passed": not problems,
        "problems": problems,
        "canonical_methods": list(METHODS),
        "main_figure_order": [
            "simulation_six_method",
            "picmus_full_phantom_six_method",
            "microbubble_power_doppler",
            "carotid_six_method",
            "computation_benchmark",
        ],
        "results_root": str(results),
        "robustness_source": str(robustness_root),
        "generated_files": sorted(
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        ),
    }
    write_text(
        output / "revision_asset_manifest.json",
        json.dumps(manifest, indent=2),
    )
    print(
        f"Materialized {len(manifest['generated_files']) + 1} revision assets "
        f"in {output}"
    )


if __name__ == "__main__":
    main()

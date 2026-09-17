#!/usr/bin/env python3
"""Build manuscript-ready LaTeX tables, macros, and figures from revision runs.

The script deliberately refuses incomplete or short timing runs by default.  It
is the single bridge between numerical outputs and the revised manuscript, so
reported values do not need to be transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable


METHODS = ("DAS", "Angular CF-DAS", "Receive-NSI", "Angle-NSI")
CONVENTIONAL_METHODS = (
    "DAS", "Receive CF-DAS", "MV", "F-DMAS", "Receive-NSI", "Angle-NSI"
)
TIMING_METHODS = {
    "DAS": "DAS / coherent compounding",
    "Angular CF-DAS": "Angular CF-DAS",
    "Receive-NSI": "Conventional NSI (two fields)",
    "Angle-NSI": "Angular NSI (streaming)",
}
EXPECTED_C_VALUES = {0.02, 0.05, 0.10, 0.20}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Materialize PMB revision results as LaTeX assets"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=root / "results" / "generated" / "revision",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "revision" / "manuscript_assets",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Create a diagnostic asset set even when publication checks fail.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {label}: {value!r}")
    return number


def first(rows: Iterable[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    joined = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
    raise KeyError(f"No result row matched {joined}")


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


def method_label(method: str) -> str:
    return {
        "DAS": "DAS",
        "Angular CF-DAS": "Angular CF-DAS",
        "Receive CF-DAS": "Receive CF-DAS",
        "MV": "MV",
        "F-DMAS": "F-DMAS",
        "Receive-NSI": "Receive-NSI",
        "Angle-NSI": "Angle-NSI",
    }[method]


def selected_c_values(
    rows: Iterable[dict[str, Any]], method: str
) -> set[float]:
    """Return finite c values for one method, rounded for JSON/CSV stability."""

    return {
        round(finite(row["c"], f"{method} c value"), 8)
        for row in rows
        if row.get("method") == method
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    results = args.results_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    point = read_json(results / "point_target" / "simulation_psf_fwhm_summary.json")
    doppler_rows = read_csv(results / "doppler" / "mbtrace_four_method_comparison.csv")
    displacement_rows = read_csv(
        results / "doppler" / "microbubble_peak_concordance_displacements.csv"
    )
    doppler_c = read_json(results / "doppler" / "mbtrace_c_sensitivity.json")
    bmode = read_json(results / "bmode" / "bmode_nsi_results.json")
    experimental = read_json(
        results / "experimental_psf" / "picmus_experimental_psf_summary.json"
    )
    timing = read_json(
        results / "timing_scaling" / "nsi_timing_scaling_summary.json"
    )
    conventional = read_json(
        results / "conventional_baselines" / "conventional_baseline_summary.json"
    )
    conventional_timing = read_json(
        results / "conventional_timing" / "conventional_timing_summary.json"
    )

    problems: list[str] = []
    point_rows = point.get("final_results", [])
    if {row.get("method") for row in point_rows} != set(METHODS):
        problems.append("point-target output does not contain exactly four methods")
    point_c_rows = point.get("c_sensitivity_results", [])
    for method in ("Receive-NSI", "Angle-NSI"):
        if selected_c_values(point_c_rows, method) != EXPECTED_C_VALUES:
            problems.append(
                f"point-target c sweep is incomplete for {method}"
            )
    if experimental.get("metadata_only") is not False:
        problems.append("experimental PSF output is metadata-only")
    experimental_summaries = experimental.get("summaries", {})
    for method in METHODS:
        if int(experimental_summaries.get("all", {}).get(method, {}).get(
            "lateral_width_mm", {}
        ).get("n", 0)) != 7:
            problems.append(
                f"experimental PSF output does not contain seven widths for {method}"
            )
        if int(experimental_summaries.get("on-axis", {}).get(method, {}).get(
            "lateral_width_mm", {}
        ).get("n", 0)) != 5:
            problems.append(
                f"experimental PSF on-axis summary does not contain five widths for {method}"
            )
        if int(experimental_summaries.get("37.5-mm depth", {}).get(
            method, {}
        ).get("lateral_width_mm", {}).get("n", 0)) != 3:
            problems.append(
                f"experimental PSF co-depth summary does not contain three widths for {method}"
            )
    experimental_c_values = {
        round(finite(value, "experimental PSF c value"), 8)
        for value in experimental.get("reconstruction", {}).get("c_values", [])
    }
    if experimental_c_values != EXPECTED_C_VALUES:
        problems.append("experimental PSF c sweep is incomplete")
    for method in ("Receive-NSI", "Angle-NSI"):
        if selected_c_values(doppler_c.get("rows", []), method) != EXPECTED_C_VALUES:
            problems.append(f"MBTrace c sweep is incomplete for {method}")
        if selected_c_values(bmode.get("c_sensitivity", []), method) != EXPECTED_C_VALUES:
            problems.append(f"carotid c sweep is incomplete for {method}")
    if timing.get("failed_cases"):
        problems.append("one or more timing-scaling cases failed")
    if timing.get("quick_engineering_run"):
        problems.append("timing suite was run with --quick")
    if int(timing.get("requested_warmups", 0)) < 10:
        problems.append("timing suite used fewer than 10 warm-ups")
    if int(timing.get("requested_repetitions", 0)) < 50:
        problems.append("timing suite used fewer than 50 repetitions")
    if any(int(row.get("n", 0)) < 50 for row in timing.get("rows", [])):
        problems.append("one or more timing rows contains fewer than 50 samples")
    primary_cases = {
        row.get("run_id") for row in timing.get("rows", [])
        if row.get("family") in {
            "grid_pixels", "receive_elements", "transmit_angles"
        }
    }
    required_cases = {
        "grid_64x128", "grid_128x256", "grid_256x512",
        "elements_64", "elements_128", "elements_256",
        "angles_9", "angles_17", "angles_33", "angles_75",
    }
    if primary_cases != required_cases:
        problems.append("timing suite is missing one or more primary scaling cases")
    required_timing_methods = set(TIMING_METHODS.values())
    for run_id in sorted(required_cases):
        case_methods = {
            row.get("method") for row in timing.get("rows", [])
            if row.get("run_id") == run_id
        }
        if case_methods != required_timing_methods:
            problems.append(
                f"timing suite primary case {run_id} does not contain all four methods"
            )
    for control_id in ("baseline_preloaded", "baseline_host_stacked"):
        control_methods = {
            row.get("method") for row in timing.get("rows", [])
            if row.get("run_id") == control_id
        }
        if control_methods != required_timing_methods:
            problems.append(
                f"timing suite control {control_id} does not contain all four methods"
            )
    expected_conventional = set(CONVENTIONAL_METHODS)
    if conventional.get("metadata_only") is not False:
        problems.append("conventional-baseline output is metadata-only")
    if conventional.get("quick_engineering_run"):
        problems.append("conventional-baseline comparison was run with --quick")
    if conventional.get("publication_ready") is not True:
        problems.append("conventional-baseline comparison is not publication-ready")
    if set(conventional.get("methods", [])) != expected_conventional:
        problems.append("conventional-baseline comparison is missing one or more methods")
    conventional_psf_rows = conventional.get("experimental_psf", {}).get(
        "metrics", []
    )
    if {row.get("method") for row in conventional_psf_rows} != expected_conventional:
        problems.append("conventional experimental PSF comparison is incomplete")
    conventional_views = conventional.get("carotid_views", [])
    conventional_view_methods = {
        view.get("view"): {row.get("method") for row in view.get("metrics", [])}
        for view in conventional_views
    }
    if set(conventional_view_methods) != {"CC", "CL"} or any(
        methods != expected_conventional
        for methods in conventional_view_methods.values()
    ):
        problems.append("conventional carotid comparison is incomplete")
    if conventional_timing.get("quick_engineering_run"):
        problems.append("conventional timing was run with --quick")
    if conventional_timing.get("publication_ready") is not True:
        problems.append("conventional timing is not publication-ready")
    if int(conventional_timing.get("requested_warmups", 0)) < 10:
        problems.append("conventional timing used fewer than 10 warm-ups")
    if int(conventional_timing.get("requested_repetitions", 0)) < 50:
        problems.append("conventional timing used fewer than 50 repetitions")
    conventional_timing_rows = conventional_timing.get("rows", [])
    if {row.get("method") for row in conventional_timing_rows} != expected_conventional:
        problems.append("conventional timing is missing one or more methods")
    if any(int(row.get("n", 0)) < 50 for row in conventional_timing_rows):
        problems.append("one or more conventional timing rows has fewer than 50 samples")
    if problems and not args.allow_incomplete:
        raise SystemExit(
            "Revision assets were not materialized:\n- " + "\n- ".join(problems)
        )

    point_by_method = {row["method"]: row for row in point_rows}
    doppler_by_method = {row["method"]: row for row in doppler_rows}
    bmode_by_view = {
        view["view"]: {row["method"]: row for row in view["metrics"]}
        for view in bmode.get("views", [])
    }
    baseline_rows = [
        row for row in timing.get("rows", []) if row.get("run_id") == "grid_128x256"
    ]
    timing_by_method = {row["method"]: row for row in baseline_rows}
    conventional_psf_by_method = {
        row["method"]: row for row in conventional_psf_rows
    }
    conventional_by_view = {
        view["view"]: {row["method"]: row for row in view["metrics"]}
        for view in conventional_views
    }
    conventional_timing_by_method = {
        row["method"]: row for row in conventional_timing_rows
    }

    for method in METHODS:
        if method not in point_by_method:
            raise KeyError(f"Missing point-target row for {method}")
        if method not in doppler_by_method:
            raise KeyError(f"Missing MBTrace row for {method}")
        if TIMING_METHODS[method] not in timing_by_method:
            raise KeyError(f"Missing baseline timing row for {method}")
        for view in ("CC", "CL"):
            if method not in bmode_by_view.get(view, {}):
                raise KeyError(f"Missing {view} carotid row for {method}")
    for method in CONVENTIONAL_METHODS:
        if method not in conventional_psf_by_method:
            raise KeyError(f"Missing conventional PSF row for {method}")
        if method not in conventional_timing_by_method:
            raise KeyError(f"Missing conventional timing row for {method}")
        for view in ("CC", "CL"):
            if method not in conventional_by_view.get(view, {}):
                raise KeyError(f"Missing conventional {view} row for {method}")

    das_width = finite(point_by_method["DAS"]["lateral_fwhm_mm"], "DAS width")
    receive_width = finite(
        point_by_method["Receive-NSI"]["lateral_fwhm_mm"], "Receive-NSI width"
    )
    angle_width = finite(
        point_by_method["Angle-NSI"]["lateral_fwhm_mm"], "Angle-NSI width"
    )
    receive_timing = finite(
        timing_by_method[TIMING_METHODS["Receive-NSI"]]["median_ms"],
        "Receive-NSI timing",
    )
    angle_timing = finite(
        timing_by_method[TIMING_METHODS["Angle-NSI"]]["median_ms"],
        "Angle-NSI timing",
    )
    signed_angle_advantage_percent = 100.0 * (
        receive_timing - angle_timing
    ) / receive_timing
    angle_receive_comparison = (
        f"{abs(signed_angle_advantage_percent):.1f}\\% "
        + ("lower" if signed_angle_advantage_percent >= 0.0 else "higher")
    )
    experimental_summaries = experimental["summaries"]

    macros = [
        "% Generated by scripts/materialize_revision_assets.py; do not edit.",
        tex_macro("RevisionAnalysisStatus", "complete"),
        tex_macro("PointDasWidth", fmt(das_width)),
        tex_macro("PointCfWidth", fmt(point_by_method["Angular CF-DAS"]["lateral_fwhm_mm"])),
        tex_macro("PointReceiveWidth", fmt(receive_width, 4)),
        tex_macro("PointAngleWidth", fmt(angle_width)),
        tex_macro("PointAngleDasReduction", fmt(100.0 * (1.0 - angle_width / das_width), 1)),
        tex_macro("PointAngleReceiveRatio", fmt(angle_width / receive_width, 1)),
        tex_macro("TimingDasMedian", fmt(timing_by_method[TIMING_METHODS["DAS"]]["median_ms"], 2)),
        tex_macro("TimingCfMedian", fmt(timing_by_method[TIMING_METHODS["Angular CF-DAS"]]["median_ms"], 2)),
        tex_macro("TimingReceiveMedian", fmt(receive_timing, 2)),
        tex_macro("TimingAngleMedian", fmt(angle_timing, 2)),
        tex_macro("TimingAngleLowerLatency", fmt(signed_angle_advantage_percent, 1)),
        tex_macro("TimingAngleReceiveComparison", angle_receive_comparison),
        tex_macro("TimingAngleDasOverhead", fmt(100.0 * (angle_timing / finite(timing_by_method[TIMING_METHODS['DAS']]['median_ms'], 'DAS timing') - 1.0), 1)),
        tex_macro("MbtraceAngleMatch", pct(doppler_by_method["Angle-NSI"]["das_concordant_fraction_at_0p15_mm"])),
        tex_macro("MbtraceReceiveMatch", pct(doppler_by_method["Receive-NSI"]["das_concordant_fraction_at_0p15_mm"])),
        tex_macro("MbtraceCfMatch", pct(doppler_by_method["Angular CF-DAS"]["das_concordant_fraction_at_0p15_mm"])),
        tex_macro("MbtraceAngleWidth", fmt(doppler_by_method["Angle-NSI"]["mean_matched_width_mm"])),
        tex_macro("MbtraceReceiveWidth", fmt(doppler_by_method["Receive-NSI"]["mean_matched_width_mm"])),
        tex_macro("MbtraceCfWidth", fmt(doppler_by_method["Angular CF-DAS"]["mean_matched_width_mm"])),
        tex_macro("ExpPsfDasOnAxisWidth", fmt(experimental_summaries["on-axis"]["DAS"]["lateral_width_mm"]["mean"])),
        tex_macro("ExpPsfCfOnAxisWidth", fmt(experimental_summaries["on-axis"]["Angular CF-DAS"]["lateral_width_mm"]["mean"])),
        tex_macro("ExpPsfReceiveOnAxisWidth", fmt(experimental_summaries["on-axis"]["Receive-NSI"]["lateral_width_mm"]["mean"])),
        tex_macro("ExpPsfAngleOnAxisWidth", fmt(experimental_summaries["on-axis"]["Angle-NSI"]["lateral_width_mm"]["mean"])),
        tex_macro("CarotidCcCfGcnr", fmt(bmode_by_view["CC"]["Angular CF-DAS"]["gcnr"])),
        tex_macro("CarotidClCfGcnr", fmt(bmode_by_view["CL"]["Angular CF-DAS"]["gcnr"])),
        tex_macro("TimingDasTransferMib", fmt(finite(timing_by_method[TIMING_METHODS["DAS"]]["timed_total_transfer_bytes"], "DAS bytes") / 2**20, 2)),
        tex_macro("TimingReceiveTransferMib", fmt(finite(timing_by_method[TIMING_METHODS["Receive-NSI"]]["timed_total_transfer_bytes"], "Receive bytes") / 2**20, 2)),
        tex_macro("TimingAngleTransferMib", fmt(finite(timing_by_method[TIMING_METHODS["Angle-NSI"]]["timed_total_transfer_bytes"], "Angle bytes") / 2**20, 2)),
    ]
    conventional_macro_stems = {
        "DAS": "ConventionalDas",
        "Receive CF-DAS": "ConventionalCf",
        "MV": "ConventionalMv",
        "F-DMAS": "ConventionalFdmas",
        "Receive-NSI": "ConventionalReceiveNsi",
        "Angle-NSI": "ConventionalAngleNsi",
    }
    for method in CONVENTIONAL_METHODS:
        stem = conventional_macro_stems[method]
        macros.extend(
            [
                tex_macro(
                    f"{stem}PsfWidth",
                    fmt(conventional_psf_by_method[method]["lateral_width_mm"], 4),
                ),
                tex_macro(
                    f"{stem}CcGcnr",
                    fmt(conventional_by_view["CC"][method]["gcnr"]),
                ),
                tex_macro(
                    f"{stem}ClGcnr",
                    fmt(conventional_by_view["CL"][method]["gcnr"]),
                ),
                tex_macro(
                    f"{stem}KernelTime",
                    fmt(conventional_timing_by_method[method]["median_ms"], 3),
                ),
            ]
        )
    write_text(output / "revision_results.tex", "\n".join(macros))

    tradeoff_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Point width & MBTrace width & DAS concordance & Baseline time \\",
        r" & (mm) & (mm) & (\%) & (ms) \\",
        r"\midrule",
    ]
    for method in METHODS:
        p = point_by_method[method]
        d = doppler_by_method[method]
        t = timing_by_method[TIMING_METHODS[method]]
        match = 1.0 if method == "DAS" else finite(
            d["das_concordant_fraction_at_0p15_mm"], f"{method} concordance"
        )
        tradeoff_lines.append(
            f"{method_label(method)} & {fmt(p['lateral_fwhm_mm'], 4)} & "
            f"{fmt(d['mean_matched_width_mm'])} & {100.0 * match:.1f} & "
            f"{fmt(t['median_ms'], 2)} \\\\"
        )
    tradeoff_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_method_tradeoff.tex", "\n".join(tradeoff_lines))

    exp_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & On-axis width & At 37.5 mm & Axial width & Localization error \\",
        r" & (mm) & (mm) & (mm) & (mm) \\",
        r"\midrule",
    ]
    for method in METHODS:
        summaries = experimental["summaries"]
        exp_lines.append(
            f"{method_label(method)} & "
            f"{fmt(summaries['on-axis'][method]['lateral_width_mm']['mean'])} & "
            f"{fmt(summaries['37.5-mm depth'][method]['lateral_width_mm']['mean'])} & "
            f"{fmt(summaries['all'][method]['axial_width_mm']['mean'])} & "
            f"{fmt(summaries['all'][method]['localization_error_mm']['mean'])} \\\\"
        )
    exp_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_experimental_psf.tex", "\n".join(exp_lines))

    carotid_lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Metric & View & DAS & Angular CF-DAS & Receive-NSI & Angle-NSI \\",
        r"\midrule",
    ]
    for metric, label, digits in (
        ("contrast_ratio_db", "CR (dB)", 2),
        ("cnr", "CNR", 3),
        ("gcnr", "gCNR", 3),
    ):
        for index, view in enumerate(("CC", "CL")):
            label_cell = label if index == 0 else ""
            values = " & ".join(
                fmt(bmode_by_view[view][method][metric], digits) for method in METHODS
            )
            carotid_lines.append(f"{label_cell} & {view} & {values} \\\\ ")
    carotid_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_carotid_metrics.tex", "\n".join(carotid_lines))

    timing_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Median & 95\% CI & Relative to DAS & Timed transfer \\",
        r" & (ms) & (ms) & (ratio) & (MiB) \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = timing_by_method[TIMING_METHODS[method]]
        bytes_value = finite(row["timed_total_transfer_bytes"], f"{method} bytes")
        timing_lines.append(
            f"{method_label(method)} & {fmt(row['median_ms'], 2)} & "
            f"[{fmt(row['median_95ci_lower_ms'], 2)}, {fmt(row['median_95ci_upper_ms'], 2)}] & "
            f"{fmt(row['ratio_to_das_median'], 3)} & {bytes_value / 2**20:.2f} \\\\"
        )
    timing_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_timing_baseline.tex", "\n".join(timing_lines))

    conventional_lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Method & Lateral width & Peak/background & CC gCNR & CL gCNR & Kernel time \\",
        r" & (mm) & (dB) & & & (ms) \\",
        r"\midrule",
    ]
    for method in CONVENTIONAL_METHODS:
        psf_row = conventional_psf_by_method[method]
        conventional_lines.append(
            f"{method_label(method)} & {fmt(psf_row['lateral_width_mm'], 4)} & "
            f"{fmt(psf_row['peak_to_median_background_db'], 2)} & "
            f"{fmt(conventional_by_view['CC'][method]['gcnr'])} & "
            f"{fmt(conventional_by_view['CL'][method]['gcnr'])} & "
            f"{fmt(conventional_timing_by_method[method]['median_ms'], 3)} \\\\"
        )
    conventional_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(
        output / "table_conventional_baselines.tex",
        "\n".join(conventional_lines),
    )

    complexity_labels = {
        "DAS": r"$O(M)$",
        "Receive CF-DAS": r"$O(M)$",
        "MV": r"$O((2K+1)(M-L+1)L^2+L^3)$",
        "F-DMAS": r"$O(M^2)$ direct; $O(M)$ exact identity",
        "Receive-NSI": r"$O(M)$",
        "Angle-NSI": r"$O(M)$",
    }
    conventional_das_time = finite(
        conventional_timing_by_method["DAS"]["median_ms"],
        "conventional DAS timing",
    )
    conventional_timing_lines = [
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        r"Method & Median (ms) & Relative to DAS & Leading order per pixel/angle \\",
        r"\midrule",
    ]
    for method in CONVENTIONAL_METHODS:
        median = finite(
            conventional_timing_by_method[method]["median_ms"],
            f"{method} conventional timing",
        )
        conventional_timing_lines.append(
            f"{method_label(method)} & {median:.3f} & "
            f"{median / conventional_das_time:.2f} & {complexity_labels[method]} \\\\"
        )
    conventional_timing_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(
        output / "table_conventional_timing.tex",
        "\n".join(conventional_timing_lines),
    )

    mv_view_windows = {
        view["view"]: int(view["mv_temporal_half_window_samples"])
        for view in conventional_views
    }
    parameter_lines = [
        r"\begin{tabular}{@{}lp{0.74\linewidth}@{}}",
        r"\toprule",
        r"Method & Fixed implementation choices \\",
        r"\midrule",
        r"Receive CF-DAS & Receive-aperture CF per transmit angle; no fitted parameter. \\",
        (
            r"MV & $L=\lfloor M_{\mathrm{active}}/2\rfloor$; all overlapping "
            r"subarrays; loading $\mathrm{tr}(R)/(100L)$; $K=0$ for the point "
            f"target and axial half-windows $K={mv_view_windows['CC']}$ (CC), "
            f"$K={mv_view_windows['CL']}$ (CL). \\\\"
        ),
        (
            r"F-DMAS & Signed square-root products for every $i<j$ pair; Kaiser FIR "
            r"edges $(1.5,1.75,2.5,2.75)f_0$; analytic signal after filtering. \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_text(
        output / "table_conventional_parameters.tex",
        "\n".join(parameter_lines),
    )

    c_lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & $c$ & Detected peaks & Match at 0.15 mm & Match at 0.35 mm \\",
        r"\midrule",
    ]
    for row in doppler_c.get("rows", []):
        c_lines.append(
            f"{method_label(row['method'])} & {fmt(row['c'], 2)} & "
            f"{int(row['detected_peak_count'])} & "
            f"{pct(row['matched_fraction_at_0p15_mm'])}\\% & "
            f"{pct(row['matched_fraction_at_0p35_mm'])}\\% \\\\"
        )
    c_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_mbtrace_c_sensitivity.tex", "\n".join(c_lines))

    displacement_lines = [
        r"\begin{tabular}{@{}lrrrrl@{}}",
        r"\toprule",
        r"Method & DAS peak & Assigned peak & Signed shift & Absolute shift & Classification \\",
        r" & (mm) & (mm) & (mm) & (mm) & \\",
        r"\midrule",
    ]
    for row in displacement_rows:
        target = row.get("target_x_mm", "")
        signed = row.get("signed_displacement_mm", "")
        absolute = row.get("absolute_displacement_mm", "")
        label = {
            "receive_nsi": "Receive-NSI",
            "angle_nsi": "Angle-NSI",
        }.get(row["method"], row["method"])
        status = row["status"].replace("_", " ")
        if row.get("possible_split", "").lower() == "true":
            status += "; possible split"
        displacement_lines.append(
            f"{label} & {fmt(row['reference_x_mm'])} & "
            f"{fmt(target) if target else '--'} & "
            f"{fmt(signed) if signed else '--'} & "
            f"{fmt(absolute) if absolute else '--'} & {status} \\\\"
        )
    displacement_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_mbtrace_displacements.tex", "\n".join(displacement_lines))

    control_lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Control & Method & Median (ms) & Relative to DAS & Transfer (MiB) \\",
        r"\midrule",
    ]
    for run_id, control_label in (
        ("baseline_preloaded", "Preloaded kernel"),
        ("baseline_host_stacked", "Host-stacked input"),
    ):
        rows = [row for row in timing.get("rows", []) if row.get("run_id") == run_id]
        for row in rows:
            canonical = next(
                (name for name, timing_name in TIMING_METHODS.items()
                 if timing_name == row["method"]),
                row["method"],
            )
            label = method_label(canonical) if canonical in METHODS else canonical
            control_lines.append(
                f"{control_label} & {label} & {fmt(row['median_ms'], 2)} & "
                f"{fmt(row['ratio_to_das_median'], 3)} & "
                f"{finite(row['timed_total_transfer_bytes'], 'control bytes') / 2**20:.2f} \\\\"
            )
    control_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text(output / "table_timing_controls.tex", "\n".join(control_lines))

    figure_map = {
        results / "point_target" / "single_scatterer_fine_psf_contours.png": output / "figures" / "figure1a_psf_contours_revision.png",
        results / "point_target" / "single_scatterer_fwhm_profiles.png": output / "figures" / "figure1b_psf_profiles_revision.png",
        results / "experimental_psf" / "picmus_experimental_psf.png": output / "figures" / "figure2_experimental_psf.png",
        results / "doppler" / "power_doppler_four_method_comparison.png": output / "figures" / "figure3_microbubble_power_doppler_revision.png",
        results / "bmode" / "Fig_BMode_NSI_Comparison_CL.png": output / "figures" / "figure4a_carotid_longitudinal_revision.png",
        results / "bmode" / "Fig_BMode_NSI_Comparison_CC.png": output / "figures" / "figure4b_carotid_cross_section_revision.png",
        results / "timing_scaling" / "nsi_timing_scaling_summary.png": output / "figures" / "figure5_timing_scaling.png",
        results / "theory" / "angular_null_small_angle_derivation.png": output / "supplementary" / "figureS1_angular_null_theory.png",
        results / "point_target" / "single_scatterer_c_sensitivity.png": output / "supplementary" / "figureS5_point_target_c_sensitivity.png",
        results / "doppler" / "microbubble_peak_concordance_tolerance_sweep.png": output / "supplementary" / "figureS6_mbtrace_concordance.png",
        results / "doppler" / "mbtrace_c_sensitivity.png": output / "supplementary" / "figureS7_mbtrace_c_sensitivity.png",
        results / "bmode" / "bmode_nsi_c_sensitivity.png": output / "supplementary" / "figureS8_carotid_c_sensitivity.png",
        results / "experimental_psf" / "picmus_experimental_psf_c_sensitivity.png": output / "supplementary" / "figureS9_experimental_psf_c_sensitivity.png",
        results / "conventional_baselines" / "conventional_baseline_experimental_psf.png": output / "reviewer_comparison" / "conventional_experimental_psf.png",
        results / "conventional_baselines" / "conventional_baseline_experimental_psf_profiles.png": output / "reviewer_comparison" / "conventional_experimental_psf_profiles.png",
        results / "conventional_baselines" / "conventional_baseline_carotid_CL.png": output / "reviewer_comparison" / "conventional_carotid_longitudinal.png",
        results / "conventional_baselines" / "conventional_baseline_carotid_CC.png": output / "reviewer_comparison" / "conventional_carotid_cross_section.png",
        results / "conventional_timing" / "conventional_timing_summary.png": output / "reviewer_comparison" / "conventional_timing.png",
    }
    for source, destination in figure_map.items():
        copy_required(source, destination)

    manifest = {
        "publication_checks_passed": not problems,
        "problems": problems,
        "results_root": str(results),
        "generated_files": sorted(
            str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
        ),
    }
    write_text(output / "revision_asset_manifest.json", json.dumps(manifest, indent=2))
    print(f"Materialized {len(manifest['generated_files']) + 1} revision assets in {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run and aggregate the publication timing-sensitivity suite.

Each case delegates numerical validation and synchronized interleaving to
``benchmark_nsi.py``.  Completed cases are resumable.  The primary suite uses
identical raw-IQ host-to-device payloads for DAS, CF-DAS, Receive-NSI, and
Angle-NSI; two baseline controls isolate preloaded-kernel timing and the older
host-stacked Receive-NSI API.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_METHODS = {
    "DAS / coherent compounding",
    "Angular CF-DAS",
    "Conventional NSI (two fields)",
    "Angular NSI (streaming)",
}


@dataclass(frozen=True)
class TimingCase:
    run_id: str
    family: str
    value: int
    angles: int = 17
    elements: int = 128
    nx: int = 128
    nz: int = 256
    scope: str = "reconstruction"
    transfers: str = "both"
    receive_input_mode: str = "device-weighted"


def build_cases() -> list[TimingCase]:
    """Return the declared, ordered revision benchmark matrix."""

    return [
        TimingCase("grid_64x128", "grid_pixels", 64 * 128, nx=64, nz=128),
        TimingCase("grid_128x256", "grid_pixels", 128 * 256),
        TimingCase("grid_256x512", "grid_pixels", 256 * 512, nx=256, nz=512),
        TimingCase("elements_64", "receive_elements", 64, elements=64),
        TimingCase("elements_128", "receive_elements", 128),
        TimingCase("elements_256", "receive_elements", 256, elements=256),
        TimingCase("angles_9", "transmit_angles", 9, angles=9),
        TimingCase("angles_17", "transmit_angles", 17),
        TimingCase("angles_33", "transmit_angles", 33, angles=33),
        TimingCase("angles_75", "transmit_angles", 75, angles=75),
        TimingCase(
            "baseline_preloaded",
            "timing_scope",
            0,
            scope="kernel",
            transfers="none",
        ),
        TimingCase(
            "baseline_host_stacked",
            "receive_transfer_mode",
            2,
            receive_input_mode="host-stacked",
        ),
    ]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the NSI timing scaling suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "generated" / "timing_scaling",
    )
    parser.add_argument(
        "--benchmark-script", type=Path, default=root / "src" / "benchmark_nsi.py"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument(
        "--suite",
        choices=("all", "primary", "baseline"),
        default="all",
        help=(
            "primary runs dimension sweeps; baseline runs the equal-transfer "
            "baseline plus preloaded and host-stacked controls."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 1 warm-up and 3 repetitions, retaining every matrix dimension.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def selected_cases(cases: list[TimingCase], suite: str) -> list[TimingCase]:
    if suite == "all":
        return cases
    if suite == "primary":
        return [case for case in cases if case.family in {
            "grid_pixels", "receive_elements", "transmit_angles"
        }]
    baseline_ids = {
        "grid_128x256", "baseline_preloaded", "baseline_host_stacked"
    }
    return [case for case in cases if case.run_id in baseline_ids]


def case_command(case: TimingCase, args: argparse.Namespace, run_dir: Path) -> list[str]:
    warmups = 1 if args.quick else args.warmups
    repetitions = 3 if args.quick else args.repetitions
    return [
        args.python,
        str(args.benchmark_script.resolve()),
        "--output-dir", str(run_dir),
        "--device", str(args.device),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
        "--bootstrap-resamples", str(args.bootstrap_resamples),
        "--configuration-label", case.run_id,
        "--angles", str(case.angles),
        "--elements", str(case.elements),
        "--nx", str(case.nx),
        "--nz", str(case.nz),
        "--scope", case.scope,
        "--transfers", case.transfers,
        "--receive-input-mode", case.receive_input_mode,
        "--angle-storage-mode", "streaming",
    ]


def load_completed_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "nsi_timing_summary.json"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return result if result.get("summary") else None


def completed_summary_matches(
    payload: dict[str, Any], case: TimingCase, args: argparse.Namespace
) -> bool:
    """Return whether a cached case exactly matches the requested run.

    This prevents a short engineering run, a different transfer scope, or a
    differently sized matrix from being silently reused by a publication run.
    """

    warmups = 1 if args.quick else int(args.warmups)
    repetitions = 3 if args.quick else int(args.repetitions)
    configuration = payload.get("configuration", {})
    expected = {
        "warmups": warmups,
        "repetitions": repetitions,
        "configuration_label": case.run_id,
        "receive_elements": case.elements,
        "angle_count": case.angles,
    }
    if any(configuration.get(key) != value for key, value in expected.items()):
        return False
    image_grid = configuration.get("image_grid", {})
    if image_grid.get("nx") != case.nx or image_grid.get("nz") != case.nz:
        return False
    if payload.get("scope") != case.scope:
        return False
    included = payload.get("included_transfers", {})
    expected_input = case.transfers in {"input", "both"}
    expected_output = case.transfers in {"output", "both"}
    if included.get("channel_host_to_device") is not expected_input:
        return False
    if included.get("final_image_device_to_host") is not expected_output:
        return False
    if payload.get("receive_input_mode") != case.receive_input_mode:
        return False
    if payload.get("angle_image_storage") != "streaming":
        return False
    summaries = payload.get("summary", [])
    if {row.get("method") for row in summaries} != EXPECTED_METHODS:
        return False
    return all(int(row.get("n", 0)) == repetitions for row in summaries)


def flatten_summaries(
    completed: list[tuple[TimingCase, dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, payload in completed:
        for summary in payload["summary"]:
            rows.append(
                {
                    "run_id": case.run_id,
                    "family": case.family,
                    "family_value": case.value,
                    "angles": case.angles,
                    "elements": case.elements,
                    "nx": case.nx,
                    "nz": case.nz,
                    "grid_pixels": case.nx * case.nz,
                    "scope": case.scope,
                    "transfers": case.transfers,
                    "receive_input_mode": case.receive_input_mode,
                    **summary,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No completed benchmark rows are available.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(path: Path, rows: list[dict[str, Any]]) -> None:
    families = ["grid_pixels", "receive_elements", "transmit_angles"]
    labels = {
        "grid_pixels": "Image-grid points",
        "receive_elements": "Receive elements",
        "transmit_angles": "Transmit angles",
    }
    method_order: list[str] = []
    for row in rows:
        if row["method"] not in method_order:
            method_order.append(row["method"])
    colors = ["#6a3d9a", "#2ca02c", "#d62728", "#1f77b4", "#17becf"]

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), dpi=180)
    for axis, family in zip(axes, families):
        family_rows = [row for row in rows if row["family"] == family]
        for color, method in zip(colors, method_order):
            selected = sorted(
                (row for row in family_rows if row["method"] == method),
                key=lambda row: int(row["family_value"]),
            )
            if not selected:
                continue
            x = [int(row["family_value"]) for row in selected]
            y = [float(row["median_ms"]) for row in selected]
            lower = [
                y_value - float(row.get("median_95ci_lower_ms") or y_value)
                for row, y_value in zip(selected, y)
            ]
            upper = [
                float(row.get("median_95ci_upper_ms") or y_value) - y_value
                for row, y_value in zip(selected, y)
            ]
            axis.errorbar(
                x, y, yerr=[lower, upper], marker="o", linewidth=1.5,
                capsize=2, color=color, label=method,
            )
        axis.set_xlabel(labels[family])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Median synchronized time [ms]")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.warmups < 1 or args.repetitions < 1:
        raise SystemExit("Warm-ups and repetitions must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(build_cases(), args.suite)
    manifest_path = output_dir / "timing_scaling_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quick_engineering_run": bool(args.quick),
        "requested_warmups": 1 if args.quick else args.warmups,
        "requested_repetitions": 3 if args.quick else args.repetitions,
        "suite": args.suite,
        "cases": [asdict(case) for case in cases],
        "interpretation": (
            "Primary cases use equal raw-IQ transfer bytes and form Receive-NSI "
            "weights on device. Controls isolate preloaded execution and the "
            "older host-stacked two-frame transfer path."
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    completed: list[tuple[TimingCase, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        run_dir = output_dir / "runs" / case.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = None if args.force else load_completed_summary(run_dir)
        if existing is not None and not completed_summary_matches(
            existing, case, args
        ):
            print(
                f"[{index}/{len(cases)}] Ignoring stale or incompatible "
                f"cached result for {case.run_id}"
            )
            existing = None
        if existing is not None:
            print(f"[{index}/{len(cases)}] Reusing completed {case.run_id}")
            completed.append((case, existing))
            continue
        command = case_command(case, args, run_dir)
        print(f"[{index}/{len(cases)}] Running {case.run_id}")
        print("  " + " ".join(command))
        if args.dry_run:
            continue
        try:
            subprocess.run(command, check=True)
            payload = load_completed_summary(run_dir)
            if payload is None:
                raise RuntimeError("benchmark completed without a readable summary")
            completed.append((case, payload))
        except (subprocess.CalledProcessError, RuntimeError) as error:
            failures.append({"run_id": case.run_id, "error": str(error)})
            if not args.continue_on_error:
                raise SystemExit(f"Timing case {case.run_id} failed: {error}") from error

    if args.dry_run:
        print(f"Dry run complete; manifest: {manifest_path}")
        return
    if not completed:
        raise SystemExit("No timing cases completed.")

    rows = flatten_summaries(completed)
    csv_path = output_dir / "nsi_timing_scaling_summary.csv"
    json_path = output_dir / "nsi_timing_scaling_summary.json"
    figure_path = output_dir / "nsi_timing_scaling_summary.png"
    write_csv(csv_path, rows)
    plot_scaling(figure_path, rows)
    aggregate = {
        **manifest,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_case_count": len(completed),
        "failed_cases": failures,
        "rows": rows,
        "output_files": {
            "summary_csv": csv_path.name,
            "summary_figure": figure_path.name,
            "manifest": manifest_path.name,
        },
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for path in (manifest_path, csv_path, json_path, figure_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Expected output was not written correctly: {path}")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

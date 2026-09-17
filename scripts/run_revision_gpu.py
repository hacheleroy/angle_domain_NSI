#!/usr/bin/env python3
"""Run the complete major-revision GPU analysis as a resumable workflow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STEPS = (
    "theory",
    "point-target",
    "simulation-six-method",
    "mbtrace",
    "carotid",
    "experimental-psf",
    "picmus-full-phantom",
    "conventional-baselines",
    "timing",
    "conventional-timing",
    "manuscript-assets",
)
EXPECTED_METHODS = {
    "DAS / coherent compounding",
    "Angular CF-DAS",
    "Conventional NSI (two fields)",
    "Angular NSI (streaming)",
}
EXPECTED_TIMING_CASES = {
    "grid_64x128", "grid_128x256", "grid_256x512",
    "elements_64", "elements_128", "elements_256",
    "angles_9", "angles_17", "angles_33", "angles_75",
    "baseline_preloaded", "baseline_host_stacked",
}
EXPECTED_C_VALUES = {0.02, 0.05, 0.10, 0.20}
EXPECTED_BASELINE_METHODS = {
    "DAS", "CF-DAS", "MV", "DMAS", "Receive-NSI", "Angle-NSI"
}
EXPECTED_PUBLICATION_METHODS = {
    "DAS", "CF-DAS", "MV", "DMAS", "Receive-NSI", "Angle-NSI"
}
CONVENTIONAL_BASELINE_SCHEMA_VERSION = 3
PICMUS_FULL_PHANTOM_SCHEMA_VERSION = 3


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run all PMB revision GPU analyses")
    parser.add_argument("--device", default="0")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument(
        "--output-root", type=Path, default=root / "results" / "generated" / "revision"
    )
    parser.add_argument(
        "--steps",
        default=",".join(DEFAULT_STEPS),
        help=f"Comma-separated subset of: {','.join(DEFAULT_STEPS)},robustness",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--timing-warmups", type=int, default=10)
    parser.add_argument("--timing-repetitions", type=int, default=50)
    return parser.parse_args()


def step_definitions(
    root: Path, data_root: Path, output_root: Path, args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    python = sys.executable
    common_env = {"CUDA_VISIBLE_DEVICES": str(args.device), "NSI_CUDA_DEVICE": str(args.device)}
    return {
        "theory": {
            "command": [
                python,
                str(root / "src" / "angular_null_theory.py"),
                "--output-dir", str(output_root / "theory"),
            ],
            "env": common_env,
            "expected": output_root / "theory" / "angular_null_theory.json",
        },
        "point-target": {
            "command": [python, str(root / "src" / "simulation_point_target.py")],
            "env": {**common_env, "NSI_OUTPUT_DIR": str(output_root / "point_target")},
            "expected": output_root / "point_target" / "simulation_psf_fwhm_summary.json",
        },
        "simulation-six-method": {
            "command": [
                python,
                str(root / "src" / "simulation_six_method_comparison.py"),
                "--output-dir", str(output_root / "simulation_six_method"),
                "--device", str(args.device),
            ],
            "env": common_env,
            "expected": output_root / "simulation_six_method" / "simulation_six_method_summary.json",
        },
        "mbtrace": {
            "command": [python, str(root / "src" / "doppler_mbtrace.py")],
            "env": {
                **common_env,
                "OPEN_NSI_MBTRACE_FILE": str(
                    data_root / "Open-NSI" / "Basic" / "data" / "MBTrace.mat"
                ),
                "NSI_OUTPUT_DIR": str(output_root / "doppler"),
            },
            "expected": output_root / "doppler" / "power_doppler_three_method_comparison.png",
        },
        "carotid": {
            "command": [python, str(root / "src" / "bmode_picmus.py")],
            "env": {
                **common_env,
                "PICMUS_DATA_DIR": str(data_root / "PICMUS" / "in_vivo"),
                "NSI_OUTPUT_DIR": str(output_root / "bmode"),
            },
            "expected": output_root / "bmode" / "bmode_nsi_c_sensitivity.json",
        },
        "experimental-psf": {
            "command": [
                python,
                str(root / "src" / "picmus_experimental_psf.py"),
                "--dataset", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_dataset_rf.hdf5"),
                "--phantom", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_phantom.hdf5"),
                "--scan", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_scan.hdf5"),
                "--output-dir", str(output_root / "experimental_psf"),
                "--device", str(args.device),
            ],
            "env": common_env,
            "expected": output_root / "experimental_psf" / "picmus_experimental_psf_summary.json",
        },
        "picmus-full-phantom": {
            "command": [
                python,
                str(root / "src" / "picmus_full_phantom_comparison.py"),
                "--dataset", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_dataset_rf.hdf5"),
                "--phantom", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_phantom.hdf5"),
                "--local-profile-summary", str(output_root / "experimental_psf" / "picmus_experimental_psf_summary.json"),
                "--local-profiles", str(output_root / "experimental_psf" / "picmus_experimental_psf_profiles.npz"),
                "--output-dir", str(output_root / "picmus_full_phantom"),
                "--device", str(args.device),
            ] + (["--force"] if args.force else []),
            "env": common_env,
            "expected": output_root / "picmus_full_phantom" / "picmus_full_phantom_summary.json",
        },
        "conventional-baselines": {
            "command": [
                python,
                str(root / "src" / "conventional_baseline_comparison.py"),
                "--resolution-dataset", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_dataset_rf.hdf5"),
                "--resolution-phantom", str(data_root / "PICMUS" / "resolution_distorsion" / "resolution_distorsion_expe_phantom.hdf5"),
                "--carotid-long", str(data_root / "PICMUS" / "in_vivo" / "carotid_long" / "carotid_long_expe_dataset_rf.hdf5"),
                "--carotid-cross", str(data_root / "PICMUS" / "in_vivo" / "carotid_cross" / "carotid_cross_expe_dataset_rf.hdf5"),
                "--output-dir", str(output_root / "conventional_baselines"),
                "--device", str(args.device),
            ] + (["--force"] if args.force else []),
            "env": common_env,
            "expected": output_root / "conventional_baselines" / "conventional_baseline_summary.json",
        },
        "timing": {
            "command": [
                python,
                str(root / "src" / "benchmark_scaling.py"),
                "--output-dir", str(output_root / "timing_scaling"),
                "--device", str(args.device),
                "--warmups", str(args.timing_warmups),
                "--repetitions", str(args.timing_repetitions),
            ] + (["--force"] if args.force else []),
            "env": common_env,
            "expected": output_root / "timing_scaling" / "nsi_timing_scaling_summary.json",
        },
        "conventional-timing": {
            "command": [
                python,
                str(root / "src" / "benchmark_conventional.py"),
                "--output-dir", str(output_root / "conventional_timing"),
                "--device", str(args.device),
                "--warmups", str(args.timing_warmups),
                "--repetitions", str(args.timing_repetitions),
            ],
            "env": common_env,
            "expected": output_root / "conventional_timing" / "conventional_timing_summary.json",
        },
        "manuscript-assets": {
            "command": [
                python,
                str(root / "scripts" / "materialize_revision_assets.py"),
                "--results-root", str(output_root),
                "--output-dir", str(output_root / "manuscript_assets"),
            ],
            "env": common_env,
            "expected": output_root / "manuscript_assets" / "revision_asset_manifest.json",
        },
        "robustness": {
            "command": [
                python,
                str(root / "src" / "simulation_robustness.py"),
                "--output-dir", str(output_root / "robustness"),
                "--device", str(args.device),
            ],
            "env": common_env,
            "expected": output_root / "robustness" / "robustness_summary.json",
        },
    }


def reusable_output(
    name: str, expected: Path, args: argparse.Namespace
) -> bool:
    """Return whether an existing step output is safe to reuse."""

    if not expected.is_file() or expected.stat().st_size == 0:
        return False
    # Assets are cheap to regenerate and must always reflect the newest
    # upstream JSON/CSV files.
    if name == "manuscript-assets":
        return False
    if name not in {
        "timing", "experimental-psf", "conventional-baselines",
        "conventional-timing", "simulation-six-method", "picmus-full-phantom",
    }:
        return True
    try:
        with expected.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    if name == "experimental-psf":
        summaries = payload.get("summaries", {})
        try:
            c_values = {
                round(float(value), 8)
                for value in payload.get("reconstruction", {}).get(
                    "c_values", []
                )
            }
        except (TypeError, ValueError):
            return False
        return bool(
            payload.get("metadata_only") is False
            and len(payload.get("target_positions_mm", [])) == 7
            and c_values == EXPECTED_C_VALUES
            and all(
                int(summaries.get(group, {}).get(method, {}).get(
                    "lateral_width_mm", {}
                ).get("n", 0)) == expected_count
                for group, expected_count in (
                    ("all", 7), ("on-axis", 5), ("37.5-mm depth", 3)
                )
                for method in (
                    "DAS", "Angular CF-DAS", "Receive-NSI", "Angle-NSI"
                )
            )
        )
    if name == "simulation-six-method":
        return bool(
            payload.get("metadata_only") is False
            and payload.get("publication_ready") is True
            and set(payload.get("methods", [])) == EXPECTED_PUBLICATION_METHODS
        )
    if name == "picmus-full-phantom":
        profile_rows = payload.get("profile_diagnostics", [])
        return bool(
            int(payload.get("schema_version", 0))
            == PICMUS_FULL_PHANTOM_SCHEMA_VERSION
            and payload.get("metadata_only") is False
            and payload.get("publication_ready") is True
            and set(payload.get("methods", [])) == EXPECTED_PUBLICATION_METHODS
            and len(profile_rows) == 21
            and all(
                "central_notch_depth_db" in row
                and "central_notch_detected" in row
                for row in profile_rows
            )
        )
    if name == "conventional-baselines":
        psf_methods = {
            row.get("method")
            for row in payload.get("experimental_psf", {}).get("metrics", [])
        }
        views = payload.get("carotid_views", [])
        view_methods = {
            view.get("view"): {
                row.get("method") for row in view.get("metrics", [])
            }
            for view in views
        }
        return bool(
            int(payload.get("schema_version", 0))
            == CONVENTIONAL_BASELINE_SCHEMA_VERSION
            and payload.get("metadata_only") is False
            and payload.get("quick_engineering_run") is False
            and payload.get("publication_ready") is True
            and set(payload.get("methods", [])) == EXPECTED_BASELINE_METHODS
            and psf_methods == EXPECTED_BASELINE_METHODS
            and set(view_methods) == {"CC", "CL"}
            and all(
                methods == EXPECTED_BASELINE_METHODS
                for methods in view_methods.values()
            )
        )
    if name == "conventional-timing":
        rows = payload.get("rows", [])
        return bool(
            payload.get("quick_engineering_run") is False
            and payload.get("publication_ready") is True
            and int(payload.get("requested_warmups", 0)) == args.timing_warmups
            and int(payload.get("requested_repetitions", 0))
            == args.timing_repetitions
            and {row.get("method") for row in rows}
            == EXPECTED_BASELINE_METHODS
            and all(
                int(row.get("n", 0)) == args.timing_repetitions
                and row.get("scope") == "post-delay beamformer kernel"
                for row in rows
            )
        )
    rows = payload.get("rows", [])
    rows_by_case = {
        run_id: [row for row in rows if row.get("run_id") == run_id]
        for run_id in EXPECTED_TIMING_CASES
    }
    basic_checks = bool(
        payload.get("quick_engineering_run") is False
        and payload.get("suite") == "all"
        and int(payload.get("requested_warmups", 0)) == args.timing_warmups
        and int(payload.get("requested_repetitions", 0))
        == args.timing_repetitions
        and not payload.get("failed_cases")
        and rows
        and {row.get("run_id") for row in rows} == EXPECTED_TIMING_CASES
        and all(
            int(row.get("n", 0)) == args.timing_repetitions for row in rows
        )
        and all(
            {row.get("method") for row in case_rows} == EXPECTED_METHODS
            for case_rows in rows_by_case.values()
        )
    )
    if not basic_checks:
        return False
    primary = EXPECTED_TIMING_CASES - {
        "baseline_preloaded", "baseline_host_stacked"
    }
    return bool(
        all(
            row.get("scope") == "reconstruction"
            and row.get("transfers") == "both"
            and row.get("receive_input_mode") == "device-weighted"
            for run_id in primary
            for row in rows_by_case[run_id]
        )
        and all(
            row.get("scope") == "kernel"
            and row.get("transfers") == "none"
            and row.get("receive_input_mode") == "device-weighted"
            for row in rows_by_case["baseline_preloaded"]
        )
        and all(
            row.get("scope") == "reconstruction"
            and row.get("transfers") == "both"
            and row.get("receive_input_mode") == "host-stacked"
            for row in rows_by_case["baseline_host_stacked"]
        )
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    definitions = step_definitions(root, data_root, output_root, args)
    steps = tuple(part.strip() for part in args.steps.split(",") if part.strip())
    unknown = [step for step in steps if step not in definitions]
    if not steps or unknown:
        raise SystemExit(f"Unknown or empty --steps selection: {unknown}")
    manifest_path = output_root / "revision_gpu_run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(args.device),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "requested_steps": list(steps),
        "steps": [],
    }
    write_manifest(manifest_path, manifest)

    for index, name in enumerate(steps, start=1):
        definition = definitions[name]
        expected = Path(definition["expected"])
        if not args.force and reusable_output(name, expected, args):
            print(f"[{index}/{len(steps)}] Reusing completed {name}: {expected}")
            manifest["steps"].append({
                "name": name, "status": "reused", "expected_output": str(expected)
            })
            write_manifest(manifest_path, manifest)
            continue
        expected.parent.mkdir(parents=True, exist_ok=True)
        command = definition["command"]
        print(f"[{index}/{len(steps)}] Running {name}")
        print("  " + shlex.join(command))
        environment = os.environ.copy()
        environment.update(definition["env"])
        started = time.perf_counter()
        record = {
            "name": name,
            "command": command,
            "expected_output": str(expected),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            subprocess.run(command, check=True, cwd=root, env=environment)
            if not expected.is_file() or expected.stat().st_size == 0:
                raise RuntimeError(f"Expected output was not created: {expected}")
            record["status"] = "completed"
        except (subprocess.CalledProcessError, RuntimeError) as error:
            record["status"] = "failed"
            record["error"] = str(error)
            if not args.continue_on_error:
                record["duration_seconds"] = time.perf_counter() - started
                manifest["steps"].append(record)
                manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
                manifest["all_requested_steps_completed"] = False
                write_manifest(manifest_path, manifest)
                raise SystemExit(f"Revision step {name} failed: {error}") from error
        record["duration_seconds"] = time.perf_counter() - started
        manifest["steps"].append(record)
        write_manifest(manifest_path, manifest)

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["all_requested_steps_completed"] = all(
        row["status"] in {"completed", "reused"} for row in manifest["steps"]
    )
    write_manifest(manifest_path, manifest)
    print(f"Revision GPU workflow complete: {manifest_path}")


if __name__ == "__main__":
    main()

"""Regression checks against the frozen final paper, including all 42 PSFs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from method_names import METHODS
from picmus_experimental_psf import metrics_for_method

SPEC = importlib.util.spec_from_file_location("paper_reproduction", ROOT / "scripts/reproduce_paper.py")
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)


class FinalPaperTests(unittest.TestCase):
    def test_frozen_inputs_and_used_assets_are_intact(self):
        manifest = replay.read_manifest()
        replay.verify_sources(manifest)
        replay.verify_reference_assets(manifest)
        self.assertEqual(manifest["methods"], list(METHODS))
        self.assertEqual(len(manifest["assets"]), 25)

    def test_reference_archive_contains_the_declared_asset_set(self):
        manifest = replay.read_manifest()
        names = set(manifest["assets"])
        observed = {
            p.relative_to(replay.REFERENCE).as_posix()
            for p in replay.REFERENCE.rglob("*")
            if p.is_file() and p.suffix == ".png"
        }
        self.assertEqual(observed, {name for name in names if name.endswith(".png")})
        self.assertEqual(sum(n.endswith(".png") for n in names), 15)
        self.assertEqual(sum(Path(n).name.startswith("table_") for n in names), 9)
        self.assertFalse((replay.REFERENCE / "main.tex").exists())
        self.assertEqual(list(replay.REFERENCE.rglob("*.tex")), [])
        self.assertNotIn("source_files", manifest)

    def test_all_42_psf_measurements_recompute_from_saved_arrays(self):
        directory = replay.INPUTS / "seven_target_metrics"
        summary = json.loads((directory / "seven_target_summary.json").read_text())
        records = {(r["target_id"], r["method"]): r for r in summary["records"]}
        self.assertEqual(set(records), {(target, method) for target in range(1, 8) for method in METHODS})
        for target in range(1, 8):
            with np.load(directory / f"target_{target}_arrays.npz", allow_pickle=False) as archive:
                nx, nz = archive["map_x_mm"].size, archive["map_z_mm"].size
                lateral_size = archive["lateral_x_mm"].size
                grid = SimpleNamespace(
                    map_x_mm=archive["map_x_mm"], map_z_mm=archive["map_z_mm"],
                    lateral_x_mm=archive["lateral_x_mm"], axial_z_mm=archive["axial_z_mm"],
                    map_slice=slice(0, nx*nz), lateral_slice=slice(nx*nz, nx*nz+lateral_size),
                    axial_slice=slice(nx*nz+lateral_size, None),
                )
                for method in METHODS:
                    key = method.lower().replace("-", "_")
                    values = np.concatenate([archive[key+"_map"].ravel(), archive[key+"_lateral"], archive[key+"_axial"]])
                    expected = records[target, method]
                    observed, _ = metrics_for_method(values, grid, expected["target_x_mm"], expected["target_z_mm"])
                    for metric in ("lateral_width_mm", "axial_width_mm", "localization_error_mm", "peak_to_median_background_db"):
                        self.assertAlmostEqual(observed[metric], expected[metric], places=11, msg=f"{target} {method} {metric}")

    def test_raster_tolerance_is_bounded_grayscale_only_and_reported(self):
        import hashlib
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="nsi-raster-test-") as directory:
            reference = Path(directory) / "reference"
            output = Path(directory) / "output"
            reference.mkdir()
            output.mkdir()
            name = "figure.png"
            pixels = np.full((4, 4, 4), 100, dtype=np.uint8)
            pixels[..., 3] = 255
            Image.fromarray(pixels).save(reference / name)
            entry = {
                "sha256": replay.sha256(reference / name),
                "rgba_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "dimensions": [4, 4],
                "rendering_tolerance": {"max_changed_pixels": 1, "max_channel_difference": 1},
            }
            manifest = {"assets": {name: entry}}
            accepted = pixels.copy()
            accepted[0, 0, :3] += 1
            Image.fromarray(accepted).save(output / name)
            with patch.object(replay, "REFERENCE", reference):
                report = replay.verify_assets(output, manifest, restore_reference_bytes=True)
            self.assertEqual(report["metadata_only_png_differences"], [])
            self.assertEqual(report["rendering_tolerance_differences"], [
                {"asset": name, "changed_pixels": 1, "max_channel_difference": 1},
            ])
            self.assertEqual(replay.sha256(output / name), entry["sha256"])

            for change in ("magnitude", "count", "color", "alpha", "no_tolerance"):
                candidate = accepted.copy()
                if change == "magnitude":
                    candidate[0, 0, :3] += 1
                elif change == "count":
                    candidate[0, 1, :3] += 1
                elif change == "color":
                    candidate[0, 0, 0] -= 1
                elif change == "alpha":
                    candidate[0, 0, 3] -= 1
                else:
                    entry.pop("rendering_tolerance")
                Image.fromarray(candidate).save(output / name)
                with self.subTest(change=change), patch.object(replay, "REFERENCE", reference), self.assertRaises(ValueError):
                    replay.verify_assets(output, manifest, restore_reference_bytes=True)

    def test_seven_target_aggregates_use_sample_standard_deviation(self):
        summary = json.loads((replay.INPUTS / "seven_target_metrics/seven_target_summary.json").read_text())
        for row in summary["aggregate"]:
            records = [r for r in summary["records"] if r["method"] == row["method"]]
            self.assertEqual(len(records), 7)
            for metric in ("lateral_width_mm", "axial_width_mm", "localization_error_mm", "peak_to_median_background_db"):
                values = np.asarray([r[metric] for r in records])
                self.assertAlmostEqual(values.mean(), row[metric]["mean"], places=12)
                self.assertAlmostEqual(values.std(ddof=1), row[metric]["sample_sd"], places=12)

    def test_archived_cpu_gpu_agreement_supports_reported_precision(self):
        rows = json.loads((replay.INPUTS / "seven_target_metrics/cpu_gpu_validation.json").read_text())
        self.assertGreaterEqual(len(rows), 11)
        self.assertLess(max(r["maximum_relative_amplitude_error"] for r in rows), 2e-5)

    def test_raw_workflow_includes_both_final_measurement_stages(self):
        spec = importlib.util.spec_from_file_location("final_paper_runner", ROOT / "scripts/run_revision_gpu.py")
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(runner)
        stages = runner.DEFAULT_STEPS
        for name in ("seven-target-metrics", "phantom-inclusion"):
            self.assertIn(name, stages)
            self.assertLess(stages.index(name), stages.index("manuscript-assets"))
        self.assertLess(stages.index("picmus-full-phantom"), stages.index("phantom-inclusion"))

    def test_complete_cpu_replay_matches_all_25_publication_assets(self):
        with tempfile.TemporaryDirectory(prefix="nsi-paper-test-") as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/reproduce_paper.py"), "--output-dir", directory],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((Path(directory) / "paper_reproduction_manifest.json").read_text())
            self.assertTrue(report["all_25_assets_byte_identical"])
            for name, row in replay.read_manifest()["assets"].items():
                self.assertEqual(replay.sha256(Path(directory) / name), row["sha256"])

    def test_raw_branch_does_not_require_a_reference_manifest(self):
        runner = SimpleNamespace(DEFAULT_STEPS=("theory", "manuscript-assets"))
        with patch.object(sys, "argv", ["reproduce_paper.py", "--from-raw"]), \
             patch.dict(sys.modules, {"run_revision_gpu": runner}), \
             patch.object(replay, "read_manifest", side_effect=AssertionError("raw branch read reference")), \
             patch.object(replay.subprocess, "run") as run:
            replay.main()
        self.assertIn("theory,robustness,manuscript-assets", run.call_args.args[0])

    def test_replay_refuses_to_overwrite_frozen_results(self):
        for destination in (replay.REFERENCE, replay.INPUTS, ROOT / "results/reported/robustness"):
            with self.subTest(destination=destination), \
                 patch.object(sys, "argv", ["reproduce_paper.py", "--output-dir", str(destination)]), \
                 patch.object(replay.subprocess, "run") as run, \
                 self.assertRaises(SystemExit):
                replay.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

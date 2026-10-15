"""Regression checks against the frozen final paper, including all 42 PSFs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

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
        replay.verify_assets(ROOT / "paper", manifest, preserve_original_metadata=False)
        self.assertEqual(manifest["methods"], list(METHODS))
        self.assertEqual(len(manifest["assets"]), 25)

    def test_manifest_contains_only_latex_referenced_figures_and_tables(self):
        sources = "\n".join((ROOT / "paper" / name).read_text() for name in ("main.tex", "supplementary_material.tex"))
        figures = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", sources))
        tables = {name + ".tex" for name in re.findall(r"\\input\{(table_[^}]+)\}", sources)}
        observed = set(replay.read_manifest()["assets"])
        self.assertEqual(observed, figures | tables | {"revision_results.tex"})
        self.assertEqual(len(figures), 15)
        self.assertEqual(len(tables), 9)

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


if __name__ == "__main__":
    unittest.main()

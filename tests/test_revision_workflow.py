#!/usr/bin/env python3
"""CPU-only tests for revision orchestration safeguards."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_revision_gpu", ROOT / "scripts" / "run_revision_gpu.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

ASSET_SPEC = importlib.util.spec_from_file_location(
    "materialize_revision_assets",
    ROOT / "scripts" / "materialize_revision_assets.py",
)
assets = importlib.util.module_from_spec(ASSET_SPEC)
assert ASSET_SPEC.loader is not None
sys.modules[ASSET_SPEC.name] = assets
ASSET_SPEC.loader.exec_module(assets)


class RevisionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = Namespace(timing_warmups=10, timing_repetitions=50)

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def complete_timing_payload(self) -> dict:
        rows = []
        for run_id in sorted(runner.EXPECTED_TIMING_CASES):
            if run_id == "baseline_preloaded":
                scope, transfers, receive_mode = "kernel", "none", "device-weighted"
            elif run_id == "baseline_host_stacked":
                scope, transfers, receive_mode = (
                    "reconstruction", "both", "host-stacked"
                )
            else:
                scope, transfers, receive_mode = (
                    "reconstruction", "both", "device-weighted"
                )
            for method in runner.EXPECTED_METHODS:
                rows.append({
                    "run_id": run_id,
                    "method": method,
                    "n": 50,
                    "scope": scope,
                    "transfers": transfers,
                    "receive_input_mode": receive_mode,
                })
        return {
            "quick_engineering_run": False,
            "suite": "all",
            "requested_warmups": 10,
            "requested_repetitions": 50,
            "failed_cases": [],
            "rows": rows,
        }

    @staticmethod
    def complete_psf_payload() -> dict:
        methods = ("DAS", "Angular CF-DAS", "Receive-NSI", "Angle-NSI")
        summaries = {
            group: {
                method: {"lateral_width_mm": {"n": count}}
                for method in methods
            }
            for group, count in (
                ("all", 7), ("on-axis", 5), ("37.5-mm depth", 3)
            )
        }
        return {
            "metadata_only": False,
            "target_positions_mm": [[0.0, 0.0, float(index)] for index in range(7)],
            "reconstruction": {"c_values": [0.02, 0.05, 0.10, 0.20]},
            "summaries": summaries,
        }

    def test_timing_reuse_requires_exact_sample_counts(self):
        payload = self.complete_timing_payload()
        path = self.write_json("timing.json", payload)
        self.assertTrue(runner.reusable_output("timing", path, self.args))
        payload["rows"][0]["n"] = 3
        path = self.write_json("timing.json", payload)
        self.assertFalse(runner.reusable_output("timing", path, self.args))

    def test_metadata_only_psf_is_not_reused(self):
        path = self.write_json("psf.json", {"metadata_only": True})
        self.assertFalse(
            runner.reusable_output("experimental-psf", path, self.args)
        )
        path = self.write_json("psf.json", self.complete_psf_payload())
        self.assertTrue(
            runner.reusable_output("experimental-psf", path, self.args)
        )

    def test_timing_reuse_rejects_a_missing_method(self):
        payload = self.complete_timing_payload()
        payload["rows"].pop()
        path = self.write_json("timing.json", payload)
        self.assertFalse(runner.reusable_output("timing", path, self.args))

    @staticmethod
    def complete_conventional_payload() -> dict:
        methods = sorted(runner.EXPECTED_BASELINE_METHODS)
        return {
            "schema_version": runner.CONVENTIONAL_BASELINE_SCHEMA_VERSION,
            "metadata_only": False,
            "quick_engineering_run": False,
            "publication_ready": True,
            "methods": methods,
            "experimental_psf": {
                "metrics": [{"method": method} for method in methods]
            },
            "carotid_views": [
                {
                    "view": view,
                    "metrics": [{"method": method} for method in methods],
                }
                for view in ("CC", "CL")
            ],
        }

    def test_conventional_baseline_reuse_requires_all_six_methods(self):
        payload = self.complete_conventional_payload()
        path = self.write_json("conventional.json", payload)
        self.assertTrue(
            runner.reusable_output("conventional-baselines", path, self.args)
        )
        payload["carotid_views"][0]["metrics"].pop()
        path = self.write_json("conventional.json", payload)
        self.assertFalse(
            runner.reusable_output("conventional-baselines", path, self.args)
        )

    def test_conventional_baseline_reuse_rejects_stale_schema(self):
        payload = self.complete_conventional_payload()
        payload["schema_version"] -= 1
        path = self.write_json("conventional.json", payload)
        self.assertFalse(
            runner.reusable_output("conventional-baselines", path, self.args)
        )

    def test_conventional_timing_reuse_requires_publication_counts(self):
        payload = {
            "quick_engineering_run": False,
            "publication_ready": True,
            "requested_warmups": 10,
            "requested_repetitions": 50,
            "rows": [
                {
                    "method": method,
                    "n": 50,
                    "scope": "post-delay beamformer kernel",
                }
                for method in runner.EXPECTED_BASELINE_METHODS
            ],
        }
        path = self.write_json("conventional_timing.json", payload)
        self.assertTrue(
            runner.reusable_output("conventional-timing", path, self.args)
        )
        payload["publication_ready"] = False
        path = self.write_json("conventional_timing.json", payload)
        self.assertFalse(
            runner.reusable_output("conventional-timing", path, self.args)
        )

    def test_manuscript_assets_are_always_regenerated(self):
        path = self.write_json("assets.json", {"publication_checks_passed": True})
        self.assertFalse(
            runner.reusable_output("manuscript-assets", path, self.args)
        )

    def test_publication_c_sweep_requires_all_declared_values(self):
        complete = [
            {"method": "Angle-NSI", "c": value}
            for value in (0.02, 0.05, 0.10, 0.20)
        ]
        self.assertEqual(
            assets.selected_c_values(complete, "Angle-NSI"),
            assets.EXPECTED_C_VALUES,
        )
        self.assertNotEqual(
            assets.selected_c_values(complete[:-1], "Angle-NSI"),
            assets.EXPECTED_C_VALUES,
        )


if __name__ == "__main__":
    unittest.main()

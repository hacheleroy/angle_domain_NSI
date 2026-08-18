#!/usr/bin/env python3
"""CPU-only tests for the timing and robustness helper functions."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"


def load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(
        name, SOURCE_DIR / filename
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


benchmark = load_module("benchmark_nsi", "benchmark_nsi.py")
robustness = load_module(
    "simulation_robustness", "simulation_robustness.py"
)


class ValidationHelperTests(unittest.TestCase):
    class FakeCupy:
        complex64 = np.complex64
        float32 = np.float32

        @staticmethod
        def asarray(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def asnumpy(value):
            return np.asarray(value)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None):
            return np.empty(shape, dtype=dtype)

        @staticmethod
        def abs(value):
            return np.abs(value)

        @staticmethod
        def maximum(left, right):
            return np.maximum(left, right)

        @staticmethod
        def sum(value, axis=None):
            return np.sum(value, axis=axis)

    def test_symmetric_angles_have_zero_broadside_weight(self):
        angles = benchmark.make_angles(17, 8.0)
        weights = np.sign(angles)
        self.assertAlmostEqual(float(np.sum(weights)), 0.0)
        self.assertEqual(float(weights[len(weights) // 2]), 0.0)

    def test_timing_summary_statistics(self):
        row = benchmark.summarize_times("test", [0.001, 0.002, 0.003])
        self.assertAlmostEqual(row["median_ms"], 2.0)
        self.assertAlmostEqual(row["mean_ms"], 2.0)
        self.assertAlmostEqual(row["sample_sd_ms"], 1.0)
        self.assertGreater(row["iqr_ms"], 0.0)

    def test_benchmark_gpu_method_logic_with_numpy_backend(self):
        args = Namespace(
            elements=4,
            dc_offset=0.05,
            include_naive_reference=True,
            transfers="none",
            f_number=1.0,
            rx_start_us=0.0,
            sampling_frequency_mhz=20.0,
            sound_speed_m_s=1540.0,
            centre_frequency_mhz=5.0,
        )
        angles = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        weights = np.sign(angles).astype(np.float32)
        rng = np.random.default_rng(3)
        iq = (
            rng.standard_normal((3, 8, 4))
            + 1j * rng.standard_normal((3, 8, 4))
        ).astype(np.complex64)
        cp = self.FakeCupy()
        prepared = benchmark.prepare_channel_data(iq, args, cp)
        geometry = {
            "grid_shape": (2, 3),
            "receive_gpu": None,
            "scan_gpu": None,
            "arrivals_gpu": [None, None, None],
        }

        def fake_beamform(channel_data, **_):
            frame_sums = np.sum(channel_data, axis=(0, 1))
            pixel_scale = np.linspace(0.8, 1.2, 6)[:, None]
            return pixel_scale * frame_sums[None, :]

        outputs = {}
        for method in (
            benchmark.METHOD_DAS,
            benchmark.METHOD_RECEIVE,
            benchmark.METHOD_ANGLE_STREAM,
            benchmark.METHOD_ANGLE_STORED,
            benchmark.METHOD_NAIVE,
        ):
            outputs[method] = benchmark.execute_method(
                method,
                args,
                angles,
                weights,
                prepared,
                geometry,
                cp,
                fake_beamform,
                lambda: geometry,
            )
            self.assertEqual(outputs[method].shape, (2, 3))
            self.assertTrue(np.all(np.isfinite(outputs[method])))

        np.testing.assert_allclose(
            outputs[benchmark.METHOD_ANGLE_STREAM],
            outputs[benchmark.METHOD_ANGLE_STORED],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            outputs[benchmark.METHOD_RECEIVE],
            outputs[benchmark.METHOD_NAIVE],
            rtol=2e-5,
            atol=2e-5,
        )

    def test_gaussian_half_amplitude_width(self):
        sigma_mm = 0.100
        axis = np.linspace(-1.0, 1.0, 20001)
        profile = np.exp(-0.5 * (axis / sigma_mm) ** 2)
        measured = robustness.measure_connected_width(axis, profile)
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_mm
        self.assertAlmostEqual(measured["width_mm"], expected, places=6)

    def test_recentered_missing_weights_are_zero_mean_and_l1_matched(self):
        angles = robustness.make_angles(17, 8.0)
        missing = np.delete(angles, np.flatnonzero(angles > 0.0)[3])
        raw = np.sign(missing)
        recentered = robustness.recentered_missing_angle_weights(missing)
        self.assertAlmostEqual(float(np.sum(recentered)), 0.0, places=6)
        self.assertAlmostEqual(
            float(np.sum(np.abs(recentered))),
            float(np.sum(np.abs(raw))),
            places=6,
        )

    def test_two_target_dip_criterion(self):
        axis = np.linspace(-0.5, 0.5, 20001)
        separated_profile = np.exp(-0.5 * ((axis + 0.10) / 0.020) ** 2)
        separated_profile += np.exp(-0.5 * ((axis - 0.10) / 0.020) ** 2)
        separated = robustness.two_target_metrics(
            axis, separated_profile, -0.10, 0.10
        )
        self.assertTrue(separated["resolved_by_minus6db_dip"])

        merged_profile = np.exp(-0.5 * ((axis + 0.01) / 0.050) ** 2)
        merged_profile += np.exp(-0.5 * ((axis - 0.01) / 0.050) ** 2)
        merged = robustness.two_target_metrics(axis, merged_profile, -0.01, 0.01)
        self.assertFalse(merged["resolved_by_minus6db_dip"])

    def test_all_summary_figures_render_from_schema(self):
        angle_rows = []
        for method_index, method in enumerate(robustness.METHOD_ORDER, start=1):
            for count in (5, 9):
                angle_rows.append(
                    {
                        "scenario_type": "angle_count",
                        "scenario_label": f"{count} angles",
                        "angle_count": count,
                        "total_angle_span_deg": 8.0,
                        "method": method,
                        "lateral_fwhm_mm": 0.01 * method_index,
                        "peak_to_background_db": 20.0,
                    }
                )
            for span in (4.0, 8.0):
                angle_rows.append(
                    {
                        "scenario_type": "angle_span",
                        "scenario_label": f"{span:g} deg total span",
                        "angle_count": 9,
                        "total_angle_span_deg": span,
                        "method": method,
                        "lateral_fwhm_mm": 0.01 * method_index,
                        "peak_to_background_db": 20.0,
                    }
                )
            for label in (
                "symmetric baseline",
                "missing: raw sign",
                "missing: recentered",
            ):
                angle_rows.append(
                    {
                        "scenario_type": "missing_angle",
                        "scenario_label": label,
                        "method": method,
                        "lateral_fwhm_mm": 0.01 * method_index,
                        "peak_to_background_db": 20.0 - method_index,
                    }
                )

        perturbation_rows = []
        for method_index, method in enumerate(robustness.METHOD_ORDER, start=1):
            for snr in ("infinity", 20.0):
                perturbation_rows.append(
                    {
                        "perturbation_type": "noise",
                        "noise_snr_db": snr,
                        "phase_jitter_sd_deg": 0.0,
                        "method": method,
                        "lateral_fwhm_mm": 0.01 * method_index,
                    }
                )
            for phase in (0.0, 10.0):
                perturbation_rows.append(
                    {
                        "perturbation_type": "phase",
                        "noise_snr_db": "infinity",
                        "phase_jitter_sd_deg": phase,
                        "method": method,
                        "lateral_fwhm_mm": 0.01 * method_index,
                    }
                )

        spatial_rows = [
            {
                "target_z_mm": depth,
                "target_x_mm": lateral,
                "method": method,
                "lateral_fwhm_mm": 0.01 * (index + 1),
            }
            for index, method in enumerate(robustness.METHOD_ORDER)
            for depth in (15.0, 20.0)
            for lateral in (-4.0, 4.0)
        ]

        x_axis = np.linspace(-0.3, 0.3, 101)
        two_rows, profile_rows = [], []
        for index, method in enumerate(robustness.METHOD_ORDER):
            two_rows.append(
                {
                    "method": method,
                    "target_separation_mm": 0.15,
                    "valley_dip_relative_to_weaker_peak_db": -8.0 - index,
                }
            )
            for x_value in x_axis:
                profile_rows.append(
                    {
                        "method": method,
                        "target_separation_mm": 0.15,
                        "x_mm": float(x_value),
                        "normalized_envelope_db": float(-40.0 * abs(x_value)),
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            benchmark.plot_summary(
                directory / "benchmark.png",
                [
                    {
                        "method": benchmark.METHOD_DAS,
                        "median_ms": 1.0,
                        "q1_ms": 0.9,
                        "q3_ms": 1.1,
                    },
                    {
                        "method": benchmark.METHOD_RECEIVE,
                        "median_ms": 2.0,
                        "q1_ms": 1.8,
                        "q3_ms": 2.2,
                    },
                ],
            )
            robustness.plot_angle_sweeps(directory / "angles.png", angle_rows)
            robustness.plot_perturbations(
                directory / "perturbations.png", perturbation_rows
            )
            robustness.plot_spatial_psf(directory / "spatial.png", spatial_rows)
            robustness.plot_two_target(
                directory / "two_target.png",
                two_rows,
                profile_rows,
                0.15,
            )
            for filename in (
                "benchmark.png",
                "angles.png",
                "perturbations.png",
                "spatial.png",
                "two_target.png",
            ):
                self.assertGreater((directory / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

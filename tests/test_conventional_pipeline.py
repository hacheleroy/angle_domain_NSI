#!/usr/bin/env python3
"""CPU tests for the six-method comparison and benchmark plumbing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_beamforming import (  # noqa: E402
    MvConfiguration,
    design_fdmas_fir,
    signed_sqrt_pair_sum,
)
from benchmark_conventional import METHODS, execute_method  # noqa: E402
from conventional_baseline_comparison import (  # noqa: E402
    compute_gcnr,
    reconstruct_fdmas,
    reconstruct_iq_methods,
    resample_regular_image,
    usable_complete_image,
    usable_positive_image,
)
from picmus_io import PicmusDataset  # noqa: E402


class ConventionalPipelineTests(unittest.TestCase):
    class FakeCupy:
        """Small eager NumPy stand-in for end-to-end reconstruction flow."""

        complex64 = np.complex64
        float32 = np.float32
        int32 = np.int32
        int64 = np.int64
        fft = np.fft
        linalg = np.linalg

        class cuda:
            class Stream:
                class null:
                    @staticmethod
                    def synchronize():
                        return None

        class _Pool:
            @staticmethod
            def free_all_blocks():
                return None

        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def asnumpy(value):
            return np.asarray(value)

        @classmethod
        def get_default_memory_pool(cls):
            return cls._Pool()

    def test_regular_image_resampling_is_bilinear(self):
        source_x = np.asarray([0.0, 1.0], dtype=float)
        source_z = np.asarray([0.0, 2.0], dtype=float)
        image = source_x[:, None] + 2.0 * source_z[None, :]
        target_x = np.asarray([0.25, 0.75])
        target_z = np.asarray([0.5, 1.5])
        actual = resample_regular_image(
            image, source_x, source_z, target_x, target_z
        )
        expected = target_x[:, None] + 2.0 * target_z[None, :]
        np.testing.assert_allclose(actual, expected)

    def test_gcnr_extremes(self):
        samples = np.linspace(-1.0, 1.0, 101)
        self.assertAlmostEqual(compute_gcnr(samples, samples, 20), 0.0)
        self.assertAlmostEqual(
            compute_gcnr(np.zeros(100), np.ones(100), 20), 1.0
        )

    def test_positive_image_validation_rejects_empty_or_invalid_cache(self):
        self.assertFalse(usable_positive_image(None))
        self.assertFalse(usable_positive_image(np.zeros((2, 3))))
        self.assertFalse(usable_positive_image(np.asarray([0.0, np.nan])))
        self.assertTrue(usable_positive_image(np.asarray([0.0, 1.0])))

    def test_complete_image_validation_rejects_zero_lateral_line(self):
        complete = np.asarray([[0.0, 1.0], [2.0, 0.0]])
        incomplete = np.asarray([[0.0, 1.0], [0.0, 0.0]])
        self.assertTrue(usable_complete_image(complete))
        self.assertFalse(usable_complete_image(incomplete))
        self.assertFalse(usable_complete_image(np.asarray([0.0, 1.0])))

    def test_every_benchmark_method_runs_with_numpy_backend(self):
        rng = np.random.default_rng(8)
        shape = (3, 2, 16, 8)
        delayed_iq = (
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        ).astype(np.complex64)
        delayed_rf = delayed_iq.real.astype(np.float32)
        mask = np.ones(shape[1:], dtype=np.float32)
        receive_sign = np.ones_like(mask)
        receive_sign[..., :4] = -1.0
        angular_weights = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        coefficients, _ = design_fdmas_fir(1.0e6, 10.0e6)
        for method in METHODS:
            with self.subTest(method=method):
                output = execute_method(
                    method,
                    delayed_iq,
                    delayed_rf,
                    mask,
                    receive_sign,
                    angular_weights,
                    coefficients,
                    MvConfiguration(chunk_pixels=4),
                    0.05,
                    np,
                )
                self.assertEqual(output.shape, shape[1:3])
                self.assertTrue(np.all(np.isfinite(output)))

    def test_shared_picmus_reconstruction_flow_with_numpy_standin(self):
        samples, elements, angles = 96, 8, 3
        fs_hz = 20.0e6
        time_s = np.arange(samples, dtype=np.float32) / fs_hz
        carrier = np.sin(2.0 * np.pi * 1.0e6 * time_s)
        data = np.broadcast_to(
            carrier[:, None, None], (samples, elements, angles)
        ).astype(np.complex64, copy=True)
        geometry = np.column_stack(
            [
                np.linspace(-1.0e-3, 1.0e-3, elements),
                np.zeros(elements),
                np.zeros(elements),
            ]
        ).astype(np.float32)
        dataset = PicmusDataset(
            data=data,
            angles_rad=np.asarray([-0.1, 0.0, 0.1]),
            initial_time_s=0.0,
            sampling_frequency_hz=fs_hz,
            sound_speed_m_s=1540.0,
            probe_geometry_m=geometry,
            modulation_frequency_hz=1.0e6,
        )
        x_m = np.asarray([-0.1e-3, 0.1e-3], dtype=np.float32)
        z_m = np.asarray([1.0e-3, 1.1e-3, 1.2e-3], dtype=np.float32)
        x_grid, z_grid = np.meshgrid(x_m, z_m, indexing="ij")
        points = np.column_stack(
            [x_grid.ravel(), np.zeros(x_grid.size), z_grid.ravel()]
        ).astype(np.float32)
        cp = self.FakeCupy()
        iq = reconstruct_iq_methods(
            dataset,
            np.arange(angles),
            points,
            grid_shape=(x_m.size, z_m.size),
            carrier_frequency_hz=1.0e6,
            f_number=1.0,
            nsi_c=0.05,
            mv_configuration=MvConfiguration(chunk_pixels=3),
            cp=cp,
            label="test",
        )
        self.assertEqual(
            set(iq), {"DAS", "CF-DAS", "MV", "Receive-NSI", "Angle-NSI"}
        )
        self.assertTrue(all(value.shape == (points.shape[0],) for value in iq.values()))
        iq_without_mv = reconstruct_iq_methods(
            dataset,
            np.arange(angles),
            points,
            grid_shape=(x_m.size, z_m.size),
            carrier_frequency_hz=1.0e6,
            f_number=1.0,
            nsi_c=0.05,
            mv_configuration=MvConfiguration(chunk_pixels=3),
            cp=cp,
            label="test without MV",
            compute_mv=False,
        )
        self.assertEqual(
            set(iq_without_mv),
            {"DAS", "CF-DAS", "Receive-NSI", "Angle-NSI"},
        )
        iq_chunked = reconstruct_iq_methods(
            dataset,
            np.arange(angles),
            points,
            grid_shape=(x_m.size, z_m.size),
            carrier_frequency_hz=1.0e6,
            f_number=1.0,
            nsi_c=0.05,
            mv_configuration=MvConfiguration(chunk_pixels=3),
            cp=cp,
            label="chunked test",
            focus_chunk_pixels=2,
        )
        self.assertEqual(set(iq_chunked), set(iq))
        for method in iq:
            np.testing.assert_allclose(
                iq_chunked[method], iq[method], rtol=3e-6, atol=1e-7
            )
        fine_z = np.linspace(0.8e-3, 1.5e-3, 36, dtype=np.float32)
        fdmas, metadata = reconstruct_fdmas(
            dataset,
            np.arange(angles),
            x_m,
            fine_z,
            carrier_frequency_hz=1.0e6,
            f_number=1.0,
            filter_configuration=None,
            cp=cp,
            label="test",
        )
        self.assertEqual(fdmas.shape, (x_m.size, fine_z.size))
        self.assertTrue(np.all(np.isfinite(fdmas)))
        self.assertGreater(metadata["fir_num_taps"], 1)
        fdmas_one_line_chunks, chunk_metadata = reconstruct_fdmas(
            dataset,
            np.arange(angles),
            x_m,
            fine_z,
            carrier_frequency_hz=1.0e6,
            f_number=1.0,
            filter_configuration=None,
            cp=cp,
            label="test chunked",
            x_chunk_lines=1,
        )
        np.testing.assert_allclose(fdmas_one_line_chunks, fdmas)
        self.assertEqual(chunk_metadata["requested_lateral_chunk_lines"], 1)

        def fail_multiline_batches(delayed_rf, active_mask, *, xp):
            if delayed_rf.shape[0] > fine_z.size:
                return xp.zeros(delayed_rf.shape[0], dtype=delayed_rf.dtype)
            return signed_sqrt_pair_sum(delayed_rf, active_mask, xp=xp)

        with patch(
            "conventional_baseline_comparison.signed_sqrt_pair_sum",
            side_effect=fail_multiline_batches,
        ):
            fdmas_fallback, fallback_metadata = reconstruct_fdmas(
                dataset,
                np.arange(angles),
                x_m,
                fine_z,
                carrier_frequency_hz=1.0e6,
                f_number=1.0,
                filter_configuration=None,
                cp=cp,
                label="test adaptive fallback",
                x_chunk_lines=2,
            )
        np.testing.assert_allclose(fdmas_fallback, fdmas_one_line_chunks)
        self.assertGreater(fallback_metadata["adaptive_batch_retries"], 0)
        self.assertEqual(
            fallback_metadata["validated_lateral_batch_sizes"], [1, 1]
        )


if __name__ == "__main__":
    unittest.main()

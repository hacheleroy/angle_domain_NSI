#!/usr/bin/env python3
"""CPU checks for the conventional beamformers added for Reviewer 1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_beamforming import (  # noqa: E402
    FdmasFilterConfiguration,
    MvConfiguration,
    analytic_signal,
    capon_minimum_variance,
    design_fdmas_fir,
    direct_signed_sqrt_pair_sum,
    fdmas_analytic_image,
    receive_cf_das,
    receive_coherence_factor,
    signed_sqrt_pair_sum,
    temporal_half_window_from_wavelengths,
)


class ConventionalBeamformingTests(unittest.TestCase):
    def test_receive_cf_matches_explicit_definition(self):
        delayed = np.asarray(
            [[1.0 + 1.0j, 2.0 - 1.0j, 0.0], [1.0j, -1.0j, 5.0]],
            dtype=np.complex64,
        )
        mask = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
        coherent = np.sum(delayed * mask, axis=-1)
        expected = np.abs(coherent) ** 2 / (
            np.sum(mask, axis=-1) * np.sum(np.abs(delayed * mask) ** 2, axis=-1)
        )
        actual = receive_coherence_factor(delayed, mask)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        weighted, factor = receive_cf_das(delayed, mask, return_factor=True)
        np.testing.assert_allclose(factor, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(weighted, expected * coherent, rtol=1e-6)

    def test_cf_is_one_for_identical_active_channels(self):
        delayed = np.full((7, 8), 2.0 - 3.0j, dtype=np.complex64)
        mask = np.ones_like(delayed.real, dtype=np.float32)
        np.testing.assert_allclose(receive_coherence_factor(delayed, mask), 1.0)

    def test_fast_dmas_is_the_direct_matrone_pair_sum(self):
        rng = np.random.default_rng(12)
        delayed = rng.normal(size=(5, 4, 9)).astype(np.float32)
        expected = direct_signed_sqrt_pair_sum(delayed)
        actual = signed_sqrt_pair_sum(delayed)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_dmas_mask_removes_inactive_elements(self):
        delayed = np.asarray([[1.0, 4.0, 9.0, 16.0]], dtype=np.float32)
        mask = np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        actual = signed_sqrt_pair_sum(delayed, mask)
        expected = direct_signed_sqrt_pair_sum(delayed[..., :2])
        np.testing.assert_allclose(actual, expected)

    def test_fdmas_filter_requires_depth_nyquist_above_upper_stop(self):
        with self.assertRaisesRegex(ValueError, "sampling is too coarse"):
            design_fdmas_fir(5.0e6, 20.0e6)

    def test_fdmas_filter_and_hilbert_preserve_shape(self):
        coefficients, metadata = design_fdmas_fir(
            5.0e6, 40.0e6, FdmasFilterConfiguration()
        )
        depth = np.arange(320, dtype=np.float32) / 40.0e6
        pair_sum = np.sin(2.0 * np.pi * 10.0e6 * depth)[None, :]
        result = fdmas_analytic_image(pair_sum, coefficients, depth_axis=1)
        self.assertEqual(result.shape, pair_sum.shape)
        self.assertTrue(np.iscomplexobj(result))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(metadata["fir_num_taps"], coefficients.size)

    def test_analytic_signal_rejects_negative_frequency_component(self):
        count = 65
        phase = 2.0 * np.pi * 5.0 * np.arange(count) / count
        result = analytic_signal(np.cos(phase).astype(np.float32))
        np.testing.assert_allclose(result, np.exp(1j * phase), rtol=2e-5, atol=2e-5)

    def test_mv_preserves_a_focused_equal_channel_signal(self):
        delayed = np.full((3, 5, 8), 2.0 + 0.5j, dtype=np.complex64)
        mask = np.ones(delayed.shape, dtype=np.float32)
        result = capon_minimum_variance(
            delayed,
            mask,
            grid_shape=(3, 5),
            configuration=MvConfiguration(chunk_pixels=4),
        )
        expected = np.sum(delayed, axis=-1)
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)

    def test_mv_accepts_dynamic_contiguous_apertures_and_temporal_average(self):
        rng = np.random.default_rng(4)
        delayed = (
            rng.normal(size=(2, 5, 8)) + 1j * rng.normal(size=(2, 5, 8))
        ).astype(np.complex64)
        mask = np.zeros(delayed.shape, dtype=np.float32)
        mask[:, :2, 2:6] = 1.0
        mask[:, 2:, 1:7] = 1.0
        result = capon_minimum_variance(
            delayed,
            mask,
            grid_shape=(2, 5),
            configuration=MvConfiguration(
                temporal_half_window_samples=1, chunk_pixels=3
            ),
        )
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_mv_rejects_an_aperture_with_holes(self):
        delayed = np.ones((2, 4), dtype=np.complex64)
        mask = np.asarray([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            capon_minimum_variance(delayed, mask)

    def test_temporal_window_uses_physical_wavelengths(self):
        # lambda = 0.3 mm and dz = 0.1 mm; 1.5 lambda is 4.5 samples.
        self.assertEqual(
            temporal_half_window_from_wavelengths(1.5, 0.1e-3, 5.0e6, 1500.0),
            4,
        )


if __name__ == "__main__":
    unittest.main()

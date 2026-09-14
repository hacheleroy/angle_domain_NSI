"""Small dependency-free checks for the algebra used in the PMB Note."""

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nsi_core import (
    angular_sign_weights,
    coherence_factor,
    coherence_factor_from_moments,
    coherence_weighted_das,
    nsi_envelope,
)
from angular_null_theory import angular_fields, normalized_null_slope_per_m


class TestNsiIdentities(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = rng.normal(size=(9, 128)) + 1j * rng.normal(size=(9, 128))
        self.dc = 0.05

    def test_receive_dc_fields_need_only_uniform_and_null(self):
        w = np.r_[-np.ones(64), np.ones(64)]
        uniform = np.ones(9) @ self.data @ np.ones(128)
        null = np.ones(9) @ self.data @ w

        direct_dc1 = np.ones(9) @ self.data @ (w + self.dc)
        direct_dc2 = np.ones(9) @ self.data @ (-w + self.dc)

        self.assertTrue(np.allclose(direct_dc1, null + self.dc * uniform))
        self.assertTrue(np.allclose(direct_dc2, -null + self.dc * uniform))

    def test_broadside_angle_has_zero_weight(self):
        angles = np.linspace(-4.0, 4.0, 9)
        weights, diagnostics = angular_sign_weights(angles)
        self.assertEqual(weights[4], 0.0)
        self.assertAlmostEqual(weights.sum(), 0.0)
        self.assertTrue(diagnostics.symmetric_angle_pairs)
        self.assertTrue(diagnostics.antisymmetric_weights)

    def test_nonuniform_symmetric_angles_are_supported(self):
        weights, diagnostics = angular_sign_weights([-4.0, -1.5, 0.0, 1.5, 4.0])
        self.assertTrue(np.array_equal(weights, [-1, -1, 0, 1, 1]))
        self.assertEqual(diagnostics.broadside_count, 1)

    def test_asymmetric_angle_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            angular_sign_weights([-4.0, -1.0, 0.0, 2.0, 4.0])

    def test_receive_and_angle_nulls_are_not_generally_equal(self):
        w = np.r_[-np.ones(64), np.ones(64)]
        v = np.sign(np.linspace(-4.0, 4.0, 9))
        receive_null = np.ones(9) @ self.data @ w
        angle_null = v @ self.data @ np.ones(128)
        self.assertFalse(np.allclose(receive_null, angle_null))

    def test_functional_is_invariant_to_null_sign(self):
        uniform = np.ones(9) @ self.data @ np.ones(128)
        v = np.sign(np.linspace(-4.0, 4.0, 9))
        null = v @ self.data @ np.ones(128)
        self.assertTrue(np.allclose(
            nsi_envelope(uniform, null, self.dc),
            nsi_envelope(uniform, -null, self.dc),
        ))

    def test_coherence_factor_limits_and_moment_form(self):
        coherent = np.ones((4, 7), dtype=np.complex64) * (2.0 + 1.0j)
        factor = coherence_factor(coherent, axis=0)
        self.assertTrue(np.allclose(factor, 1.0))
        summed = coherent.sum(axis=0)
        power = (np.abs(coherent) ** 2).sum(axis=0)
        self.assertTrue(np.allclose(
            factor,
            coherence_factor_from_moments(summed, power, 4),
        ))

        cancelling = np.asarray([1.0, -1.0, 1.0, -1.0])[:, None]
        self.assertTrue(np.allclose(coherence_factor(cancelling, axis=0), 0.0))

    def test_cf_weighted_das_preserves_perfectly_coherent_envelope(self):
        contributions = np.ones((5, 3), dtype=np.complex64) * (1.0 - 2.0j)
        expected = np.abs(contributions.sum(axis=0))
        self.assertTrue(np.allclose(
            coherence_weighted_das(contributions, axis=0), expected
        ))

    def test_small_angle_derivative_matches_finite_difference(self):
        angles = np.linspace(-4.0, 4.0, 17)
        k0 = 2.0 * np.pi / 0.0003
        dx = np.asarray([-1e-9, 1e-9])
        uniform, null = angular_fields(dx, angles, wavenumber_rad_m=k0)
        numerical = abs((null[1] - null[0]) / (2e-9)) / abs(uniform.mean())
        analytical = normalized_null_slope_per_m(angles, k0)
        self.assertAlmostEqual(numerical / analytical, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()

"""Small dependency-free checks for the algebra used in the PMB Note."""

import unittest

import numpy as np


def nsi_functional(uniform, null, dc):
    """NSI envelope from the two independent complex fields U and Z."""
    return 0.5 * (np.abs(null + dc * uniform) + np.abs(-null + dc * uniform)) - np.abs(null)


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
        weights = np.sign(angles)
        weights[np.isclose(angles, 0.0)] = 0.0
        self.assertEqual(weights[4], 0.0)
        self.assertAlmostEqual(weights.sum(), 0.0)

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
            nsi_functional(uniform, null, self.dc),
            nsi_functional(uniform, -null, self.dc),
        ))


if __name__ == "__main__":
    unittest.main()

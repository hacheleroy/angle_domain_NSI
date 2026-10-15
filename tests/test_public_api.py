"""Behavior of the installed stack API and the NPZ command-line interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from angle_nsi import angle_nsi


class PublicApiTests(unittest.TestCase):
    def test_stack_matches_the_direct_three_apodization_definition(self):
        rng = np.random.default_rng(4)
        angles = np.asarray([-4, -1.5, 0, 1.5, 4])
        for dtype in (np.complex64, np.complex128):
            images = (rng.normal(size=(3, 5, 7)) + 1j*rng.normal(size=(3, 5, 7))).astype(dtype)
            weights = np.sign(angles).astype(images.real.dtype)
            c = 0.05
            plus = np.einsum("ikj,k->ij", images, weights+c)
            minus = np.einsum("ikj,k->ij", images, -weights+c)
            null = np.einsum("ikj,k->ij", images, weights)
            expected = np.maximum((abs(plus)+abs(minus))/2-abs(null), 0)
            actual = angle_nsi(images, angles, c, angle_axis=1)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
            self.assertEqual(actual.dtype, np.dtype(dtype).type(0).real.dtype)

    def test_identical_coherent_images_have_the_documented_raw_amplitude(self):
        images = np.full((2, 4, 5), 2+3j, dtype=np.complex64)
        actual = angle_nsi(images, [-4,-2,0,2,4], c=0.1)
        np.testing.assert_allclose(actual, 0.1*5*abs(2+3j), rtol=1e-6)
        self.assertEqual(actual.shape, (2,4))

    def test_angle_axis_and_batch_axes_are_preserved(self):
        rng = np.random.default_rng(12)
        images = (rng.normal(size=(2,3,4,5))+1j*rng.normal(size=(2,3,4,5))).astype(np.complex64)
        expected = angle_nsi(images, [-4,-2,0,2,4])
        actual = angle_nsi(np.moveaxis(images,-1,0), [-4,-2,0,2,4], angle_axis=0)
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        self.assertEqual(actual.shape,(2,3,4))

    def test_invalid_or_unsuitable_inputs_fail_clearly(self):
        valid = np.ones((2,3),dtype=np.complex64)
        cases = [
            (valid.real, [-2,0,2], {}, "complex"),
            (valid, [-2,2], {}, "length"),
            (valid, [-2,0,3], {}, "symmetric"),
            (valid, [2,0,-2], {}, "increasing"),
            (valid, [[-2,0,2]], {}, "one-dimensional"),
            (valid, [-2,0,2], {"angle_axis":2}, "invalid"),
            (valid, [-2,0,2], {"c":0}, "positive"),
            (valid, [-2,0,2], {"c":[0.05]}, "scalar"),
            (valid, [-2,0,2], {"c":float('inf')}, "finite"),
            (valid*np.nan, [-2,0,2], {}, "finite"),
        ]
        for images,angles,options,message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message):
                angle_nsi(images,angles,**options)

    def test_zero_input_stays_zero(self):
        np.testing.assert_array_equal(angle_nsi(np.zeros((3,4),complex),[-2,0,2],angle_axis=0),np.zeros(4))

    def test_cli_round_trip_and_input_protection(self):
        from angle_nsi.cli import main

        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"input.npz"
            output=Path(temp)/"output.npz"
            images=np.full((4,7,3),1+2j,dtype=np.complex64)
            np.savez(source,images=images,angles_deg=[-2,0,2],x_mm=np.arange(7))
            original=source.read_bytes()
            main([str(source),"--output",str(output),"--c","0.1"])
            with np.load(output,allow_pickle=False) as result:
                np.testing.assert_allclose(result['angle_nsi'],angle_nsi(images,[-2,0,2],0.1))
                np.testing.assert_allclose(result['das'],abs(images.sum(axis=-1)))
                np.testing.assert_array_equal(result['x_mm'],np.arange(7))
            with self.assertRaises(SystemExit): main([str(source),"--output",str(source),"--overwrite"])
            with self.assertRaises(SystemExit): main([str(source),"--output",str(output)])
            self.assertEqual(source.read_bytes(),original)

    def test_acquisition_scripts_import_without_gpu_or_data_side_effects(self):
        program = """
import importlib, pathlib, sys
sys.path.insert(0, sys.argv[1])
for name in ['simulation_point_target', 'doppler_mbtrace']:
    assert callable(importlib.import_module(name).main)
assert 'cupy' not in sys.modules
assert 'pymust' not in sys.modules
assert list(pathlib.Path.cwd().iterdir()) == []
"""
        with tempfile.TemporaryDirectory() as temp:
            result=subprocess.run([sys.executable,"-c",program,str(ROOT/'src')],cwd=temp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()

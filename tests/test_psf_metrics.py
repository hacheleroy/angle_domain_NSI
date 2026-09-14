"""Tests for experimental and simulated PSF measurement helpers."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psf_metrics import amplitude_to_db, measure_connected_width, peak_to_background_db
from picmus_experimental_psf import (
    belongs_to_summary_group,
    build_target_grid,
    choose_angle_indices,
)
from picmus_io import load_picmus_dataset, load_picmus_phantom, load_picmus_scan


class PsfMetricTests(unittest.TestCase):
    def test_gaussian_minus6db_width(self):
        sigma = 0.1
        axis = np.linspace(-1.0, 1.0, 20001)
        profile = np.exp(-0.5 * (axis / sigma) ** 2)
        result = measure_connected_width(axis, profile, level_db=-6.0)
        expected = 2.0 * sigma * np.sqrt(6.0 * np.log(10.0) / 10.0)
        self.assertAlmostEqual(result.width_mm, expected, places=6)

    def test_expected_peak_window_avoids_larger_remote_peak(self):
        axis = np.linspace(-2.0, 2.0, 4001)
        local = np.exp(-0.5 * ((axis + 0.5) / 0.05) ** 2)
        remote = 2.0 * np.exp(-0.5 * ((axis - 1.0) / 0.05) ** 2)
        result = measure_connected_width(
            axis,
            local + remote,
            expected_peak_mm=-0.5,
            search_radius_mm=0.2,
        )
        self.assertAlmostEqual(result.peak_axis_mm, -0.5, places=6)

    def test_local_background_metric(self):
        x = np.linspace(-1.0, 1.0, 101)
        z = np.linspace(9.0, 11.0, 101)
        image = np.ones((101, 101))
        image[50, 50] = 10.0
        self.assertAlmostEqual(
            peak_to_background_db(
                image, x, z, target_x_mm=0.0, target_z_mm=10.0
            ),
            20.0,
        )

    def test_db_normalization(self):
        values = amplitude_to_db([1.0, 0.5, 0.1])
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], -20.0)

    def test_target_grid_contains_direct_fine_profiles(self):
        grid = build_target_grid(
            -0.2,
            37.6,
            map_half_width_mm=0.8,
            map_spacing_mm=0.02,
            profile_half_width_mm=1.0,
            lateral_spacing_mm=0.002,
            axial_spacing_mm=0.005,
        )
        self.assertAlmostEqual(np.median(np.diff(grid.lateral_x_mm)), 0.002)
        self.assertAlmostEqual(np.median(np.diff(grid.axial_z_mm)), 0.005)
        self.assertEqual(grid.points_m.shape[1], 3)
        np.testing.assert_array_equal(choose_angle_indices(75, 17), np.arange(29, 46))

    def test_same_depth_summary_includes_on_and_off_axis_targets(self):
        near_axis = {
            "position_group": "on-axis", "target_z_mm": 37.6
        }
        off_axis = {
            "position_group": "37.5-mm depth", "target_z_mm": 37.5
        }
        shallow = {
            "position_group": "on-axis", "target_z_mm": 28.0
        }
        self.assertTrue(belongs_to_summary_group(near_axis, "37.5-mm depth"))
        self.assertTrue(belongs_to_summary_group(off_axis, "37.5-mm depth"))
        self.assertFalse(belongs_to_summary_group(shallow, "37.5-mm depth"))

    def test_picmus_readers_canonicalize_official_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.hdf5"
            phantom_path = root / "phantom.hdf5"
            scan_path = root / "scan.hdf5"
            with h5py.File(dataset_path, "w") as h5:
                group = h5.create_group("US/US_DATASET0000")
                data = group.create_group("data")
                real = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
                data.create_dataset("real", data=real)
                data.create_dataset("imag", data=np.zeros_like(real))
                group.create_dataset("angles", data=[-0.1, 0.0, 0.1])
                group.create_dataset("initial_time", data=[0.0])
                group.create_dataset("sampling_frequency", data=[20e6])
                group.create_dataset("sound_speed", data=[1540.0])
                group.create_dataset("modulation_frequency", data=[0.0])
                geometry = np.vstack([
                    np.linspace(-0.00045, 0.00045, 4), np.zeros(4), np.zeros(4)
                ]).astype(np.float32)
                group.create_dataset("probe_geometry", data=geometry)
            with h5py.File(phantom_path, "w") as h5:
                group = h5.create_group("US/US_DATASET0000")
                group.create_dataset(
                    "scatterers_positions", data=np.asarray([[0.0], [0.0], [0.02]])
                )
                group.create_dataset("scatterers_amplitude", data=[1.0])
            with h5py.File(scan_path, "w") as h5:
                group = h5.create_group("US/US_DATASET0000")
                group.create_dataset("x_axis", data=[-0.01, 0.01])
                group.create_dataset("z_axis", data=[0.005, 0.04])

            dataset = load_picmus_dataset(dataset_path)
            phantom = load_picmus_phantom(phantom_path)
            scan_x, scan_z = load_picmus_scan(scan_path)
            self.assertEqual(dataset.data.shape, (8, 4, 3))
            self.assertEqual(phantom.positions_m.shape, (1, 3))
            np.testing.assert_allclose(scan_x, [-0.01, 0.01])
            np.testing.assert_allclose(scan_z, [0.005, 0.04])


if __name__ == "__main__":
    unittest.main()

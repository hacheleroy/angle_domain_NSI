import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from method_names import ANGLE_NSI, CF_DAS, DAS, RECEIVE_NSI
from picmus_full_phantom_comparison import (
    canonicalize_images,
    load_staged_cache,
    profile_diagnostics,
    save_staged_cache,
)


class PicmusFullPhantomTests(unittest.TestCase):
    def test_partial_method_cache_is_resumable_and_signature_checked(self) -> None:
        x_axis = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        z_axis = np.asarray([2.0, 3.0], dtype=np.float32)
        image = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.npz"
            metadata = root / "cache.json"
            save_staged_cache(
                cache,
                metadata,
                "current-signature",
                {DAS: image},
                x_axis,
                z_axis,
            )
            loaded, loaded_metadata = load_staged_cache(
                cache,
                metadata,
                "current-signature",
                x_axis,
                z_axis,
            )
            stale, _ = load_staged_cache(
                cache,
                metadata,
                "different-signature",
                x_axis,
                z_axis,
            )

        self.assertEqual(set(loaded), {DAS})
        np.testing.assert_array_equal(loaded[DAS], image)
        self.assertEqual(loaded_metadata["methods"], [DAS])
        self.assertEqual(stale, {})

    def test_flat_reconstruction_arrays_are_reshaped_for_display(self) -> None:
        arrays = {
            DAS: np.arange(6),
            CF_DAS: np.arange(6) + 10,
            RECEIVE_NSI: np.arange(6) + 20,
            ANGLE_NSI: np.arange(6) + 30,
        }
        images = canonicalize_images(arrays, (2, 3))
        self.assertEqual(set(images), set(arrays))
        self.assertTrue(all(image.shape == (2, 3) for image in images.values()))

    def test_profile_diagnostic_distinguishes_notch_from_shift(self) -> None:
        axis = np.linspace(-0.2, 0.2, 401)
        das = np.exp(-0.5 * (axis / 0.045) ** 2)
        split = np.exp(-0.5 * ((axis + 0.045) / 0.010) ** 2)
        split += 0.7 * np.exp(-0.5 * ((axis - 0.045) / 0.010) ** 2)
        shifted = np.exp(-0.5 * ((axis + 0.035) / 0.030) ** 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            profiles = root / "profiles.npz"
            summary.write_text(
                json.dumps({"target_positions_mm": [[0.0, 0.0, 20.0]]}),
                encoding="utf-8",
            )
            np.savez_compressed(
                profiles,
                lateral_axes_mm=axis[None, :],
                target_1_das_lateral=das,
                target_1_receive_nsi_lateral=split,
                target_1_angle_nsi_lateral=shifted,
            )

            rows = profile_diagnostics(summary, profiles)

        by_method = {row["method"]: row for row in rows}
        self.assertFalse(by_method[DAS]["central_notch_detected"])
        self.assertTrue(by_method[RECEIVE_NSI]["central_notch_detected"])
        self.assertGreaterEqual(
            by_method[RECEIVE_NSI]["central_notch_depth_db"], 6.0
        )
        self.assertFalse(by_method[ANGLE_NSI]["central_notch_detected"])
        self.assertAlmostEqual(
            by_method[ANGLE_NSI]["local_peak_offset_mm"], -0.035, places=3
        )


if __name__ == "__main__":
    unittest.main()

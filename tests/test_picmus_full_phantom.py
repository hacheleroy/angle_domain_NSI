import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from method_names import ANGLE_NSI, CF_DAS, DAS, RECEIVE_NSI
from picmus_full_phantom_comparison import (
    canonicalize_images,
    lateral_tiles,
    load_tile_cache,
    load_staged_cache,
    profile_diagnostics,
    reconstruct_isolated_stage,
    save_staged_cache,
    save_tile_cache,
    tile_cache_path,
)


class PicmusFullPhantomTests(unittest.TestCase):
    def test_lateral_tiles_cover_axis_once(self) -> None:
        self.assertEqual(lateral_tiles(9, 4), [(0, 4), (4, 8), (8, 9)])
        with self.assertRaises(ValueError):
            lateral_tiles(9, 0)

    def test_tile_cache_round_trip_validates_identity_and_signal(self) -> None:
        x_axis = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        z_axis = np.asarray([2.0, 3.0], dtype=np.float32)
        image = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tile.npz"
            save_tile_cache(
                path,
                signature="signature",
                stage="iq",
                begin=0,
                end=3,
                x_axis=x_axis,
                z_axis=z_axis,
                images={DAS: image},
            )
            loaded = load_tile_cache(
                path,
                signature="signature",
                stage="iq",
                begin=0,
                end=3,
                x_axis=x_axis,
                z_axis=z_axis,
                methods=(DAS,),
            )
            stale = load_tile_cache(
                path,
                signature="other",
                stage="iq",
                begin=0,
                end=3,
                x_axis=x_axis,
                z_axis=z_axis,
                methods=(DAS,),
            )
            with self.assertRaises(ValueError):
                save_tile_cache(
                    path,
                    signature="signature",
                    stage="iq",
                    begin=0,
                    end=3,
                    x_axis=x_axis,
                    z_axis=z_axis,
                    images={DAS: np.zeros_like(image)},
                )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        np.testing.assert_array_equal(loaded[0][DAS], image)
        self.assertIsNone(stale)

    def test_isolated_stage_resumes_valid_tiles(self) -> None:
        x_axis = np.arange(5, dtype=np.float32)
        z_axis = np.arange(2, dtype=np.float32)
        signature = "resume-signature"
        args = Namespace(
            force=False,
            tile_x_lines=2,
            device="0",
            dataset=Path("dataset.hdf5"),
            phantom=Path("phantom.hdf5"),
            output_dir=Path("output"),
            carrier_frequency_mhz=5.0,
            f_number=1.0,
            nsi_c=0.05,
            x_margin_mm=3.0,
            z_margin_mm=3.0,
            x_spacing_mm=0.05,
            z_spacing_mm=0.10,
            mv_x_spacing_mm=0.15,
            dmas_x_spacing_mm=0.10,
            dmas_z_spacing_mm=0.02,
            mv_chunk_pixels=32,
            iq_focus_chunk_pixels=1024,
            dmas_x_chunk_lines=16,
            angle_count=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            tile_root = Path(directory)
            first_path = tile_cache_path(tile_root, "iq", 0, 2)
            save_tile_cache(
                first_path,
                signature=signature,
                stage="iq",
                begin=0,
                end=2,
                x_axis=x_axis[0:2],
                z_axis=z_axis,
                images={DAS: np.full((2, 2), 1.0, dtype=np.float32)},
            )
            calls: list[tuple[int, int]] = []

            def create_missing_tile(command, **_kwargs):
                begin = int(command[command.index("--worker-x-start") + 1])
                end = int(command[command.index("--worker-x-stop") + 1])
                calls.append((begin, end))
                save_tile_cache(
                    tile_cache_path(tile_root, "iq", begin, end),
                    signature=signature,
                    stage="iq",
                    begin=begin,
                    end=end,
                    x_axis=x_axis[begin:end],
                    z_axis=z_axis,
                    images={
                        DAS: np.full(
                            (end - begin, z_axis.size),
                            float(begin + 1),
                            dtype=np.float32,
                        )
                    },
                )

            with patch(
                "picmus_full_phantom_comparison.subprocess.run",
                side_effect=create_missing_tile,
            ):
                assembled, _ = reconstruct_isolated_stage(
                    args,
                    stage="iq",
                    native_x=x_axis,
                    z_axis=z_axis,
                    methods=(DAS,),
                    run_signature=signature,
                    tile_root=tile_root,
                )

            self.assertEqual(calls, [(2, 4), (4, 5)])
            np.testing.assert_array_equal(assembled[DAS][0:2], 1.0)
            np.testing.assert_array_equal(assembled[DAS][2:4], 3.0)
            np.testing.assert_array_equal(assembled[DAS][4:5], 5.0)
            manifest = json.loads(
                (tile_root / "iq_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["completed_tile_count"], 3)

            with patch(
                "picmus_full_phantom_comparison.subprocess.run"
            ) as subprocess_run:
                resumed, _ = reconstruct_isolated_stage(
                    args,
                    stage="iq",
                    native_x=x_axis,
                    z_axis=z_axis,
                    methods=(DAS,),
                    run_signature=signature,
                    tile_root=tile_root,
                )
            subprocess_run.assert_not_called()
            np.testing.assert_array_equal(resumed[DAS], assembled[DAS])

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

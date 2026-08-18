"""CPU tests for the matched microbubble trace-width analysis."""

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace_width_analysis import (
    PeakWidth,
    analyse_three_profiles,
    match_three_methods,
    measure_peak_widths,
    save_analysis_outputs,
)


def _record(x_mm: float, width_mm: float = 0.2) -> PeakWidth:
    return PeakWidth(
        sample_index=0,
        x_mm=x_mm,
        peak_db=0.0,
        left_crossing_mm=x_mm - width_mm / 2.0,
        right_crossing_mm=x_mm + width_mm / 2.0,
        width_mm=width_mm,
    )


class TestTraceWidthAnalysis(unittest.TestCase):
    def test_linear_crossings_recover_known_width(self):
        x_mm = np.linspace(-1.0, 1.0, 2001)
        # The -6 dB crossings are exactly at x = +/-0.2 mm.
        profile_db = -6.0 * (x_mm / 0.2) ** 2
        records = measure_peak_widths(
            x_mm,
            profile_db,
            prominence_db=6.0,
            min_distance_mm=0.2,
            min_height_db=-1.0,
        )
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0].width_mm, 0.4, places=6)

    def test_matching_never_reuses_a_target_peak(self):
        das = [_record(0.00), _record(0.20)]
        receive = [_record(0.10)]
        angle = [_record(0.01), _record(0.21)]
        matches, pair_maps = match_three_methods(
            das, receive, angle, tolerance_mm=0.15
        )
        self.assertEqual(len(pair_maps["receive_nsi"]), 1)
        self.assertEqual(len(set(pair_maps["receive_nsi"].values())), 1)
        self.assertEqual(len(matches), 1)

    def test_summary_uses_only_complete_triplets_and_sample_sd(self):
        x_mm = np.linspace(-2.0, 2.0, 4001)

        def two_peaks(width_a, width_b, shift=0.0):
            first = -6.0 * ((x_mm - (-0.8 + shift)) / (width_a / 2.0)) ** 2
            second = -6.0 * ((x_mm - (0.8 + shift)) / (width_b / 2.0)) ** 2
            return np.maximum(first, second)

        analysis = analyse_three_profiles(
            x_mm,
            two_peaks(0.20, 0.40),
            two_peaks(0.10, 0.20, shift=0.01),
            two_peaks(0.15, 0.25, shift=-0.01),
            prominence_db=6.0,
            min_distance_mm=0.2,
            min_height_db=-1.0,
            matching_tolerance_mm=0.15,
        )
        self.assertEqual(analysis["counts"]["complete_triplets"], 2)
        self.assertAlmostEqual(analysis["width_summary"]["das"]["mean_mm"], 0.3, places=5)
        self.assertAlmostEqual(
            analysis["width_summary"]["das"]["sample_sd_mm"],
            np.std([0.2, 0.4], ddof=1),
            places=5,
        )

    def test_csv_and_json_outputs_are_created(self):
        x_mm = np.linspace(-1.0, 1.0, 2001)
        profile_db = -6.0 * (x_mm / 0.2) ** 2
        analysis = analyse_three_profiles(
            x_mm,
            profile_db,
            profile_db,
            profile_db,
            prominence_db=6.0,
            min_height_db=-1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = save_analysis_outputs(
                analysis, directory, depth_mm=11.0
            )
            self.assertTrue(Path(csv_path).is_file())
            self.assertTrue(Path(json_path).is_file())


if __name__ == "__main__":
    unittest.main()

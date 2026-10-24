import json
import unittest

import numpy as np
import pandas as pd

from LLM.world import build_world_frame, transform_pose_dataframe, validate_world_frame
from LLM.takeoff import detect_takeoff


class WorldTests(unittest.TestCase):
    def test_tilted_reference_is_orthogonal_and_round_trips(self):
        frame = build_world_frame([10, 20, 30], [0.3, -0.8, 0.2], [1, 0.1, 0.4])
        origin, rotation = validate_world_frame(frame)
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3)))
        self.assertAlmostEqual(np.linalg.det(rotation), 1)
        points = np.array([[1, 2, 3], [4, 5, 6]])
        axes = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        df = pd.DataFrame((points @ rotation.T) + origin, columns=["x_mm", "y_mm", "z_mm"])
        df[["nx", "ny", "nz"]] = axes @ rotation.T
        df["position_valid"], df["orientation_valid"], df["frame"] = True, True, [17, 18]
        result = transform_pose_dataframe(df, frame)
        self.assertTrue(np.allclose(result[["x_mm", "y_mm", "z_mm"]], points))
        self.assertTrue(np.allclose(result[["nx", "ny", "nz"]], axes))
        self.assertTrue(np.allclose(result.alpha_x_deg, [45, 0]))
        self.assertTrue(np.allclose(result.alpha_y_deg, [0, 45]))
        self.assertEqual(result.frame.tolist(), [17, 18])
        json.dumps(frame, allow_nan=False)

    def test_reject_degenerate_or_left_handed_frame(self):
        for up, hint in [([1, 0, 0], [1, 0, 0]), ([0, 0, 0], [1, 0, 0]),
                         ([0, 0, np.nan], [1, 0, 0]), ([0, 0, 1], [0, 0, 0])]:
            with self.subTest(up=up), self.assertRaises(ValueError):
                build_world_frame([0, 0, 0], up, hint)
        frame = build_world_frame([0, 0, 0], [0, 0, 1])
        frame["R_camera_from_world"][0][0] = -1
        with self.assertRaises(ValueError):
            validate_world_frame(frame)

    def test_independent_validity_and_csv_flags(self):
        df = pd.DataFrame({"x_mm": [1, np.nan, 2], "y_mm": 0, "z_mm": 0,
                           "nx": [0, 0, 0], "ny": 0, "nz": [1, 1, 0],
                           "position_valid": ["False", "True", "True"],
                           "orientation_valid": ["True", "True", "True"]})
        before = df.copy(deep=True)
        result = transform_pose_dataframe(df, build_world_frame([0, 0, 0], [0, 0, 1]))
        self.assertEqual(result.position_valid.tolist(), [False, False, True])
        self.assertEqual(result.orientation_valid.tolist(), [True, True, False])
        self.assertTrue(np.isnan(result.iloc[0].x_mm))
        self.assertTrue(np.isnan(result.iloc[2].alpha_x_deg))
        pd.testing.assert_frame_equal(before, df)


class TakeoffTests(unittest.TestCase):
    @staticmethod
    def flight():
        t = np.arange(101) / 100
        return pd.DataFrame({"t_s": t, "z_mm": 12 + np.maximum(t - 0.4, 0) * 20,
                             "position_valid": True})

    def test_rise_detected_with_confirmation_delay_and_reference_invariance(self):
        df = self.flight()
        result = detect_takeoff(df)
        self.assertEqual(result["status"], "detected")
        self.assertGreaterEqual(result["onset_s"], 0.45)
        self.assertLessEqual(result["onset_s"], 0.47)
        self.assertGreaterEqual(result["confirmed_s"] - result["onset_s"], 0.05 - 1e-10)
        self.assertFalse(result["physical_bearing_clearance_measured"])
        df.z_mm += 123
        self.assertEqual(detect_takeoff(df)["onset_s"], result["onset_s"])
        json.dumps(result, allow_nan=False)

    def test_stationary_noise_and_short_spike_do_not_count(self):
        df = self.flight()
        df.z_mm = 12 + 0.05 * np.sin(df.t_s * 100)
        df.loc[45:46, "z_mm"] = [15, 16]
        self.assertEqual(detect_takeoff(df)["status"], "no_takeoff")

    def test_no_confirmation_across_invalid_row_or_time_gap(self):
        df = self.flight()
        df.loc[48, "position_valid"] = False
        self.assertEqual(detect_takeoff(df)["status"], "uncertain")
        df = self.flight().drop(index=np.arange(48, 60))
        result = detect_takeoff(df)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["onset_s"])

    def test_empty_missing_baseline_short_tail_and_invalid_config(self):
        df = self.flight()
        self.assertEqual(detect_takeoff(df.iloc[0:0])["status"], "uncertain")
        self.assertEqual(detect_takeoff(df.iloc[20:])["reason"], "insufficient_baseline")
        self.assertEqual(detect_takeoff(df.iloc[:49])["status"], "uncertain")
        for cfg in [{"sustain_s": 0}, {"baseline_start_s": 2}, {"noise_sigma_factor": -1},
                    {"end_s": 0.1}, {"max_gap_s": float("nan")}, {"min_baseline_samples": 2.5},
                    {"misspelled": 1}]:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                detect_takeoff(df, cfg)
        df.loc[4, "t_s"] = df.loc[3, "t_s"]
        with self.assertRaises(ValueError):
            detect_takeoff(df)

    def test_above_threshold_after_gap_is_not_witnessed_takeoff(self):
        df = self.flight()
        df.loc[25:55, "position_valid"] = False
        self.assertEqual(detect_takeoff(df)["status"], "uncertain")

    def test_invalid_last_baseline_row_cannot_be_skipped_for_onset(self):
        df = self.flight()
        df.loc[20, "position_valid"] = False
        df.loc[21:, "z_mm"] += 2
        self.assertEqual(detect_takeoff(df)["status"], "uncertain")


if __name__ == "__main__":
    unittest.main()

"""Recorded-frame, geometry and experimental-validity contract checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from LLM.axis_pose import run_axis_pose, _view
from LLM.record.io import write_json, read_json
from LLM.tests.test_capture_pose import _rig
from controller.calib.rig import StereoRig
from controller.pose import axis_pose as core


class AxisPoseChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='llm-axis-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rig_path = self.root/'rig.json'
        _rig(self.rig_path)
        self.rig = StereoRig.load(self.rig_path)
        self.take = self.root/'run'
        capture = self.take/'raw/camera'
        capture.mkdir(parents=True)
        self.backgrounds = {}
        self.truth = np.array([.2, -.9, .38]); self.truth /= np.linalg.norm(self.truth)
        for tag, cam in zip('AB', self.rig.cameras):
            (capture/tag).mkdir()
            p = self.root/f'{tag}.png'
            cv2.imwrite(str(p), np.zeros((240, 320), np.uint8))
            self.backgrounds[tag] = p
            gray = core._render(cam, np.array([5., 4., 200.]), self.truth,
                                radius=22., thick=.3, size=(240, 320))
            # Frame 1 disappears; frame 3 has excessive source skew.
            w = cv2.VideoWriter(str(capture/tag/f'{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (320, 240))
            self.assertTrue(w.isOpened())
            for k in range(5):
                f = np.zeros_like(gray) if k == 1 and tag == 'B' else gray
                w.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
            w.release()
        pd.DataFrame(dict(index=[0, 1, 2, 3, 4, 5], written=[1, 0, 1, 1, 1, 1],
                          t_a=[10., 10.01, 10.02, 10.04, 10.06, 10.1],
                          t_b=[10.001, 10.011, 10.021, 10.041, 10.081, 10.101])).to_csv(capture/'frames.csv', index=False)
        write_json(capture/'meta.json', dict(n_frames=6, dropped=1, encoder_errors=0, mode=[320, 240], rotate180=False))
        # Existing official pose output must not be changed by this diagnostic.
        (self.take/'processed').mkdir()
        (self.take/'processed/pose_camera.csv').write_text('untouched\n')
        self.opts = dict(robot_rois={'A': [60, 50, 200, 150], 'B': [60, 50, 200, 150]},
                         background_paths=self.backgrounds, background_confirmed=True)

    def test_video_geometry_missing_skew_reacquisition_and_outputs(self):
        report = run_axis_pose(self.take, self.rig_path, **self.opts)
        df = pd.read_csv(report['csv'])
        self.assertEqual(df.frame_index.tolist(), [0, 2, 3, 4, 5])
        self.assertEqual(df.axis_estimate_valid.tolist(), [True, False, True, False, True])
        self.assertEqual(df.loc[3, 'reason'], 'stereo_skew_exceeds_limit')
        self.assertTrue(df.loc[2, 'reference_reset'])
        self.assertTrue(df.loc[4, 'reference_reset'])
        self.assertTrue(df.loc[[1, 3], ['nx_est', 'x_est_mm']].isna().all().all())
        self.assertFalse(df.orientation_valid.any() or df.position_valid.any())
        axis = df.loc[0, ['nx_est', 'ny_est', 'nz_est']].to_numpy(float)
        # Coarse raster + parallel cameras are not the core's calibrated-rig
        # benchmark. This guard catches wrong frames/branches, not physical accuracy.
        self.assertLess(np.degrees(np.arccos(np.clip(axis @ self.truth, -1, 1))), 5.)
        np.testing.assert_allclose(df.t_source_a_s, [10, 10.02, 10.04, 10.06, 10.1])
        self.assertEqual((self.take/'processed/pose_camera.csv').read_text(), 'untouched\n')
        for name in ('curves', 'overlays'):
            self.assertIsNotNone(cv2.imread(report[name]))
        meta = read_json(Path(report['directory'])/'metadata.json')
        self.assertIn(str(self.rig_path.resolve()), meta['input_sha256'])
        self.assertTrue((Path(report['directory'])/'inputs/background_A.png').exists())

    def test_explicit_threshold_preview_and_bad_lens_model(self):
        opts = self.opts | dict(background_mode='threshold', background_paths=None,
                               background_confirmed=False, max_frames=1)
        report = run_axis_pose(self.take, self.rig_path, **opts)
        self.assertEqual(report['frames'], 1)
        self.assertEqual(report['accepted'], 1)
        data = read_json(self.rig_path)
        data['cameras'][0]['dist'] = [0.] * 8
        write_json(self.rig_path, data)
        with self.assertRaisesRegex(ValueError, '5-term'):
            run_axis_pose(self.take, self.rig_path, **opts)

    def test_configuration_and_clipped_masks(self):
        for changes, message in [({'background_confirmed': False}, 'empty-scene'),
                                 ({'robot_rois': {'A': [0, 0, 10, 10]}}, 'both A and B'),
                                 ({'robot_rois': {'A': [0, 0, 400, 400], 'B': [0, 0, 10, 10]}}, 'outside'),
                                 ({'reference_axis_camera': [0, 0, 0]}, 'nonzero'),
                                 ({'max_gap_s': 0}, 'Invalid')]:
            with self.assertRaisesRegex(ValueError, message):
                run_axis_pose(self.take, self.rig_path, **(self.opts | changes))
        image = np.full((40, 50), 255, np.uint8)
        self.assertEqual(_view(image, None, (0, 0, 50, 40), 90)[2], 'mask_touches_roi_border')
        with self.assertRaisesRegex(ValueError, 'rotation'):
            data = read_json(self.rig_path); data['rotate180'] = True
            write_json(self.rig_path, data)
            run_axis_pose(self.take, self.rig_path, **self.opts)

    def test_jump_is_rejected_and_no_state_is_carried_through_failure(self):
        original = core.pose
        calls = []
        def solve(cams, views, ref):
            calls.append(ref.copy())
            out = original(cams, views, ref)
            if len(calls) == 3 and out is not None:
                return out[0], np.array([1., 0., 0.]), out[2]
            return out
        # Keep all five frames visible and paired so there is a continuity chain.
        capture = self.take/'raw/camera'
        rows = pd.read_csv(capture/'frames.csv'); rows['t_b'] = rows.t_a + .001
        rows.to_csv(capture/'frames.csv', index=False)
        with patch('LLM.axis_pose.core.pose', side_effect=solve):
            result = run_axis_pose(self.take, self.rig_path, **self.opts)
        df = pd.read_csv(result['csv'])
        self.assertEqual(df.loc[3, 'reason'], 'axis_jump')
        self.assertTrue(df.loc[4, 'reference_reset'])
        np.testing.assert_allclose(calls[-1], [0., -1., 0.])


if __name__ == '__main__':
    unittest.main()

"""Offline joint-holder regressions with independent labelled camera views."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from controller.calib.rig import Camera, StereoRig, look_at
from LLM.analyse.holder_geometry import fit_holder, localize_holder, match_holder
from LLM.record.io import sha256, write_json
from LLM.analyse.rig import camera_key


CAMERA = dict(indices=[1, 0], width=1280, height=800, fps=120,
              rotate180=True, backend='auto', codec='auto')


def scene():
    K = np.array([[900., 0, 640], [0, 900, 400], [0, 0, 1]])
    ca = look_at([20, -380, 120], [49.5, 10.5, 20])
    cb = look_at([420, 40, 140], [49.5, 10.5, 20])
    true_pose = np.linalg.inv(ca)
    rig = StereoRig((Camera(K, np.zeros(5), np.eye(4), 'A'),
                     Camera(K, np.zeros(5), true_pose @ cb, 'B')),
                    meta=dict(image_size=[1280, 800], rotate180=True))
    layout = dict(pattern_type='checker_quadrants', units='mm', survey_confirmed=True,
                  bearing_center_fixture_mm=[49.5, 10.5, 15], x_hint_fixture=[1, 0, 0], points={}, surfaces={})
    for block, x, y in [('P1', 1.5, -37.5), ('P2', 97.5, -37.5),
                        ('P3', 97.5, 58.5), ('P4', 1.5, 58.5)]:
        layout['points'][block] = [x, y, 40]
        layout['surfaces'][block] = dict(normal=[0, 0, 1])
        for axis in 'XY':
            negative = block in (('P1', 'P4') if axis == 'X' else ('P1', 'P2'))
            normal = [-1 if negative else 1, 0, 0] if axis == 'X' else [0, -1 if negative else 1, 0]
            for level, z in [('L', 12), ('H', 30)]:
                label = f'{block}_{axis}_{level}'
                layout['points'][label] = [(-10.5 if negative else 109.5), y, z] if axis == 'X' else [x, (-49.5 if negative else 70.5), z]
                layout['surfaces'][label] = dict(outward_normal=normal)
    labels = dict(A=['P1_Y_L', 'P1_Y_H', 'P2_Y_L', 'P2_Y_H'],
                  B=['P2_X_L', 'P2_X_H', 'P3_X_L', 'P3_X_H'])
    observations = {}
    for tag, cam in zip('AB', rig.cameras):
        xyz = np.array([layout['points'][name] for name in labels[tag]]) @ true_pose[:3, :3].T + true_pose[:3, 3]
        uv = project(cam, xyz)
        observations[tag] = dict(zip(labels[tag], uv))
    return rig, layout, observations, true_pose


def project(cam, xyz):
    local = (np.asarray(xyz, float) - cam.position) @ cam.R
    return cv2.projectPoints(local, np.zeros(3), np.zeros(3), cam.K, cam.dist)[0].reshape(-1, 2)


def files(root):
    rig, layout, observations, truth = scene()
    calibration = root / 'rig.json'; rig.save(calibration)
    layout_path = root / 'layout.json'; write_json(layout_path, layout)
    regions = root / 'regions.json'; write_json(regions, {'schema_version': 1})
    pair = root / 'pair'; pair.mkdir()
    write_json(pair / 'meta.json', dict(session_id='test', camera=camera_key(CAMERA), times=[1., 1.001]))
    for tag in 'AB':
        cv2.imwrite(str(pair / f'{tag}.png'), np.zeros((800, 1280), np.uint8))
    report = dict(cameras={tag: dict(landmarks={label: dict(status='detected', center_px=uv.tolist())
                                               for label, uv in observations[tag].items()}) for tag in 'AB'},
                  physical_accuracy_validated=False)
    plumb = {}
    for key, xyz in [('bottom', [49.5, 10.5, 55]), ('top', [49.5, 10.5, 95])]:
        camera_xyz = np.asarray(xyz) @ truth[:3, :3].T + truth[:3, 3]
        plumb[key] = {tag.lower(): project(cam, [camera_xyz])[0].tolist() for tag, cam in zip('AB', rig.cameras)}
    args = (pair, calibration, layout_path, regions, plumb, CAMERA)
    return args, report, truth, layout


class HolderGeometryTests(unittest.TestCase):
    def test_rendered_top_checkers_detect_and_localize_end_to_end(self):
        from LLM.analyse.holder import make_holder_regions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, layout, _, _ = scene()
            K = np.array([[900., 0, 640], [0, 900, 400], [0, 0, 1]])
            baseline = np.eye(4); baseline[0, 3] = 60
            rig = StereoRig((Camera(K, np.zeros(5), np.eye(4), 'A'),
                             Camera(K, np.zeros(5), baseline, 'B')),
                            meta=dict(image_size=[1280, 800], rotate180=True))
            pose = np.eye(4); pose[:3, :3] = np.diag([1., -1., -1.])
            pose[:3, 3] = [-49.5, 10.5, 440]
            calibration = root / 'rig.json'; rig.save(calibration)
            layout_path = root / 'layout.json'; write_json(layout_path, layout)
            write_json(root / 'meta.json', dict(session_id='rendered', camera=camera_key(CAMERA), times=[2., 2.001]))
            regions = {'A': {}, 'B': {}}
            for tag, cam in zip('AB', rig.cameras):
                image = np.full((3200, 5120), 30, np.uint8)
                for label in ['P1', 'P2', 'P3', 'P4']:
                    center = np.array(layout['points'][label], float)
                    for du, dv in [(-8, -8), (0, 0)]:
                        corners = center + np.array([[du, dv, 0], [du+8, dv, 0],
                                                     [du+8, dv+8, 0], [du, dv+8, 0]])
                        uv = project(cam, corners @ pose[:3, :3].T + pose[:3, 3])
                        cv2.fillConvexPoly(image, np.rint(uv * 4).astype(np.int32), 230)
                    uv_center = project(cam, [center @ pose[:3, :3].T + pose[:3, 3]])[0]
                    x, y = np.rint(uv_center).astype(int)
                    regions[tag][label] = [int(x-15), int(y-15), 31, 31]
                cv2.imwrite(str(root / f'{tag}.png'), cv2.resize(image, (1280, 800), interpolation=cv2.INTER_AREA))
            regions_path = root / 'regions.json'
            specification = make_holder_regions(root, regions_path, layout_path=layout_path)
            specification['regions'] = regions; write_json(regions_path, specification)
            plumb = {}
            for name, point in [('bottom', [49.5, 10.5, 50]), ('top', [49.5, 10.5, 80])]:
                xyz = np.array(point) @ pose[:3, :3].T + pose[:3, 3]
                plumb[name] = {tag.lower(): project(cam, [xyz])[0].tolist() for tag, cam in zip('AB', rig.cameras)}
            world = localize_holder(root, calibration, layout_path, regions_path, plumb, CAMERA, session_id='rendered')
            expected = pose[:3, :3] @ layout['bearing_center_fixture_mm'] + pose[:3, 3]
            np.testing.assert_allclose(world['origin_camera_mm'], expected, atol=.5)
            np.testing.assert_allclose(np.asarray(world['R_camera_from_world'])[:, 2], [0, 0, -1], atol=1e-7)
            self.assertLess(world['evidence']['fixture_fit']['rms_px'], .2)
            self.assertEqual(world['evidence']['fixture_fit']['n_observations'], 8)
            self.assertTrue((root / 'holder_detections.json').exists())
            for tag in 'AB':
                self.assertEqual(cv2.imread(str(root / f'{tag}_holder_landmarks.png')).shape, (800, 1280, 3))
            layout['survey_confirmed'] = False; write_json(layout_path, layout)
            with self.assertRaisesRegex(ValueError, 'surveyed'):
                localize_holder(root, calibration, layout_path, regions_path, plumb, CAMERA, session_id='rendered')

    def test_disjoint_side_faces_recover_one_rigid_pose(self):
        rig, layout, observations, truth = scene()
        self.assertFalse(set(observations['A']) & set(observations['B']))
        before = rig.b.T_world_cam.copy()
        pose, report = fit_holder(rig, layout, observations)
        np.testing.assert_allclose(pose, truth, atol=1e-6)
        np.testing.assert_array_equal(rig.b.T_world_cam, before)
        self.assertEqual(report['shared_labels'], [])
        self.assertEqual(report['n_observations'], 8)
        self.assertLess(report['rms_px'], 1e-7)
        self.assertFalse(report['physical_accuracy_validated'])

    def test_pixel_noise_preserves_pose_and_reports_actual_error(self):
        rig, layout, observations, truth = scene()
        rng = np.random.default_rng(91)
        for view in observations.values():
            for label in view:
                view[label] += rng.normal(0, .08, 2)
        pose, report = fit_holder(rig, layout, observations)
        self.assertGreater(report['rms_px'], .01)
        self.assertLess(report['rms_px'], .2)
        np.testing.assert_allclose(pose[:3, 3], truth[:3, 3], atol=.4)

    def test_wrong_identity_and_changed_rig_are_rejected(self):
        rig, layout, observations, _ = scene()
        a = observations['A']
        a['P1_Y_L'], a['P2_Y_H'] = a['P2_Y_H'], a['P1_Y_L']
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            fit_holder(rig, layout, observations)
        rig, layout, observations, _ = scene()
        rig.b.T_world_cam[0, 3] += 20
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            fit_holder(rig, layout, observations)

    def test_degenerate_unknown_and_duplicate_centers_are_rejected(self):
        rig, layout, observations, _ = scene()
        for i, label in enumerate(observations['A']):
            layout['points'][label] = [i * 20, 0, 0]
        with self.assertRaisesRegex(ValueError, 'collinear'):
            fit_holder(rig, layout, observations)
        rig, layout, observations, _ = scene()
        observations['A']['unknown'] = observations['A'].pop('P1_Y_L')
        with self.assertRaisesRegex(ValueError, 'unknown CAD'):
            fit_holder(rig, layout, observations)
        rig, layout, observations, _ = scene()
        observations['A']['P1_Y_L'] = observations['A']['P1_Y_H'].copy()
        with self.assertRaisesRegex(ValueError, 'duplicate or unresolved'):
            fit_holder(rig, layout, observations)

    def test_back_facing_names_do_not_pass_a_low_residual(self):
        rig, layout, observations, _ = scene()
        for surface in layout['surfaces'].values():
            field = 'normal' if 'normal' in surface else 'outward_normal'
            surface[field] = (-np.asarray(surface[field])).tolist()
        with self.assertRaisesRegex(ValueError, 'back-facing'):
            fit_holder(rig, layout, observations)

    def test_world_uses_surveyed_bearing_and_plumb_and_binds_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, report, truth, layout = files(Path(tmp))
            with patch('LLM.analyse.holder.inspect_holder', return_value=report):
                world = localize_holder(*args, session_id='test')
            expected = truth[:3, :3] @ layout['bearing_center_fixture_mm'] + truth[:3, 3]
            np.testing.assert_allclose(world['origin_camera_mm'], expected, atol=1e-6)
            np.testing.assert_allclose(np.asarray(world['R_camera_from_world'])[:, 2], truth[:3, 2], atol=1e-6)
            self.assertEqual(world['evidence']['fixture_fit']['shared_labels'], [])
            self.assertFalse(world['evidence']['physical_accuracy_validated'])
            for p in [args[1], args[2], args[3], args[0] / 'A.png', args[0] / 'B.png', args[0] / 'meta.json']:
                self.assertEqual(world['evidence']['input_sha256'][str(p.resolve())], sha256(p))

    def test_unconfirmed_survey_unknown_bearing_and_wrong_session_refuse_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, _, layout = files(Path(tmp))
            layout['survey_confirmed'] = False; write_json(args[2], layout)
            with self.assertRaisesRegex(ValueError, 'surveyed'):
                localize_holder(*args, session_id='test')
            layout['survey_confirmed'] = True; layout['bearing_center_fixture_mm'] = None
            write_json(args[2], layout)
            with self.assertRaisesRegex(ValueError, 'bearing'):
                localize_holder(*args, session_id='test')
            with self.assertRaisesRegex(ValueError, 'different camera session'):
                localize_holder(*args, session_id='other')

    def test_failed_detection_invalid_plumb_and_wrong_image_mode_refuse_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, report, _, _ = files(Path(tmp))
            failed = copy.deepcopy(report)
            failed['cameras']['A']['landmarks']['P1_Y_L'] = dict(status='ambiguous')
            with patch('LLM.analyse.holder.inspect_holder', return_value=failed):
                with self.assertRaisesRegex(ValueError, 'ambiguous'):
                    localize_holder(*args, session_id='test')
            args[4]['top']['a'] = [-10, 30]
            with patch('LLM.analyse.holder.inspect_holder', return_value=report):
                with self.assertRaisesRegex(ValueError, 'Plumb top'):
                    localize_holder(*args, session_id='test')
            cv2.imwrite(str(args[0] / 'B.png'), np.zeros((400, 640), np.uint8))
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                localize_holder(*args, session_id='test')

    def test_unlabelled_candidates_are_named_and_dots_break_the_symmetry(self):
        import json
        rig, layout, _, truth = scene()
        layout['id_dots'] = json.loads((Path(__file__).parents[1] / 'analyse/holder_layout.json').read_text())['id_dots']
        labels = list(layout['points'])
        X = np.array([layout['points'][k] for k in labels]) @ truth[:3, :3].T + truth[:3, 3]
        dots = np.array([d['center_mm'] for d in layout['id_dots']]) @ truth[:3, :3].T + truth[:3, 3]
        rng = np.random.default_rng(0)
        cands, images = [], []
        for cam in rig.cameras:
            uv = project(cam, X)
            keep = rng.random(len(uv)) > .3       # 30 % missed, 10 outliers, 0.1 px noise
            cands.append(rng.permutation(np.r_[uv[keep] + rng.normal(0, .1, (keep.sum(), 2)),
                                               rng.uniform([0, 0], [1280, 800], (10, 2))]))
            image = np.full((800, 1280), 40, np.uint8)
            for u, v in np.rint(project(cam, dots)).astype(int):
                cv2.circle(image, (u, v), 2, 230, -1)
            images.append(image)
        up = truth[:3, :3] @ [0, 0, 1]
        _, info = match_holder(rig, *cands, layout, up)
        self.assertTrue(info['ambiguous'], 'without the dots a 96 mm square ties at 90 deg')
        self.assertEqual(len(info['tied_yaws_deg']), 4)
        observations, info = match_holder(rig, *cands, layout, up, images=images)
        self.assertFalse(info['ambiguous'])
        for tag, cam in zip('AB', rig.cameras):
            for label, uv in observations[tag].items():
                self.assertLess(np.linalg.norm(uv - project(cam, [X[labels.index(label)]])[0]), 1, (tag, label))
        T, _ = fit_holder(rig, layout, observations)
        self.assertLess(np.abs(T[:3, 3] - truth[:3, 3]).max(), .5)
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1))), .2)


def run():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HolderGeometryTests))
    if not result.wasSuccessful():
        raise AssertionError('Holder geometry regression failed')


if __name__ == '__main__':
    run()

"""File-level provenance, identity and CLI checks for holder inspection."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from LLM.analyse.holder import inspect_holder, load_holder_layout, make_holder_regions
from LLM.record.io import read_json, sha256, write_json


def pair(root):
    image = np.full((200, 300), 25, np.uint8)
    image[60:100, 100:140] = 230
    image[100:140, 140:180] = 230
    for tag in 'AB':
        cv2.imwrite(str(root/f'{tag}.png'), image)


class HolderInspectionTests(unittest.TestCase):
    def test_cad_geometry_and_no_unearned_survey(self):
        layout = load_holder_layout()
        self.assertEqual(len(layout['points']), 20)
        self.assertEqual(layout['points']['P4_X_H'], [-10.5, 58.5, 30])
        self.assertEqual(layout['points']['P2'], [97.5, -37.5, 40])
        self.assertFalse(layout['survey_confirmed'])
        self.assertIsNone(layout['bearing_center_fixture_mm'])
        self.assertEqual(sum(d['face'] == 'top' for d in layout['id_dots']), 10)

    def test_image_binding_duplicate_labels_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pair(root)
            original = {t: sha256(root/f'{t}.png') for t in 'AB'}
            path = root/'regions.json'
            data = make_holder_regions(root, path)
            data['regions']['A'] = {'P1': [115, 75, 50, 50], 'P2': [115, 75, 50, 50]}
            data['regions']['B'] = {'P1': [115, 75, 50, 50]}
            write_json(path, data)
            report = inspect_holder(root, regions_path=path)
            self.assertEqual(report['cameras']['A']['landmarks']['P1']['status'], 'ambiguous')
            self.assertEqual(report['cameras']['A']['landmarks']['P2']['status'], 'ambiguous')
            self.assertNotIn('center_px', report['cameras']['A']['landmarks']['P1'])
            self.assertEqual(report['cameras']['B']['landmarks']['P1']['status'], 'detected')
            self.assertFalse(report['automatic_id_decoding'])
            self.assertIn(str(path.resolve()), report['input_sha256'])
            for tag in 'AB':
                self.assertEqual(sha256(root/f'{tag}.png'), original[tag])
                self.assertEqual(cv2.imread(str(root/f'{tag}_holder_landmarks.png')).shape, (200, 300, 3))
            with self.assertRaises(FileExistsError):
                make_holder_regions(root, path)
            changed = cv2.imread(str(root/'A.png')); changed[0, 0] = 255
            cv2.imwrite(str(root/'A.png'), changed)
            with self.assertRaisesRegex(ValueError, 'different reference'):
                inspect_holder(root, regions_path=path)

    def test_template_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pair(root)
            path = root/'regions.json'
            make_holder_regions(root, path)
            inspect_holder(root, regions_path=path)
            report = read_json(root/'holder_detections.json')
            self.assertTrue(report['cameras']['A']['candidates'])
            self.assertEqual(report['cameras']['A']['landmarks'], {})
            data = read_json(path); data['regions']['A']['not_a_landmark'] = [100, 50, 80, 100]
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, 'unknown CAD'):
                inspect_holder(root, regions_path=path)


if __name__ == '__main__':
    unittest.main()

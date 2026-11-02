"""Inspect the printed holder's checker centers without claiming decoded IDs.

The CAD targets are repeated 2x2 checker patterns, not AprilTags. Named regions
are explicit operator correspondences tied to the saved images by SHA-256.
See controller/calib/theory.md, 'Printed holder checker landmarks'.
"""
from pathlib import Path

import cv2
import numpy as np

from LLM.io import read_json, sha256, write_json
from LLM.holder_detection import detect_checker_centers, detect_labeled_centers


DEFAULT_LAYOUT = Path(__file__).parent / 'config/holder_layout.json'


def load_holder_layout(path=DEFAULT_LAYOUT):
    layout = read_json(path)
    if layout.get('pattern_type') != 'checker_quadrants' or layout.get('units') != 'mm':
        raise ValueError('Expected checker_quadrants holder layout in mm')
    points = layout.get('points')
    if not isinstance(points, dict) or not points:
        raise ValueError('Holder layout needs named CAD points')
    for name, xyz in points.items():
        p = np.asarray(xyz, float)
        if not name or p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Each holder landmark needs a name and finite CAD xyz')
    return layout


def _images(pair_dir):
    images = {}
    for tag in 'AB':
        path = Path(pair_dir) / f'{tag}.png'
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Save a static reference pair first: missing/invalid {path}')
        images[tag] = image
    return images


def make_holder_regions(pair_dir, destination, *, layout_path=DEFAULT_LAYOUT):
    """Create an empty correspondence file; existing operator work is preserved."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Regions file already exists: {destination}')
    images = _images(pair_dir)
    layout = load_holder_layout(layout_path)
    data = {
        'schema_version': 1,
        'image_sha256': {tag: sha256(Path(pair_dir)/f'{tag}.png') for tag in 'AB'},
        'image_size': {tag: [im.shape[1], im.shape[0]] for tag, im in images.items()},
        'regions': {'A': {}, 'B': {}},
        'available_labels': list(layout['points']),
        'note': 'Fill each camera with visible CAD label: [x, y, width, height] in saved-image pixels. '
                'Use one tight region per checker intersection. Candidate numbers are not CAD IDs. '
                'The same block on different faces/heights is a different physical point.',
    }
    write_json(destination, data)
    return data


def inspect_holder(pair_dir, *, regions_path=None, layout_path=DEFAULT_LAYOUT,
                   detector_options=None):
    """Detect all candidates and optionally refine explicitly named regions.

    Overlays and holder_detections.json are rebuildable outputs. A.png, B.png,
    CAD geometry, and the operator's correspondence file are never modified.
    """
    pair_dir = Path(pair_dir)
    layout = load_holder_layout(layout_path)
    images = _images(pair_dir)
    options = dict(detector_options or {})
    if 'roi' in options:
        raise ValueError('Use named regions instead of a global roi detector option')
    regions = {'A': {}, 'B': {}}
    paths = [Path(layout_path), pair_dir/'A.png', pair_dir/'B.png', Path(__file__),
             Path(__file__).with_name('holder_detection.py')]
    if (pair_dir/'meta.json').exists():
        paths.append(pair_dir/'meta.json')
    if regions_path is not None:
        data = read_json(regions_path)
        if data.get('schema_version') != 1 or set(data.get('regions', {})) != {'A', 'B'}:
            raise ValueError('Regions must contain schema_version=1 and regions.A / regions.B')
        regions = data['regions']
        for tag in 'AB':
            if data.get('image_sha256', {}).get(tag) != sha256(pair_dir/f'{tag}.png'):
                raise ValueError(f'Camera {tag} regions belong to different reference images; recreate/review them')
            if not isinstance(regions[tag], dict) or set(regions[tag]) - set(layout['points']):
                raise ValueError(f'Camera {tag} has invalid/unknown CAD landmark names')
        paths.append(Path(regions_path))
    report = {
        'schema_version': 1, 'kind': 'holder_checker_detection',
        'pattern_type': 'checker_quadrants', 'layout_version': layout.get('version'),
        'physical_accuracy_validated': False, 'automatic_id_decoding': False,
        'identity_method': 'operator_named_regions' if regions_path else 'unassigned_candidates',
        'detector_options': options, 'cameras': {},
        'input_sha256': {str(p.resolve()): sha256(p) for p in paths},
    }
    for tag, image in images.items():
        candidates = detect_checker_centers(image, **options)
        landmarks = detect_labeled_centers(image, regions[tag], **options)
        overlay = image.copy()
        for i, row in enumerate(candidates):
            point = tuple(np.rint(row['center_px']).astype(int))
            cv2.drawMarker(overlay, point, (255, 220, 0), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(overlay, f'c{i}', (point[0]+5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (255, 220, 0), 1, cv2.LINE_AA)
        for name, row in landmarks.items():
            x, y, w, h = map(int, regions[tag][name])
            color = (0, 220, 0) if row['status'] == 'detected' else (0, 140, 255)
            cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), color, 1)
            if row['status'] == 'detected':
                point = tuple(np.rint(row['center_px']).astype(int))
                cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(overlay, f"{name}: {row['status']}", (x, max(15, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        cv2.putText(overlay, 'Checker candidates; c# is NOT a CAD ID', (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 140, 255), 1, cv2.LINE_AA)
        output = pair_dir / f'{tag}_holder_landmarks.png'
        if not cv2.imwrite(str(output), overlay):
            raise OSError(f'Cannot write holder overlay: {output}')
        report['cameras'][tag] = {
            'image_size': [image.shape[1], image.shape[0]],
            'candidates': candidates, 'landmarks': landmarks, 'overlay': str(output),
        }
    write_json(pair_dir/'holder_detections.json', report)
    return report


def _self_check():
    import unittest
    from LLM.tests import test_holder
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_holder))
    if not result.wasSuccessful():
        raise AssertionError('Holder inspection checks failed')


if __name__ == '__main__':
    _self_check()

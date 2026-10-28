"""Fixed-rig verification and surveyed fixture localization; no hardware access.

A verification fits only target pose, never camera K/dist/relative R,t. Thus a
changed stereo rig cannot pass by recalibrating itself against the check images.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from controller.calib.calibrate import CharucoSpec, detect
from controller.calib.rig import StereoRig
from LLM.io import read_json, write_json, sha256
from LLM.world import build_world_frame
from LLM.pose import _point


def camera_key(camera):
    return {k: list(camera[k]) if k == 'indices' else camera[k]
            for k in ('indices', 'width', 'height', 'fps', 'rotate180', 'backend', 'codec')}


def fixed_rig(path, camera):
    data = read_json(path)
    if data.get('image_size') != [camera['width'], camera['height']]:
        raise ValueError('Camera mode differs from calibration')
    if data.get('rotate180') != camera['rotate180']:
        raise ValueError('Image rotation differs from calibration')
    rig = StereoRig.load(path)
    if len(rig.cameras) != 2 or not np.allclose(rig.a.T_world_cam, np.eye(4)):
        raise ValueError('Expected two cameras, calibration frame = camera A')
    for cam in rig.cameras:
        T = cam.T_world_cam
        if (not np.isfinite(T).all() or not np.isfinite(cam.K).all()
                or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(cam.R.T @ cam.R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(cam.R), 1)
                or min(cam.K[0, 0], cam.K[1, 1]) <= 0
                or (cam.dist is not None and not np.isfinite(cam.dist).all())):
            raise ValueError('Invalid camera matrices')
    if np.linalg.norm(rig.b.position) < 1e-6:
        raise ValueError('Zero stereo baseline')
    return rig, data


def fit_target(rig, objects, uv_a, uv_b):
    """Fit six target-pose parameters with both camera models held fixed."""
    obj = np.asarray(objects, dtype=float).reshape(-1, 3)
    uv = [np.asarray(x, dtype=float).reshape(-1, 2) for x in (uv_a, uv_b)]
    if len(obj) < 8 or any(len(x) != len(obj) for x in uv):
        raise ValueError('Need at least eight corresponding points')
    if not all(np.isfinite(x).all() for x in [obj, *uv]):
        raise ValueError('Non-finite target observations')
    ok, rv, tv = cv2.solvePnP(obj, uv[0], rig.a.K, rig.a.dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError('Target PnP failed')
    def project(p):
        xyz_a = obj @ cv2.Rodrigues(p[:3])[0].T + p[3:]
        return [xyz_a, (xyz_a - rig.b.position) @ rig.b.R]
    def residual(p):
        return np.concatenate([(cv2.projectPoints(x, np.zeros(3), np.zeros(3), c.K, c.dist)[0]
                                .reshape(-1, 2) - u).ravel()
                               for x, c, u in zip(project(p), rig.cameras, uv)])
    fit = least_squares(residual, np.r_[rv.ravel(), tv.ravel()], max_nfev=150)
    if not fit.success or any(np.any(x[:, 2] <= 0) for x in project(fit.x)):
        raise ValueError('Target solve failed or is behind camera')
    errors = residual(fit.x).reshape(2, -1, 2)
    lengths = np.linalg.norm(errors, axis=2)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(fit.x[:3])[0]; T[:3, 3] = fit.x[3:]
    return T, {'rms_px': float(np.sqrt(np.mean(lengths**2))),
               'max_point_px': float(lengths.max()), 'n_points': len(obj)}


def verify_bag(bag, calibration, camera, *, session_id, rms_limit=1.0):
    """Use a NEW stationary ChArUco bag; reject failures, do not trim outliers."""
    bag = Path(bag)
    rig, data = fixed_rig(calibration, camera)
    meta = read_json(bag / 'meta.json')
    if meta.get('camera_indices') != list(camera['indices']) or meta.get('rotate180') != camera['rotate180']:
        raise ValueError('Check bag was captured with different A/B mapping or rotation')
    board = meta['board']
    spec = CharucoSpec(*board['squares'], board['square_mm'], board['marker_mm'], board['dictionary'])
    for key in ('squares', 'square_mm', 'marker_mm', 'dictionary'):
        if board[key] != data[key]:
            raise ValueError(f'Verification board differs from calibration: {key}')
    rows, normals, centres = [], [], []
    hashes = {str((bag/'meta.json').resolve()): sha256(bag/'meta.json')}
    if (bag/'pairs.csv').exists():
        hashes[str((bag/'pairs.csv').resolve())] = sha256(bag/'pairs.csv')
    for a in sorted((bag / 'A').glob('pair_*.png')):
        b = bag / 'B' / a.name
        images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (a, b)]
        row = {'pair': a.stem}
        try:
            if any(im is None or im.shape != (camera['height'], camera['width']) for im in images):
                raise ValueError('missing image or incorrect dimensions')
            observations = [detect(spec, im) for im in images]
            maps = [dict(zip(ids.tolist(), corners)) if ids is not None else {} for corners, ids in observations]
            ids = sorted(set(maps[0]) & set(maps[1]))
            T, result = fit_target(rig, spec.corners_mm[ids], *[np.array([m[i] for i in ids]) for m in maps])
            row.update(result, passed=result['rms_px'] < rms_limit and result['max_point_px'] < 3.0)
            normals.append(T[:3, 2]); centres.append(T[:3, 3])
        except (ValueError, cv2.error) as exc:
            row.update(passed=False, reason=str(exc))
        rows.append(row)
        for p in (a, b):
            if p.exists(): hashes[str(p.resolve())] = sha256(p)
    spread = max((np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a in normals for b in normals), default=0)
    travel = max((np.linalg.norm(a-b) for a in centres for b in centres), default=0)
    report = {'schema_version': 1, 'kind': 'fixed_rig_verification', 'session_id': session_id,
              'camera': camera_key(camera), 'rig_sha256': sha256(calibration), 'input_sha256': hashes,
              'pairs': rows, 'normal_spread_deg': float(spread), 'translation_spread_mm': float(travel),
              'limits': {'rms_px': rms_limit, 'max_point_px': 3, 'min_pairs': 5,
                         'min_normal_spread_deg': 10, 'min_translation_spread_mm': 20},
              'physical_accuracy_validated': False,
              'passed': bool(len(rows) >= 5 and all(r['passed'] for r in rows) and spread >= 10 and travel >= 20)}
    write_json(bag.parent / 'verification.json', report)
    return report


def require_verified(report, calibration, camera, session_id):
    if (report.get('kind') != 'fixed_rig_verification' or report.get('passed') is not True
            or report.get('session_id') != session_id or report.get('camera') != camera_key(camera)
            or report.get('rig_sha256') != sha256(calibration)):
        raise ValueError('Missing/stale verification: verify this session, A/B order, mode and rig again')
    for p, digest in report['input_sha256'].items():
        if sha256(p) != digest:
            raise ValueError('Verification evidence changed')


def localize_fixture(pair_dir, calibration, layout_path, plumb, camera, *, session_id):
    """Known AprilTag CAD corners -> bearing origin; stereo plumb -> gravity.

    Marker corner order: canonical decoded tag TL,TR,BR,BL (NOT image-axis order).
    All lengths in mm. Unknown CAD-to-ID mapping must never be guessed.
    """
    pair_dir = Path(pair_dir)
    meta = read_json(pair_dir / 'meta.json')
    if meta['session_id'] != session_id or meta['camera'] != camera_key(camera):
        raise ValueError('Fixture images belong to a different camera session')
    if abs(meta['times'][0] - meta['times'][1]) > .01:
        raise ValueError('Fixture image pair skew exceeds 10 ms')
    rig, data = fixed_rig(calibration, camera)
    layout = read_json(layout_path)
    if layout.get('survey_confirmed') is not True:
        raise ValueError('Fill and confirm surveyed tag corners + bearing in fixture_layout.json first')
    if layout.get('dictionary') != 'DICT_APRILTAG_36h11' or layout.get('units') != 'mm':
        raise ValueError('Expected tag36h11 layout in mm')
    markers = layout['markers']
    if len({m['id'] for m in markers}) != len(markers): raise ValueError('Duplicate tag IDs in layout')
    geometry = {}
    for marker in markers:
        p = np.asarray(marker['corners_fixture_mm'], dtype=float)
        if p.shape != (4, 3) or not np.isfinite(p).all(): raise ValueError('Each tag needs four surveyed 3D corners')
        edges = np.roll(p, -1, axis=0) - p
        if (not np.allclose(np.linalg.norm(edges, axis=1), layout['tag_black_edge_mm'], atol=.1)
                or abs(np.dot(edges[0], edges[1])) > .1
                or not np.allclose(p[0]+p[2], p[1]+p[3], atol=.1)):
            raise ValueError('Tag corners do not form the declared square')
        geometry[int(marker['id'])] = p
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    maps = []
    for tag in 'AB':
        im = cv2.imread(str(pair_dir / f'{tag}.png'), cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (camera['height'], camera['width']): raise ValueError('Invalid fixture image')
        corners, ids, _ = detector.detectMarkers(im)
        if ids is not None and len(set(ids.ravel())) != len(ids): raise ValueError('Duplicate tag ID in image')
        maps.append({int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)} if ids is not None else {})
    ids = sorted(set(maps[0]) & set(maps[1]) & set(geometry))
    if len(ids) < 2: raise ValueError('Need at least two surveyed tags visible in BOTH cameras')
    T, report = fit_target(rig, np.concatenate([geometry[i] for i in ids]),
                           *[np.concatenate([m[i] for i in ids]) for m in maps])
    if report['rms_px'] >= 1 or report['max_point_px'] >= 3:
        raise ValueError(f'Fixture verification failed: {report}')
    points = []
    for key in ('bottom', 'top'):
        entry = plumb[key]
        xyz, rms, angle, reason = _point(entry['a'], entry['b'], rig, data['image_size'], 2, 1)
        if xyz is None: raise ValueError(f'Plumb {key}: {reason}')
        points.append(xyz)
    bearing = np.asarray(layout['bearing_center_fixture_mm'], float)
    hint = np.asarray(layout['x_hint_fixture'], float)
    world = build_world_frame(T[:3, :3] @ bearing + T[:3, 3], points[1]-points[0], T[:3, :3] @ hint)
    hashes = {str(Path(p).resolve()): sha256(p) for p in [layout_path, calibration,
              pair_dir/'A.png', pair_dir/'B.png', pair_dir/'meta.json']}
    world['evidence'] = {'method': 'surveyed_apriltags_plus_stereo_plumb', 'session_id': session_id,
                         'camera': camera_key(camera), 'rig_sha256': sha256(calibration),
                         'input_sha256': hashes, 'tag_ids': ids, 'fixture_fit': report,
                         'T_camera_A_from_fixture': T.tolist(), 'plumb_pixels': plumb,
                         'physical_accuracy_validated': False}
    return world


def inspect_tags(pair_dir):
    """Write ID overlays so the operator can associate physical faces with CAD."""
    pair_dir = Path(pair_dir)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    result = {}
    for tag in 'AB':
        image = cv2.imread(str(pair_dir/f'{tag}.png'))
        if image is None: raise ValueError('Save a static reference pair first')
        corners, ids, _ = detector.detectMarkers(image)
        result[tag] = [] if ids is None else [int(i) for i in ids.ravel()]
        if ids is not None: cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imwrite(str(pair_dir/f'{tag}_tag_ids.png'), image)
    return result

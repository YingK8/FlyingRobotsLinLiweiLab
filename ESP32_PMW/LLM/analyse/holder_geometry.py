"""Fixed-stereo pose of named checker centers on the custom coil holder.

Each camera keeps its own 2D/3D correspondences, including disjoint side faces.
Only the plumb endpoints are triangulated. Labels come from reviewed image ROIs,
not from the appearance of the identical checker targets.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from LLM.record.io import read_json, sha256
from LLM.analyse.pose import _point
from LLM.analyse.rig import camera_key, fixed_rig
from LLM.analyse.world import build_world_frame


# Conservative software acceptance limits, not measured physical accuracy.
LIMITS = dict(rms_px=1.0, max_point_px=3.0, min_points_per_camera=4,
              max_jacobian_condition=1e5, distinct_pose_mm=2.0,
              distinct_pose_deg=1.0)


def _vector(value, size, name):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain {size} finite coordinates') from exc
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must contain {size} finite coordinates')
    return vector


def _noncollinear(points, name):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[0] < 1e-6 or singular[1] < max(1e-6, singular[0] * 1e-3):
        raise ValueError(f'{name}: degenerate or nearly collinear points')


def _pose_starts(rig, objects, pixels):
    """Seed both camera views; IPPE adds the alternate planar solution."""
    starts = []
    for cam, obj, uv in zip(rig.cameras, objects, pixels):
        singular = np.linalg.svd(obj - obj.mean(axis=0), compute_uv=False)
        flags = [cv2.SOLVEPNP_SQPNP]
        if singular[2] < max(1e-7, singular[0] * 1e-6):
            flags.append(cv2.SOLVEPNP_IPPE)
        for flag in flags:
            try:
                solutions = cv2.solvePnPGeneric(obj, uv, cam.K, cam.dist, flags=flag)
            except cv2.error:
                continue
            if not solutions[0]:
                continue
            for rv, tv in zip(solutions[1], solutions[2]):
                # Camera-to-A transforms are fixed by the calibration.
                rotation = cam.R @ cv2.Rodrigues(rv)[0]
                translation = cam.R @ tv.ravel() + cam.position
                seed = np.r_[cv2.Rodrigues(rotation)[0].ravel(), translation]
                if np.isfinite(seed).all():
                    starts.append(seed)
    return starts


def fit_holder(rig, layout, observations):
    """Return a gated CAD-to-camera-A transform from independently named views.

    ``observations`` is ``{'A': {label: [u,v]}, 'B': {...}}``. This mathematical
    fit does not accept a physical survey or create a world frame; the public
    ``localize_holder`` entry point separately requires that evidence.
    """
    if layout.get('units') != 'mm' or layout.get('pattern_type') != 'checker_quadrants':
        raise ValueError('Expected custom checker_quadrants layout in mm')
    if len(rig.cameras) != 2 or not np.allclose(rig.a.T_world_cam, np.eye(4)):
        raise ValueError('Expected fixed stereo rig expressed in camera A')
    geometry = {label: _vector(point, 3, f'CAD point {label}')
                for label, point in layout.get('points', {}).items()}
    labels, objects, pixels, normals = [], [], [], []
    for tag in 'AB':
        view = observations.get(tag, {})
        if len(view) < LIMITS['min_points_per_camera']:
            raise ValueError(f'{tag}: need at least four named checker centers')
        names = sorted(view)
        unknown = set(names) - set(geometry)
        if unknown:
            raise ValueError(f'{tag}: unknown CAD labels {sorted(unknown)}')
        obj = np.array([geometry[name] for name in names])
        uv = np.array([_vector(view[name], 2, f'{tag}/{name}') for name in names])
        _noncollinear(obj, f'{tag} CAD')
        _noncollinear(uv, f'{tag} image')
        if len(np.unique(obj, axis=0)) != len(obj):
            raise ValueError(f'{tag}: duplicate physical CAD points')
        separations = np.linalg.norm(uv[:, None] - uv[None, :], axis=2)
        np.fill_diagonal(separations, np.inf)
        if separations.min() < 2:
            raise ValueError(f'{tag}: duplicate or unresolved checker centers')
        normal = []
        for name in names:
            surface = layout.get('surfaces', {}).get(name, {})
            value = surface.get('outward_normal', surface.get('normal'))
            if value is None:
                normal.append(None)
            else:
                n = _vector(value, 3, f'{name} surface normal')
                if np.linalg.norm(n) < 1e-9:
                    raise ValueError(f'{name}: zero surface normal')
                normal.append(n / np.linalg.norm(n))
        labels.append(names); objects.append(obj); pixels.append(uv); normals.append(normal)
    if sum(map(len, labels)) < 8 or len(set(labels[0]) | set(labels[1])) < 4:
        raise ValueError('Need eight observations and four distinct CAD landmarks')
    all_obj = np.concatenate(objects)
    scale_mm = max(float(np.linalg.norm(np.ptp(all_obj, axis=0))), 1.)

    def project(parameters):
        rotation = cv2.Rodrigues(parameters[:3])[0]
        return [((obj @ rotation.T + parameters[3:]) - cam.position) @ cam.R
                for obj, cam in zip(objects, rig.cameras)]

    def residual(parameters):
        return np.concatenate([
            (cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), cam.K, cam.dist)[0]
             .reshape(-1, 2) - uv).ravel()
            for xyz, cam, uv in zip(project(parameters), rig.cameras, pixels)])

    starts = _pose_starts(rig, objects, pixels)
    if not starts:
        raise ValueError('Holder PnP initialization failed')
    accepted, rejected = [], []
    for seed in starts:
        try:
            fit = least_squares(residual, seed, x_scale=[1, 1, 1, scale_mm, scale_mm, scale_mm],
                                max_nfev=250, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            local_points = project(fit.x)
            if not fit.success or not np.isfinite(fit.x).all() or any(np.any(x[:, 2] <= 0) for x in local_points):
                rejected.append('failed solve or negative depth'); continue
            rotation = cv2.Rodrigues(fit.x[:3])[0]
            if any(n is not None and np.dot(cam.R.T @ rotation @ n, -xyz) <= 0
                   for cam, points, ns in zip(rig.cameras, local_points, normals)
                   for xyz, n in zip(points, ns)):
                rejected.append('back-facing named surface'); continue
            errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
            rms = float(np.sqrt(np.mean(errors ** 2)))
            if rms >= LIMITS['rms_px'] or errors.max() >= LIMITS['max_point_px']:
                rejected.append('reprojection residual'); continue
            # Scale translation columns so mm versus radians do not set conditioning.
            singular = np.linalg.svd(fit.jac * [1, 1, 1, scale_mm, scale_mm, scale_mm], compute_uv=False)
            condition = float(singular[0] / max(singular[-1], 1e-30))
            if condition >= LIMITS['max_jacobian_condition']:
                rejected.append('weak pose observability'); continue
            T = np.eye(4); T[:3, :3] = rotation; T[:3, 3] = fit.x[3:]
            accepted.append((rms, T, errors, condition))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            rejected.append('numerical solve failure')
    if not accepted:
        raise ValueError('Holder verification failed: ' + ', '.join(sorted(set(rejected))))
    accepted.sort(key=lambda item: item[0])
    rms, best, errors, condition = accepted[0]
    for _, alternative, _, _ in accepted[1:]:
        angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(best[:3, :3].T @ alternative[:3, :3])[0])))
        displacement = np.linalg.norm(all_obj @ best[:3, :3].T + best[:3, 3]
                                      - all_obj @ alternative[:3, :3].T - alternative[:3, 3], axis=1).max()
        if angle > LIMITS['distinct_pose_deg'] or displacement > LIMITS['distinct_pose_mm']:
            raise ValueError('Ambiguous holder pose: materially different solutions pass the image gates')
    split = len(labels[0])
    per_camera = {}
    for tag, names, err in zip('AB', labels, (errors[:split], errors[split:])):
        per_camera[tag] = dict(labels=names, n_points=len(names), rms_px=float(np.sqrt(np.mean(err**2))),
                               max_point_px=float(err.max()), point_error_px=dict(zip(names, map(float, err))))
    return best, dict(rms_px=rms, max_point_px=float(errors.max()), n_observations=len(errors),
                     n_unique_landmarks=len(set(labels[0]) | set(labels[1])), cameras=per_camera,
                     shared_labels=sorted(set(labels[0]) & set(labels[1])),
                     jacobian_condition=condition, initializations=len(starts),
                     accepted_initializations=len(accepted), limits=dict(LIMITS),
                     correspondence_method='operator_named_ROIs', stereo_extrinsics_refitted=False,
                     physical_accuracy_validated=False)


def localize_holder(pair_dir, calibration, layout_path, regions_path, plumb, camera, *, session_id):
    """Named checker centers -> surveyed bearing; stereo plumb -> gravity."""
    from LLM.analyse.holder import inspect_holder

    pair_dir = Path(pair_dir)
    meta = read_json(pair_dir / 'meta.json')
    if meta.get('session_id') != session_id or meta.get('camera') != camera_key(camera):
        raise ValueError('Holder images belong to a different camera session')
    times = _vector(meta.get('times'), 2, 'Reference timestamps')
    if abs(times[0] - times[1]) > .01:
        raise ValueError('Holder image pair skew exceeds 10 ms')
    rig, data = fixed_rig(calibration, camera)
    layout = read_json(layout_path)
    if layout.get('survey_confirmed') is not True:
        raise ValueError('Confirm surveyed holder landmarks and bearing in the layout before world localization')
    bearing = _vector(layout.get('bearing_center_fixture_mm'), 3, 'Surveyed bearing center')
    hint = _vector(layout.get('x_hint_fixture'), 3, 'Fixture x direction')
    for tag in 'AB':
        image = cv2.imread(str(pair_dir / f'{tag}.png'), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (camera['height'], camera['width']):
            raise ValueError(f'{tag}: holder image missing or dimensions differ from calibration')
    inputs = [layout_path, regions_path, calibration, pair_dir / 'A.png', pair_dir / 'B.png', pair_dir / 'meta.json']
    hashes = {str(Path(p).resolve()): sha256(p) for p in inputs}
    inspected = inspect_holder(pair_dir, regions_path=regions_path, layout_path=layout_path)
    observations = {}
    for tag in 'AB':
        observations[tag] = {}
        landmarks = inspected['cameras'][tag]['landmarks']
        for label, entry in landmarks.items():
            if entry.get('status') != 'detected' or entry.get('center_px') is None:
                raise ValueError(f'{tag}/{label}: checker detection is {entry.get("status", "missing")}')
            center = _vector(entry['center_px'], 2, f'{tag}/{label} center')
            if np.any(center < 0) or np.any(center >= data['image_size']):
                raise ValueError(f'{tag}/{label}: checker center outside image')
            observations[tag][label] = center
    T, report = fit_holder(rig, layout, observations)
    points, diagnostics, plumb_pixels = [], {}, {}
    for key in ('bottom', 'top'):
        entry = plumb.get(key, {})
        uv_a = _vector(entry.get('a'), 2, f'Plumb {key} A')
        uv_b = _vector(entry.get('b'), 2, f'Plumb {key} B')
        xyz, rms, angle, reason = _point(uv_a, uv_b, rig, data['image_size'], 2, 1)
        if xyz is None:
            raise ValueError(f'Plumb {key}: {reason}')
        points.append(xyz)
        plumb_pixels[key] = dict(a=uv_a.tolist(), b=uv_b.tolist())
        diagnostics[key] = dict(camera_A_mm=xyz.tolist(), reprojection_rms_px=float(rms), ray_angle_deg=float(angle))
    if np.linalg.norm(points[1] - points[0]) <= 1e-6:
        raise ValueError('Plumb top and bottom must be distinct physical points')
    world = build_world_frame(T[:3, :3] @ bearing + T[:3, 3], points[1] - points[0], T[:3, :3] @ hint)
    for path, digest in inspected.get('input_sha256', {}).items():
        if path in hashes and hashes[path] != digest:
            raise ValueError('Holder reference changed during detection')
        hashes[path] = digest
    hashes[str(Path(__file__).resolve())] = sha256(__file__)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Holder reference changed during localization')
    world['evidence'] = dict(method='custom_holder_checker_plus_stereo_plumb', session_id=session_id,
                             camera=camera_key(camera), rig_sha256=sha256(calibration), input_sha256=hashes,
                             landmark_labels={tag: list(observations[tag]) for tag in 'AB'}, fixture_fit=report,
                             T_camera_A_from_fixture=T.tolist(), plumb_pixels=plumb_pixels,
                             plumb_triangulation=diagnostics, physical_accuracy_validated=False,
                             operator_assumptions='Reviewed full face/height labels, surveyed printed landmarks and bearing; stationary plumb with correct top/bottom.')
    return world


if __name__ == '__main__':
    from LLM.tests.test_holder_geometry import run
    run()

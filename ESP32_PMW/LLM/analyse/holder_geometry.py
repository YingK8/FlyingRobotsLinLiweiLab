"""Fixed-stereo pose of named checker centers on the custom coil holder.

Each camera keeps its own 2D/3D correspondences, including disjoint side faces.
Only the plumb endpoints are triangulated. Labels come from reviewed image ROIs, or
from `match_holder`, which names unlabelled candidates by their 3D fit to the CAD.
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


def _project(cam, xyz):
    local = (np.asarray(xyz, float) - cam.position) @ cam.R
    return cv2.projectPoints(local, np.zeros(3), np.zeros(3), cam.K, cam.dist)[0].reshape(-1, 2)


def _dot_evidence(rig, images, dots, T):
    """Mean (dot - surrounding ring) brightness at the projected white ID dots."""
    world = dots @ T[:3, :3].T + T[:3, 3]
    out = []
    for cam, img in zip(rig.cameras, images):
        h, w = img.shape[:2]
        for u, v in np.rint(_project(cam, world)).astype(int):
            if 4 <= u < w - 4 and 4 <= v < h - 4:
                patch = img[v - 4:v + 5, u - 4:u + 5].astype(float)
                out.append(patch[3:6, 3:6].mean() - np.median(np.r_[patch[0], patch[-1], patch[:, 0], patch[:, -1]]))
    return float(np.mean(out)) if out else -np.inf


def _label(rig, layout, cand, T, tol_px):
    """{'A': {label: uv}, 'B': ...}: each CAD point posed by T, on a face turned toward
    the camera (as `fit_holder` requires), names the candidate within `tol_px` of it."""
    labels, surfaces = list(layout['points']), layout.get('surfaces', {})
    world = np.array([layout['points'][k] for k in labels], float) @ T[:3, :3].T + T[:3, 3]
    observations = {'A': {}, 'B': {}}
    for tag, cam, c in zip('AB', rig.cameras, cand):
        if not len(c):
            continue
        for label, p, xyz in zip(labels, _project(cam, world), world):
            n = surfaces.get(label, {})
            n = n.get('normal', n.get('outward_normal'))
            if n is not None and np.dot(T[:3, :3] @ n, cam.position - xyz) <= 0:
                continue
            d = np.linalg.norm(c - p, axis=1)
            if d.min() < tol_px:
                observations[tag][label] = c[int(np.argmin(d))]
    return observations


def match_holder(rig, cand_a, cand_b, layout, up, *, images=None, epipolar_px=2.0,
                 reproject_px=3.0, coarse_px=8.0, yaw_step_deg=3.0, min_named=6):
    """Name unlabelled checker centres by fitting the CAD points, z held on `up`.

    The detector finds the targets but cannot tell them apart (`identity_method`).
    Candidates paired across the views (epipolar, mutual best) and triangulated are
    the anchors: each anchor x CAD point x yaw (CAD +Z on the plumb `up`, so 4 DOF)
    poses the holder, and the pose is scored by projecting ALL CAD points into BOTH
    views and counting single-view candidates within `coarse_px`. One anchor is
    enough. Scoring in 3D needed >= 3 triangulated points, and on 2026-11-05 camera B
    found only 5 candidates against A's 11 (its auto exposure brightened under a new
    black cloth), 2 triangulated, and the fit refused. `coarse_px` covers the yaw step
    at ~70 mm from the anchor; a level holder to ~2 deg is assumed (a guess).

    The four posts sit on a 96 mm square, so the targets are 4-fold symmetric about the
    boss axis and yaw ties at 90 deg steps. Each tied pose is refined by `fit_holder`
    and the white ID dots (1-4 per post, `layout['id_dots']`) break the tie: with
    ``images`` the pose whose projected dots are brightest against their ring wins.
    Without images, or if the dots do not separate them, ``info['ambiguous']`` is True
    and the yaw is not to be trusted (z and the boss-axis origin are unaffected).
    Returns ``({'A': {label: uv}, 'B': ...}, info)``.
    """
    from scipy.spatial import cKDTree
    cand = [np.asarray(c, float).reshape(-1, 2) for c in (cand_a, cand_b)]
    empty = {'A': {}, 'B': {}}
    if min(len(c) for c in cand) < 1:
        return empty, dict(n3d=0, ambiguous=True)
    a, b = rig.cameras
    norm = [cv2.undistortPoints(c.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
            for c, cam in zip(cand, rig.cameras)]
    R = a.R.T @ b.R                     # B -> A rotation, camera A at the origin
    t = a.R.T @ (b.position - a.position)
    E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
    xa, xb = (np.c_[n, np.ones(len(n))] for n in norm)
    lines = xb @ E.T                    # epipolar lines in A, one per B candidate
    dist = np.abs(xa @ lines.T) / np.linalg.norm(lines[:, :2], axis=1) * a.K[0, 0]
    size = rig.meta.get('image_size', [1e9, 1e9])
    P = []
    for i in range(len(cand[0])):
        j = int(np.argmin(dist[i]))
        if dist[i, j] < epipolar_px and int(np.argmin(dist[:, j])) == i:
            xyz = _point(cand[0][i], cand[1][j], rig, size, reproject_px, 1)[0]
            if xyz is not None:
                P.append(xyz)
    if not P:
        return empty, dict(n3d=0, ambiguous=True)
    P = np.array(P)
    labels = list(layout['points'])
    Q = np.array([layout['points'][k] for k in labels], float)
    surf = layout.get('surfaces', {})
    N = np.array([(surf.get(k, {}).get('normal') or surf.get(k, {}).get('outward_normal') or [np.nan] * 3)
                  for k in labels], float)
    trees = [cKDTree(c) for c in cand]
    z = np.asarray(up, float) / np.linalg.norm(up)
    x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < .9 else [0, 1.0, 0]); x /= np.linalg.norm(x)
    R0 = np.c_[x, np.cross(z, x), z]   # CAD z -> up
    hypotheses = []
    for yaw in np.radians(np.arange(0, 360, yaw_step_deg)):
        c, s_ = np.cos(yaw), np.sin(yaw)
        Rq = R0 @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        Qr, Nr = Q @ Rq.T, N @ Rq.T
        shifts = (P[:, None, :] - Qr[None, :, :]).reshape(-1, 3)        # every anchor x CAD point
        X = Qr[None] + shifts[:, None]                                    # (S, points, 3)
        score, cost = np.zeros(len(shifts)), np.zeros(len(shifts))
        for cam, tree in zip(rig.cameras, trees):
            local = (X - cam.position) @ cam.R
            facing = np.einsum('spk,pk->sp', cam.position - X, Nr)
            ok = (local[..., 2] > 0) & ~(facing <= 0)                     # NaN normal: no face test
            uv = cv2.projectPoints(local.reshape(-1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)[0]
            d, _ = tree.query(uv.reshape(-1, 2), distance_upper_bound=coarse_px)
            hit = ok & np.isfinite(d.reshape(ok.shape))
            score += hit.sum(1)
            cost += np.where(hit, d.reshape(ok.shape), 0).sum(1)
        k = int(np.lexsort((cost, -score))[0])
        hypotheses.append((int(score[k]), float(cost[k]), np.degrees(yaw), np.r_[np.c_[Rq, shifts[k]], [[0, 0, 0, 1]]]))
    top = max(h[0] for h in hypotheses)
    if top < min_named:
        return empty, dict(n3d=len(P), score=top, ambiguous=True)
    tied = []                           # best of each distinct yaw that reaches the top score
    for h in sorted((h for h in hypotheses if h[0] == top), key=lambda h: h[1]):
        if all(abs((h[2] - g[0] + 180) % 360 - 180) > 10 for g in tied):   # tied: (yaw, T, obs)
            obs = _label(rig, layout, cand, h[3], coarse_px)
            try:                        # refine on its own labels before reading the dots
                T = fit_holder(rig, layout, obs)[0]
                obs = _label(rig, layout, cand, T, reproject_px)
            except ValueError:
                T = h[3]
            tied.append((h[2], T, obs))
    dots = np.array([d['center_mm'] for d in layout.get('id_dots', [])], float).reshape(-1, 3)
    evidence = ([_dot_evidence(rig, images, dots, T) for _, T, _ in tied]
                if images is not None and len(dots) and len(tied) > 1 else [0.0])
    order = np.argsort(evidence)[::-1]
    # guess: a 10-grey-level lead is a white dot against a dark face, not noise
    ambiguous = len(tied) > 1 and (len(evidence) < 2 or evidence[order[0]] - evidence[order[1]] < 10)
    return tied[int(order[0])][2], dict(n3d=len(P), score=top, tied_yaws_deg=[round(y) for y, _, _ in tied],
                                        dot_evidence=[round(e, 1) for e in evidence], ambiguous=ambiguous)


def thin_lines(gray, cam, n=8, max_tilt_deg=40.0, min_px=120, ridge_grey=25):
    """The `n` longest thin bright straight lines within `max_tilt_deg` of image-vertical.

    A 9 px white top-hat keeps ridges a few px wide (the rope) and drops broad bright
    areas (windows, walls), which is what defeated `scene.segment_scene`'s whole-frame
    Otsu on the 2026-11-05 bench: it picked a checker column in A and a cable in B.
    40 deg covers a rolled camera -- the rope leans ~21 deg in B's image there -- and
    is a guess, as are `min_px` and `ridge_grey`. Each line is returned as image
    endpoints plus the undistorted ``(centroid, unit_dir)`` `disc_pose.mast_plane` takes.
    """
    ridge = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    segments = cv2.HoughLinesP(((ridge > ridge_grey) * 255).astype(np.uint8), 1, np.pi / 360, 80,
                               minLineLength=min_px, maxLineGap=8)
    out = []
    for x1, y1, x2, y2 in (segments.reshape(-1, 4) if segments is not None else []):
        d = np.array([x2 - x1, y2 - y1], float)
        if np.degrees(np.arccos(abs(d[1]) / np.linalg.norm(d))) > max_tilt_deg:
            continue
        p = cv2.undistortPoints(np.array([[x1, y1], [x2, y2]], float).reshape(-1, 1, 2),
                                cam.K, cam.dist, P=cam.K).reshape(-1, 2)
        out.append(dict(length_px=float(np.linalg.norm(d)), ends=[[int(x1), int(y1)], [int(x2), int(y2)]],
                        line=(p.mean(0), (p[1] - p[0]) / np.linalg.norm(p[1] - p[0]))))
    return sorted(out, key=lambda o: -o['length_px'])[:n]


def find_rope_and_holder(rig, images, cands, layout):
    """The rope and a level holder confirm each other: every A x B thin-line pair
    proposes an `up`, and the one along which `match_holder` names the most targets wins.

    Any truly vertical line gives the same `up` (a holder edge would too), so the winner
    is the rope only in the sense that it is the longest vertical pair; it is drawn for
    the operator to check. Returns ``(up, (line_a, line_b), observations, info)``.
    """
    from controller.pose.disc_pose import mast_direction
    lines = [thin_lines(img, cam) for img, cam in zip(images, rig.cameras)]
    image_up = rig.a.R @ [0.0, -1.0, 0.0]      # sign only: camera A is roughly upright
    best = None
    for la in lines[0]:
        for lb in lines[1]:
            up = mast_direction([la['line'], lb['line']], rig.cameras, up=image_up)
            if up is None:
                continue
            obs, info = match_holder(rig, *cands, layout, up, images=images)
            key = (info.get('score', 0), sum(map(len, obs.values())))
            if best is None or key > best[0]:
                best = (key, up, (la, lb), obs, info)
    if best is None:
        return None, None, {'A': {}, 'B': {}}, dict(lines=[len(l) for l in lines], ambiguous=True)
    return best[1], best[2], best[3], best[4] | dict(lines=[len(l) for l in lines])


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

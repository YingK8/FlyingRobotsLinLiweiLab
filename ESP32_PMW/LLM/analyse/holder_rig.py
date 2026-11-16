"""Camera-to-camera pose from the coil holder: each camera's pose relative to the holder's
CAD marks, then ``T_ba = T_B @ inv(T_A)`` (`controller/calib/theory.md` 17a).

The board cannot be shown to both cameras on this rig (axes ~99.6 deg apart, operator
2026-11-15), so only the lenses come from the board, one camera at a time. The holder is in
both views by construction. Unlike `holder_geometry.match_holder`, nothing here needs an
existing stereo pose: each camera names its own marks.

    name_in_camera  one camera: which candidate is which CAD mark, and its pose
    resolve         the 4-fold symmetry left after the ID dots, by the LED (or refuse)
    solve_extrinsic joint fit of CAD->A and A->B to both cameras' named marks
    holder_rig      the StereoRig (A = identity), meta marked unverified
    led_epipolar_px the independent check: the LED is not in the fit
"""
import itertools
from datetime import datetime

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from controller.calib.rig import Camera, StereoRig
from LLM.analyse.holder_geometry import LIMITS

# Symmetric labellings the ID dots leave tied differ by 90 deg of yaw in ONE camera. The
# LED -- one point both cameras see, off the holder axis -- lands tens of px off its
# epipolar line under a wrong one. Thresholds are guesses: accept <= LED_ACCEPT_PX, and only
# if the runner-up is LED_RATIO x worse and at least LED_GAP_PX.
LED_ACCEPT_PX, LED_RATIO, LED_GAP_PX = 5.0, 3.0, 15.0
# NEGATIVE RESULTS (synthetic, 2026-11-15), so without the LED `resolve` refuses and the
# operator names marks by hand:
# - the expected axis separation (~99.6 deg, capture.PAIR_MARGIN_DEG comment): a 90 deg
#   relabel moves the separation only ~16-36 deg when both cameras look down, and with a
#   wrong expectation it confidently picked a wrong labelling;
# - the rope: each view's rope plane has a horizontal normal and the symmetry is a turn
#   about the vertical, so every labelling reconstructs a vertical rope.
COARSE_PX = 8.0        # a hypothesis' projected mark this close to a candidate counts (as match_holder)
FINE_PX = 3.0          # relabel radius after refinement (= LIMITS max_point_px)
MAX_CANDIDATES = 15    # best-quality candidates tried as P3P anchors; more only costs time
MAX_TRIPLE_PX = 500.0  # three marks of one post never span more than this in the image (a guess)
DOT_LEAD = 10.0        # grey levels: a white dot against a dark face, not noise (as match_holder)
DISTINCT_DEG = 10.0    # poses closer than this are the same hypothesis


def _points(layout):
    labels = list(layout["points"])
    P = np.array([layout["points"][k] for k in labels], float)
    surf = layout.get("surfaces", {})
    N = np.array([(surf.get(k, {}).get("outward_normal") or surf.get(k, {}).get("normal") or [np.nan] * 3)
                  for k in labels], float)
    return labels, P, N


def _post_triplets(labels):
    """Index triplets within one post (top + its four side marks): the P3P anchors."""
    posts = {}
    for i, k in enumerate(labels):
        posts.setdefault(k.split("_")[0], []).append(i)
    return [t for idx in posts.values() for t in itertools.combinations(idx, 3)]


def _project(K, dist, R, t, P):
    return cv2.projectPoints(P @ R.T + t, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)


def _seen(R, t, P, N):
    """In front of the camera and on a face turned toward it (NaN normal: no face test)."""
    X = P @ R.T + t
    facing = np.einsum("pk,pk->p", -X, N @ R.T)
    return (X[:, 2] > 0) & ~(facing <= 0)


def _label(K, dist, R, t, P, N, labels, cands, tol):
    """{label: uv} -- each visible mark names the candidate within `tol` px, one-to-one."""
    uv, ok = _project(K, dist, R, t, P), _seen(R, t, P, N)
    out, used = {}, set()
    for i in np.argsort([np.linalg.norm(cands - p, axis=1).min() for p in uv]):
        if not ok[i]:
            continue
        d = np.linalg.norm(cands - uv[i], axis=1)
        j = int(np.argmin(d))
        if d[j] < tol and j not in used:
            out[labels[i]], used = cands[j], used | {j}
    return out


def _refine(K, dist, R, t, P, N, labels, cands):
    """Label coarsely, PnP on those, relabel finely, PnP again; (R, t, obs, rms) or None."""
    index = {k: i for i, k in enumerate(labels)}
    for tol in (COARSE_PX, FINE_PX):
        obs = _label(K, dist, R, t, P, N, labels, cands, tol)
        if len(obs) < LIMITS["min_points_per_camera"]:
            return None
        obj = P[[index[k] for k in obs]]
        uv = np.array(list(obs.values()), float)
        ok, rv, tv = cv2.solvePnP(obj, uv, K, dist, cv2.Rodrigues(R)[0], t.reshape(3, 1), True,
                                  cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, t = cv2.Rodrigues(rv)[0], tv.ravel()
    err = np.linalg.norm(_project(K, dist, R, t, obj) - uv, axis=1)
    return R, t, obs, float(np.sqrt(np.mean(err ** 2)))


def _dot_evidence(K, dist, R, t, dots, image):
    """Mean (dot - surrounding ring) brightness at the projected ID dots (one camera)."""
    if image is None or not len(dots):
        return 0.0
    h, w = image.shape[:2]
    out = []
    for (u, v), z in zip(np.rint(_project(K, dist, R, t, dots)).astype(int), (dots @ R.T + t)[:, 2]):
        if z > 0 and 4 <= u < w - 4 and 4 <= v < h - 4:
            patch = image[v - 4:v + 5, u - 4:u + 5].astype(float)
            out.append(patch[3:6, 3:6].mean() - np.median(np.r_[patch[0], patch[-1], patch[:, 0], patch[:, -1]]))
    return float(np.mean(out)) if out else -np.inf


def name_in_camera(K, dist, cands, layout, image=None, quality=None):
    """One camera alone: name the holder marks among unlabelled ``cands`` (N x 2 px).

    Hypotheses are P3P poses from three candidates taken as three marks of one post (top,
    X_L/H, Y_L/H), every post and ordering; each is scored by how many visible CAD marks
    land within COARSE_PX of a candidate, then refined by PnP on what it named. The holder
    is 4-fold symmetric, so every pose has three twins that score identically; the white
    ID dots (1-4 per post) in ``image`` rank them. Returns ``(obs, (R, t), info)`` where
    ``info["poses"]`` lists every tied pose, best dot evidence first, and
    ``info["ambiguous"]`` says the dots did not separate them. ``(None, None, info)`` when
    nothing reaches LIMITS["min_points_per_camera"] marks.
    """
    K, dist = np.asarray(K, float), np.asarray(dist, float).ravel()
    labels, P, N = _points(layout)
    cands = np.asarray(cands, float).reshape(-1, 2)
    order = np.argsort(-np.asarray(quality)) if quality is not None else np.arange(len(cands))
    anchors = cands[order[:MAX_CANDIDATES]]
    tree = cKDTree(cands) if len(cands) else None
    best, hyps = 0, []
    if len(anchors) >= 3:
        for ia, ib, ic in itertools.permutations(range(len(anchors)), 3):
            img = anchors[[ia, ib, ic]]
            if max(np.linalg.norm(img[0] - img[1]), np.linalg.norm(img[0] - img[2]),
                   np.linalg.norm(img[1] - img[2])) > MAX_TRIPLE_PX:
                continue
            for tri in _post_triplets(labels):
                try:
                    n, rvs, tvs = cv2.solveP3P(P[list(tri)], img, K, dist, flags=cv2.SOLVEPNP_P3P)
                except cv2.error:
                    continue
                for rv, tv in zip(rvs[:n], tvs[:n]):
                    R, t = cv2.Rodrigues(rv)[0], tv.ravel()
                    ok = _seen(R, t, P, N)
                    if not ok.any():
                        continue
                    d, _ = tree.query(_project(K, dist, R, t, P[ok]), distance_upper_bound=COARSE_PX)
                    score = int(np.isfinite(d).sum())
                    if score >= max(best - 1, LIMITS["min_points_per_camera"]):
                        best = max(best, score)
                        hyps.append((score, float(d[np.isfinite(d)].sum()), R, t))
    info = dict(n_candidates=len(cands), best_score=best)
    poses = []
    for score, _, R, t in sorted(hyps, key=lambda h: (-h[0], h[1])):
        if score < best - 1:
            continue
        if any(np.degrees(np.linalg.norm(cv2.Rodrigues(R @ q[0].T)[0])) < DISTINCT_DEG for q in poses):
            continue
        got = _refine(K, dist, R, t, P, N, labels, cands)
        if got is None or got[3] > LIMITS["rms_px"]:
            continue
        if any(np.degrees(np.linalg.norm(cv2.Rodrigues(got[0] @ q[0].T)[0])) < DISTINCT_DEG for q in poses):
            continue
        poses.append(got)
        if len(poses) >= 8:
            break
    if not poses:
        return None, None, dict(info, ambiguous=True, poses=[])
    n_top = max(len(p[2]) for p in poses)
    poses = [p for p in poses if len(p[2]) == n_top]          # the symmetric twins name equally many
    dots = np.array([d["center_mm"] for d in layout.get("id_dots", [])], float).reshape(-1, 3)
    evidence = [_dot_evidence(K, dist, R, t, dots, image) for R, t, _, _ in poses]
    rank = np.argsort(evidence)[::-1]
    poses = [poses[i] for i in rank]
    evidence = [evidence[i] for i in rank]
    ambiguous = len(poses) > 1 and (image is None or evidence[0] - evidence[1] < DOT_LEAD)
    R, t, obs, rms = poses[0]
    info.update(n_named=len(obs), rms_px=rms, ambiguous=ambiguous, poses=[(p[0], p[1], p[2]) for p in poses],
                dot_evidence=[round(e, 1) for e in evidence])
    return obs, (R, t), info


def separation_deg(T_ba):
    """Angle between the two optical axes, from x_B = T_ba x_A."""
    return float(np.degrees(np.arccos(np.clip(np.asarray(T_ba)[2, 2], -1.0, 1.0))))


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def _epipolar_px(K_a, dist_a, K_b, dist_b, T_ba, uv_a, uv_b):
    """Distance (px, camera A) of uv_a from the epipolar line of uv_b; x_B = T_ba x_A."""
    xa, xb = (cv2.undistortPoints(np.asarray(uv, float).reshape(1, 1, 2), np.asarray(K, float),
                                  np.asarray(d, float).ravel()).reshape(2)
              for uv, K, d in ((uv_a, K_a, dist_a), (uv_b, K_b, dist_b)))
    R, t = np.asarray(T_ba)[:3, :3], np.asarray(T_ba)[:3, 3]
    E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R   # x_b^T E x_a = 0
    line = E.T @ np.r_[xb, 1.0]
    return float(abs(np.r_[xa, 1.0] @ line) / np.linalg.norm(line[:2]) * np.asarray(K_a)[0, 0])


def resolve(info_a, info_b, lenses=None, led=None):
    """Pick one tied pose per camera. Only the RELATIVE choice moves T_ba (renaming both
    cameras by the same symmetry leaves it unchanged). With ``led`` = (uv_a, uv_b) and
    ``lenses`` = (K_a, dist_a, K_b, dist_b) the choice whose T_ba puts the LED on its
    epipolar line wins; without the LED it refuses (see the negative results above).
    Returns ``(obs_a, obs_b, how)``, or ``(None, None, why)``."""
    if not (info_a["ambiguous"] or info_b["ambiguous"]):
        return info_a["poses"][0][2], info_b["poses"][0][2], "ID dots"
    if led is None or lenses is None:
        return None, None, "ID dots did not decide and no LED was found: name marks by hand"
    combos = []
    for i, pa in enumerate(info_a["poses"]):
        for j, pb in enumerate(info_b["poses"]):
            T_ba = _T(*pb[:2]) @ np.linalg.inv(_T(*pa[:2]))
            if any(np.allclose(T_ba, c[3], atol=1e-3) for c in combos):
                continue                        # the same relative choice: the same T_ba
            combos.append((_epipolar_px(*lenses, T_ba, *led), i, j, T_ba))
    combos.sort(key=lambda c: c[0])
    best, i, j = combos[0][:3]
    second = combos[1][0] if len(combos) > 1 else np.inf
    if best <= LED_ACCEPT_PX and second >= max(LED_RATIO * best, LED_GAP_PX):
        return info_a["poses"][i][2], info_b["poses"][j][2], f"LED epipolar {best:.1f} px (next {second:.0f} px)"
    return None, None, f"LED epipolar {[round(c[0], 1) for c in combos[:3]]} px: no clear winner"


def solve_extrinsic(K_a, dist_a, K_b, dist_b, obs, layout):
    """Joint fit of CAD->A and A->B to both cameras' named marks (12 parameters).

    ``obs`` = {'A': {label: uv}, 'B': {...}}; the views need share no label. Seeds are
    each camera's own PnP. Accepted under holder_geometry.LIMITS (rms, worst point,
    marks per camera, Jacobian condition); raises ValueError with the reason otherwise.
    Returns ``(T_ba, T_a_from_cad, report)`` with x_B = T_ba x_A."""
    labels, P, N = _points(layout)
    index = {k: i for i, k in enumerate(labels)}
    Ks = [np.asarray(K_a, float), np.asarray(K_b, float)]
    ds = [np.asarray(dist_a, float).ravel(), np.asarray(dist_b, float).ravel()]
    obj, uv, seeds = [], [], []
    for tag, K, d in zip("AB", Ks, ds):
        named = obs.get(tag, {})
        if len(named) < LIMITS["min_points_per_camera"]:
            raise ValueError(f"camera {tag}: {len(named)} named marks, need {LIMITS['min_points_per_camera']}")
        o = P[[index[k] for k in named]]
        u = np.array(list(named.values()), float)
        if np.linalg.matrix_rank(o - o.mean(0), tol=1e-6) < 2:
            raise ValueError(f"camera {tag}: named marks are collinear")
        ok, rv, tv = cv2.solvePnP(o, u, K, d, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise ValueError(f"camera {tag}: PnP failed")
        obj.append(o), uv.append(u), seeds.append(_T(cv2.Rodrigues(rv)[0], tv))
    T_ba0 = seeds[1] @ np.linalg.inv(seeds[0])

    def unpack(x):
        return _T(cv2.Rodrigues(x[:3])[0], x[3:6]), _T(cv2.Rodrigues(x[6:9])[0], x[9:12])

    def residual(x):
        T_a, T_ba = unpack(x)
        out = []
        for k, T in enumerate((T_a, T_ba @ T_a)):
            out.append((_project(Ks[k], ds[k], T[:3, :3], T[:3, 3], obj[k]) - uv[k]).ravel())
        return np.concatenate(out)

    x0 = np.r_[cv2.Rodrigues(seeds[0][:3, :3])[0].ravel(), seeds[0][:3, 3],
               cv2.Rodrigues(T_ba0[:3, :3])[0].ravel(), T_ba0[:3, 3]]
    s = float(np.ptp(P, axis=0).max())
    sol = least_squares(residual, x0, x_scale=np.r_[[1] * 3, [s] * 3, [1] * 3, [s] * 3],
                        xtol=1e-12, ftol=1e-12, max_nfev=500)
    T_a, T_ba = unpack(sol.x)
    report, rms_all, worst = dict(cameras={}), [], 0.0
    for k, (tag, T) in enumerate(zip("AB", (T_a, T_ba @ T_a))):
        X = obj[k] @ T[:3, :3].T + T[:3, 3]
        if (X[:, 2] <= 0).any():
            raise ValueError(f"camera {tag}: a named mark is behind the camera")
        idx = [index[l] for l in obs[tag]]
        if not _seen(T[:3, :3], T[:3, 3], P[idx], N[idx]).all():
            raise ValueError(f"camera {tag}: a named mark is on a face turned away")
        err = np.linalg.norm(_project(Ks[k], ds[k], T[:3, :3], T[:3, 3], obj[k]) - uv[k], axis=1)
        rms_all.append(err)
        worst = max(worst, float(err.max()))
        report["cameras"][tag] = dict(labels=list(obs[tag]), n_points=len(err),
                                      rms_px=float(np.sqrt(np.mean(err ** 2))), max_point_px=float(err.max()))
    J = sol.jac * np.r_[[1] * 3, [s] * 3, [1] * 3, [s] * 3]
    cond = float(np.linalg.cond(J))
    rms = float(np.sqrt(np.mean(np.concatenate(rms_all) ** 2)))
    report.update(rms_px=rms, max_point_px=worst, jacobian_condition=cond,
                  axis_separation_deg=separation_deg(T_ba), baseline_mm=float(np.linalg.norm(
                      np.linalg.inv(T_ba)[:3, 3])), limits=LIMITS, physical_accuracy_validated=False)
    if rms > LIMITS["rms_px"] or worst > LIMITS["max_point_px"]:
        raise ValueError(f"holder fit rms {rms:.2f} px, worst {worst:.2f} px (limits {LIMITS['rms_px']}, "
                         f"{LIMITS['max_point_px']})")
    if cond > LIMITS["max_jacobian_condition"]:
        raise ValueError(f"holder fit ill-conditioned (Jacobian condition {cond:.3g})")
    return T_ba, T_a, report


def holder_rig(K_a, dist_a, K_b, dist_b, T_ba, report, *, image_size, rotate180, camera_indices,
               lenses, board=None):
    """The StereoRig of a holder fit: A at the origin, B at inv(T_ba); never verified."""
    meta = dict(image_size=list(image_size), rotate180=rotate180, camera_indices=list(camera_indices),
                rms_px=report["rms_px"], method="holder_cad", verified=False, holder_fit=report,
                lenses_source=str(lenses), world_frame="camera_A",
                created=datetime.now().isoformat(timespec="seconds"), **(board or {}))
    return StereoRig((Camera(np.asarray(K_a, float), np.asarray(dist_a, float).ravel(), np.eye(4), "A"),
                      Camera(np.asarray(K_b, float), np.asarray(dist_b, float).ravel(), np.linalg.inv(T_ba), "B")),
                     meta=meta)


def led_epipolar_px(rig, uv_a, uv_b):
    """Distance (px, camera A) of the LED in A from the epipolar line of the LED in B.
    The LED is not in the holder fit: an independent check, unless `resolve` already
    used it to pick between symmetric labellings (then it only chose, it did not fit)."""
    a, b = rig.cameras
    return _epipolar_px(a.K, a.dist, b.K, b.dist, np.linalg.inv(b.T_world_cam) @ a.T_world_cam, uv_a, uv_b)


def _self_check():
    from LLM.tests.test_holder_geometry import scene
    from LLM.record.io import read_json
    from LLM.analyse.holder_geometry import _project as cam_project
    rig, _, _, truth = scene()
    layout = read_json(__import__("pathlib").Path(__file__).with_name("holder_layout.json"))
    labels, P, N = _points(layout)
    dots = np.array([d["center_mm"] for d in layout["id_dots"]], float)
    rng = np.random.default_rng(0)
    T_true = np.linalg.inv(rig.b.T_world_cam)          # x_B = T_ba x_A
    seen, images, cands = {}, {}, {}
    for tag, cam in zip("AB", rig.cameras):
        T = (np.linalg.inv(cam.T_world_cam)) @ truth   # CAD -> this camera
        ok = _seen(T[:3, :3], T[:3, 3], P, N)
        uv = cam_project(cam, P[ok] @ truth[:3, :3].T + truth[:3, 3])
        inside = (uv[:, 0] > 0) & (uv[:, 0] < 1280) & (uv[:, 1] > 0) & (uv[:, 1] < 800)
        seen[tag] = dict(zip([l for l, o in zip(labels, ok) if o and True], uv))
        seen[tag] = {l: p for (l, p), keep in zip(seen[tag].items(), inside) if keep}
        c = np.array(list(seen[tag].values())) + rng.normal(0, 0.15, (len(seen[tag]), 2))
        cands[tag] = np.r_[c, [[100.0, 700.0], [1200.0, 90.0]]]           # two false candidates
        img = np.full((800, 1280), 30, np.uint8)                          # dark faces, white ID dots
        X = dots @ T[:3, :3].T + T[:3, 3]
        for (u, v), z in zip(cv2.projectPoints(X, np.zeros(3), np.zeros(3), cam.K, cam.dist)[0].reshape(-1, 2),
                             X[:, 2]):
            if z > 0:
                cv2.circle(img, (int(round(u)), int(round(v))), 2, 230, -1)
        images[tag] = img
    assert min(len(s) for s in seen.values()) >= 6, {t: len(s) for t, s in seen.items()}
    # With the dots: each camera names its marks correctly on its own.
    named, infos = {}, {}
    for tag, cam in zip("AB", rig.cameras):
        obs, _, info = name_in_camera(cam.K, cam.dist, cands[tag], layout, images[tag])
        assert obs is not None and not info["ambiguous"], (tag, info)
        assert set(obs) <= set(seen[tag]) and len(obs) >= len(seen[tag]) - 1, (tag, sorted(obs), sorted(seen[tag]))
        assert all(np.linalg.norm(obs[l] - seen[tag][l]) < 1.0 for l in obs), tag
        named[tag], infos[tag] = obs, info
    T_ba, T_a, report = solve_extrinsic(rig.a.K, rig.a.dist, rig.b.K, rig.b.dist, named, layout)
    dR = np.degrees(np.linalg.norm(cv2.Rodrigues(T_ba[:3, :3] @ T_true[:3, :3].T)[0]))
    dt = np.linalg.norm(T_ba[:3, 3] - T_true[:3, 3])
    assert dR < 0.2 and dt < 0.5, (dR, dt)
    assert np.allclose(T_a, truth, atol=1e-1) or np.linalg.norm(T_a[:3, 3] - truth[:3, 3]) < 1.0
    # Without the dots: tied, and the expected separation picks the relative labelling.
    blind = {tag: name_in_camera(cam.K, cam.dist, cands[tag], layout, None)[2] for tag, cam in zip("AB", rig.cameras)}
    assert all(i["ambiguous"] and len(i["poses"]) >= 2 for i in blind.values()), [len(i["poses"]) for i in blind.values()]
    lenses = (rig.a.K, rig.a.dist, rig.b.K, rig.b.dist)
    p = truth[:3, :3] @ [150.0, -60.0, 25.0] + truth[:3, 3]  # the LED: off the holder axis, seen by both
    led = [cam_project(c, [p])[0] for c in rig.cameras]
    oa, ob, how = resolve(blind["A"], blind["B"], lenses, led)
    assert oa is not None and how.startswith("LED"), how
    T2 = solve_extrinsic(*lenses, {"A": oa, "B": ob}, layout)[0]
    assert np.allclose(T2, T_ba, atol=1e-3), "the LED picks the relative labelling that keeps T_ba"
    oa, ob, why = resolve(blind["A"], blind["B"], lenses, None)
    assert oa is None and "by hand" in why, why               # no dots, no LED: refuse -> clicks
    far = [led[0] + [0, 40.0], led[1]]                         # an LED that matches no labelling
    assert resolve(blind["A"], blind["B"], lenses, far)[0] is None
    # The independent check: a point both cameras see lands on its epipolar line.
    rig2 = holder_rig(rig.a.K, rig.a.dist, rig.b.K, rig.b.dist, T_ba, report, image_size=(1280, 800),
                      rotate180=True, camera_indices=(1, 0), lenses="synthetic")
    e = led_epipolar_px(rig2, *led)
    assert e < 0.5, e                                          # 0.15 px candidate noise in the fit
    assert led_epipolar_px(rig2, led[0] + [0, 5.0], led[1]) > 1.0
    assert rig2.meta["verified"] is False and rig2.meta["method"] == "holder_cad"
    print(f"holder_rig: named {len(named['A'])}/{len(named['B'])} marks per camera, T_ba to {dR:.3f} deg / "
          f"{dt:.3f} mm; without dots, {how}; LED epipolar check ok")


if __name__ == "__main__":
    _self_check()

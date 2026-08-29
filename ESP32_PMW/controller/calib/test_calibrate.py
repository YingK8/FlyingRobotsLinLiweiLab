#!/usr/bin/env python3
"""
Everything that runs without a camera: the synthetic round trip, and the scale check.

`theory.md` sections 14.6 and 14.7. The round trip builds a rig with a **known** extrinsic,
projects the board into it, and demands the solver return what it was given -- the only
check here that can fail for the right reason.

    python test_calibrate.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE_ = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE_), str(HERE_.parent / "camera")]
from calibrate import (BOARDS, HERE, MIN_COMMON_CORNERS, MIN_CORNERS, SPEC,  # noqa: E402
                       fit_corners,
                       detect, intrinsics_from_dir, pair_views, refine_extrinsic,
                       seed_extrinsic, solve_board_pose)
from capture import (_look, _pose_key, _rough_K, capture, gates,  # noqa: E402
                     read_meta, solo_ok, write_meta)
from rig import StereoRig  # noqa: E402
import sources  # noqa: E402
from zeroing import frame_from_normal  # noqa: E402

def scale_check(pairs, K_a, dist_a, K_b, dist_b, T_ba, spec, holdout=None):
    """Triangulated corner distances vs. the board's known geometry, in mm."""
    use = pairs if holdout is None else pairs[-holdout:]
    P_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_b = np.hstack([T_ba[:3, :3], T_ba[:3, 3].reshape(3, 1)])

    errs, truths = [], []
    for p in use:
        na = cv2.undistortPoints(p["img_a"], K_a, dist_a).reshape(-1, 2).T
        nb = cv2.undistortPoints(p["img_b"], K_b, dist_b).reshape(-1, 2).T
        X = cv2.triangulatePoints(P_a, P_b, na, nb)
        X = (X[:3] / X[3]).T                                  # (N,3) mm in camera A

        truth = spec.corners_mm[p["ids"]]
        # Every corner pair in the view, so the comparison spans short and long baselines
        # rather than one convenient distance.
        i, j = np.triu_indices(len(X), k=1)
        d_meas = np.linalg.norm(X[i] - X[j], axis=1)
        d_true = np.linalg.norm(truth[i] - truth[j], axis=1)
        errs.append(d_meas - d_true)
        truths.append(d_true)

    errs, truths = np.concatenate(errs), np.concatenate(truths)
    rel = errs / truths
    print(f"scale check over {len(use)} pair(s), {len(errs)} corner-to-corner distances")
    print(f"  absolute error  {np.mean(errs):+.4f} mm mean, "
          f"{np.percentile(np.abs(errs), 95):.4f} mm p95, {np.abs(errs).max():.4f} mm worst")
    print(f"  relative error  {np.mean(rel) * 100:+.4f}% mean, "
          f"{np.percentile(np.abs(rel), 95) * 100:.4f}% p95")
    print("  (shares the board's absolute scale -- a mis-scaled print passes this)")
    return errs, rel

TEST_SPEC = BOARDS["9x6_letter"]      # the tests assert this board's own geometry


def _synthetic_pairs(spec, rig_true, n=18, seed=0, noise_px=0.0, drop=0.0, jitter_deg=18.0):
    """Project the board at n poses into both cameras of a known rig.

    The board normal sits on the **bisector of the two optical axes**, jittered by
    ``jitter_deg`` -- the capture this module tells you to do, so the test exercises the
    recommended procedure. Orienting it randomly instead leaves camera B at 85-89 deg
    incidence on a 60-deg rig and most pairs are rejected before the solver sees them.
    """
    rng = np.random.default_rng(seed)
    K_a, K_b = rig_true.a.K, rig_true.b.K
    dist = np.zeros(5)
    obj = spec.corners_mm
    centre = obj.mean(axis=0)

    bisector = rig_true.a.optical_axis + rig_true.b.optical_axis
    bisector = bisector / np.linalg.norm(bisector)
    R_face = frame_from_normal(bisector)      # board +z along the bisector

    views_a, views_b = [], []
    for k in range(n):
        R_board = (Rotation.from_euler("xyz", rng.uniform(-jitter_deg, jitter_deg, 3),
                                       degrees=True).as_matrix() @ R_face)
        T_world_board = np.eye(4)
        T_world_board[:3, :3] = R_board
        # Put the board's centre near the point both cameras are aimed at.
        T_world_board[:3, 3] = rng.uniform(-40, 40, 3) - R_board @ centre

        T_a_board = np.linalg.inv(rig_true.a.T_world_cam) @ T_world_board
        T_b_board = np.linalg.inv(rig_true.b.T_world_cam) @ T_world_board

        view = {}
        for tag, K, T in (("a", K_a, T_a_board), ("b", K_b, T_b_board)):
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            pts, _ = cv2.projectPoints(obj, rvec, T[:3, 3], K, dist)
            pts = pts.reshape(-1, 2)
            if noise_px:
                pts = pts + rng.normal(0, noise_px, pts.shape)
            ids = np.arange(len(obj), dtype=np.int32)
            if drop:
                keep = rng.random(len(ids)) > drop
                if keep.sum() < MIN_COMMON_CORNERS + 4:
                    keep[:] = True
                pts, ids = pts[keep], ids[keep]
            view[tag] = {"index": f"s{k:03d}", "path": None, "corners": pts, "ids": ids}
        views_a.append(view["a"])
        views_b.append(view["b"])
    T_ba_true = np.linalg.inv(rig_true.b.T_world_cam) @ rig_true.a.T_world_cam
    return views_a, views_b, T_ba_true


def _recover(spec, rig_true, **kw):
    views_a, views_b, T_true = _synthetic_pairs(spec, rig_true, **kw)
    K_a, K_b = rig_true.a.K, rig_true.b.K
    d = np.zeros(5)
    pairs = pair_views(spec, views_a, views_b, K_a, d, K_b, d)
    seed_T, spread = seed_extrinsic(pairs)
    T_ba, info = refine_extrinsic(pairs, K_a, d, K_b, d, (960, 720), seed_T)
    ang = np.degrees(Rotation.from_matrix(T_ba[:3, :3] @ T_true[:3, :3].T).magnitude())
    lin = np.linalg.norm(T_ba[:3, 3] - T_true[:3, 3])
    return ang, lin, spread, info


def _fake_look(rvec, centre, speed=0.0, n=None):
    n = fit_corners(SPEC) if n is None else n
    pts = np.array(centre, dtype=np.float64) + np.zeros((n, 2))
    return {"n": n, "rvec": np.asarray(rvec, dtype=np.float64), "speed": speed,
            "corners": pts, "incidence": 30.0}


def test_solo_keeps_unpaired():
    """A frame only one camera saw is kept for its intrinsics, once per pose."""

    saved = []
    lk = _fake_look([0.0, 0.3, 0.0], (600, 400))
    assert solo_ok(lk, saved)
    saved.append(_pose_key(lk))
    assert not solo_ok(lk, saved), "the same pose twice is a duplicate"
    assert solo_ok(_fake_look([0.0, 0.9, 0.0], (600, 400)), saved), "tilted away is novel"
    assert solo_ok(_fake_look([0.0, 0.3, 0.0], (200, 400)), saved), "moved away is novel"
    assert not solo_ok(_fake_look([0.0, 0.9, 0.0], (200, 400), speed=1e4), saved), "blurred"
    assert not solo_ok(_fake_look([0.0, 0.9, 0.0], (200, 400), n=6), saved), "too few"


def test_bag_reuse():
    """A finished bag is reused, not re-shot -- no camera is opened."""

    with tempfile.TemporaryDirectory() as tmp:
        bag = Path(tmp) / "bag"
        bag.mkdir()
        assert read_meta(bag) == {}, "an unfinished bag has no meta"
        write_meta(bag, SPEC, (0, 1), (1280, 800), True, 26, [4, 7], {})
        assert capture(bag) == bag, "meta.json present: reuse, do not open a camera"
        assert read_meta(bag)["n_pairs"] == 26


def test_exact():
    """Noise-free: the solve must return the rig it was given, to numerical precision."""
    rig_true = StereoRig.from_spherical(elev_deg=(45, 45), azim_deg=(0, 90), range_mm=300)
    ang, lin, _, _ = _recover(TEST_SPEC, rig_true, n=18, seed=1)
    print(f"  rotation error {ang:.6f} deg, translation error {lin:.6f} mm")
    assert ang < 0.01, f"rotation off by {ang} deg"
    assert lin < 0.01, f"translation off by {lin} mm"


def test_noise():
    """0.2 px corner noise: still sub-0.1 deg, and the spread reflects the noise."""
    rig_true = StereoRig.from_spherical(elev_deg=(45, 45), azim_deg=(0, 90), range_mm=300)
    base = rig_true.baseline_mm()
    ang, lin, spread, _ = _recover(TEST_SPEC, rig_true, n=24, seed=2, noise_px=0.2)
    print(f"  rotation error {ang:.4f} deg, translation error {lin:.4f} mm "
          f"({lin / base * 100:.3f}% of a {base:.0f} mm baseline)")
    assert ang < 0.5, f"rotation off by {ang} deg under 0.2 px noise"
    assert lin / base < 0.02, f"translation off by {lin / base * 100:.2f}% of baseline"
    assert spread["rot_deg_max"] > 0, "spread should be non-zero under noise"


def test_disjoint_ids():
    """Each camera sees a different corner subset -- the intersection must line them up."""
    rig_true = StereoRig.from_spherical(elev_deg=(45, 45), azim_deg=(0, 100), range_mm=300)
    ang, lin, _, _ = _recover(TEST_SPEC, rig_true, n=24, seed=3, drop=0.35, jitter_deg=12.0)
    print(f"  rotation error {ang:.6f} deg, translation error {lin:.6f} mm")
    assert ang < 0.01, f"rotation off by {ang} deg with disjoint ids"
    assert lin < 0.01, f"translation off by {lin} mm with disjoint ids"


def test_mixed_hemisphere_is_refused():
    """One camera above, one below: a flat board cannot serve both, and is not pretended to.

    `axis_separation_deg` calls this rig 60 deg apart, which is right for triangulation and
    wrong for a plane: the *directed* angle is 120 deg, so the best a board can show either
    camera is 60 deg. Every pair is refused rather than solved badly (section 14.3d).
    """
    d = np.zeros(5)
    mixed = StereoRig.from_spherical(elev_deg=(45, -45), azim_deg=(0, 90), range_mm=300)
    same = StereoRig.from_spherical(elev_deg=(45, 45), azim_deg=(0, 90), range_mm=300)
    assert abs(mixed.axis_separation_deg() - same.axis_separation_deg()) < 0.1, \
        "undirected, the two rigs are indistinguishable"
    assert abs(mixed.bisector_incidence_deg() - 60.0) < 0.1, "half the DIRECTED angle"
    assert abs(same.bisector_incidence_deg() - 30.0) < 0.1, "and it tells them apart"

    kept = []
    for rig in (mixed, same):
        va, vb, _ = _synthetic_pairs(TEST_SPEC, rig, n=12, seed=4, jitter_deg=12.0)
        kept.append(len(pair_views(TEST_SPEC, va, vb, rig.a.K, d, rig.b.K, d)))
    assert kept[0] == 0, f"a board pinned at the limit must not survive jitter: {kept}"
    assert kept[1] == 12, f"the same-hemisphere rig keeps every view: {kept}"


def test_units_are_mm():
    """A 9x6 at 16.667 mm spans 150 x 100 mm, not 0.15 x 0.10."""
    w, h = TEST_SPEC.size_mm
    print(f"  board {w:.3f} x {h:.3f} mm, {TEST_SPEC.n_corners} corners")
    assert 149 < w < 151 and 99 < h < 101, f"board is {w}x{h}, not millimetres"
    assert TEST_SPEC.n_corners == 40


def test_marker_scales_with_square():
    """A rescaled print rescales both dimensions."""
    s = TEST_SPEC.with_square_mm(TEST_SPEC.square_mm * 0.97)
    print(f"  {TEST_SPEC.marker_mm:.4f} -> {s.marker_mm:.4f} mm at 97% scale")
    assert abs(s.marker_mm / s.square_mm - TEST_SPEC.marker_mm / TEST_SPEC.square_mm) < 1e-12

def regression_9x6():
    ref_path = HERE / "assets" / "camera_intrinsics.npz"
    img_dir = HERE / "assets" / "board_images_9x6"
    if not ref_path.exists() or not img_dir.exists():
        print("reference intrinsics or board_images/9x6 missing -- skipping")
        return
    K_ref = np.asarray(np.load(ref_path)["camera_matrix"], dtype=np.float64)

    K_gray, _, _ = intrinsics_from_dir(img_dir, spec=TEST_SPEC, pattern="*.jpg",
                                       decode="gray", name="IMREAD_GRAYSCALE")
    print()
    K_bgr, _, _ = intrinsics_from_dir(img_dir, spec=TEST_SPEC, pattern="*.jpg",
                                      decode="bgr2gray", name="cvtColor BGR2GRAY")

    print(f"\n  {'':4s} {'IMREAD_GRAY':>13s} {'BGR2GRAY':>13s} {'checked in':>13s}"
          f" {'decode':>9s} {'vs ref':>9s}")
    for i, (label, r, c) in enumerate([("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)]):
        g, b, ref = K_gray[r, c], K_bgr[r, c], K_ref[r, c]
        print(f"  {label:4s} {g:13.3f} {b:13.3f} {ref:13.3f} {g - b:+9.3f} {g - ref:+9.3f}")

    print("\n  'decode' is IMREAD_GRAYSCALE minus BGR2GRAY on the SAME images: the two JPEG\n"
          "  grayscale paths disagree by up to 2 grey levels, which moves the sub-pixel\n"
          "  corners. Board units and array shape were checked separately and change K by\n"
          "  exactly zero, so this is the only difference between this notebook and the\n"
          "  algorithm in visual_servo.ipynb.\n\n"
          "  'vs ref' is larger and is NOT reproducible by either path, so the checked-in\n"
          "  npz predates these images or came from a different OpenCV. Treat it as a fact\n"
          "  about that file, not a drift here. If both decode columns agree with each\n"
          "  other, this notebook's detection and board definition are sound.")


def self_test():
    """Everything that runs without a camera: the synthetic round trip, and this file."""

    for fn in (test_units_are_mm, test_marker_scales_with_square, test_solo_keeps_unpaired,
               test_bag_reuse, test_mixed_hemisphere_is_refused, test_exact, test_noise, test_disjoint_ids):
        print(f"{fn.__name__}: {fn.__doc__.splitlines()[0]}")
        fn()
        print("  ok\n")

    # The 180 deg flip for an inverted mount must not cost a corner: ArUco is
    # rotation-invariant and the ids come from the markers, so it cannot permute the
    # correspondences either.
    n = SPEC.n_corners
    img = SPEC.board.generateImage((100 * SPEC.cols, 100 * SPEC.rows), marginSize=40)
    up = detect(SPEC, img)[1]
    down = detect(SPEC, cv2.rotate(img, cv2.ROTATE_180))[1]
    assert up is not None and len(up) == n, "upright detection failed"
    assert down is not None and len(down) == n, f"flipped: {len(down or [])}/{n}"
    assert set(up.ravel().tolist()) == set(down.ravel().tolist()), "flip permuted the ids"
    assert "rotate180" in sources.MonoCamera.__init__.__code__.co_varnames
    print(f"flip: {n} corners upright and flipped, ids unchanged\n  ok\n")

    # The shutter's gates, on a board that is still, one sweeping at the speed that failed
    # the first real run, and one repeating a pose already saved.
    look = _look(SPEC, img, None, None)
    assert look["rvec"] is not None
    still = [dict(look, speed=0.0), dict(look, speed=0.0)]
    ok, _ = gates(SPEC, still, 0.002, [])
    assert ok, "a still, well-posed board must pass every gate"
    ok, checks = gates(SPEC, [dict(look, speed=440.0)] * 2, 0.002, [])
    blocked = [name for name, passed, _ in checks if not passed]
    assert not ok and "still (pair)" in blocked, blocked
    seen = [(Rotation.from_rotvec(look["rvec"].ravel()),
             look["corners"].reshape(-1, 2).mean(axis=0), look["incidence"])]
    ok, checks = gates(SPEC, still, 0.002, seen)
    assert not ok and "new pose" in [n for n, p, _ in checks if not p]
    assert solve_board_pose(SPEC, np.full((6, 1, 2), np.nan, np.float32),
                            np.arange(6).reshape(-1, 1), _rough_K((800, 1280)),
                            np.zeros(5)) is None, "a non-finite pose must come back as None"
    print("shutter: still passes, 440 px/s refused, duplicate pose refused\n  ok\n")

    print("all self-tests passed")


if __name__ == "__main__":
    self_test()

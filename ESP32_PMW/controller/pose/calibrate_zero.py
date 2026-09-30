"""
Set the pose datum from a reference image.

    calibrate_zero.from_image("ref.png")
    calibrate_zero.from_source("camera", frames=30)   # averaged; prefer this
    calibrate_zero.clear()

Estimate the pose in a reference view and store it, so every later run reports
"how far from there" instead of "how far from the lens".  Re-running the
estimator on the reference then reads zero on all six channels; `test_zeroing.py`
asserts exactly that.

Averaging several frames is worth it on a live camera: the datum is subtracted
from every subsequent measurement, so noise in it becomes a fixed bias in the
whole run rather than something that averages out.

**Do not use a dead-on reference pose.**  Tilt is read from foreshortening, so
its sensitivity goes as ``1/sin(theta)`` and near face-on it is ill-conditioned
-- on a face-on render the recovered axis came out about 10 degrees off.  The
datum is a rotation, so that error tilts every later reading.  Measured, the
axis error collapses as soon as you leave face-on -- 10.4 deg at 0, 2.3 deg by
10 -- while position error then climbs with tilt as the mast inflates the minor
axis.  **Zero at roughly 10-20 degrees of tilt**, which is the best of both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.

from controller.pose.estimator import RADIUS_MM, PoseEstimator, load_intrinsics
from controller.calib.zeroing import DEFAULT_PATH, Zero, average_poses


def collect(estimator, source, n_frames, max_attempts=None):
    """
    Gather ``n_frames`` successful reference observations from a source.
    """

    max_attempts = max_attempts or n_frames * 10
    centers, normals, psis = [], [], []

    for _ in range(max_attempts):
        item = source.read()
        if item is None:
            break
        solved = estimator.solve_camera_frame(item[1])
        if solved is None:
            continue
        c, n, psi = solved
        centers.append(c)
        normals.append(n)
        psis.append(psi)
        if len(centers) >= n_frames:
            break

    return centers, normals, psis


def _save(centers, normals, psis, origin, out, radius_mm):
    """
    Average the observations into a datum and write it. Returns the `Zero`.
    """

    center, normal = average_poses(centers, normals)
    psi = float(np.mean(psis))

    # Pin the in-plane term too, so phi reads zero at the reference rather than
    # whatever the camera mounting happens to be.
    in_plane = np.array(
        [np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0], dtype=np.float64
    )
    zero = Zero.from_pose(
        center,
        normal,
        psi_deg=psi,
        in_plane=in_plane,
        meta={
            "source": str(origin),
            "n_frames": len(centers),
            "radius_mm": radius_mm,
            "center_mm": np.round(center, 4).tolist(),
            "normal": np.round(normal, 6).tolist(),
        },
    )
    path = zero.save(out)

    print(f"datum from {len(centers)} frame(s) of {origin}")
    print(
        f"  centre {np.round(center, 3)} mm   normal {np.round(normal, 4)}   psi {psi:.2f} deg"
    )
    if len(centers) > 1:
        spread = float(np.max(np.linalg.norm(np.array(centers) - center, axis=1)))
        print(f"  worst frame-to-frame spread in centre: {spread:.3f} mm")
    print(f"  written to {path}")
    return zero


def _estimator(intrinsics=None, radius_mm=RADIUS_MM, thresh=None):
    K, dist = load_intrinsics(intrinsics) if intrinsics else load_intrinsics()
    kw = {} if thresh is None else {"thresh": thresh}
    return PoseEstimator(camera_matrix=K, dist_coeffs=dist, radius_mm=radius_mm, **kw)


def from_image(
    path, out=DEFAULT_PATH, intrinsics=None, radius_mm=RADIUS_MM, thresh=None
):
    """
    Datum from one reference image.
    """

    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(f"could not read {path}")
    est = _estimator(intrinsics, radius_mm, thresh)
    solved = est.solve_camera_frame(frame)
    if solved is None:
        raise ValueError(
            "no detection in the reference image -- check threshold and framing"
        )
    return _save([solved[0]], [solved[1]], [solved[2]], path, out, radius_mm)


def from_source(
    spec, frames=30, out=DEFAULT_PATH, intrinsics=None, radius_mm=RADIUS_MM, thresh=None
):
    """
    Datum averaged over live frames. Worth preferring over a single image.

        The datum inherits whatever error the frames had, permanently, so averaging a
        few dozen is cheap insurance against calibrating to one bad frame.
    """

    from controller.camera import sources

    est = _estimator(intrinsics, radius_mm, thresh)
    with sources.open_source(spec) as s:
        centers, normals, psis = collect(est, s, frames)
    if not centers:
        raise OSError("no usable frames from the source")
    return _save(centers, normals, psis, spec, out, radius_mm)


def clear(out=DEFAULT_PATH):
    """
    Reset the datum to identity, i.e. report raw camera coordinates.
    """

    path = Zero.identity().save(out)
    print(f"datum cleared -> {path}")
    return path

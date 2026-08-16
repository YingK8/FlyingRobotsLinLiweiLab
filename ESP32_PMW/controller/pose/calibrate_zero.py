"""Set the pose datum from a reference image.

    uv run python controller/pose/calibrate_zero.py --image ref.png
    uv run python controller/pose/calibrate_zero.py --source camera --frames 30
    uv run python controller/pose/calibrate_zero.py --clear

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

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

from estimator import RADIUS_MM, PoseEstimator, load_intrinsics  # noqa: E402
from zeroing import DEFAULT_PATH, Zero, average_poses  # noqa: E402


def collect(estimator, source, n_frames, max_attempts=None):
    """Gather ``n_frames`` successful reference observations from a source."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", help="reference image file")
    src.add_argument("--source", help="frame source: 'camera', 'camera:1', a video, or a directory")
    ap.add_argument("--frames", type=int, default=30, help="frames to average (source mode)")
    ap.add_argument("--out", default=str(DEFAULT_PATH), help="where to write the datum")
    ap.add_argument("--intrinsics", default=None, help="camera_intrinsics.npz override")
    ap.add_argument("--radius-mm", type=float, default=RADIUS_MM)
    ap.add_argument("--thresh", type=int, default=None)
    ap.add_argument("--clear", action="store_true", help="reset the datum to identity")
    args = ap.parse_args(argv)

    if args.clear:
        path = Zero.identity().save(args.out)
        print(f"datum cleared -> {path}")
        return 0

    if not args.image and not args.source:
        ap.error("give --image, --source, or --clear")

    K, dist = load_intrinsics(args.intrinsics) if args.intrinsics else load_intrinsics()
    kw = {} if args.thresh is None else {"thresh": args.thresh}
    est = PoseEstimator(camera_matrix=K, dist_coeffs=dist, radius_mm=args.radius_mm, **kw)

    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"could not read {args.image}", file=sys.stderr)
            return 2
        solved = est.solve_camera_frame(frame)
        if solved is None:
            print("no detection in the reference image -- check threshold and framing",
                  file=sys.stderr)
            return 3
        centers, normals, psis = [solved[0]], [solved[1]], [solved[2]]
        origin = args.image
    else:
        import sources

        with sources.open_source(args.source) as s:
            centers, normals, psis = collect(est, s, args.frames)
        origin = args.source
        if not centers:
            print("no usable frames from the source", file=sys.stderr)
            return 3

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
            "source": origin,
            "n_frames": len(centers),
            "radius_mm": args.radius_mm,
            "center_mm": np.round(center, 4).tolist(),
            "normal": np.round(normal, 6).tolist(),
        },
    )
    path = zero.save(args.out)

    spread = float(np.max(np.linalg.norm(np.array(centers) - center, axis=1))) if len(centers) > 1 else 0.0
    print(f"datum from {len(centers)} frame(s) of {origin}")
    print(f"  centre {np.round(center, 3)} mm   normal {np.round(normal, 4)}   psi {psi:.2f} deg")
    if len(centers) > 1:
        print(f"  worst frame-to-frame spread in centre: {spread:.3f} mm")
    print(f"  written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Stereo calibration: what each camera sees, and where the two of them are.

A bag of board pair images in, `stereo_rig.json` out. `calibrate_intrinsics` solves one
camera's K from its views; `calibrate_extrinsics` solves the frame the two share, from the
views they took together. The theory is `theory.md` chapter 2.

**World frame is camera A**, ``T_world_camA = I``. The board only exists during
calibration and the robot is the thing being measured, so camera A is the only datum that
is both physically persistent and independent of the measurement.

Three things cost real accuracy when got wrong. Shoot at the sensor's **native** mode --
1280x720 is slower than 1280x800 and is a crop, and only 640x400 is a true rescale. Let
**nothing lossy** touch a frame: two JPEG grayscale decode paths two grey levels apart
moved f_x by 0.34 px (`test_calibrate.py`). And the board must be **still** and the print
**measured** -- the refusing shutter in `capture.py`, the ruler bar `sheet.py` draws.

    python sheet.py                       # a printable board, cut marks and ruler bar
    python calibrate.py                   # capture (or reuse the bag), solve, write
    python calibrate.py --no-capture      # re-solve the bag already on disk
    python test_calibrate.py              # no cameras, no board
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ESP32 = HERE.parents[1]
OUT_DIR = ESP32 / "results" / "stereo_calibration"
PAIR_DIR = OUT_DIR / "pairs"
RIG_PATH = HERE / "stereo_rig.json"

# ---- the board ----------------------------------------------------------------------
MIN_CORNERS = 6  # solvePnP needs 4, and returns nonsense from 4 noisy ones
MIN_FIT_FRAC = 0.5  # ...but posing a view and *calibrating* from it are different
# questions. Half the board before a view joins the fit: on a
# 4x12 only 3 corners wide, a 12-corner view is a thin strip and
# a badly conditioned homography. Measured on a 63-view set, the
# views that blew up to 1.9 px all had 11-18 of 33 (section 14.3a).
MAX_INCIDENCE_DEG = 60.0  # past this the corners crowd together and the pose goes soft
# along the ray. Section 14.3 derives 70 from rotation
# uncertainty; 60 is what two measured sets say for a board this
# small, and 70 was applied in `pair_views` only, so solo views at
# 77 deg walked into the intrinsics fit. Camera A: 1.075 px over
# 65 views uncapped, 0.404 over 30 capped (section 14.3c).


@dataclass
class CharucoSpec:
    """One physical board, in millimetres.

    ``square_mm`` is the *measured* pitch of the print in front of you, not the nominal one.
    ``marker_mm`` must be smaller -- OpenCV insets the marker inside the white square.
    """

    cols: int
    rows: int
    square_mm: float
    marker_mm: float
    dict_name: str = "DICT_4X4_100"
    name: str = ""

    def __post_init__(self):
        if self.marker_mm >= self.square_mm:
            raise ValueError(f"marker {self.marker_mm} >= square {self.square_mm} mm")

    @cached_property
    def board(self):
        return cv2.aruco.CharucoBoard(
            (self.cols, self.rows),
            float(self.square_mm),
            float(self.marker_mm),
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name)),
        )

    @cached_property
    def detector(self):
        return cv2.aruco.CharucoDetector(self.board)  # sub-pixel refines internally

    @property
    def n_corners(self):
        return (self.cols - 1) * (self.rows - 1)

    @property
    def size_mm(self):
        return (self.cols * self.square_mm, self.rows * self.square_mm)

    @property
    def corners_mm(self):
        """Interior corners in board coordinates, ``(n_corners, 3)``, z = 0."""
        return np.asarray(self.board.getChessboardCorners(), dtype=np.float64)

    def with_square_mm(self, square_mm):
        """Copy at a measured pitch, scaling the marker by the same factor."""
        k = float(square_mm) / self.square_mm
        return CharucoSpec(
            self.cols,
            self.rows,
            float(square_mm),
            self.marker_mm * k,
            self.dict_name,
            self.name,
        )

    def summary(self):
        w, h = self.size_mm
        return {
            "board": self.name,
            "squares": [self.cols, self.rows],
            "square_mm": round(self.square_mm, 4),
            "marker_mm": round(self.marker_mm, 4),
            "dictionary": self.dict_name,
            "n_corners": self.n_corners,
            "size_mm": [round(w, 3), round(h, 3)],
        }


BOARDS = {
    # 8 corners across, not 3: at 119 mm this determines f_x 2.3x better than the slide
    # from the same 40 views, at the same reprojection RMS (section 14.3a).
    #
    # Measured with calipers off this print, not nominal. The generator asked for 6.0113 mm
    # squares and 4.5 mm markers; the sheet came out at 5.95 and 4.45, a uniform 98.98%.
    # The two readings agree to 4 um at that scale, which is what makes them believable.
    # Using the nominal figures would inflate every distance the rig reports by 1.03% --
    # a 100 mm span reading 101.03 -- with every acceptance gate still green (section 14.4).
    "9x6_6mm": CharucoSpec(9, 6, 5.95, 4.45, "DICT_4X4_100", "9x6_6mm"),
    "9x6_6mm_nominal": CharucoSpec(
        9, 6, 6.0113, 4.5, "DICT_4X4_100", "9x6_6mm_nominal"
    ),
    "4x12_6mm": CharucoSpec(4, 12, 6.0113, 4.5, "DICT_4X4_100", "4x12_6mm"),
    "9x6_letter": CharucoSpec(9, 6, 16.667, 12.5, "DICT_4X4_100", "9x6_letter"),
}


# The board in front of us -- change this line, or pass --board. 4x12 at 6 mm fits a
# 25 x 75 mm microscope slide, the flattest cheap backing there is, and pays 2.3x in
# focal-length uncertainty for it (section 14.3a). Every pitch here is nominal until the
# print is measured with the sheet's ruler bar: the letter print came out at 16.52 mm
# against a nominal 16.667, and nothing in the acceptance gate can see that.
BOARD = "9x6_6mm"
COLS, ROWS = 9, 6
SQUARE_MM, MARKER_MM = 5.95, 4.45


def make_spec(square_mm=SQUARE_MM, marker_mm=MARKER_MM, cols=COLS, rows=ROWS):
    """The board at its measured pitch."""
    return CharucoSpec(
        cols,
        rows,
        square_mm,
        marker_mm,
        "DICT_4X4_100",
        f"{cols}x{rows}_{square_mm:g}mm",
    )


SPEC = BOARDS[BOARD]


# ---- detection ----------------------------------------------------------------------
def detect(spec, image):
    """ChArUco corners in one image: ``(corners (N,2), ids (N,))`` or ``(None, None)``."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = spec.detector.detectBoard(gray)
    if corners is None or ids is None or len(corners) == 0:
        return None, None
    return (
        np.asarray(corners, dtype=np.float64).reshape(-1, 2),
        np.asarray(ids, dtype=np.int32).reshape(-1),
    )


def match_points(spec, corners, ids):
    """``(object (N,1,3), image (N,1,2))`` float32 for cv2.calibrateCamera/stereoCalibrate."""
    obj, img = spec.board.matchImagePoints(
        np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(ids, dtype=np.int32).reshape(-1, 1),
    )
    if obj is None or img is None or len(obj) == 0:
        return None, None
    return (
        np.asarray(obj, dtype=np.float32).reshape(-1, 1, 3),
        np.asarray(img, dtype=np.float32).reshape(-1, 1, 2),
    )


def solve_board_pose(spec, corners, ids, K, dist):
    """``(rvec, tvec)`` of the board in camera coordinates, mm. ``None`` if unsolvable.

    IPPE is the planar-target solver, exact for z = 0 object points; the LM refinement then
    minimises the reprojection error, which is the quantity being gated on. A degenerate
    view lets that diverge to NaN, and "unsolvable" is the answer every caller handles.
    """
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj, img = match_points(spec, corners, ids)
    if obj is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None
    return rvec.reshape(3), tvec.reshape(3)


def board_incidence_deg(rvec):
    """Angle from face-on, degrees. 0 = square to the camera, 90 = edge-on."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return math.degrees(math.acos(float(np.clip(abs(R[2, 2]), 0.0, 1.0))))


def board_normal(rvec):
    """The board's normal in camera coordinates, in the ``+z`` hemisphere."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R[:, 2] * (1.0 if R[2, 2] >= 0 else -1.0)


def orientation_spread_deg(normals):
    """RMS angle of the board normals about their mean. 0 = every view the same way up.

    Not the spread of `board_incidence_deg`, which is what this gate measured until it was
    measured itself: incidence is *unsigned*, so tilting left 40 deg and right 40 deg score
    identically, and a set covering the hemisphere reads the same as one that only ever
    tilts one way. The textbook set -- face-on plus 40 deg tilted four ways -- scored 16 deg
    on it and was told to tilt more (section 14.3b).
    """
    N = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    mean = N.mean(axis=0)
    mean = mean / max(np.linalg.norm(mean), 1e-12)
    return float(
        np.sqrt(np.mean(np.degrees(np.arccos(np.clip(N @ mean, -1.0, 1.0))) ** 2))
    )


def pose_matrix(rvec, tvec):
    """``T_cam_board`` (4x4): board coordinates -> camera coordinates."""
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


NATIVE_W, NATIVE_H = 1280, 800  # ELP OV9281 native: 119 fps, full field of view
# Which index is which camera: `camera/elp.py :: probe_indices`. USB cameras enumerate
# BEFORE the built-in FaceTime, so the ELP is normally index 0.


# ---- load what was shot -------------------------------------------------------------
def fit_corners(spec, frac=MIN_FIT_FRAC):
    """How many corners a view must deliver to be worth calibrating from."""

    return max(MIN_CORNERS, int(round(frac * spec.n_corners)))


def _load_dir(
    spec, img_dir, pattern="*.png", decode="gray", label="", size=None, min_corners=None
):
    """Detect the board in every image in one directory. ``(views, image_size)``.

    ``decode`` selects the JPEG grayscale reader, which is not cosmetic: the two paths move
    f_x by 0.34 px (`test_calibrate.regression_9x6`).
    """
    need = MIN_CORNERS if min_corners is None else min_corners
    views = []
    for path in sorted(Path(img_dir).glob(pattern)):
        if decode == "gray":
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            colour = cv2.imread(str(path))
            img = None if colour is None else cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        if img is None:
            print(f"  skip {label}{path.name}: unreadable")
            continue
        wh = (img.shape[1], img.shape[0])
        if size is None:
            size = wh
        elif wh != size:
            # Intrinsics are per-mode, and on this sensor most modes are crops rather than
            # rescales, so a mixed-resolution set cannot be calibrated together.
            raise ValueError(f"{path}: size {wh} != {size}")
        corners, ids = detect(spec, img)
        if ids is None or len(ids) < need:
            print(
                f"  skip {label}{path.name}: {0 if ids is None else len(ids)} corners"
            )
            continue
        views.append({"index": path.stem, "path": path, "corners": corners, "ids": ids})
    return views, size


def load_views(spec, pair_dir=PAIR_DIR, pattern="*.png", min_corners=None):
    """Both cameras' views from a capture bag: ``pair_dir/A`` and ``pair_dir/B``.

    Pairs and solo frames alike (section 16.5a). Solo frames constrain that camera's
    intrinsics but cannot contribute to the extrinsic, and their stems carry the camera
    tag so `pair_views` can never match two of them.
    """
    out, size = {}, None
    for tag in "AB":
        out[tag], size = _load_dir(
            spec,
            Path(pair_dir) / tag,
            pattern,
            label=f"{tag}/",
            size=size,
            min_corners=min_corners,
        )
        print(
            f"camera {tag}: {len(out[tag])} usable views, "
            f"{np.mean([len(v['ids']) for v in out[tag]]):.1f} corners mean"
            if out[tag]
            else f"camera {tag}: no usable views"
        )
    return out["A"], out["B"], size


# ---- per-camera intrinsics ----------------------------------------------------------
MIN_VIEWS = 8  # 4 is the algebraic minimum; 4 is not a calibration
MIN_ORIENT_SPREAD_DEG = 25.0  # of `orientation_spread_deg`. The textbook set -- face-on
# plus 40 deg tilted four ways -- scores 35.8 deg, and a
# set that only ever tilts one way about 10.


def calibrate_intrinsics(
    spec,
    views,
    image_size,
    name="",
    min_corners=None,
    max_incidence_deg=MAX_INCIDENCE_DEG,
):
    """``(K, dist, info)`` for one camera. The *Extended* form returns the per-view RMS and
    the parameter standard deviations that the plain one makes you recompute.

    Views under `fit_corners` are dropped **here** and not at load: a thin strip of corners
    is a badly conditioned homography for *this* fit (section 14.3a), while `pair_views`
    wants the same image and only needs eight corners shared with the other camera.
    """
    need = fit_corners(spec) if min_corners is None else min_corners
    obj_pts, img_pts, thin = [], [], 0
    for v in views:
        if len(v["ids"]) < need:
            thin += 1
            continue
        o, i = match_points(spec, v["corners"], v["ids"])
        if o is not None:
            obj_pts.append(o)
            img_pts.append(i)
    if len(obj_pts) < MIN_VIEWS:
        raise RuntimeError(
            f"camera {name}: {len(obj_pts)} usable views, need {MIN_VIEWS}"
        )

    # Two passes: incidence needs a pose, a pose needs K, and K is what is being solved.
    # The first fit is only ever used to measure the angles the second one drops.
    rms, K, dist, rvecs, _, std_K, _, per_view = cv2.calibrateCameraExtended(
        obj_pts, img_pts, image_size, None, None
    )
    steep = [
        i for i, rv in enumerate(rvecs) if board_incidence_deg(rv) > max_incidence_deg
    ]
    if steep and len(obj_pts) - len(steep) >= MIN_VIEWS:
        obj_pts = [o for i, o in enumerate(obj_pts) if i not in set(steep)]
        img_pts = [p for i, p in enumerate(img_pts) if i not in set(steep)]
        rms, K, dist, rvecs, _, std_K, _, per_view = cv2.calibrateCameraExtended(
            obj_pts, img_pts, image_size, None, None
        )
    dist = np.asarray(dist, dtype=np.float64).ravel()
    per_view = np.asarray(per_view, dtype=np.float64).ravel().tolist()
    std_K = np.asarray(std_K, dtype=np.float64).ravel()  # fx, fy, cx, cy, then dist

    incidences = [board_incidence_deg(rv) for rv in rvecs]
    spread = orientation_spread_deg([board_normal(rv) for rv in rvecs])
    info = {
        "n_views": len(obj_pts),
        "rms_px": float(rms),
        "per_view_rms_px": per_view,
        "incidence_deg": incidences,
        "orientation_spread_deg": spread,
        "std_fx_fy_cx_cy": std_K[:4].tolist(),
        "image_size": list(image_size),
    }

    print(
        f"camera {name}: {len(obj_pts)} views, RMS {rms:.4f} px, "
        f"worst view {max(per_view):.4f} px"
        + (f"  ({thin} under {need} corners" if thin else "")
        + (f", {len(steep)} over {max_incidence_deg:.0f} deg" if steep else "")
        + (")" if thin or steep else "")
    )
    print(f"  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(
        f"  +/-  {std_K[0]:.2f}      {std_K[1]:.2f}      {std_K[2]:.2f}     "
        f"{std_K[3]:.2f}   (1 sigma)"
    )
    print(f"  dist {np.array2string(dist, precision=5)}")
    print(
        f"  incidence {min(incidences):.1f}-{max(incidences):.1f} deg, "
        f"orientation spread {spread:.1f} deg"
    )
    if spread < MIN_ORIENT_SPREAD_DEG:
        print(
            f"  WARNING: orientation spread {spread:.1f} deg is under "
            f"{MIN_ORIENT_SPREAD_DEG}. Coplanar points at near-identical angles leave "
            f"focal length and distortion poorly separated. Tilt the board "
            f"more, and in more than one direction."
        )
    return K, dist, info


def intrinsics_from_dir(img_dir, spec=SPEC, pattern="*.png", name=None, decode="gray"):
    """``(K, dist, info)`` for one camera from a flat directory of board photos."""
    img_dir = Path(img_dir)
    views, size = _load_dir(spec, img_dir, pattern, decode)
    if not views:
        raise FileNotFoundError(
            f"no usable board photos matching {pattern} in {img_dir}"
        )
    return calibrate_intrinsics(spec, views, size, name or img_dir.name)


# ---- extrinsics ---------------------------------------------------------------------
MIN_COMMON_CORNERS = 8


def pair_views(
    spec,
    views_a,
    views_b,
    K_a,
    dist_a,
    K_b,
    dist_b,
    max_incidence_deg=MAX_INCIDENCE_DEG,
):
    """Views the two cameras took together, sharing enough corners at usable incidence."""
    by_b = {v["index"]: v for v in views_b}
    pairs, rejected = [], []

    for va in views_a:
        vb = by_b.get(va["index"])
        if vb is None:
            continue
        common = np.intersect1d(va["ids"], vb["ids"])
        if len(common) < MIN_COMMON_CORNERS:
            rejected.append((va["index"], f"{len(common)} common corners"))
            continue

        sel_a = np.isin(va["ids"], common)
        sel_b = np.isin(vb["ids"], common)
        # intersect1d returns sorted ids, and isin preserves each view's own order, so
        # re-sort both by id to guarantee row i is the same physical corner in both.
        ord_a = np.argsort(va["ids"][sel_a])
        ord_b = np.argsort(vb["ids"][sel_b])
        ids = va["ids"][sel_a][ord_a]
        img_a = va["corners"][sel_a][ord_a]
        img_b = vb["corners"][sel_b][ord_b]
        assert np.array_equal(ids, vb["ids"][sel_b][ord_b])

        pose_a = solve_board_pose(spec, img_a, ids, K_a, dist_a)
        pose_b = solve_board_pose(spec, img_b, ids, K_b, dist_b)
        if pose_a is None or pose_b is None:
            rejected.append((va["index"], "solvePnP failed"))
            continue
        inc_a, inc_b = board_incidence_deg(pose_a[0]), board_incidence_deg(pose_b[0])
        if max(inc_a, inc_b) > max_incidence_deg:
            rejected.append((va["index"], f"incidence {inc_a:.0f}/{inc_b:.0f} deg"))
            continue

        pairs.append(
            {
                "index": va["index"],
                "ids": ids,
                "obj": spec.corners_mm[ids].astype(np.float32).reshape(-1, 1, 3),
                "img_a": img_a.astype(np.float32).reshape(-1, 1, 2),
                "img_b": img_b.astype(np.float32).reshape(-1, 1, 2),
                "T_a": pose_matrix(*pose_a),
                "T_b": pose_matrix(*pose_b),
                "incidence_a": inc_a,
                "incidence_b": inc_b,
            }
        )

    print(f"{len(pairs)} usable pairs, {len(rejected)} rejected")
    for idx, why in rejected:
        print(f"  reject {idx}: {why}")
    return pairs


def calibrate_extrinsics(spec, views_a, views_b, K_a, dist_a, K_b, dist_b, image_size):
    """``(T_ba, pairs, seed_T, spread, info)`` from the paired views, intrinsics held fixed.

    The peer of `calibrate_intrinsics`: the two cameras' views and the K each was solved
    for, in; the frame they share, out. Views present on only one side carry no extrinsic
    information, which is exactly why `capture` keeps them anyway -- for the intrinsics.
    """

    pairs = pair_views(spec, views_a, views_b, K_a, dist_a, K_b, dist_b)
    if not pairs:
        raise RuntimeError("no usable pairs -- check the rejection reasons above")
    seed_T, spread = seed_extrinsic(pairs)
    T_ba, info = refine_extrinsic(pairs, K_a, dist_a, K_b, dist_b, image_size, seed_T)
    return T_ba, pairs, seed_T, spread, info


def seed_extrinsic(pairs):
    """Closed-form ``T_camB_camA``: chordal mean rotation, median translation.

    The board pose cancels pair by pair (section 14.1), so this owes the bundle nothing.
    It is not fed to `refine_extrinsic` -- OpenCV 5 refuses `CALIB_USE_EXTRINSIC_GUESS` --
    which is what makes their agreement evidence rather than a tautology (section 14.5).
    """
    if not pairs:
        raise RuntimeError("no pairs to seed from -- every pair was rejected above")
    Ts = np.array([p["T_b"] @ np.linalg.inv(p["T_a"]) for p in pairs])
    rots = Rotation.from_matrix(Ts[:, :3, :3])
    mean_rot, t_med = rots.mean(), np.median(Ts[:, :3, 3], axis=0)

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = mean_rot.as_matrix(), t_med

    ang = np.degrees((rots * mean_rot.inv()).magnitude())
    lin = np.linalg.norm(Ts[:, :3, 3] - t_med, axis=1)
    spread = {
        "rot_deg_median": float(np.median(ang)),
        "rot_deg_max": float(ang.max()),
        "trans_mm_median": float(np.median(lin)),
        "trans_mm_max": float(lin.max()),
    }
    print(f"seed from {len(pairs)} pairs: baseline {np.linalg.norm(t_med):.2f} mm")
    print(
        f"  pair-to-pair spread: rotation {spread['rot_deg_median']:.3f} deg median / "
        f"{spread['rot_deg_max']:.3f} worst, translation "
        f"{spread['trans_mm_median']:.3f} mm median / {spread['trans_mm_max']:.3f} worst"
    )
    if ang.max() > 5 * max(np.median(ang), 1e-6):
        print(
            f"  pair {pairs[int(np.argmax(ang))]['index']} is the outlier at "
            f"{ang.max():.2f} deg -- look at that image before trusting the bundle"
        )
    return T, spread


def refine_extrinsic(
    pairs, K_a, dist_a, K_b, dist_b, image_size, seed_T, fix_intrinsic=True
):
    """cv2.stereoCalibrateExtended over all pairs. ``(T_camB_camA, info)``.

    The *Extended* form also returns the board pose per pair **from this bundle**, which is
    the joint fit `stereo_residuals` needs and would otherwise have to be re-solved.
    """
    out = cv2.stereoCalibrateExtended(
        [p["obj"] for p in pairs],
        [p["img_a"] for p in pairs],
        [p["img_b"] for p in pairs],
        K_a,
        dist_a,
        K_b,
        dist_b,
        image_size,
        seed_T[:3, :3].copy(),
        seed_T[:3, 3].copy().reshape(3, 1),
        flags=cv2.CALIB_FIX_INTRINSIC if fix_intrinsic else 0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10),
    )
    rms, K_a, dist_a, K_b, dist_b, R, T, E, F, rvecs, tvecs, per_view = out

    T_ba = np.eye(4)
    T_ba[:3, :3], T_ba[:3, 3] = R, np.asarray(T).reshape(3)
    per_view = np.asarray(per_view, dtype=np.float64).reshape(-1, 2)

    gap_deg = float(
        np.degrees(Rotation.from_matrix(T_ba[:3, :3] @ seed_T[:3, :3].T).magnitude())
    )
    gap_mm = float(np.linalg.norm(T_ba[:3, 3] - seed_T[:3, 3]))
    info = {
        "rms_px": float(rms),
        "n_pairs": len(pairs),
        "per_pair_rms_px": per_view.tolist(),
        "fix_intrinsic": bool(fix_intrinsic),
        "seed_gap_deg": gap_deg,
        "seed_gap_mm": gap_mm,
        "rvecs": [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs],
        "tvecs": [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs],
    }
    print(
        f"stereoCalibrate: RMS {rms:.4f} px over {len(pairs)} pairs "
        f"(worst pair {per_view.max():.4f} px)"
    )
    print(f"  baseline {np.linalg.norm(T_ba[:3, 3]):.3f} mm")
    print(
        f"  agreement with the independent closed-form seed: "
        f"{gap_deg:.4f} deg, {gap_mm:.4f} mm"
    )
    return T_ba, info


# ---- solve --------------------------------------------------------------------------
def run_calibration(spec, pair_dir=PAIR_DIR):
    """A capture bag -> a gated result. Everything downstream reads the returned dict."""

    views_a, views_b, image_size = load_views(spec, pair_dir)
    print(f"image size {image_size}\n")

    K_a, dist_a, intr_a = calibrate_intrinsics(spec, views_a, image_size, "A")
    print()
    K_b, dist_b, intr_b = calibrate_intrinsics(spec, views_b, image_size, "B")
    print()

    T_ba, pairs, seed_T, spread, stereo_info = calibrate_extrinsics(
        spec, views_a, views_b, K_a, dist_a, K_b, dist_b, image_size
    )
    print()

    from results import acceptance, stereo_residuals, structure_report, uncertainty_um

    resid = stereo_residuals(
        pairs,
        K_a,
        dist_a,
        K_b,
        dist_b,
        T_ba,
        stereo_info["rvecs"],
        stereo_info["tvecs"],
    )
    struct_a = structure_report(resid["A"]["rad"], resid["A"]["res"])
    struct_b = structure_report(resid["B"]["rad"], resid["B"]["res"])
    print()
    z = float(np.median([np.linalg.norm(p["T_a"][:3, 3]) for p in pairs]))
    sep = math.degrees(math.acos(float(np.clip(T_ba[:3, :3].T[2, 2], -1.0, 1.0))))
    passed = acceptance(
        stereo_info,
        resid,
        intr_a,
        intr_b,
        struct_a,
        struct_b,
        spread,
        um=lambda px: uncertainty_um(px, z, K_a[0, 0], sep),
    )

    return {
        "pairs": pairs,
        "image_size": image_size,
        "K_a": K_a,
        "dist_a": dist_a,
        "K_b": K_b,
        "dist_b": dist_b,
        "intr_a": intr_a,
        "intr_b": intr_b,
        "T_ba": T_ba,
        "seed_T": seed_T,
        "spread": spread,
        "resid": resid,
        "stereo_info": stereo_info,
        "struct_a": struct_a,
        "struct_b": struct_b,
        "passed": passed,
    }


# ---- driver -------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument(
        "--bag",
        type=Path,
        default=PAIR_DIR,
        help="capture directory: shot if new, reused if it already exists",
    )
    p.add_argument(
        "--override", action="store_true", help="re-shoot the bag, replacing it"
    )
    p.add_argument("--append", action="store_true", help="add to the bag")
    p.add_argument(
        "--no-capture", action="store_true", help="solve on disk, no cameras"
    )
    p.add_argument(
        "--indices",
        nargs=2,
        type=int,
        default=(0, 1),
        help="camera indices; the first is the world frame",
    )
    p.add_argument(
        "--board",
        choices=sorted(BOARDS),
        default=None,
        help="which printed board is in front of you",
    )
    p.add_argument(
        "--square-mm",
        type=float,
        default=None,
        help="the print's measured pitch, off the sheet's ruler bar",
    )
    p.add_argument("--no-flip", action="store_true", help="cameras are not inverted")
    p.add_argument("--rig", type=Path, default=RIG_PATH)
    p.add_argument("--out", type=Path, default=OUT_DIR)  # npz and figures
    a = p.parse_args(argv)

    spec = BOARDS[a.board] if a.board else make_spec()
    if a.square_mm:  # a scaled print scales the markers with it
        spec = spec.with_square_mm(a.square_mm)
    print(
        f"board {spec.name}: {spec.square_mm} mm squares, {spec.marker_mm} mm markers"
    )
    if not a.no_capture:
        from capture import capture  # cameras only when actually shooting

        capture(
            a.bag,
            tuple(a.indices),
            spec,
            rotate180=not a.no_flip,
            append=a.append,
            override=a.override,
        )

    cal = run_calibration(spec, a.bag)
    from results import write_results
    from plots import coverage_figure, figures, undistort_figure

    coverage_figure(spec, a.bag, a.out)
    figures(cal["pairs"], cal["image_size"], cal["resid"], out_dir=a.out)
    undistort_figure(cal, a.bag, a.out)
    if not cal["passed"]:
        print("\nacceptance failed -- nothing written. Fix the capture, not the limit.")
        return 1
    write_results(
        cal,
        spec,
        a.rig,
        a.out,
        meta={"camera_indices": list(a.indices), "rotate180": not a.no_flip},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

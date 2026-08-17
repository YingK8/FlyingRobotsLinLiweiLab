"""
Where the two cameras are, and how to move a pose between their frames.

`conic.py` works entirely in one camera's coordinates.  Two cameras need a
shared frame and the transforms into it, and that is all this module is: a pair
of intrinsics plus a pair of poses, with the bookkeeping to carry a
``(center, normal)`` across.

Conventions, all of them OpenCV to match the rest of the package:

* ``T_world_cam`` maps **camera coordinates into world coordinates**.  So a
  point at the lens is ``T_world_cam @ [0,0,0,1]`` -- the camera's position in
  the world.  The camera looks down its own +z, with +y down.
* The world frame depends on where the rig came from, and the two producers do
  not agree -- check ``meta["world_frame"]`` before reading any angle off a
  loaded rig:

  - **A measured rig** (`stereo_calibration.ipynb`) uses **camera A** as the
    datum, ``T_world_camA = I``, so +z is A's optical axis and +y is image-down.
    `baseline_mm` and `axis_separation_deg` are frame-independent and stay
    correct; `tilt_seen_deg`, `tilt_information` and anything phrased as an
    elevation do **not**, because they assume +z is the rotor axis.  Those become
    meaningful once the rig is rebased onto the robot's disk frame at rest, which
    happens when the estimator is wired into visual servoing -- deliberately not
    part of the calibration, so the extrinsic never depends on the robot.
  - **A nominal rig** (`StereoRig.from_spherical`) centres a right-handed frame
    on the hover point with **+z up**, which is *not* a camera convention -- the
    robot's world is up, and forcing it into a y-down frame would make every
    elevation angle read backwards.  Everything below assumes this case.

Terminology warning.  "Azimuth" here is a **camera bearing** about the world +z
axis.  `render.pose_matrix` uses ``azimuth_deg`` for something else entirely --
which way the *robot* leans.  They are unrelated and the collision has already
caused one confusion, so both are named explicitly at every call site.

Nothing here is a rotation implementation: `scipy.spatial.transform.Rotation`
does the work, as it already does in `zeroing.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_PATH = Path(__file__).resolve().parent / "stereo_rig.json"
DEFAULT_INTRINSICS = (
    Path(__file__).resolve().parent / "assets" / "camera_intrinsics.npz"
)

# Nominal working distance, mm.  The middle of the 140-360 mm band that
# `validation/make_dataset.sample_poses` draws from.
DEFAULT_RANGE_MM = 250.0


@dataclass
class Camera:
    """
    One camera: intrinsics, distortion, and pose in the world frame.
    """

    K: np.ndarray
    dist: np.ndarray | None
    T_world_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    name: str = ""

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if self.dist is not None:
            self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        self.T_world_cam = np.asarray(self.T_world_cam, dtype=np.float64).reshape(4, 4)

    @property
    def R(self):
        """
        Rotation taking camera coordinates to world coordinates.
        """

        return self.T_world_cam[:3, :3]

    @property
    def K_inv(self):
        """
        Cached ``inv(K)``.

                Cached because the stereo refinement needs it once per view per residual
                evaluation -- tens of times a frame at 420 Hz -- and inverting a 3x3
                there was measurable next to the frame budget.
        """

        inv = getattr(self, "_K_inv", None)
        if inv is None:
            inv = np.linalg.inv(self.K)
            object.__setattr__(self, "_K_inv", inv)
        return inv

    @property
    def position(self):
        """
        Lens position in world coordinates, mm.
        """

        return self.T_world_cam[:3, 3]

    @property
    def optical_axis(self):
        """
        Unit world vector the camera looks along (its own +z).
        """

        return self.R[:, 2]

    def scaled(self, scale):
        """
        Copy with intrinsics scaled for a resized image.

                Mirrors the inline form ``K[:2, :] *= scale``: fx, fy,
                cx and cy all scale together, distortion coefficients do not -- they are
                defined on normalised coordinates and are resolution-free.
        """

        K = self.K.copy()
        K[:2, :] *= float(scale)
        return Camera(K=K, dist=self.dist, T_world_cam=self.T_world_cam, name=self.name)

    def to_camera(self, center_world, normal_world):
        """
        World ``(center, normal)`` -> this camera's frame.
        """

        R = self.R
        c = np.asarray(center_world, dtype=np.float64).reshape(3)
        n = np.asarray(normal_world, dtype=np.float64).reshape(3)
        return R.T @ (c - self.T_world_cam[:3, 3]), R.T @ n

    def to_world(self, center_cam, normal_cam):
        """
        This camera's frame -> world ``(center, normal)``.
        """

        R = self.R
        c = np.asarray(center_cam, dtype=np.float64).reshape(3)
        n = np.asarray(normal_cam, dtype=np.float64).reshape(3)
        return R @ c + self.T_world_cam[:3, 3], R @ n


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """
    ``T_world_cam`` for a camera at ``eye`` aimed at ``target``.

        Built in OpenCV camera axes: +z toward the target, +y **down** in the image,
        +x to the right.  ``up`` is the world direction that should appear as image
        "up", so the camera's +y is its negation projected perpendicular to the view.

        Degenerates when the view direction is parallel to ``up`` -- looking straight
        down the world +z with ``up = +z`` has no defined roll.  A perpendicular seed
        is substituted there rather than raising, so a top-down rig is constructible;
        the resulting roll is arbitrary but deterministic.
    """

    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    z = target - eye
    nz = np.linalg.norm(z)
    if nz < 1e-12:
        raise ValueError("camera position coincides with its target")
    z = z / nz

    if abs(float(z @ (up / np.linalg.norm(up)))) > 0.999:
        up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    # Image +y points "down", i.e. against world up.
    y = -up + (up @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    T = np.eye(4)
    # Re-orthonormalise through SVD, the same cheap insurance `zeroing.py` takes.
    T[:3, :3] = Rotation.from_matrix(np.column_stack([x, y, z])).as_matrix()
    T[:3, 3] = eye
    return T


def spherical_position(elev_deg, azim_deg, range_mm, target=(0.0, 0.0, 0.0)):
    """
    Camera position from bearing and height angle about the world +z axis.

        ``elev_deg`` is the angle **above the horizontal plane** through the target;
        negative means the camera sits below and looks up.  ``azim_deg`` is the
        bearing about world +z, measured from +x toward +y.
    """

    e, a = math.radians(elev_deg), math.radians(azim_deg)
    offset = range_mm * np.array(
        [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
    )
    return np.asarray(target, dtype=np.float64).reshape(3) + offset


@dataclass
class StereoRig:
    """
    Two cameras in a shared world frame.

        ``cameras`` is ordered; index 0 is "A" and 1 is "B" everywhere downstream,
        including the ``_a`` / ``_b`` CSV column suffixes.
    """

    cameras: tuple
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, i):
        return self.cameras[i]

    @property
    def a(self):
        return self.cameras[0]

    @property
    def b(self):
        return self.cameras[1]

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_spherical(
        cls,
        elev_deg=(45.0, 45.0),
        azim_deg=(0.0, 90.0),
        range_mm=DEFAULT_RANGE_MM,
        K=None,
        dist=None,
        intrinsics_path=DEFAULT_INTRINSICS,
        meta=None,
    ):
        """
        A nominal rig from placement angles, sharing one set of intrinsics.

                The default is the configuration the plan settles on: both cameras 45
                degrees above horizontal, 90 degrees apart in bearing.  That gives 60
                degrees between the optical axes and shows the rotor at 45 degrees of
                tilt in both views -- the balance point between triangulation geometry
                (which wants the axes far apart) and the flat-circle model (which wants
                the rotor near face-on).

                A mixed-hemisphere rig is ``elev_deg=(45, -45)``: optically identical,
                but it splits the occlusion risk, because the takeoff stand blocks the
                view from below and the coils block it from above.  Bearings 0 and 180
                are the configuration to avoid -- they put both cameras in one plane
                with the target, collapsing the baseline.
        """

        if K is None:
            d = np.load(Path(intrinsics_path))
            K = np.asarray(d["camera_matrix"], dtype=np.float64)
            if dist is None:
                dist = np.asarray(d["dist_coeffs"], dtype=np.float64).ravel()

        cams = []
        for i, (e, a) in enumerate(zip(elev_deg, azim_deg)):
            cams.append(
                Camera(
                    K=K,
                    dist=dist,
                    T_world_cam=look_at(spherical_position(e, a, range_mm)),
                    name="AB"[i] if i < 2 else str(i),
                )
            )
        return cls(
            cameras=tuple(cams),
            meta=dict(
                meta or {},
                elev_deg=list(map(float, elev_deg)),
                azim_deg=list(map(float, azim_deg)),
                range_mm=float(range_mm),
                source="from_spherical",
            ),
        )

    @classmethod
    def monocular(cls, K=None, dist=None, intrinsics_path=DEFAULT_INTRINSICS):
        """
        A one-camera rig at the world origin -- the identity case.

                Lets stereo code paths run against a single view for A/B comparison
                without a separate branch.
        """

        if K is None:
            d = np.load(Path(intrinsics_path))
            K = np.asarray(d["camera_matrix"], dtype=np.float64)
            if dist is None:
                dist = np.asarray(d["dist_coeffs"], dtype=np.float64).ravel()
        return cls(
            cameras=(Camera(K=K, dist=dist, name="A"),), meta={"source": "monocular"}
        )

    def scaled(self, scale):
        """
        Copy with every camera's intrinsics scaled for a resized image.
        """

        return StereoRig(
            cameras=tuple(c.scaled(scale) for c in self.cameras),
            meta=dict(self.meta, scale=float(scale)),
        )

    # ---- geometry ---------------------------------------------------------

    def relative(self, i=0, j=1):
        """
        ``T_cj_ci``: maps camera ``i``'s coordinates into camera ``j``'s.

                This is what the renderer needs -- it holds the camera at the origin and
                moves the mesh, so view ``j`` is produced by re-posing the model with
                this transform applied.
        """

        return np.linalg.inv(self.cameras[j].T_world_cam) @ self.cameras[i].T_world_cam

    def axis_separation_deg(self, i=0, j=1):
        """
        Angle between two optical axes, treated as **undirected lines**.

                Undirected because triangulation conditioning depends on the axes'
                directions only up to sign: two cameras facing each other from opposite
                sides have the same depth-axis geometry as two facing the same way.
                Reporting the directed angle would call a perfectly good
                mixed-hemisphere rig "120 degrees apart" when it behaves like 60.
        """

        d = float(
            np.clip(
                abs(self.cameras[i].optical_axis @ self.cameras[j].optical_axis), 0, 1
            )
        )
        return math.degrees(math.acos(d))

    def baseline_mm(self, i=0, j=1):
        return float(
            np.linalg.norm(self.cameras[i].position - self.cameras[j].position)
        )

    def tilt_seen_deg(self, normal_world=(0.0, 0.0, 1.0)):
        """
        Rotor tilt each camera would report for a given world normal.

                Tilt is the angle between the rotor axis and the optical axis, both as
                lines.  Useful for checking a candidate rig against the measured
                good band (roughly 10-45 degrees) before rendering anything.
        """

        n = np.asarray(normal_world, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return [
            math.degrees(math.acos(float(np.clip(abs(n @ c.optical_axis), 0, 1))))
            for c in self.cameras
        ]

    def tilt_information(self, normal_world=(0.0, 0.0, 1.0)):
        """
        Summed ``sin^2`` sensitivity of the axis-ratio channel to tilt.

                The ellipse's axis ratio reads ``|cos(tilt)|``, so its derivative with
                respect to tilt goes as ``sin(tilt)`` and vanishes face-on.  Summing
                ``sin^2`` over the cameras is the Fisher-information trace for that
                channel: it is what stays bounded away from zero once there are two
                views, and it is the quantitative reason the face-on singularity
                disappears.  A single camera at 45 degrees scores 0.5; the default rig
                scores 1.0.

                Only the axis-ratio channel -- the ellipse also carries orientation and
                centre information, so this is a lower bound on what is observable, not
                the whole story.
        """

        n = np.asarray(normal_world, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return float(sum(1.0 - float(n @ c.optical_axis) ** 2 for c in self.cameras))

    def position_covariance(self, sigma_lat_mm, sigma_depth_mm):
        """
        Fused position covariance predicted for per-view error scales.

                Each view is modelled as anisotropic in its own frame -- ``sigma_lat``
                across the optical axis, ``sigma_depth`` along it -- and the views are
                combined in information form.  The point of the function is the
                prediction it makes before any rendering: with a depth-to-lateral ratio
                near 11, the fused worst axis lands near ``sigma_lat`` rather than
                anywhere near ``sigma_depth``.

                Returns the 3x3 covariance in world coordinates.  ``sqrt`` of its
                eigenvalues are the per-axis sigmas.
        """

        info = np.zeros((3, 3))
        eye = np.eye(3)
        for c in self.cameras:
            d = c.optical_axis
            cov = (sigma_lat_mm**2) * eye + (
                sigma_depth_mm**2 - sigma_lat_mm**2
            ) * np.outer(d, d)
            info += np.linalg.inv(cov)
        return np.linalg.inv(info)

    def predicted_sigma_mm(self, sigma_lat_mm, sigma_depth_mm):
        """
        Per-axis fused sigma, ascending. ``[-1]`` is the worst axis.
        """

        return np.sqrt(
            np.linalg.eigvalsh(self.position_covariance(sigma_lat_mm, sigma_depth_mm))
        )

    def summary(self):
        """
        One-line-per-fact description, for provenance headers and logs.
        """

        out = {
            "n_cameras": len(self.cameras),
            "axis_separation_deg": (
                round(self.axis_separation_deg(), 3) if len(self) > 1 else 0.0
            ),
            "baseline_mm": round(self.baseline_mm(), 3) if len(self) > 1 else 0.0,
            "tilt_seen_deg": [round(t, 3) for t in self.tilt_seen_deg()],
            "tilt_information": round(self.tilt_information(), 4),
        }
        out.update({k: v for k, v in self.meta.items() if k not in out})
        return out

    # ---- persistence ------------------------------------------------------

    def save(self, path=DEFAULT_PATH):
        path = Path(path)
        path.write_text(
            json.dumps(
                {
                    "cameras": [
                        {
                            "name": c.name,
                            "K": c.K.tolist(),
                            "dist": None if c.dist is None else c.dist.tolist(),
                            "T_world_cam": c.T_world_cam.tolist(),
                        }
                        for c in self.cameras
                    ],
                    "convention": (
                        "T_world_cam maps camera coords to world; OpenCV camera axes "
                        "(+z forward, +y down); world +z up"
                    ),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **self.meta,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        """
        Load a rig.  Unlike `zeroing.Zero`, a missing file is an error.

                There is no sensible default extrinsic: running stereo against a guessed
                camera-to-camera transform produces confident, wrong answers rather than
                obviously broken ones, which is the worst failure mode available.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no stereo rig at {path}; measure one with "
                f"stereo_calibration.ipynb, or build a nominal one with "
                f"StereoRig.from_spherical(...)"
            )
        d = json.loads(path.read_text())
        cams = tuple(
            Camera(
                K=np.array(c["K"], dtype=np.float64),
                dist=(
                    None
                    if c.get("dist") is None
                    else np.array(c["dist"], dtype=np.float64)
                ),
                T_world_cam=np.array(c["T_world_cam"], dtype=np.float64),
                name=c.get("name", ""),
            )
            for c in d["cameras"]
        )
        known = {"cameras", "convention"}
        return cls(cameras=cams, meta={k: v for k, v in d.items() if k not in known})


def report(
    rig=None,
    elev_deg=(45.0, 45.0),
    azim_deg=(0.0, 90.0),
    range_mm=DEFAULT_RANGE_MM,
    scale=1.0,
    sigma_lat_mm=0.078,
    sigma_depth_mm=0.857,
    out=None,
):
    """
    Print a rig's geometry and its predicted fused precision. Returns the rig.

        For choosing angles before committing to a mount. ``rig`` loads one from
        JSON; otherwise it is built from the spherical description. The default
        per-view sigmas are the held-out values at 1024x768, and they are scaled by
        ``1/scale`` because per-view precision goes inversely with pixel density.
    """

    r = (
        StereoRig.load(rig)
        if rig
        else StereoRig.from_spherical(
            elev_deg=elev_deg, azim_deg=azim_deg, range_mm=range_mm
        )
    )
    if scale != 1.0:
        r = r.scaled(scale)

    for k, v in r.summary().items():
        print(f"{k:24s} {v}")

    s = 1.0 / scale
    sig = r.predicted_sigma_mm(sigma_lat_mm * s, sigma_depth_mm * s)
    print(f"{'per-view sigma_lat_mm':24s} {sigma_lat_mm * s:.4f}")
    print(f"{'per-view sigma_depth_mm':24s} {sigma_depth_mm * s:.4f}")
    print(f"{'fused sigma per axis':24s} {np.array2string(sig, precision=4)}")
    print(f"{'fused worst axis':24s} {sig.max():.4f} mm")

    if out:
        print(f"wrote {r.save(out)}")
    return r

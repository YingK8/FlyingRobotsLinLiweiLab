"""
Express pose relative to a reference image instead of the camera.

Raw output from `conic.py` is in camera coordinates: position in millimetres
from the lens, normal in the camera's axes.  That is rarely what you want to
plot or feed a controller.  What you want is "how far from where it started",
which means picking a datum.

So: point the estimator at a reference image -- the robot sitting on the pad, or
a rendered nominal pose -- and store the pose it reports.  Every later pose is
then reported in that frame, and re-running the estimator on the reference image
reads exactly zero on all six channels.  `test_zeroing.py` asserts precisely
that.

The datum is a full rotation, not just a translation.  It is built from the
reference normal plus the reference in-plane direction, so the tilt azimuth is
pinned too; zeroing only the position would leave phi reading whatever the
camera mounting happened to be.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_PATH = Path(__file__).resolve().parent / "pose_zero.json"


def frame_from_normal(normal, in_plane=None):
    """
    Right-handed rotation whose +z is ``normal``.

        ``in_plane`` optionally fixes the remaining spin about that axis; without it
        an arbitrary but deterministic perpendicular is chosen, so the frame is
        reproducible run to run.

        The result is passed through `Rotation.from_matrix`, which orthonormalises
        via SVD -- cheap insurance that accumulated round-off never yields a matrix
        that is almost, but not quite, a rotation.
    """

    z = np.asarray(normal, dtype=np.float64)
    z = z / np.linalg.norm(z)

    if in_plane is None:
        seed = (
            np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
    else:
        seed = np.asarray(in_plane, dtype=np.float64)
        if (
            np.linalg.norm(np.cross(z, seed)) < 1e-8
        ):  # parallel: unusable as a reference
            seed = (
                np.array([1.0, 0.0, 0.0])
                if abs(z[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )

    x = seed - (seed @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.column_stack([x, y, z])).as_matrix()


@dataclass
class Zero:
    """
    A pose datum: rotation, translation, and the reference in-plane angle.

        ``identity()`` gives the pass-through datum, i.e. report raw camera
        coordinates.  That is what an estimator uses when no zero file is supplied.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    psi_ref_deg: float = 0.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def identity(cls):
        return cls()

    @property
    def is_identity(self):
        return np.allclose(self.R, np.eye(3)) and np.allclose(self.t, 0.0)

    @classmethod
    def from_pose(cls, center, normal, psi_deg=0.0, in_plane=None, meta=None):
        """
        Build a datum from one estimated reference pose.
        """

        return cls(
            R=frame_from_normal(normal, in_plane),
            t=np.asarray(center, dtype=np.float64).reshape(3).copy(),
            psi_ref_deg=float(psi_deg),
            meta=dict(meta or {}),
        )

    def apply(self, center, normal):
        """
        Map a camera-frame pose into the datum frame.

                ``xyz = R' (c - t)`` and ``n = R' n`` -- so at the reference itself the
                position is the zero vector and the normal is +z.
        """

        c = np.asarray(center, dtype=np.float64).reshape(3)
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        return self.R.T @ (c - self.t), self.R.T @ n

    def apply_psi(self, psi_deg):
        """
        Offset an image-plane angle by the reference, wrapped to +-180 deg.
        """

        return (float(psi_deg) - self.psi_ref_deg + 180.0) % 360.0 - 180.0

    def save(self, path=DEFAULT_PATH):
        path = Path(path)
        path.write_text(
            json.dumps(
                {
                    "R": self.R.tolist(),
                    "t": self.t.tolist(),
                    "psi_ref_deg": self.psi_ref_deg,
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
        Load a datum; returns `identity()` if the file is absent.

                Missing is not an error -- running without a zero is a legitimate mode
                (raw camera coordinates), and forcing a calibration step before the
                first run would be annoying.
        """

        path = Path(path)
        if not path.exists():
            return cls.identity()
        d = json.loads(path.read_text())
        known = {"R", "t", "psi_ref_deg"}
        return cls(
            R=np.array(d["R"], dtype=np.float64),
            t=np.array(d["t"], dtype=np.float64),
            psi_ref_deg=float(d.get("psi_ref_deg", 0.0)),
            meta={k: v for k, v in d.items() if k not in known},
        )


def average_poses(centers, normals):
    """
    Mean of several reference observations, for a less noisy datum.

        Positions average directly.  Normals are unit vectors, so they get the
        standard treatment: sum, then renormalise, after flipping any that point
        the opposite way (the estimator's sign choice can flap between frames on a
        near head-on view, and averaging those raw would cancel to nothing).
    """

    c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    ref = n[0]
    n = np.where((n @ ref)[:, None] < 0, -n, n)
    mean_n = n.mean(axis=0)
    norm = np.linalg.norm(mean_n)
    if norm < 1e-9:
        raise ValueError("reference normals do not agree; cannot average")
    return c.mean(axis=0), mean_n / norm

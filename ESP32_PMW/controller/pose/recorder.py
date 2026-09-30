"""
CSV logging for pose runs.

Follows the convention already used in `ai/picoscope_capture.py`: a block of
``# key, value`` comment lines carrying the run's provenance, then a normal
header row.  `pandas.read_csv(path, comment="#")` reads it directly, and the
metadata travels with the data instead of in someone's notes.

Writes are buffered.  At 420 fps a per-row flush would put a syscall in the
control path for no benefit, and an unflushed tail is only ever a few
milliseconds of data -- while `close()` (and the context manager) always
flushes, including on the way out of an exception.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

COLUMNS = [
    "frame",
    "t_capture",
    "t_host",
    "x_mm",
    "y_mm",
    "z_mm",
    "theta_deg",
    "phi_deg",
    "psi_deg",
    "nx",
    "ny",
    "nz",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "ellipse_cx",
    "ellipse_cy",
    "major_px",
    "minor_px",
    "ellipse_deg",
    "area_px",
    "fit_rms_px",
    "ambiguity_margin_deg",
    "n_solutions",
    "jump_deg",
    "discrepancy_mm",
    "margin",
    "refine_rms_px",
    "union_coverage",
    "pred_pos_mm",
    "pred_ang_deg",
    "t_seg_ms",
    "t_est_ms",
    "t_total_ms",
]

#: Columns that only a `stereo.StereoPose` carries.  A monocular `estimator.Pose`
#: leaves them blank rather than absent, so one reader handles both logs.
#:
#: `discrepancy_mm` earns its place twice over: it is the two views disagreeing,
#: which is an estimate of the error that needs no stationary robot and no ground
#: truth at all -- see `noise.py`.
STEREO_COLUMNS = (
    "discrepancy_mm",
    "margin",
    "refine_rms_px",
    "union_coverage",
    "pred_pos_mm",
    "pred_ang_deg",
)


def _stereo_cells(pose):
    """
    The stereo-only columns for ``pose``, blank on anything that lacks them.
    """

    out = []
    for name in STEREO_COLUMNS:
        v = getattr(pose, name, None)
        try:
            out.append("" if v is None or not np.isfinite(v) else f"{float(v):.4f}")
        except TypeError:
            out.append("")
    return out


def write_metadata(fh, meta):
    """
    Emit the ``# key, value`` provenance block.
    """

    fh.write(
        f"# generated, {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    )
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            v = np.array2string(
                v.ravel(), precision=6, separator=" ", max_line_width=10**6
            )
        fh.write(f"# {k}, {v}\n")


class PoseRecorder:
    """
    Append `estimator.Pose` rows to a CSV.

        Lost frames are written too, with the pose columns blank.  A gap in the file
        is a real event -- the robot left frame, or segmentation failed -- and
        silently dropping those rows would make a run look cleaner than it was and
        hide dropout bursts from anyone reading the log later.
    """

    def __init__(self, path, meta=None, flush_every=200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="")
        write_metadata(self._fh, meta or {})
        self._w = csv.writer(self._fh)
        self._w.writerow(COLUMNS)
        self._flush_every = flush_every
        self.n_rows = 0
        self.n_lost = 0

    def write(self, pose, t_capture=None, frame_index=None, velocity=None):
        """
        Record one frame. ``pose`` may be ``None`` for a lost frame.

                ``velocity`` is the filtered rate from `filter.PoseFilter`; blank when no
                filter is running. It is logged separately from position because the two
                come from different places -- position is best taken raw, velocity is
                only usable filtered. See `filter.py`.
        """

        if pose is None:
            self.n_lost += 1
            row = [frame_index if frame_index is not None else "", t_capture or ""]
            row += [""] * (len(COLUMNS) - len(row))
            self._w.writerow(row)
        else:
            (ecx, ecy), (major, minor), edeg = pose.ellipse
            x, y, z = pose.xyz_mm
            nx, ny, nz = pose.normal
            self._w.writerow(
                [
                    pose.frame_index,
                    "" if t_capture is None else f"{t_capture:.6f}",
                    f"{pose.t:.6f}",
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{z:.4f}",
                    f"{pose.theta_deg:.4f}",
                    f"{pose.phi_deg:.4f}",
                    f"{pose.psi_deg:.4f}",
                    f"{nx:.6f}",
                    f"{ny:.6f}",
                    f"{nz:.6f}",
                    *(
                        ["", "", ""]
                        if velocity is None
                        else [f"{v:.4f}" for v in velocity]
                    ),
                    f"{ecx:.3f}",
                    f"{ecy:.3f}",
                    f"{major:.3f}",
                    f"{minor:.3f}",
                    f"{edeg:.3f}",
                    f"{pose.area_px:.1f}",
                    f"{pose.fit_rms_px:.4f}",
                    f"{pose.ambiguity_margin_deg:.3f}",
                    pose.n_solutions,
                    "" if not np.isfinite(pose.jump_deg) else f"{pose.jump_deg:.3f}",
                    *_stereo_cells(pose),
                    f"{pose.t_seg_ms:.4f}",
                    f"{pose.t_est_ms:.4f}",
                    f"{pose.t_total_ms:.4f}",
                ]
            )

        self.n_rows += 1
        if self.n_rows % self._flush_every == 0:
            self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

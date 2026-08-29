#!/usr/bin/env python3
"""
Measure the estimator's own noise on a robot that is not moving.

Every measurement-noise number the pipeline uses today is a render or a floor:
`filter.py`'s sigmas come from a held-out split of synthetic data, `stereo.py`'s
fusion weights are the Cramer-Rao bound of `theory.md` S13, and `z_track.py`
prices its step-out against an assumed 0.5 mm per frame. A stationary target
settles all of them at once -- whatever scatter comes back **is** the noise,
because the truth is constant.

**The scatter is not white, and that is the whole difficulty.** Depth
autocorrelates at r = 0.966 from one frame to the next and stays above 0.5 for
408 ms (`filter.py`). The sample standard deviation of a static run is therefore
not a Kalman R: it is the total wander, most of which is a slowly varying bias
no filter removes. Two numbers are reported for every channel, and on this
pipeline they differ by a lot:

    sigma_total   the stationary scatter. **Ships as R** -- under-trusting a
                  correlated measurement is safe, over-trusting it lets the
                  filter chase a bias it cannot remove.
    sigma_white   MAD of the frame-to-frame difference over sqrt(2). What a
                  differencing velocity estimator actually sees, and the only
                  part that averaging can remove.

Where:
    z_mm       range, the median of the run. Depth noise is quoted against it
    rho1       lag-1 autocorrelation of the residual
    tau_s      correlation time, -dt/ln(rho1) for an AR(1)
    n_eff      independent samples, N (1-rho)/(1+rho); the error bar on sigma
    sigma_lat  per-view error across the optical axis, mm
    sigma_dep  per-view error along it, mm

Capture needs no new code. A static session is a flight in which nothing moves:

    python camera/record.py --out results/static      # one take per height
    python pose/noise.py --from results/static        # fit and write the model
    python pose/noise.py --show                       # against the CRLB floor

**Several heights, not one.** The shipped model is `sigma_depth = frac * z`, and
one station cannot separate that fraction from a constant. Three or four across
the envelope fit it, and the fitted exponent says whether the shape holds at all.

Recorded coils-off, so this is the *vision* noise floor: it excludes
drive-induced vibration and EMI. The artifact records that in `condition`, and a
coils-on set can be added later and differenced against this one.

Self-check: uv run python controller/pose/noise.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it. pose is stage 3 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

DEFAULT_PATH = HERE / "noise_model.json"

#: 1.4826 turns a median absolute deviation into a standard deviation for a
#: Gaussian.  MAD first, std second: a static run still drops the occasional
#: blunder frame (`theory.md` 16.23) and one of those moves a std but not a
#: median.  Both are reported -- their ratio is the contamination figure
#: `theory.md` 13.7 already tabulates for the boundary.
MAD_TO_SIGMA = 1.4826

#: Fallbacks, used when no measured model is on disk.  These are the values the
#: pipeline shipped before this module existed, and they are **rendered, not
#: measured** -- `filter.py` for the first three, `stereo.py` for the ratio.
FALLBACK_SIGMA_LAT_MM = 0.078
FALLBACK_SIGMA_DEPTH_MM = 0.857
FALLBACK_SIGMA_LATERAL_MM = 0.13
FALLBACK_SIGMA_DEPTH_FRAC = 0.0036
FALLBACK_SIGMA_NORMAL = 0.022

#: Range the fallback per-view depth scale was quoted at, so the fallback can be
#: expressed as a fraction like a measured one.
FALLBACK_REF_Z_MM = 250.0


def _robust_sigma(v):
    """
    ``1.4826 * MAD``, the Gaussian-equivalent scale that a blunder frame cannot move.
    """

    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def _channel(v, dt):
    """
    Scatter, correlation and effective sample count for one static channel.

        ``sigma_white`` comes from the first difference rather than from a fitted
        AR(1): differencing kills any drift of any shape, and for white noise
        ``var(x_t - x_{t-1}) = 2 sigma^2`` exactly.  It is the honest answer to
        "how much of this scatter could averaging ever remove", which the total
        scatter on its own cannot answer.
    """

    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3:
        return {"n": int(n)}

    r = v - np.median(v)
    sig = _robust_sigma(v)
    d = np.diff(v)

    # Lag-1 autocorrelation on the residual. Guarded: a channel that never moves
    # (a stuck value, a constant) has zero variance and no defined correlation.
    var = float(np.mean(r * r))
    rho = float(np.mean(r[1:] * r[:-1]) / var) if var > 0 else float("nan")
    rho = min(max(rho, -0.999), 0.999) if np.isfinite(rho) else float("nan")

    # tau from an AR(1): x_t = rho x_{t-1} + e, so the envelope decays as
    # exp(-t/tau) with tau = -dt/ln(rho). Only meaningful for positive rho --
    # an anticorrelated channel is not drifting, it is alternating.
    tau = -dt / math.log(rho) if np.isfinite(rho) and rho > 0 else 0.0

    # Independent samples for the mean of a correlated series. This is the error
    # bar on sigma itself: at rho = 0.966, 1000 frames are worth 17.
    n_eff = n * (1.0 - rho) / (1.0 + rho) if np.isfinite(rho) else float(n)

    return {
        "n": int(n),
        "centre": float(np.median(v)),
        "sigma": sig,
        "std": float(np.std(v)),
        "sigma_white": _robust_sigma(d) / math.sqrt(2.0),
        "rho1": rho,
        "tau_s": float(tau),
        "n_eff": float(max(n_eff, 1.0)),
    }


def measure(t, xyz, normal=None, meta=None):
    """
    One station: the scatter of a run in which the robot did not move.

        ``t`` seconds, ``xyz`` (N,3) mm in the datum frame, ``normal`` (N,3) unit.
        Rows with a lost frame are expected to be dropped by the caller -- a blank
        row is a real event but it is not a measurement.

        The 3x3 position covariance is kept whole.  `filter.py` models R as
        ``(lat, lat, frac*z)``, which is the *monocular* geometry; two cameras 90
        degrees apart in azimuth do not give a covariance aligned to the world
        axes, and the off-diagonal terms are the evidence for that.
    """

    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if xyz.shape[0] < 3:
        raise ValueError(f"a station needs at least 3 poses, got {xyz.shape[0]}")

    dt = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    axes = {name: _channel(xyz[:, i], dt) for i, name in enumerate("xyz")}

    centre = np.median(xyz, axis=0)
    cov = np.cov((xyz - centre).T)
    # Ascending, so [-1] is the worst axis -- the same convention as
    # `rig.predicted_sigma_mm`, which this is compared against.
    sigma_axes = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0.0, None))

    station = {
        "n": int(xyz.shape[0]),
        "dt_s": dt,
        "fps": (1.0 / dt) if dt and np.isfinite(dt) and dt > 0 else float("nan"),
        "z_mm": float(centre[2]),
        "range_mm": float(np.linalg.norm(centre)),
        "centre_mm": [float(c) for c in centre],
        "axes": axes,
        "cov_pos": [[float(c) for c in row] for row in cov],
        "sigma_axes_mm": [float(s) for s in sigma_axes],
    }

    if normal is not None:
        n = np.asarray(normal, dtype=float)
        mean_n = np.median(n, axis=0)
        mean_n = mean_n / max(np.linalg.norm(mean_n), 1e-12)
        # Scatter in the tangent plane only. The component along the mean normal
        # is fixed by |n| = 1 and carries no information; pooling it in would
        # dilute the number `filter.py` consumes as an isotropic sigma.
        basis = np.linalg.svd(mean_n.reshape(1, 3))[2][1:]
        tang = (n - mean_n) @ basis.T
        station["normal"] = {
            "centre": [float(c) for c in mean_n],
            "sigma": float(np.hypot(*[_robust_sigma(tang[:, i]) for i in (0, 1)])
                           / math.sqrt(2.0)),
            "sigma_deg": float("nan"),
            **{f"t{i}": _channel(tang[:, i], dt) for i in (0, 1)},
        }
        station["normal"]["sigma_deg"] = float(
            math.degrees(math.asin(min(station["normal"]["sigma"], 1.0)))
        )

    station.update(meta or {})
    return station


def per_view_scales(sigma_axes_mm, rig=None):
    """
    Invert the fusion: per-view ``(sigma_lat, sigma_depth)`` from the fused scatter.

        `rig.position_covariance` predicts the fused 3x3 from the two per-view
        scales, and this runs it backwards.  The map is homogeneous of degree one --
        doubling both scales doubles every fused sigma -- so the *shape* of the
        eigenvalue triple fixes the ratio and one scalar then fixes the size.  That
        splits a two-parameter fit into two one-parameter ones, neither of which
        needs an optimiser that can fail.

        Returns ``(sigma_lat_mm, sigma_depth_mm)``, or the fallbacks when no rig is
        available -- the ratio is a property of where the cameras are, and guessing
        it from a single fused number is not possible.
    """

    meas = np.sort(np.asarray(sigma_axes_mm, dtype=float))
    if rig is None or not np.all(np.isfinite(meas)) or meas[-1] <= 0:
        return FALLBACK_SIGMA_LAT_MM, FALLBACK_SIGMA_DEPTH_MM

    def shape_err(log_ratio):
        pred = np.sort(rig.predicted_sigma_mm(1.0, math.exp(log_ratio)))
        if pred[-1] <= 0:
            return np.inf
        return float(np.sum((pred / pred[-1] - meas / meas[-1]) ** 2))

    # Ratios from 1 (isotropic) to ~150. The rig's own geometric prediction is
    # near 11 (`rig.position_covariance`), so this brackets it by an order of
    # magnitude either side. A scan, not a solver: 200 points is instant, the
    # objective is one 3x3 eigendecomposition, and a scan cannot land in a local
    # minimum of a curve nobody has proved unimodal.
    hi = math.log(150.0)
    grid = np.linspace(0.0, hi, 200)
    best = min(grid, key=shape_err)
    if best >= grid[-2] or best <= grid[1]:
        # The scan hit its own edge, so the shape of the measured scatter is not one
        # this rig can produce at any ratio. On a static station that means the robot
        # moved, or the run is dominated by something that is not per-view noise;
        # returning the boundary value as if it were a fit would hide that.
        print(f"per-view fit hit the scan edge (ratio {math.exp(best):.1f}): the "
              f"measured anisotropy {np.round(meas / meas[-1], 3)} is not a shape "
              f"this rig makes. Not a per-view noise fit.")
    pred = np.sort(rig.predicted_sigma_mm(1.0, math.exp(best)))
    scale = float(meas[-1] / pred[-1])
    return scale, scale * math.exp(best)


def _powerlaw(z, s):
    """
    Exponent and coefficient of ``s = c z^p``, fitted in log space.

        Reported, not shipped.  The derivation in `theory.md` 12 makes lateral
        error go as ``z`` and depth as ``z^2``; the shipped model in `filter.py` is
        depth linear in ``z`` and lateral flat.  An exponent far from the derived
        one says the shape is wrong, which is worth knowing even while the shipped
        shape stays put.
    """

    z = np.asarray(z, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(z) & np.isfinite(s) & (z > 0) & (s > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    p, c = np.polyfit(np.log(z[ok]), np.log(s[ok]), 1)
    return float(p), float(math.exp(c))


@dataclass
class NoiseModel:
    """
    Measured per-frame measurement noise, and what the consumers ask of it.

        The shipped parameterisation is the one `filter.py` and `stereo.py`
        already use -- a per-view lateral scale that does not vary with range and
        a depth scale that is a fraction of it -- so adopting a measured model
        changes numbers, not code paths.  `exponents` records what the data
        actually says about that shape.
    """

    sigma_lat_mm: float = FALLBACK_SIGMA_LAT_MM
    sigma_depth_frac: float = FALLBACK_SIGMA_DEPTH_MM / FALLBACK_REF_Z_MM
    sigma_lateral_mm: float = FALLBACK_SIGMA_LATERAL_MM
    sigma_depth_frac_world: float = FALLBACK_SIGMA_DEPTH_FRAC
    sigma_normal: float = FALLBACK_SIGMA_NORMAL
    sigma_white: dict = field(default_factory=dict)
    rho1: dict = field(default_factory=dict)
    tau_s: dict = field(default_factory=dict)
    exponents: dict = field(default_factory=dict)
    stations: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    #: True when this is the built-in fallback rather than anything measured.
    #: Consumers print it; nothing should silently believe a rendered number is a
    #: bench measurement.
    measured: bool = False

    @property
    def ref_z_mm(self):
        """
        Representative range, the median of the recorded stations.

            `stereo.fuse` wants the two per-view scales as fixed millimetres, not as
            a function of range, so the depth scale has to be evaluated somewhere.
            The middle of the envelope that was actually measured is the least wrong
            single choice available.
        """

        z = [abs(s.get("z_mm", 0.0)) for s in self.stations]
        z = [v for v in z if v > 0]
        return float(np.median(z)) if z else FALLBACK_REF_Z_MM

    def sigma_depth_mm(self, z_mm):
        """Per-view depth scale at a range, the ``frac * z`` the fusion weights want."""

        return self.sigma_depth_frac * max(abs(float(z_mm)), 1.0)

    def sigma_pos(self, z_mm, rig=None):
        """
        Measurement noise for `filter.PoseFilter`, as a triple or a full 3x3.

            With a rig, `rig.position_covariance` builds the fused covariance the
            two cameras actually produce, which is not aligned to the world axes.
            Without one -- the monocular path -- fall back to the axis-aligned
            triple `filter.py` has always used.
        """

        z = max(abs(float(z_mm)), 1.0)
        if rig is None:
            return (self.sigma_lateral_mm, self.sigma_lateral_mm,
                    self.sigma_depth_frac_world * z)
        return rig.position_covariance(self.sigma_lat_mm, self.sigma_depth_mm(z))

    def velocity_sigma_mm_s(self, axis="z", tau_s=0.08, dt=1.0 / 60.0):
        """
        Noise on a differenced-and-low-passed rate, which is what control consumes.

            A first-order low-pass of a *white* difference gives
            ``sigma_v = sigma_w sqrt(2) / dt * sqrt(a / (2 - a))`` with
            ``a = dt / (tau + dt)``.  Only ``sigma_white`` belongs here: the
            correlated part of the position error is common to both samples of the
            difference and cancels out of it, which is precisely why it does not
            appear as velocity noise however large it is in position.
        """

        sw = float(self.sigma_white.get(axis, float("nan")))
        a = dt / (tau_s + dt)
        return sw * math.sqrt(2.0) / dt * math.sqrt(a / (2.0 - a))

    def tau_for_velocity_sigma(self, target_mm_s, axis="z", dt=1.0 / 60.0):
        """
        The low-pass time constant that buys ``target_mm_s`` of rate noise.

            `velocity_sigma_mm_s` run backwards, for choosing `z_track.tau_zdot` and
            `predictor.TAU_VEL_S` from the measurement instead of from taste.

            **Longer is not free, and past the correlation time it is not even
            effective.** The formula assumes the position error is white between
            samples; the correlated part of it is common to both ends of the
            difference and never reaches the rate at all. So a tau far beyond
            ``tau_s[axis]`` buys lag and nothing else. Returns ``inf`` when the
            target is below what the white noise alone permits.
        """

        sw = float(self.sigma_white.get(axis, float("nan")))
        if not np.isfinite(sw) or sw <= 0 or target_mm_s <= 0:
            return float("nan")
        k = float(target_mm_s) * dt / (sw * math.sqrt(2.0))
        a = 2.0 * k**2 / (1.0 + k**2)
        if a >= 1.0:
            return 0.0          # even an unfiltered difference is quiet enough
        return dt / a - dt

    @classmethod
    def fit(cls, stations, rig=None, meta=None):
        """
        Fit the shipped parameterisation across stations at different heights.
        """

        if not stations:
            raise ValueError("no stations to fit")

        z = np.array([s["z_mm"] for s in stations], dtype=float)
        pv = [per_view_scales(s["sigma_axes_mm"], rig) for s in stations]
        lat = np.array([p[0] for p in pv], dtype=float)
        dep = np.array([p[1] for p in pv], dtype=float)

        # Depth as a fraction of range, through the origin: that is the shape
        # `filter.py` consumes. Least squares on frac alone, not a two-parameter
        # line -- an intercept the shipped model has no slot for would be fitted
        # and then silently dropped. The exponent below is where a bad shape shows.
        good = np.isfinite(z) & np.isfinite(dep) & (z > 0)
        frac = float((z[good] @ dep[good]) / (z[good] @ z[good])) if good.any() else 0.0

        def pooled(key, sub="sigma_white"):
            vals = [s["axes"][key].get(sub) for s in stations
                    if key in s.get("axes", {})]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")

        lat_s = np.array([np.hypot(s["axes"]["x"]["sigma"], s["axes"]["y"]["sigma"])
                          / math.sqrt(2.0) for s in stations])
        dep_s = np.array([s["axes"]["z"]["sigma"] for s in stations])
        world_lat = float(np.median(lat_s))
        world_frac = float(np.median(dep_s / np.maximum(np.abs(z), 1.0)))
        normals = [s["normal"]["sigma"] for s in stations if "normal" in s]

        # Exponents off the measured world channels, not off the per-view
        # inversion: that inversion needs a rig and returns a constant without
        # one, which would report a confident z^0 for every axis. The diagnostic
        # has to work on the data alone.
        lat_p, _ = _powerlaw(np.abs(z), lat_s)
        dep_p, _ = _powerlaw(np.abs(z), dep_s)

        return cls(
            sigma_lat_mm=float(np.median(lat)),
            sigma_depth_frac=frac,
            sigma_lateral_mm=world_lat,
            sigma_depth_frac_world=world_frac,
            sigma_normal=float(np.median(normals)) if normals else FALLBACK_SIGMA_NORMAL,
            sigma_white={k: pooled(k) for k in "xyz"},
            rho1={k: pooled(k, "rho1") for k in "xyz"},
            tau_s={k: pooled(k, "tau_s") for k in "xyz"},
            exponents={"lateral": lat_p, "depth": dep_p},
            stations=list(stations),
            meta=dict(meta or {}),
            measured=True,
        )

    def save(self, path=DEFAULT_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sigma_lat_mm": self.sigma_lat_mm,
                    "sigma_depth_frac": self.sigma_depth_frac,
                    "sigma_lateral_mm": self.sigma_lateral_mm,
                    "sigma_depth_frac_world": self.sigma_depth_frac_world,
                    "sigma_normal": self.sigma_normal,
                    "sigma_white": self.sigma_white,
                    "rho1": self.rho1,
                    "tau_s": self.tau_s,
                    "exponents": self.exponents,
                    "model": (
                        "static-target scatter; per-view lateral flat in range and "
                        "depth proportional to it, inverted from the fused covariance "
                        "through rig.position_covariance. sigma is 1.4826*MAD of the "
                        "run (ships as R); sigma_white is MAD of the frame-to-frame "
                        "difference over sqrt(2)"
                    ),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n_stations": len(self.stations),
                    "n_samples": int(sum(s.get("n", 0) for s in self.stations)),
                    "stations": self.stations,
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
        Load the measured model; **a missing file means the rendered fallbacks**.

            Not an error, and deliberately not `StereoRig.load`'s refusal.  A rig
            that is guessed gives confident wrong answers, so that one raises.  A
            noise model that is guessed gives the behaviour the pipeline had
            before this file existed, which is a known, shipped, documented state
            -- and it has to be available, because the first static run has to be
            estimated by something.
        """

        path = Path(path)
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        known = {
            "sigma_lat_mm", "sigma_depth_frac", "sigma_lateral_mm",
            "sigma_depth_frac_world", "sigma_normal", "sigma_white", "rho1",
            "tau_s", "exponents", "stations",
        }
        return cls(
            **{k: d[k] for k in known if k in d},
            meta={k: v for k, v in d.items() if k not in known},
            measured=True,
        )

    def summary(self):
        """One line per fact, for provenance headers and logs."""

        src = "measured" if self.measured else "FALLBACK (rendered, not measured)"
        out = [
            f"noise model: {src}, {len(self.stations)} station(s), "
            f"{self.meta.get('condition', 'condition unrecorded')}",
            f"  per-view    lateral {self.sigma_lat_mm:.4f} mm, "
            f"depth {self.sigma_depth_frac:.5f} * z "
            f"({self.sigma_depth_mm(250.0):.3f} mm at 250 mm)",
            f"  world       lateral {self.sigma_lateral_mm:.4f} mm, "
            f"depth {self.sigma_depth_frac_world:.5f} * z",
            f"  normal      {self.sigma_normal:.4f} "
            f"({math.degrees(math.asin(min(self.sigma_normal, 1.0))):.2f} deg)",
        ]
        if self.rho1:
            out.append("  correlation " + ", ".join(
                f"{k} rho1 {self.rho1[k]:.3f} tau {self.tau_s.get(k, 0):.3f} s"
                for k in "xyz" if k in self.rho1 and np.isfinite(self.rho1[k])))
        if self.sigma_white:
            out.append("  white part  " + ", ".join(
                f"{k} {self.sigma_white[k]:.4f} mm" for k in "xyz"
                if k in self.sigma_white and np.isfinite(self.sigma_white[k])))
        if self.exponents:
            out.append(
                f"  exponents   lateral z^{self.exponents.get('lateral', float('nan')):.2f}"
                f" (shipped z^0), depth z^{self.exponents.get('depth', float('nan')):.2f}"
                f" (shipped z^1)")
        return "\n".join(out)


def station_from_csv(path, meta=None):
    """
    One station from a `recorder.PoseRecorder` log of a stationary run.

        Lost frames are written as blank rows on purpose (`recorder.py`), and they
        are dropped here: a dropout is a real event but it is not a measurement of
        anything.  How many were dropped is kept, because a static run that loses
        frames is telling you about the bench.
    """

    import csv

    path = Path(path)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(r for r in fh if not r.startswith("#")))
    t, xyz, nrm = [], [], []
    for r in rows:
        if not r.get("x_mm"):
            continue
        t.append(float(r["t_host"]))
        xyz.append([float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])])
        nrm.append([float(r["nx"]), float(r["ny"]), float(r["nz"])])
    if len(xyz) < 3:
        raise ValueError(f"{path}: only {len(xyz)} solved poses, not a station")
    return measure(t, xyz, nrm, meta=dict(
        meta or {}, source=str(path), n_lost=len(rows) - len(xyz)))


def _collect(est, read, n_frames, filt=None, progress=True):
    """
    Drive a source through the estimator until ``n_frames`` poses land.

        ``read`` returns ``(t, frames)`` or ``None`` at the end.  Shared by the live
        and the offline path so the two measure the same thing -- the offline twin
        of a live run has to agree with it or the number means nothing.
    """

    t, xyz, nrm, seen, lost = [], [], [], 0, 0
    while len(xyz) < n_frames:
        item = read()
        if item is None:
            break
        seen += 1
        t_cap, frames = item
        pose = est.update(frames, t=t_cap, frame_index=seen)
        if filt is not None:
            filt.update(pose, t=t_cap)
        if pose is None:
            lost += 1
            continue
        t.append(float(pose.t))
        xyz.append([float(v) for v in pose.xyz_mm])
        nrm.append([float(v) for v in pose.normal])
        if progress and len(xyz) % 100 == 0:
            print(f"  {len(xyz)}/{n_frames} poses, {lost} lost", flush=True)
    return t, xyz, nrm, seen, lost


def station_from_recording(rec_dir, n_frames=100000, rig_path=None, meta=None):
    """
    One station from a `camera/record.py` take in which the robot did not move.

        Runs the same estimator the live path runs, headless -- no viser, no
        window.  `live_viz.from_recording` would do this too, but it starts a
        viewer and a web server to fit three numbers.
    """

    sys.path.insert(0, str(HERE.parent / "viz"))
    from live_viz import _stereo_estimator
    from record import latest_flight, open_recording

    rec_dir = latest_flight(Path(rec_dir))
    rig, est = _stereo_estimator(rig_path, backgrounds="running")
    caps, stamps = open_recording(rec_dir)
    if stamps is None:
        print(f"{rec_dir}: no frames.csv, so the two views are assumed simultaneous")

    i = 0

    def read():
        nonlocal i
        got = [c.read() for c in caps]
        if not all(ok for ok, _ in got):
            return None
        # The recorder's own per-camera capture times, so an offline station and a
        # live one price the inter-camera skew identically (`theory.md` 17).
        row = stamps[i] if stamps is not None and i < len(stamps) else None
        i += 1
        t = float(np.mean(row)) if row is not None else i / 60.0
        return t, [f for _, f in got]

    try:
        t, xyz, nrm, seen, lost = _collect(est, read, n_frames)
    finally:
        for c in caps:
            c.release()
    print(f"{rec_dir.name}: {seen} frames, {len(xyz)} solved, {lost} lost")
    return measure(t, xyz, nrm, meta=dict(
        meta or {}, source=str(rec_dir), n_lost=lost, n_frames=seen)), rig


def station_from_source(spec="camera:0,camera:1", n_frames=600, rig_path=None,
                        width=1280, height=800, rotate180=True, meta=None):
    """
    One station straight off the cameras, with the robot clamped and not moving.
    """

    sys.path.insert(0, str(HERE.parent / "viz"))
    import sources
    from live_viz import _stereo_estimator

    rig, est = _stereo_estimator(rig_path, backgrounds="running")
    cams = [c.strip() for c in spec.split(",")]
    src = sources.open_stereo(cams, max_skew_s=None, width=width, height=height,
                              grayscale=True, rotate180=rotate180)
    try:
        t, xyz, nrm, seen, lost = _collect(est, src.read, n_frames)
    finally:
        src.close()
    print(f"{seen} frames, {len(xyz)} solved, {lost} lost")
    return measure(t, xyz, nrm, meta=dict(
        meta or {}, source=spec, n_lost=lost, n_frames=seen)), rig


def report(model, rig=None, radius_mm=None):
    """
    The measured model against the floors, which is the only way to read it.

        A sigma on its own says nothing: `theory.md` 13.7 puts the rendered
        pipeline at 23x the photon bound, so a bench run near that is healthy and
        one far above it is a bench problem -- focus, exposure, a partly occluded
        rim -- rather than a worse estimator.
    """

    import bounds

    lines = [model.summary()]
    if rig is not None:
        pred = rig.predicted_sigma_mm(model.sigma_lat_mm,
                                      model.sigma_depth_mm(250.0))
        lines.append("  fused (rig) " + ", ".join(f"{s:.4f}" for s in pred)
                     + " mm per axis at 250 mm, worst last")

    for st in model.stations:
        z = st["z_mm"]
        meas = st["sigma_axes_mm"]
        line = (f"  station z={z:7.1f} mm  n={st['n']:5d}  "
                f"axes " + " ".join(f"{s:.4f}" for s in meas) + " mm")
        if rig is not None and radius_mm is not None:
            try:
                b = bounds.budget(abs(z), radius_mm, rig.cameras[0].K,
                                  tilt_deg=30.0)
                floor = b["photon_sigma_px"]
                line += f"   rim {2 * b['semi_major_px']:.0f} px, photon {floor:.4f} px"
            except Exception as e:          # a floor that cannot be computed is not fatal
                line += f"   (no floor: {e})"
        lines.append(line)
    return "\n".join(lines)


def calibrate(sources_, out=DEFAULT_PATH, condition="coils_off", rig_path=None,
              n_frames=600, live=False):
    """
    Record or re-read the stations, fit, and write the artifact.

        ``sources_`` is a list of recording directories, CSV logs, or camera specs.
        This is the whole calibration step: the pipeline loads what it writes.
    """

    stations, rig = [], None
    for s in sources_:
        p = Path(s)
        if live or not p.exists():
            st, rig = station_from_source(str(s), n_frames=n_frames,
                                          rig_path=rig_path)
        elif p.is_dir():
            st, rig = station_from_recording(p, rig_path=rig_path)
        else:
            st = station_from_csv(p)
        stations.append(st)

    if rig is None:
        try:
            import rig as rigmod
            rig = rigmod.StereoRig.load()
        except Exception:
            rig = None

    model = NoiseModel.fit(stations, rig=rig, meta={
        "condition": condition,
        "source": "static_capture",
        "note": ("coils off: this is the vision noise floor and excludes "
                 "drive-induced vibration and EMI"),
    })
    path = model.save(out)
    print(report(model, rig))
    print(f"\nwrote {path}")
    if len(stations) < 3:
        print(f"\n{len(stations)} station(s): sigma_depth_frac is not identified by "
              "fewer than\nthree heights. The fraction written is the best fit to "
              "what was given, and\nthe exponents above are not meaningful.")
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--from", dest="src", nargs="+", default=None,
                   help="recording dirs or poses.csv logs, one per height")
    p.add_argument("--record", action="store_true",
                   help="capture live instead; --cameras gives the spec")
    p.add_argument("--cameras", default="camera:0,camera:1")
    p.add_argument("--stations", type=int, default=3,
                   help="how many heights to prompt for when recording live")
    p.add_argument("--frames", type=int, default=600, help="poses per station")
    p.add_argument("--condition", default="coils_off",
                   help="what the drive was doing; recorded in the artifact")
    p.add_argument("--rig", type=Path, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_PATH)
    p.add_argument("--show", action="store_true", help="print the saved model, fit nothing")
    a = p.parse_args(argv)

    if a.show:
        model = NoiseModel.load(a.out)
        try:
            import rig as rigmod
            r = rigmod.StereoRig.load()
        except Exception:
            r = None
        from estimator import RADIUS_BENCH_MM
        print(report(model, r, RADIUS_BENCH_MM))
        return 0

    if a.record:
        stations, rig = [], None
        for i in range(a.stations):
            input(f"\nstation {i + 1}/{a.stations}: set the height, hold the robot "
                  f"still, press ENTER ")
            st, rig = station_from_source(a.cameras, n_frames=a.frames,
                                          rig_path=a.rig)
            stations.append(st)
            print(f"  z = {st['z_mm']:.1f} mm, axes "
                  + " ".join(f"{s:.4f}" for s in st["sigma_axes_mm"]) + " mm")
        model = NoiseModel.fit(stations, rig=rig, meta={
            "condition": a.condition, "source": "static_capture_live"})
        print(report(model, rig))
        print(f"\nwrote {model.save(a.out)}")
        return 0

    if not a.src:
        p.error("give --from with the static takes, or --record to shoot them")
    calibrate(a.src, out=a.out, condition=a.condition, rig_path=a.rig,
              n_frames=a.frames)
    return 0


def _ar1(n, sigma, rho, rng):
    """
    A stationary AR(1) series with the requested marginal sigma and lag-1 rho.

        The innovation is scaled by ``sqrt(1 - rho^2)`` so that the *marginal*
        standard deviation is ``sigma`` whatever rho is.  Without that the test
        would be measuring the innovation and the correlated and white cases would
        not be comparable.
    """

    e = rng.normal(0.0, sigma * math.sqrt(1.0 - rho**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def _check():
    rng = np.random.default_rng(20260828)
    dt, n = 1.0 / 60.0, 4000

    # 1. White noise: the total scatter and the white part must agree, because
    #    for rho = 0 there is nothing for averaging to remove.
    t = np.arange(n) * dt
    xyz = np.stack([_ar1(n, 0.2, 0.0, rng) for _ in range(3)], axis=1)
    st = measure(t, xyz + [0.0, 0.0, 250.0])
    for k in "xyz":
        c = st["axes"][k]
        assert abs(c["sigma"] - 0.2) < 0.02, (k, c["sigma"])
        assert abs(c["sigma_white"] - 0.2) < 0.02, (k, c["sigma_white"])
        assert abs(c["rho1"]) < 0.05, (k, c["rho1"])
    print("white noise: sigma 0.2 recovered, white == total, rho ~ 0\n  ok")

    # 2. Correlated noise: the total scatter is still 0.2, but the white part is
    #    far smaller and the effective sample count collapses. This is the case
    #    the real depth channel is in, and the reason both numbers are reported.
    rho = 0.966
    xyz = np.stack([_ar1(n, 0.2, rho, rng) for _ in range(3)], axis=1)
    st = measure(t, xyz + [0.0, 0.0, 250.0])
    c = st["axes"]["z"]
    assert abs(c["sigma"] - 0.2) < 0.05, c["sigma"]
    assert abs(c["rho1"] - rho) < 0.02, c["rho1"]
    assert c["sigma_white"] < 0.25 * c["sigma"], (c["sigma_white"], c["sigma"])
    assert c["n_eff"] < 0.05 * n, c["n_eff"]
    tau_expected = -dt / math.log(rho)
    assert abs(c["tau_s"] - tau_expected) < 0.3 * tau_expected, c["tau_s"]
    print(f"correlated: rho {c['rho1']:.3f}, tau {c['tau_s'] * 1e3:.0f} ms, "
          f"white {c['sigma_white']:.3f} vs total {c['sigma']:.3f} mm, "
          f"{n} frames worth {c['n_eff']:.0f}\n  ok")

    # 3. The fit across heights recovers a depth fraction. Built in the world
    #    frame with no rig, so it exercises the world-frame path the monocular
    #    filter uses; the per-view inversion needs a rig and is checked by
    #    `report` against `rig.predicted_sigma_mm` on real data.
    frac, lat = 0.004, 0.15
    stations = []
    for z in (150.0, 250.0, 350.0):
        xyz = np.stack([
            _ar1(n, lat, 0.5, rng),
            _ar1(n, lat, 0.5, rng),
            _ar1(n, frac * z, 0.9, rng) + z,
        ], axis=1)
        stations.append(measure(t, xyz))
    m = NoiseModel.fit(stations)
    assert abs(m.sigma_depth_frac_world - frac) < 0.2 * frac, m.sigma_depth_frac_world
    assert abs(m.sigma_lateral_mm - lat) < 0.2 * lat, m.sigma_lateral_mm
    assert abs(m.exponents["depth"] - 1.0) < 0.3, m.exponents
    print(f"fit: depth frac {m.sigma_depth_frac_world:.5f} (built {frac}), "
          f"lateral {m.sigma_lateral_mm:.4f} mm (built {lat}), "
          f"depth exponent {m.exponents['depth']:.2f} (built 1.0)\n  ok")

    # 3b. The per-view inversion, against the forward map it inverts. Two cameras
    #     90 degrees apart in azimuth, which is this rig's geometry.
    class _Rig:
        axes = (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))

        def position_covariance(self, lat, dep):
            info = np.zeros((3, 3))
            for d in self.axes:
                cov = lat**2 * np.eye(3) + (dep**2 - lat**2) * np.outer(d, d)
                info += np.linalg.inv(cov)
            return np.linalg.inv(info)

        def predicted_sigma_mm(self, lat, dep):
            return np.sqrt(np.linalg.eigvalsh(self.position_covariance(lat, dep)))

    r = _Rig()
    for lat, dep in ((0.08, 0.86), (0.2, 2.0), (0.05, 5.0)):
        got = per_view_scales(r.predicted_sigma_mm(lat, dep), r)
        assert abs(got[0] / lat - 1) < 0.05, (lat, dep, got)
        assert abs(got[1] / dep - 1) < 0.05, (lat, dep, got)
    print("per-view inversion: (lat, depth) recovered from the fused axes\n  ok")

    # An anisotropy no ratio can make must say so rather than return the boundary.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        per_view_scales([1.0, 1.0, 1.0e6], r)
    assert "scan edge" in buf.getvalue(), buf.getvalue()
    print("per-view inversion: an impossible shape is reported, not fitted\n  ok")

    # 3c. tau <-> rate noise round-trips, which is what the control retune reads.
    m.sigma_white = {"z": 0.05}
    for tau in (0.02, 0.08, 0.30):
        sv = m.velocity_sigma_mm_s("z", tau_s=tau)
        back = m.tau_for_velocity_sigma(sv, "z")
        assert abs(back - tau) < 1e-6 * max(tau, 1.0), (tau, sv, back)
    quiet = m.velocity_sigma_mm_s("z", tau_s=0.30)
    loud = m.velocity_sigma_mm_s("z", tau_s=0.02)
    assert quiet < loud, (quiet, loud)
    print(f"tau <-> rate noise: round-trips; 20 ms gives {loud:.1f} mm/s, "
          f"300 ms gives {quiet:.1f} mm/s\n  ok")

    # 4. Round trip through the artifact, and the missing-file policy.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "noise_model.json"
        assert not NoiseModel.load(p).measured, "a missing file must give fallbacks"
        m.save(p)
        back = NoiseModel.load(p)
        assert back.measured
        assert abs(back.sigma_depth_frac - m.sigma_depth_frac) < 1e-12
        assert len(back.stations) == 3
        assert back.meta["model"], "the payload must say what it is"
    print("round trip: saved, reloaded, missing file falls back\n  ok")

    # 5. sigma_pos gives the triple filter.PoseFilter consumes, and it scales.
    s150, s300 = m.sigma_pos(150.0), m.sigma_pos(300.0)
    assert abs(s300[2] / s150[2] - 2.0) < 1e-9, (s150, s300)
    assert s150[0] == s150[1] == m.sigma_lateral_mm
    print("sigma_pos: lateral flat, depth doubles with range\n  ok")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    _check()
    print("\nall checks passed")
